// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// End-to-end serving tests

package http_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/http/internal"
	"net/url"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

type dummyAddr string
type oneConnListener struct {
	conn net.Conn
}

func (l *oneConnListener) Accept() (c net.Conn, err error) {
	c = l.conn
	if c == nil {
		err = io.EOF
		return
	}
	err = nil
	l.conn = nil
	return
}

func (l *oneConnListener) Close() error {
	return nil
}

func (l *oneConnListener) Addr() net.Addr {
	return dummyAddr("test-address")
}

func (a dummyAddr) Network() string {
	return string(a)
}

func (a dummyAddr) String() string {
	return string(a)
}

type noopConn struct{}

func (noopConn) LocalAddr() net.Addr                { return dummyAddr("local-addr") }
func (noopConn) RemoteAddr() net.Addr               { return dummyAddr("remote-addr") }
func (noopConn) SetDeadline(t time.Time) error      { return nil }
func (noopConn) SetReadDeadline(t time.Time) error  { return nil }
func (noopConn) SetWriteDeadline(t time.Time) error { return nil }

type rwTestConn struct {
	io.Reader
	io.Writer
	noopConn

	closeFunc func() error // called if non-nil
	closec    chan bool    // else, if non-nil, send value to it on close
}

func (c *rwTestConn) Close() error {
	if c.closeFunc != nil {
		return c.closeFunc()
	}
	select {
	case c.closec <- true:
	default:
	}
	return nil
}

type testConn struct {
	readMu   sync.Mutex // for TestHandlerBodyClose
	readBuf  bytes.Buffer
	writeBuf bytes.Buffer
	closec   chan bool // if non-nil, send value to it on close
	noopConn
}

func (c *testConn) Read(b []byte) (int, error) {
	c.readMu.Lock()
	defer c.readMu.Unlock()
	return c.readBuf.Read(b)
}

func (c *testConn) Write(b []byte) (int, error) {
	return c.writeBuf.Write(b)
}

func (c *testConn) Close() error {
	select {
	case c.closec <- true:
	default:
	}
	return nil
}

// reqBytes treats req as a request (with \n delimiters) and returns it with \r\n delimiters,
// ending in \r\n\r\n
func reqBytes(req string) []byte {
	return []byte(strings.ReplaceAll(strings.TrimSpace(req), "\n", "\r\n") + "\r\n\r\n")
}

type handlerTest struct {
	logbuf  bytes.Buffer
	handler Handler
}

func newHandlerTest(h Handler) handlerTest {
	return handlerTest{handler: h}
}

func (ht *handlerTest) rawResponse(req string) string {
	reqb := reqBytes(req)
	var output bytes.Buffer
	conn := &rwTestConn{
		Reader: bytes.NewReader(reqb),
		Writer: &output,
		closec: make(chan bool, 1),
	}
	ln := &oneConnListener{conn: conn}
	srv := &Server{
		ErrorLog: log.New(&ht.logbuf, "", 0),
		Handler:  ht.handler,
	}
	go srv.Serve(ln)
	<-conn.closec
	return output.String()
}

func TestConsumingBodyOnNextConn(t *testing.T) {
	t.Parallel()
	defer afterTest(t)
	conn := new(testConn)
	for i := 0; i < 2; i++ {
		conn.readBuf.Write([]byte(
			"POST / HTTP/1.1\r\n" +
				"Host: test\r\n" +
				"Content-Length: 11\r\n" +
				"\r\n" +
				"foo=1&bar=1"))
	}

	reqNum := 0
	ch := make(chan *Request)
	servech := make(chan error)
	listener := &oneConnListener{conn}
	handler := func(res ResponseWriter, req *Request) {
		reqNum++
		ch <- req
	}

	go func() {
		servech <- Serve(listener, HandlerFunc(handler))
	}()

	var req *Request
	req = <-ch
	if req == nil {
		t.Fatal("Got nil first request.")
	}
	if req.Method != "POST" {
		t.Errorf("For request #1's method, got %q; expected %q",
			req.Method, "POST")
	}

	req = <-ch
	if req == nil {
		t.Fatal("Got nil first request.")
	}
	if req.Method != "POST" {
		t.Errorf("For request #2's method, got %q; expected %q",
			req.Method, "POST")
	}

	if serveerr := <-servech; serveerr != io.EOF {
		t.Errorf("Serve returned %q; expected EOF", serveerr)
	}
}

type stringHandler string

func (s stringHandler) ServeHTTP(w ResponseWriter, r *Request) {
	w.Header().Set("Result", string(s))
}

var handlers = []struct {
	pattern string
	msg     string
}{
	{"/", "Default"},
	{"/someDir/", "someDir"},
	{"/#/", "hash"},
	{"someHost.com/someDir/", "someHost.com/someDir"},
}

var vtests = []struct {
	url      string
	expected string
}{
	{"http://localhost/someDir/apage", "someDir"},
	{"http://localhost/%23/apage", "hash"},
	{"http://localhost/otherDir/apage", "Default"},
	{"http://someHost.com/someDir/apage", "someHost.com/someDir"},
	{"http://otherHost.com/someDir/apage", "someDir"},
	{"http://otherHost.com/aDir/apage", "Default"},
	// redirections for trees
	{"http://localhost/someDir", "/someDir/"},
	{"http://localhost/%23", "/%23/"},
	{"http://someHost.com/someDir", "/someDir/"},
}

func TestHostHandlers(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	mux := NewServeMux()
	for _, h := range handlers {
		mux.Handle(h.pattern, stringHandler(h.msg))
	}
	ts := httptest.NewServer(mux)
	defer ts.Close()

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	cc := httputil.NewClientConn(conn, nil)
	for _, vt := range vtests {
		var r *Response
		var req Request
		if req.URL, err = url.Parse(vt.url); err != nil {
			t.Errorf("cannot parse url: %v", err)
			continue
		}
		if err := cc.Write(&req); err != nil {
			t.Errorf("writing request: %v", err)
			continue
		}
		r, err := cc.Read(&req)
		if err != nil {
			t.Errorf("reading response: %v", err)
			continue
		}
		switch r.StatusCode {
		case StatusOK:
			s := r.Header.Get("Result")
			if s != vt.expected {
				t.Errorf("Get(%q) = %q, want %q", vt.url, s, vt.expected)
			}
		case StatusMovedPermanently:
			s := r.Header.Get("Location")
			if s != vt.expected {
				t.Errorf("Get(%q) = %q, want %q", vt.url, s, vt.expected)
			}
		default:
			t.Errorf("Get(%q) unhandled status code %d", vt.url, r.StatusCode)
		}
	}
}

var serveMuxRegister = []struct {
	pattern string
	h       Handler
}{
	{"/dir/", serve(200)},
	{"/search", serve(201)},
	{"codesearch.google.com/search", serve(202)},
	{"codesearch.google.com/", serve(203)},
	{"example.com/", HandlerFunc(checkQueryStringHandler)},
}

// serve returns a handler that sends a response with the given code.
func serve(code int) HandlerFunc {
	return func(w ResponseWriter, r *Request) {
		w.WriteHeader(code)
	}
}

// checkQueryStringHandler checks if r.URL.RawQuery has the same value
// as the URL excluding the scheme and the query string and sends 200
// response code if it is, 500 otherwise.
func checkQueryStringHandler(w ResponseWriter, r *Request) {
	u := *r.URL
	u.Scheme = "http"
	u.Host = r.Host
	u.RawQuery = ""
	if "http://"+r.URL.RawQuery == u.String() {
		w.WriteHeader(200)
	} else {
		w.WriteHeader(500)
	}
}

var serveMuxTests = []struct {
	method  string
	host    string
	path    string
	code    int
	pattern string
}{
	{"GET", "google.com", "/", 404, ""},
	{"GET", "google.com", "/dir", 301, "/dir/"},
	{"GET", "google.com", "/dir/", 200, "/dir/"},
	{"GET", "google.com", "/dir/file", 200, "/dir/"},
	{"GET", "google.com", "/search", 201, "/search"},
	{"GET", "google.com", "/search/", 404, ""},
	{"GET", "google.com", "/search/foo", 404, ""},
	{"GET", "codesearch.google.com", "/search", 202, "codesearch.google.com/search"},
	{"GET", "codesearch.google.com", "/search/", 203, "codesearch.google.com/"},
	{"GET", "codesearch.google.com", "/search/foo", 203, "codesearch.google.com/"},
	{"GET", "codesearch.google.com", "/", 203, "codesearch.google.com/"},
	{"GET", "codesearch.google.com:443", "/", 203, "codesearch.google.com/"},
	{"GET", "images.google.com", "/search", 201, "/search"},
	{"GET", "images.google.com", "/search/", 404, ""},
	{"GET", "images.google.com", "/search/foo", 404, ""},
	{"GET", "google.com", "/../search", 301, "/search"},
	{"GET", "google.com", "/dir/..", 301, ""},
	{"GET", "google.com", "/dir/..", 301, ""},
	{"GET", "google.com", "/dir/./file", 301, "/dir/"},

	// The /foo -> /foo/ redirect applies to CONNECT requests
	// but the path canonicalization does not.
	{"CONNECT", "google.com", "/dir", 301, "/dir/"},
	{"CONNECT", "google.com", "/../search", 404, ""},
	{"CONNECT", "google.com", "/dir/..", 200, "/dir/"},
	{"CONNECT", "google.com", "/dir/..", 200, "/dir/"},
	{"CONNECT", "google.com", "/dir/./file", 200, "/dir/"},
}

func TestServeMuxHandler(t *testing.T) {
	setParallel(t)
	mux := NewServeMux()
	for _, e := range serveMuxRegister {
		mux.Handle(e.pattern, e.h)
	}

	for _, tt := range serveMuxTests {
		r := &Request{
			Method: tt.method,
			Host:   tt.host,
			URL: &url.URL{
				Path: tt.path,
			},
		}
		h, pattern := mux.Handler(r)
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, r)
		if pattern != tt.pattern || rr.Code != tt.code {
			t.Errorf("%s %s %s = %d, %q, want %d, %q", tt.method, tt.host, tt.path, rr.Code, pattern, tt.code, tt.pattern)
		}
	}
}

// Issue 24297
func TestServeMuxHandleFuncWithNilHandler(t *testing.T) {
	setParallel(t)
	defer func() {
		if err := recover(); err == nil {
			t.Error("expected call to mux.HandleFunc to panic")
		}
	}()
	mux := NewServeMux()
	mux.HandleFunc("/", nil)
}

var serveMuxTests2 = []struct {
	method  string
	host    string
	url     string
	code    int
	redirOk bool
}{
	{"GET", "google.com", "/", 404, false},
	{"GET", "example.com", "/test/?example.com/test/", 200, false},
	{"GET", "example.com", "test/?example.com/test/", 200, true},
}

// TestServeMuxHandlerRedirects tests that automatic redirects generated by
// mux.Handler() shouldn't clear the request's query string.
func TestServeMuxHandlerRedirects(t *testing.T) {
	setParallel(t)
	mux := NewServeMux()
	for _, e := range serveMuxRegister {
		mux.Handle(e.pattern, e.h)
	}

	for _, tt := range serveMuxTests2 {
		tries := 1 // expect at most 1 redirection if redirOk is true.
		turl := tt.url
		for {
			u, e := url.Parse(turl)
			if e != nil {
				t.Fatal(e)
			}
			r := &Request{
				Method: tt.method,
				Host:   tt.host,
				URL:    u,
			}
			h, _ := mux.Handler(r)
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, r)
			if rr.Code != 301 {
				if rr.Code != tt.code {
					t.Errorf("%s %s %s = %d, want %d", tt.method, tt.host, tt.url, rr.Code, tt.code)
				}
				break
			}
			if !tt.redirOk {
				t.Errorf("%s %s %s, unexpected redirect", tt.method, tt.host, tt.url)
				break
			}
			turl = rr.HeaderMap.Get("Location")
			tries--
		}
		if tries < 0 {
			t.Errorf("%s %s %s, too many redirects", tt.method, tt.host, tt.url)
		}
	}
}

// Tests for https://golang.org/issue/900
func TestMuxRedirectLeadingSlashes(t *testing.T) {
	setParallel(t)
	paths := []string{"//foo.txt", "///foo.txt", "/../../foo.txt"}
	for _, path := range paths {
		req, err := ReadRequest(bufio.NewReader(strings.NewReader("GET " + path + " HTTP/1.1\r\nHost: test\r\n\r\n")))
		if err != nil {
			t.Errorf("%s", err)
		}
		mux := NewServeMux()
		resp := httptest.NewRecorder()

		mux.ServeHTTP(resp, req)

		if loc, expected := resp.Header().Get("Location"), "/foo.txt"; loc != expected {
			t.Errorf("Expected Location header set to %q; got %q", expected, loc)
			return
		}

		if code, expected := resp.Code, StatusMovedPermanently; code != expected {
			t.Errorf("Expected response code of StatusMovedPermanently; got %d", code)
			return
		}
	}
}

// Test that the special cased "/route" redirect
// implicitly created by a registered "/route/"
// properly sets the query string in the redirect URL.
// See Issue 17841.
func TestServeWithSlashRedirectKeepsQueryString(t *testing.T) {
	setParallel(t)
	defer afterTest(t)

	writeBackQuery := func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%s", r.URL.RawQuery)
	}

	mux := NewServeMux()
	mux.HandleFunc("/testOne", writeBackQuery)
	mux.HandleFunc("/testTwo/", writeBackQuery)
	mux.HandleFunc("/testThree", writeBackQuery)
	mux.HandleFunc("/testThree/", func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%s:bar", r.URL.RawQuery)
	})

	ts := httptest.NewServer(mux)
	defer ts.Close()

	tests := [...]struct {
		path     string
		method   string
		want     string
		statusOk bool
	}{
		0: {"/testOne?this=that", "GET", "this=that", true},
		1: {"/testTwo?foo=bar", "GET", "foo=bar", true},
		2: {"/testTwo?a=1&b=2&a=3", "GET", "a=1&b=2&a=3", true},
		3: {"/testTwo?", "GET", "", true},
		4: {"/testThree?foo", "GET", "foo", true},
		5: {"/testThree/?foo", "GET", "foo:bar", true},
		6: {"/testThree?foo", "CONNECT", "foo", true},
		7: {"/testThree/?foo", "CONNECT", "foo:bar", true},

		// canonicalization or not
		8: {"/testOne/foo/..?foo", "GET", "foo", true},
		9: {"/testOne/foo/..?foo", "CONNECT", "404 page not found\n", false},
	}

	for i, tt := range tests {
		req, _ := NewRequest(tt.method, ts.URL+tt.path, nil)
		res, err := ts.Client().Do(req)
		if err != nil {
			continue
		}
		slurp, _ := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if !tt.statusOk {
			if got, want := res.StatusCode, 404; got != want {
				t.Errorf("#%d: Status = %d; want = %d", i, got, want)
			}
		}
		if got, want := string(slurp), tt.want; got != want {
			t.Errorf("#%d: Body = %q; want = %q", i, got, want)
		}
	}
}

func TestServeWithSlashRedirectForHostPatterns(t *testing.T) {
	setParallel(t)
	defer afterTest(t)

	mux := NewServeMux()
	mux.Handle("example.com/pkg/foo/", stringHandler("example.com/pkg/foo/"))
	mux.Handle("example.com/pkg/bar", stringHandler("example.com/pkg/bar"))
	mux.Handle("example.com/pkg/bar/", stringHandler("example.com/pkg/bar/"))
	mux.Handle("example.com:3000/pkg/connect/", stringHandler("example.com:3000/pkg/connect/"))
	mux.Handle("example.com:9000/", stringHandler("example.com:9000/"))
	mux.Handle("/pkg/baz/", stringHandler("/pkg/baz/"))

	tests := []struct {
		method string
		url    string
		code   int
		loc    string
		want   string
	}{
		{"GET", "http://example.com/", 404, "", ""},
		{"GET", "http://example.com/pkg/foo", 301, "/pkg/foo/", ""},
		{"GET", "http://example.com/pkg/bar", 200, "", "example.com/pkg/bar"},
		{"GET", "http://example.com/pkg/bar/", 200, "", "example.com/pkg/bar/"},
		{"GET", "http://example.com/pkg/baz", 301, "/pkg/baz/", ""},
		{"GET", "http://example.com:3000/pkg/foo", 301, "/pkg/foo/", ""},
		{"CONNECT", "http://example.com/", 404, "", ""},
		{"CONNECT", "http://example.com:3000/", 404, "", ""},
		{"CONNECT", "http://example.com:9000/", 200, "", "example.com:9000/"},
		{"CONNECT", "http://example.com/pkg/foo", 301, "/pkg/foo/", ""},
		{"CONNECT", "http://example.com:3000/pkg/foo", 404, "", ""},
		{"CONNECT", "http://example.com:3000/pkg/baz", 301, "/pkg/baz/", ""},
		{"CONNECT", "http://example.com:3000/pkg/connect", 301, "/pkg/connect/", ""},
	}

	ts := httptest.NewServer(mux)
	defer ts.Close()

	for i, tt := range tests {
		req, _ := NewRequest(tt.method, tt.url, nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)

		if got, want := w.Code, tt.code; got != want {
			t.Errorf("#%d: Status = %d; want = %d", i, got, want)
		}

		if tt.code == 301 {
			if got, want := w.HeaderMap.Get("Location"), tt.loc; got != want {
				t.Errorf("#%d: Location = %q; want = %q", i, got, want)
			}
		} else {
			if got, want := w.HeaderMap.Get("Result"), tt.want; got != want {
				t.Errorf("#%d: Result = %q; want = %q", i, got, want)
			}
		}
	}
}

func TestShouldRedirectConcurrency(t *testing.T) {
	setParallel(t)
	defer afterTest(t)

	mux := NewServeMux()
	ts := httptest.NewServer(mux)
	defer ts.Close()
	mux.HandleFunc("/", func(w ResponseWriter, r *Request) {})
}

func BenchmarkServeMux(b *testing.B)           { benchmarkServeMux(b, true) }
func BenchmarkServeMux_SkipServe(b *testing.B) { benchmarkServeMux(b, false) }
func benchmarkServeMux(b *testing.B, runHandler bool) {
	type test struct {
		path string
		code int
		req  *Request
	}

	// Build example handlers and requests
	var tests []test
	endpoints := []string{"search", "dir", "file", "change", "count", "s"}
	for _, e := range endpoints {
		for i := 200; i < 230; i++ {
			p := fmt.Sprintf("/%s/%d/", e, i)
			tests = append(tests, test{
				path: p,
				code: i,
				req:  &Request{Method: "GET", Host: "localhost", URL: &url.URL{Path: p}},
			})
		}
	}
	mux := NewServeMux()
	for _, tt := range tests {
		mux.Handle(tt.path, serve(tt.code))
	}

	rw := httptest.NewRecorder()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, tt := range tests {
			*rw = httptest.ResponseRecorder{}
			h, pattern := mux.Handler(tt.req)
			if runHandler {
				h.ServeHTTP(rw, tt.req)
				if pattern != tt.path || rw.Code != tt.code {
					b.Fatalf("got %d, %q, want %d, %q", rw.Code, pattern, tt.code, tt.path)
				}
			}
		}
	}
}

func TestServerTimeouts(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	// Try three times, with increasing timeouts.
	tries := []time.Duration{250 * time.Millisecond, 500 * time.Millisecond, 1 * time.Second}
	for i, timeout := range tries {
		err := testServerTimeouts(timeout)
		if err == nil {
			return
		}
		t.Logf("failed at %v: %v", timeout, err)
		if i != len(tries)-1 {
			t.Logf("retrying at %v ...", tries[i+1])
		}
	}
	t.Fatal("all attempts failed")
}

func testServerTimeouts(timeout time.Duration) error {
	reqNum := 0
	ts := httptest.NewUnstartedServer(HandlerFunc(func(res ResponseWriter, req *Request) {
		reqNum++
		fmt.Fprintf(res, "req=%d", reqNum)
	}))
	ts.Config.ReadTimeout = timeout
	ts.Config.WriteTimeout = timeout
	ts.Start()
	defer ts.Close()

	// Hit the HTTP server successfully.
	c := ts.Client()
	r, err := c.Get(ts.URL)
	if err != nil {
		return fmt.Errorf("http Get #1: %v", err)
	}
	got, err := ioutil.ReadAll(r.Body)
	expected := "req=1"
	if string(got) != expected || err != nil {
		return fmt.Errorf("Unexpected response for request #1; got %q ,%v; expected %q, nil",
			string(got), err, expected)
	}

	// Slow client that should timeout.
	t1 := time.Now()
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		return fmt.Errorf("Dial: %v", err)
	}
	buf := make([]byte, 1)
	n, err := conn.Read(buf)
	conn.Close()
	latency := time.Since(t1)
	if n != 0 || err != io.EOF {
		return fmt.Errorf("Read = %v, %v, wanted %v, %v", n, err, 0, io.EOF)
	}
	minLatency := timeout / 5 * 4
	if latency < minLatency {
		return fmt.Errorf("got EOF after %s, want >= %s", latency, minLatency)
	}

	// Hit the HTTP server successfully again, verifying that the
	// previous slow connection didn't run our handler.  (that we
	// get "req=2", not "req=3")
	r, err = c.Get(ts.URL)
	if err != nil {
		return fmt.Errorf("http Get #2: %v", err)
	}
	got, err = ioutil.ReadAll(r.Body)
	r.Body.Close()
	expected = "req=2"
	if string(got) != expected || err != nil {
		return fmt.Errorf("Get #2 got %q, %v, want %q, nil", string(got), err, expected)
	}

	if !testing.Short() {
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			return fmt.Errorf("long Dial: %v", err)
		}
		defer conn.Close()
		go io.Copy(ioutil.Discard, conn)
		for i := 0; i < 5; i++ {
			_, err := conn.Write([]byte("GET / HTTP/1.1\r\nHost: foo\r\n\r\n"))
			if err != nil {
				return fmt.Errorf("on write %d: %v", i, err)
			}
			time.Sleep(timeout / 2)
		}
	}
	return nil
}

// Test that the HTTP/2 server handles Server.WriteTimeout (Issue 18437)
func TestHTTP2WriteDeadlineExtendedOnNewRequest(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(res ResponseWriter, req *Request) {}))
	ts.Config.WriteTimeout = 250 * time.Millisecond
	ts.TLS = &tls.Config{NextProtos: []string{"h2"}}
	ts.StartTLS()
	defer ts.Close()

	c := ts.Client()
	if err := ExportHttp2ConfigureTransport(c.Transport.(*Transport)); err != nil {
		t.Fatal(err)
	}

	for i := 1; i <= 3; i++ {
		req, err := NewRequest("GET", ts.URL, nil)
		if err != nil {
			t.Fatal(err)
		}

		// fail test if no response after 1 second
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()
		req = req.WithContext(ctx)

		r, err := c.Do(req)
		if ctx.Err() == context.DeadlineExceeded {
			t.Fatalf("http2 Get #%d response timed out", i)
		}
		if err != nil {
			t.Fatalf("http2 Get #%d: %v", i, err)
		}
		r.Body.Close()
		if r.ProtoMajor != 2 {
			t.Fatalf("http2 Get expected HTTP/2.0, got %q", r.Proto)
		}
		time.Sleep(ts.Config.WriteTimeout / 2)
	}
}

// tryTimeouts runs testFunc with increasing timeouts. Test passes on first success,
// and fails if all timeouts fail.
func tryTimeouts(t *testing.T, testFunc func(timeout time.Duration) error) {
	tries := []time.Duration{250 * time.Millisecond, 500 * time.Millisecond, 1 * time.Second}
	for i, timeout := range tries {
		err := testFunc(timeout)
		if err == nil {
			return
		}
		t.Logf("failed at %v: %v", timeout, err)
		if i != len(tries)-1 {
			t.Logf("retrying at %v ...", tries[i+1])
		}
	}
	t.Fatal("all attempts failed")
}

// Test that the HTTP/2 server RSTs stream on slow write.
func TestHTTP2WriteDeadlineEnforcedPerStream(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	setParallel(t)
	defer afterTest(t)
	tryTimeouts(t, testHTTP2WriteDeadlineEnforcedPerStream)
}

func testHTTP2WriteDeadlineEnforcedPerStream(timeout time.Duration) error {
	reqNum := 0
	ts := httptest.NewUnstartedServer(HandlerFunc(func(res ResponseWriter, req *Request) {
		reqNum++
		if reqNum == 1 {
			return // first request succeeds
		}
		time.Sleep(timeout) // second request times out
	}))
	ts.Config.WriteTimeout = timeout / 2
	ts.TLS = &tls.Config{NextProtos: []string{"h2"}}
	ts.StartTLS()
	defer ts.Close()

	c := ts.Client()
	if err := ExportHttp2ConfigureTransport(c.Transport.(*Transport)); err != nil {
		return fmt.Errorf("ExportHttp2ConfigureTransport: %v", err)
	}

	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		return fmt.Errorf("NewRequest: %v", err)
	}
	r, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("http2 Get #1: %v", err)
	}
	r.Body.Close()
	if r.ProtoMajor != 2 {
		return fmt.Errorf("http2 Get expected HTTP/2.0, got %q", r.Proto)
	}

	req, err = NewRequest("GET", ts.URL, nil)
	if err != nil {
		return fmt.Errorf("NewRequest: %v", err)
	}
	r, err = c.Do(req)
	if err == nil {
		r.Body.Close()
		if r.ProtoMajor != 2 {
			return fmt.Errorf("http2 Get expected HTTP/2.0, got %q", r.Proto)
		}
		return fmt.Errorf("http2 Get #2 expected error, got nil")
	}
	expected := "stream ID 3; INTERNAL_ERROR" // client IDs are odd, second stream should be 3
	if !strings.Contains(err.Error(), expected) {
		return fmt.Errorf("http2 Get #2: expected error to contain %q, got %q", expected, err)
	}
	return nil
}

// Test that the HTTP/2 server does not send RST when WriteDeadline not set.
func TestHTTP2NoWriteDeadline(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	setParallel(t)
	defer afterTest(t)
	tryTimeouts(t, testHTTP2NoWriteDeadline)
}

func testHTTP2NoWriteDeadline(timeout time.Duration) error {
	reqNum := 0
	ts := httptest.NewUnstartedServer(HandlerFunc(func(res ResponseWriter, req *Request) {
		reqNum++
		if reqNum == 1 {
			return // first request succeeds
		}
		time.Sleep(timeout) // second request timesout
	}))
	ts.TLS = &tls.Config{NextProtos: []string{"h2"}}
	ts.StartTLS()
	defer ts.Close()

	c := ts.Client()
	if err := ExportHttp2ConfigureTransport(c.Transport.(*Transport)); err != nil {
		return fmt.Errorf("ExportHttp2ConfigureTransport: %v", err)
	}

	for i := 0; i < 2; i++ {
		req, err := NewRequest("GET", ts.URL, nil)
		if err != nil {
			return fmt.Errorf("NewRequest: %v", err)
		}
		r, err := c.Do(req)
		if err != nil {
			return fmt.Errorf("http2 Get #%d: %v", i, err)
		}
		r.Body.Close()
		if r.ProtoMajor != 2 {
			return fmt.Errorf("http2 Get expected HTTP/2.0, got %q", r.Proto)
		}
	}
	return nil
}

// golang.org/issue/4741 -- setting only a write timeout that triggers
// shouldn't cause a handler to block forever on reads (next HTTP
// request) that will never happen.
func TestOnlyWriteTimeout(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	var (
		mu   sync.RWMutex
		conn net.Conn
	)
	var afterTimeoutErrc = make(chan error, 1)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, req *Request) {
		buf := make([]byte, 512<<10)
		_, err := w.Write(buf)
		if err != nil {
			t.Errorf("handler Write error: %v", err)
			return
		}
		mu.RLock()
		defer mu.RUnlock()
		if conn == nil {
			t.Error("no established connection found")
			return
		}
		conn.SetWriteDeadline(time.Now().Add(-30 * time.Second))
		_, err = w.Write(buf)
		afterTimeoutErrc <- err
	}))
	ts.Listener = trackLastConnListener{ts.Listener, &mu, &conn}
	ts.Start()
	defer ts.Close()

	c := ts.Client()

	errc := make(chan error)
	go func() {
		res, err := c.Get(ts.URL)
		if err != nil {
			errc <- err
			return
		}
		_, err = io.Copy(ioutil.Discard, res.Body)
		res.Body.Close()
		errc <- err
	}()
	select {
	case err := <-errc:
		if err == nil {
			t.Errorf("expected an error from Get request")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for Get error")
	}
	if err := <-afterTimeoutErrc; err == nil {
		t.Error("expected write error after timeout")
	}
}

// trackLastConnListener tracks the last net.Conn that was accepted.
type trackLastConnListener struct {
	net.Listener

	mu   *sync.RWMutex
	last *net.Conn // destination
}

func (l trackLastConnListener) Accept() (c net.Conn, err error) {
	c, err = l.Listener.Accept()
	if err == nil {
		l.mu.Lock()
		*l.last = c
		l.mu.Unlock()
	}
	return
}

// TestIdentityResponse verifies that a handler can unset
func TestIdentityResponse(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	handler := HandlerFunc(func(rw ResponseWriter, req *Request) {
		rw.Header().Set("Content-Length", "3")
		rw.Header().Set("Transfer-Encoding", req.FormValue("te"))
		switch {
		case req.FormValue("overwrite") == "1":
			_, err := rw.Write([]byte("foo TOO LONG"))
			if err != ErrContentLength {
				t.Errorf("expected ErrContentLength; got %v", err)
			}
		case req.FormValue("underwrite") == "1":
			rw.Header().Set("Content-Length", "500")
			rw.Write([]byte("too short"))
		default:
			rw.Write([]byte("foo"))
		}
	})

	ts := httptest.NewServer(handler)
	defer ts.Close()

	c := ts.Client()

	// Note: this relies on the assumption (which is true) that
	// Get sends HTTP/1.1 or greater requests. Otherwise the
	// server wouldn't have the choice to send back chunked
	// responses.
	for _, te := range []string{"", "identity"} {
		url := ts.URL + "/?te=" + te
		res, err := c.Get(url)
		if err != nil {
			t.Fatalf("error with Get of %s: %v", url, err)
		}
		if cl, expected := res.ContentLength, int64(3); cl != expected {
			t.Errorf("for %s expected res.ContentLength of %d; got %d", url, expected, cl)
		}
		if cl, expected := res.Header.Get("Content-Length"), "3"; cl != expected {
			t.Errorf("for %s expected Content-Length header of %q; got %q", url, expected, cl)
		}
		if tl, expected := len(res.TransferEncoding), 0; tl != expected {
			t.Errorf("for %s expected len(res.TransferEncoding) of %d; got %d (%v)",
				url, expected, tl, res.TransferEncoding)
		}
		res.Body.Close()
	}

	// Verify that ErrContentLength is returned
	url := ts.URL + "/?overwrite=1"
	res, err := c.Get(url)
	if err != nil {
		t.Fatalf("error with Get of %s: %v", url, err)
	}
	res.Body.Close()

	// Verify that the connection is closed when the declared Content-Length
	// is larger than what the handler wrote.
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("error dialing: %v", err)
	}
	_, err = conn.Write([]byte("GET /?underwrite=1 HTTP/1.1\r\nHost: foo\r\n\r\n"))
	if err != nil {
		t.Fatalf("error writing: %v", err)
	}

	// The ReadAll will hang for a failing test, so use a Timer to
	// fail explicitly.
	goTimeout(t, 2*time.Second, func() {
		got, _ := ioutil.ReadAll(conn)
		expectedSuffix := "\r\n\r\ntoo short"
		if !strings.HasSuffix(string(got), expectedSuffix) {
			t.Errorf("Expected output to end with %q; got response body %q",
				expectedSuffix, string(got))
		}
	})
}

func testTCPConnectionCloses(t *testing.T, req string, h Handler) {
	setParallel(t)
	defer afterTest(t)
	s := httptest.NewServer(h)
	defer s.Close()

	conn, err := net.Dial("tcp", s.Listener.Addr().String())
	if err != nil {
		t.Fatal("dial error:", err)
	}
	defer conn.Close()

	_, err = fmt.Fprint(conn, req)
	if err != nil {
		t.Fatal("print error:", err)
	}

	r := bufio.NewReader(conn)
	res, err := ReadResponse(r, &Request{Method: "GET"})
	if err != nil {
		t.Fatal("ReadResponse error:", err)
	}

	didReadAll := make(chan bool, 1)
	go func() {
		select {
		case <-time.After(5 * time.Second):
			t.Error("body not closed after 5s")
			return
		case <-didReadAll:
		}
	}()

	_, err = ioutil.ReadAll(r)
	if err != nil {
		t.Fatal("read error:", err)
	}
	didReadAll <- true

	if !res.Close {
		t.Errorf("Response.Close = false; want true")
	}
}

func testTCPConnectionStaysOpen(t *testing.T, req string, handler Handler) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewServer(handler)
	defer ts.Close()
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	br := bufio.NewReader(conn)
	for i := 0; i < 2; i++ {
		if _, err := io.WriteString(conn, req); err != nil {
			t.Fatal(err)
		}
		res, err := ReadResponse(br, nil)
		if err != nil {
			t.Fatalf("res %d: %v", i+1, err)
		}
		if _, err := io.Copy(ioutil.Discard, res.Body); err != nil {
			t.Fatalf("res %d body copy: %v", i+1, err)
		}
		res.Body.Close()
	}
}

// TestServeHTTP10Close verifies that HTTP/1.0 requests won't be kept alive.
func TestServeHTTP10Close(t *testing.T) {
	testTCPConnectionCloses(t, "GET / HTTP/1.0\r\n\r\n", HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/file")
	}))
}

// TestClientCanClose verifies that clients can also force a connection to close.
func TestClientCanClose(t *testing.T) {
	testTCPConnectionCloses(t, "GET / HTTP/1.1\r\nHost: foo\r\nConnection: close\r\n\r\n", HandlerFunc(func(w ResponseWriter, r *Request) {
		// Nothing.
	}))
}

// TestHandlersCanSetConnectionClose verifies that handlers can force a connection to close,
// even for HTTP/1.1 requests.
func TestHandlersCanSetConnectionClose11(t *testing.T) {
	testTCPConnectionCloses(t, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n\r\n", HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Connection", "close")
	}))
}

func TestHandlersCanSetConnectionClose10(t *testing.T) {
	testTCPConnectionCloses(t, "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Connection", "close")
	}))
}

func TestHTTP2UpgradeClosesConnection(t *testing.T) {
	testTCPConnectionCloses(t, "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n", HandlerFunc(func(w ResponseWriter, r *Request) {
		// Nothing. (if not hijacked, the server should close the connection
		// afterwards)
	}))
}

func send204(w ResponseWriter, r *Request) { w.WriteHeader(204) }
func send304(w ResponseWriter, r *Request) { w.WriteHeader(304) }

// Issue 15647: 204 responses can't have bodies, so HTTP/1.0 keep-alive conns should stay open.
func TestHTTP10KeepAlive204Response(t *testing.T) {
	testTCPConnectionStaysOpen(t, "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", HandlerFunc(send204))
}

func TestHTTP11KeepAlive204Response(t *testing.T) {
	testTCPConnectionStaysOpen(t, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n", HandlerFunc(send204))
}

func TestHTTP10KeepAlive304Response(t *testing.T) {
	testTCPConnectionStaysOpen(t,
		"GET / HTTP/1.0\r\nConnection: keep-alive\r\nIf-Modified-Since: Mon, 02 Jan 2006 15:04:05 GMT\r\n\r\n",
		HandlerFunc(send304))
}

// Issue 15703
func TestKeepAliveFinalChunkWithEOF(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, false /* h1 */, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.(Flusher).Flush() // force chunked encoding
		w.Write([]byte("{\"Addr\": \"" + r.RemoteAddr + "\"}"))
	}))
	defer cst.close()
	type data struct {
		Addr string
	}
	var addrs [2]data
	for i := range addrs {
		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.NewDecoder(res.Body).Decode(&addrs[i]); err != nil {
			t.Fatal(err)
		}
		if addrs[i].Addr == "" {
			t.Fatal("no address")
		}
		res.Body.Close()
	}
	if addrs[0] != addrs[1] {
		t.Fatalf("connection not reused")
	}
}

func TestSetsRemoteAddr_h1(t *testing.T) { testSetsRemoteAddr(t, h1Mode) }
func TestSetsRemoteAddr_h2(t *testing.T) { testSetsRemoteAddr(t, h2Mode) }

func testSetsRemoteAddr(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%s", r.RemoteAddr)
	}))
	defer cst.close()

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("ReadAll error: %v", err)
	}
	ip := string(body)
	if !strings.HasPrefix(ip, "127.0.0.1:") && !strings.HasPrefix(ip, "[::1]:") {
		t.Fatalf("Expected local addr; got %q", ip)
	}
}

type blockingRemoteAddrListener struct {
	net.Listener
	conns chan<- net.Conn
}

func (l *blockingRemoteAddrListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	brac := &blockingRemoteAddrConn{
		Conn:  c,
		addrs: make(chan net.Addr, 1),
	}
	l.conns <- brac
	return brac, nil
}

type blockingRemoteAddrConn struct {
	net.Conn
	addrs chan net.Addr
}

func (c *blockingRemoteAddrConn) RemoteAddr() net.Addr {
	return <-c.addrs
}

// Issue 12943
func TestServerAllowsBlockingRemoteAddr(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "RA:%s", r.RemoteAddr)
	}))
	conns := make(chan net.Conn)
	ts.Listener = &blockingRemoteAddrListener{
		Listener: ts.Listener,
		conns:    conns,
	}
	ts.Start()
	defer ts.Close()

	c := ts.Client()
	c.Timeout = time.Second
	// Force separate connection for each:
	c.Transport.(*Transport).DisableKeepAlives = true

	fetch := func(num int, response chan<- string) {
		resp, err := c.Get(ts.URL)
		if err != nil {
			t.Errorf("Request %d: %v", num, err)
			response <- ""
			return
		}
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("Request %d: %v", num, err)
			response <- ""
			return
		}
		response <- string(body)
	}

	// Start a request. The server will block on getting conn.RemoteAddr.
	response1c := make(chan string, 1)
	go fetch(1, response1c)

	// Wait for the server to accept it; grab the connection.
	conn1 := <-conns

	// Start another request and grab its connection
	response2c := make(chan string, 1)
	go fetch(2, response2c)
	var conn2 net.Conn

	select {
	case conn2 = <-conns:
	case <-time.After(time.Second):
		t.Fatal("Second Accept didn't happen")
	}

	// Send a response on connection 2.
	conn2.(*blockingRemoteAddrConn).addrs <- &net.TCPAddr{
		IP: net.ParseIP("12.12.12.12"), Port: 12}

	// ... and see it
	response2 := <-response2c
	if g, e := response2, "RA:12.12.12.12:12"; g != e {
		t.Fatalf("response 2 addr = %q; want %q", g, e)
	}

	// Finish the first response.
	conn1.(*blockingRemoteAddrConn).addrs <- &net.TCPAddr{
		IP: net.ParseIP("21.21.21.21"), Port: 21}

	// ... and see it
	response1 := <-response1c
	if g, e := response1, "RA:21.21.21.21:21"; g != e {
		t.Fatalf("response 1 addr = %q; want %q", g, e)
	}
}

func TestIdentityResponseHeaders(t *testing.T) {
	// Not parallel; changes log output.
	defer afterTest(t)
	log.SetOutput(ioutil.Discard) // is noisy otherwise
	defer log.SetOutput(os.Stderr)

	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Transfer-Encoding", "identity")
		w.(Flusher).Flush()
		fmt.Fprintf(w, "I am an identity response.")
	}))
	defer ts.Close()

	c := ts.Client()
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	defer res.Body.Close()

	if g, e := res.TransferEncoding, []string(nil); !reflect.DeepEqual(g, e) {
		t.Errorf("expected TransferEncoding of %v; got %v", e, g)
	}
	if _, haveCL := res.Header["Content-Length"]; haveCL {
		t.Errorf("Unexpected Content-Length")
	}
	if !res.Close {
		t.Errorf("expected Connection: close; got %v", res.Close)
	}
}

// TestHeadResponses verifies that all MIME type sniffing and Content-Length
// counting of GET requests also happens on HEAD requests.
func TestHeadResponses_h1(t *testing.T) { testHeadResponses(t, h1Mode) }
func TestHeadResponses_h2(t *testing.T) { testHeadResponses(t, h2Mode) }

func testHeadResponses(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("<html>"))
		if err != nil {
			t.Errorf("ResponseWriter.Write: %v", err)
		}

		// Also exercise the ReaderFrom path
		_, err = io.Copy(w, strings.NewReader("789a"))
		if err != nil {
			t.Errorf("Copy(ResponseWriter, ...): %v", err)
		}
	}))
	defer cst.close()
	res, err := cst.c.Head(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if len(res.TransferEncoding) > 0 {
		t.Errorf("expected no TransferEncoding; got %v", res.TransferEncoding)
	}
	if ct := res.Header.Get("Content-Type"); ct != "text/html; charset=utf-8" {
		t.Errorf("Content-Type: %q; want text/html; charset=utf-8", ct)
	}
	if v := res.ContentLength; v != 10 {
		t.Errorf("Content-Length: %d; want 10", v)
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Error(err)
	}
	if len(body) > 0 {
		t.Errorf("got unexpected body %q", string(body))
	}
}

func TestTLSHandshakeTimeout(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {}))
	errc := make(chanWriter, 10) // but only expecting 1
	ts.Config.ReadTimeout = 250 * time.Millisecond
	ts.Config.ErrorLog = log.New(errc, "", 0)
	ts.StartTLS()
	defer ts.Close()
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()
	goTimeout(t, 10*time.Second, func() {
		var buf [1]byte
		n, err := conn.Read(buf[:])
		if err == nil || n != 0 {
			t.Errorf("Read = %d, %v; want an error and no bytes", n, err)
		}
	})
	select {
	case v := <-errc:
		if !strings.Contains(v, "timeout") && !strings.Contains(v, "TLS handshake") {
			t.Errorf("expected a TLS handshake timeout error; got %q", v)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("timeout waiting for logged error")
	}
}

func TestTLSServer(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.TLS != nil {
			w.Header().Set("X-TLS-Set", "true")
			if r.TLS.HandshakeComplete {
				w.Header().Set("X-TLS-HandshakeComplete", "true")
			}
		}
	}))
	ts.Config.ErrorLog = log.New(ioutil.Discard, "", 0)
	defer ts.Close()

	// Connect an idle TCP connection to this server before we run
	// our real tests. This idle connection used to block forever
	// in the TLS handshake, preventing future connections from
	// being accepted. It may prevent future accidental blocking
	// in newConn.
	idleConn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer idleConn.Close()
	goTimeout(t, 10*time.Second, func() {
		if !strings.HasPrefix(ts.URL, "https://") {
			t.Errorf("expected test TLS server to start with https://, got %q", ts.URL)
			return
		}
		client := ts.Client()
		res, err := client.Get(ts.URL)
		if err != nil {
			t.Error(err)
			return
		}
		if res == nil {
			t.Errorf("got nil Response")
			return
		}
		defer res.Body.Close()
		if res.Header.Get("X-TLS-Set") != "true" {
			t.Errorf("expected X-TLS-Set response header")
			return
		}
		if res.Header.Get("X-TLS-HandshakeComplete") != "true" {
			t.Errorf("expected X-TLS-HandshakeComplete header")
		}
	})
}

func TestServeTLS(t *testing.T) {
	// Not parallel: uses global test hooks.
	defer afterTest(t)
	defer SetTestHookServerServe(nil)

	cert, err := tls.X509KeyPair(internal.LocalhostCert, internal.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	tlsConf := &tls.Config{
		Certificates: []tls.Certificate{cert},
	}

	ln := newLocalListener(t)
	defer ln.Close()
	addr := ln.Addr().String()

	serving := make(chan bool, 1)
	SetTestHookServerServe(func(s *Server, ln net.Listener) {
		serving <- true
	})
	handler := HandlerFunc(func(w ResponseWriter, r *Request) {})
	s := &Server{
		Addr:      addr,
		TLSConfig: tlsConf,
		Handler:   handler,
	}
	errc := make(chan error, 1)
	go func() { errc <- s.ServeTLS(ln, "", "") }()
	select {
	case err := <-errc:
		t.Fatalf("ServeTLS: %v", err)
	case <-serving:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout")
	}

	c, err := tls.Dial("tcp", ln.Addr().String(), &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{"h2", "http/1.1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if got, want := c.ConnectionState().NegotiatedProtocol, "h2"; got != want {
		t.Errorf("NegotiatedProtocol = %q; want %q", got, want)
	}
	if got, want := c.ConnectionState().NegotiatedProtocolIsMutual, true; got != want {
		t.Errorf("NegotiatedProtocolIsMutual = %v; want %v", got, want)
	}
}

// Test that the HTTPS server nicely rejects plaintext HTTP/1.x requests.
func TestTLSServerRejectHTTPRequests(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Error("unexpected HTTPS request")
	}))
	var errBuf bytes.Buffer
	ts.Config.ErrorLog = log.New(&errBuf, "", 0)
	defer ts.Close()
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	io.WriteString(conn, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n")
	slurp, err := ioutil.ReadAll(conn)
	if err != nil {
		t.Fatal(err)
	}
	const wantPrefix = "HTTP/1.0 400 Bad Request\r\n"
	if !strings.HasPrefix(string(slurp), wantPrefix) {
		t.Errorf("response = %q; wanted prefix %q", slurp, wantPrefix)
	}
}

// Issue 15908
func TestAutomaticHTTP2_Serve_NoTLSConfig(t *testing.T) {
	testAutomaticHTTP2_Serve(t, nil, true)
}

func TestAutomaticHTTP2_Serve_NonH2TLSConfig(t *testing.T) {
	testAutomaticHTTP2_Serve(t, &tls.Config{}, false)
}

func TestAutomaticHTTP2_Serve_H2TLSConfig(t *testing.T) {
	testAutomaticHTTP2_Serve(t, &tls.Config{NextProtos: []string{"h2"}}, true)
}

func testAutomaticHTTP2_Serve(t *testing.T, tlsConf *tls.Config, wantH2 bool) {
	setParallel(t)
	defer afterTest(t)
	ln := newLocalListener(t)
	ln.Close() // immediately (not a defer!)
	var s Server
	s.TLSConfig = tlsConf
	if err := s.Serve(ln); err == nil {
		t.Fatal("expected an error")
	}
	gotH2 := s.TLSNextProto["h2"] != nil
	if gotH2 != wantH2 {
		t.Errorf("http2 configured = %v; want %v", gotH2, wantH2)
	}
}

func TestAutomaticHTTP2_Serve_WithTLSConfig(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ln := newLocalListener(t)
	ln.Close() // immediately (not a defer!)
	var s Server
	// Set the TLSConfig. In reality, this would be the
	// *tls.Config given to tls.NewListener.
	s.TLSConfig = &tls.Config{
		NextProtos: []string{"h2"},
	}
	if err := s.Serve(ln); err == nil {
		t.Fatal("expected an error")
	}
	on := s.TLSNextProto["h2"] != nil
	if !on {
		t.Errorf("http2 wasn't automatically enabled")
	}
}

func TestAutomaticHTTP2_ListenAndServe(t *testing.T) {
	cert, err := tls.X509KeyPair(internal.LocalhostCert, internal.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	testAutomaticHTTP2_ListenAndServe(t, &tls.Config{
		Certificates: []tls.Certificate{cert},
	})
}

func TestAutomaticHTTP2_ListenAndServe_GetCertificate(t *testing.T) {
	cert, err := tls.X509KeyPair(internal.LocalhostCert, internal.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	testAutomaticHTTP2_ListenAndServe(t, &tls.Config{
		GetCertificate: func(clientHello *tls.ClientHelloInfo) (*tls.Certificate, error) {
			return &cert, nil
		},
	})
}

func testAutomaticHTTP2_ListenAndServe(t *testing.T, tlsConf *tls.Config) {
	// Not parallel: uses global test hooks.
	defer afterTest(t)
	defer SetTestHookServerServe(nil)
	var ok bool
	var s *Server
	const maxTries = 5
	var ln net.Listener
Try:
	for try := 0; try < maxTries; try++ {
		ln = newLocalListener(t)
		addr := ln.Addr().String()
		ln.Close()
		t.Logf("Got %v", addr)
		lnc := make(chan net.Listener, 1)
		SetTestHookServerServe(func(s *Server, ln net.Listener) {
			lnc <- ln
		})
		s = &Server{
			Addr:      addr,
			TLSConfig: tlsConf,
		}
		errc := make(chan error, 1)
		go func() { errc <- s.ListenAndServeTLS("", "") }()
		select {
		case err := <-errc:
			t.Logf("On try #%v: %v", try+1, err)
			continue
		case ln = <-lnc:
			ok = true
			t.Logf("Listening on %v", ln.Addr().String())
			break Try
		}
	}
	if !ok {
		t.Fatalf("Failed to start up after %d tries", maxTries)
	}
	defer ln.Close()
	c, err := tls.Dial("tcp", ln.Addr().String(), &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{"h2", "http/1.1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if got, want := c.ConnectionState().NegotiatedProtocol, "h2"; got != want {
		t.Errorf("NegotiatedProtocol = %q; want %q", got, want)
	}
	if got, want := c.ConnectionState().NegotiatedProtocolIsMutual, true; got != want {
		t.Errorf("NegotiatedProtocolIsMutual = %v; want %v", got, want)
	}
}

type serverExpectTest struct {
	contentLength    int // of request body
	chunked          bool
	expectation      string // e.g. "100-continue"
	readBody         bool   // whether handler should read the body (if false, sends StatusUnauthorized)
	expectedResponse string // expected substring in first line of http response
}

func expectTest(contentLength int, expectation string, readBody bool, expectedResponse string) serverExpectTest {
	return serverExpectTest{
		contentLength:    contentLength,
		expectation:      expectation,
		readBody:         readBody,
		expectedResponse: expectedResponse,
	}
}

var serverExpectTests = []serverExpectTest{
	// Normal 100-continues, case-insensitive.
	expectTest(100, "100-continue", true, "100 Continue"),
	expectTest(100, "100-cOntInUE", true, "100 Continue"),

	// No 100-continue.
	expectTest(100, "", true, "200 OK"),

	// 100-continue but requesting client to deny us,
	// so it never reads the body.
	expectTest(100, "100-continue", false, "401 Unauthorized"),
	// Likewise without 100-continue:
	expectTest(100, "", false, "401 Unauthorized"),

	// Non-standard expectations are failures
	expectTest(0, "a-pony", false, "417 Expectation Failed"),

	// Expect-100 requested but no body (is apparently okay: Issue 7625)
	expectTest(0, "100-continue", true, "200 OK"),
	// Expect-100 requested but handler doesn't read the body
	expectTest(0, "100-continue", false, "401 Unauthorized"),
	// Expect-100 continue with no body, but a chunked body.
	{
		expectation:      "100-continue",
		readBody:         true,
		chunked:          true,
		expectedResponse: "100 Continue",
	},
}

// Tests that the server responds to the "Expect" request header
// correctly.
// http2 test: TestServer_Response_Automatic100Continue
func TestServerExpect(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		// Note using r.FormValue("readbody") because for POST
		// requests that would read from r.Body, which we only
		// conditionally want to do.
		if strings.Contains(r.URL.RawQuery, "readbody=true") {
			ioutil.ReadAll(r.Body)
			w.Write([]byte("Hi"))
		} else {
			w.WriteHeader(StatusUnauthorized)
		}
	}))
	defer ts.Close()

	runTest := func(test serverExpectTest) {
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatalf("Dial: %v", err)
		}
		defer conn.Close()

		// Only send the body immediately if we're acting like an HTTP client
		// that doesn't send 100-continue expectations.
		writeBody := test.contentLength != 0 && strings.ToLower(test.expectation) != "100-continue"

		go func() {
			contentLen := fmt.Sprintf("Content-Length: %d", test.contentLength)
			if test.chunked {
				contentLen = "Transfer-Encoding: chunked"
			}
			_, err := fmt.Fprintf(conn, "POST /?readbody=%v HTTP/1.1\r\n"+
				"Connection: close\r\n"+
				"%s\r\n"+
				"Expect: %s\r\nHost: foo\r\n\r\n",
				test.readBody, contentLen, test.expectation)
			if err != nil {
				t.Errorf("On test %#v, error writing request headers: %v", test, err)
				return
			}
			if writeBody {
				var targ io.WriteCloser = struct {
					io.Writer
					io.Closer
				}{
					conn,
					ioutil.NopCloser(nil),
				}
				if test.chunked {
					targ = httputil.NewChunkedWriter(conn)
				}
				body := strings.Repeat("A", test.contentLength)
				_, err = fmt.Fprint(targ, body)
				if err == nil {
					err = targ.Close()
				}
				if err != nil {
					if !test.readBody {
						// Server likely already hung up on us.
						// See larger comment below.
						t.Logf("On test %#v, acceptable error writing request body: %v", test, err)
						return
					}
					t.Errorf("On test %#v, error writing request body: %v", test, err)
				}
			}
		}()
		bufr := bufio.NewReader(conn)
		line, err := bufr.ReadString('\n')
		if err != nil {
			if writeBody && !test.readBody {
				// This is an acceptable failure due to a possible TCP race:
				// We were still writing data and the server hung up on us. A TCP
				// implementation may send a RST if our request body data was known
				// to be lost, which may trigger our reads to fail.
				// See RFC 1122 page 88.
				t.Logf("On test %#v, acceptable error from ReadString: %v", test, err)
				return
			}
			t.Fatalf("On test %#v, ReadString: %v", test, err)
		}
		if !strings.Contains(line, test.expectedResponse) {
			t.Errorf("On test %#v, got first line = %q; want %q", test, line, test.expectedResponse)
		}
	}

	for _, test := range serverExpectTests {
		runTest(test)
	}
}

// Under a ~256KB (maxPostHandlerReadBytes) threshold, the server
// should consume client request bodies that a handler didn't read.
func TestServerUnreadRequestBodyLittle(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	conn := new(testConn)
	body := strings.Repeat("x", 100<<10)
	conn.readBuf.Write([]byte(fmt.Sprintf(
		"POST / HTTP/1.1\r\n"+
			"Host: test\r\n"+
			"Content-Length: %d\r\n"+
			"\r\n", len(body))))
	conn.readBuf.Write([]byte(body))

	done := make(chan bool)

	readBufLen := func() int {
		conn.readMu.Lock()
		defer conn.readMu.Unlock()
		return conn.readBuf.Len()
	}

	ls := &oneConnListener{conn}
	go Serve(ls, HandlerFunc(func(rw ResponseWriter, req *Request) {
		defer close(done)
		if bufLen := readBufLen(); bufLen < len(body)/2 {
			t.Errorf("on request, read buffer length is %d; expected about 100 KB", bufLen)
		}
		rw.WriteHeader(200)
		rw.(Flusher).Flush()
		if g, e := readBufLen(), 0; g != e {
			t.Errorf("after WriteHeader, read buffer length is %d; want %d", g, e)
		}
		if c := rw.Header().Get("Connection"); c != "" {
			t.Errorf(`Connection header = %q; want ""`, c)
		}
	}))
	<-done
}

// Over a ~256KB (maxPostHandlerReadBytes) threshold, the server
// should ignore client request bodies that a handler didn't read
// and close the connection.
func TestServerUnreadRequestBodyLarge(t *testing.T) {
	setParallel(t)
	if testing.Short() && testenv.Builder() == "" {
		t.Log("skipping in short mode")
	}
	conn := new(testConn)
	body := strings.Repeat("x", 1<<20)
	conn.readBuf.Write([]byte(fmt.Sprintf(
		"POST / HTTP/1.1\r\n"+
			"Host: test\r\n"+
			"Content-Length: %d\r\n"+
			"\r\n", len(body))))
	conn.readBuf.Write([]byte(body))
	conn.closec = make(chan bool, 1)

	ls := &oneConnListener{conn}
	go Serve(ls, HandlerFunc(func(rw ResponseWriter, req *Request) {
		if conn.readBuf.Len() < len(body)/2 {
			t.Errorf("on request, read buffer length is %d; expected about 1MB", conn.readBuf.Len())
		}
		rw.WriteHeader(200)
		rw.(Flusher).Flush()
		if conn.readBuf.Len() < len(body)/2 {
			t.Errorf("post-WriteHeader, read buffer length is %d; expected about 1MB", conn.readBuf.Len())
		}
	}))
	<-conn.closec

	if res := conn.writeBuf.String(); !strings.Contains(res, "Connection: close") {
		t.Errorf("Expected a Connection: close header; got response: %s", res)
	}
}

type handlerBodyCloseTest struct {
	bodySize     int
	bodyChunked  bool
	reqConnClose bool

	wantEOFSearch bool // should Handler's Body.Close do Reads, looking for EOF?
	wantNextReq   bool // should it find the next request on the same conn?
}

func (t handlerBodyCloseTest) connectionHeader() string {
	if t.reqConnClose {
		return "Connection: close\r\n"
	}
	return ""
}

var handlerBodyCloseTests = [...]handlerBodyCloseTest{
	// Small enough to slurp past to the next request +
	// has Content-Length.
	0: {
		bodySize:      20 << 10,
		bodyChunked:   false,
		reqConnClose:  false,
		wantEOFSearch: true,
		wantNextReq:   true,
	},

	// Small enough to slurp past to the next request +
	// is chunked.
	1: {
		bodySize:      20 << 10,
		bodyChunked:   true,
		reqConnClose:  false,
		wantEOFSearch: true,
		wantNextReq:   true,
	},

	// Small enough to slurp past to the next request +
	// has Content-Length +
	// declares Connection: close (so pointless to read more).
	2: {
		bodySize:      20 << 10,
		bodyChunked:   false,
		reqConnClose:  true,
		wantEOFSearch: false,
		wantNextReq:   false,
	},

	// Small enough to slurp past to the next request +
	// declares Connection: close,
	// but chunked, so it might have trailers.
	// TODO: maybe skip this search if no trailers were declared
	// in the headers.
	3: {
		bodySize:      20 << 10,
		bodyChunked:   true,
		reqConnClose:  true,
		wantEOFSearch: true,
		wantNextReq:   false,
	},

	// Big with Content-Length, so give up immediately if we know it's too big.
	4: {
		bodySize:      1 << 20,
		bodyChunked:   false, // has a Content-Length
		reqConnClose:  false,
		wantEOFSearch: false,
		wantNextReq:   false,
	},

	// Big chunked, so read a bit before giving up.
	5: {
		bodySize:      1 << 20,
		bodyChunked:   true,
		reqConnClose:  false,
		wantEOFSearch: true,
		wantNextReq:   false,
	},

	// Big with Connection: close, but chunked, so search for trailers.
	// TODO: maybe skip this search if no trailers were declared
	// in the headers.
	6: {
		bodySize:      1 << 20,
		bodyChunked:   true,
		reqConnClose:  true,
		wantEOFSearch: true,
		wantNextReq:   false,
	},

	// Big with Connection: close, so don't do any reads on Close.
	// With Content-Length.
	7: {
		bodySize:      1 << 20,
		bodyChunked:   false,
		reqConnClose:  true,
		wantEOFSearch: false,
		wantNextReq:   false,
	},
}

func TestHandlerBodyClose(t *testing.T) {
	setParallel(t)
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in -short mode")
	}
	for i, tt := range handlerBodyCloseTests {
		testHandlerBodyClose(t, i, tt)
	}
}

func testHandlerBodyClose(t *testing.T, i int, tt handlerBodyCloseTest) {
	conn := new(testConn)
	body := strings.Repeat("x", tt.bodySize)
	if tt.bodyChunked {
		conn.readBuf.WriteString("POST / HTTP/1.1\r\n" +
			"Host: test\r\n" +
			tt.connectionHeader() +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n")
		cw := internal.NewChunkedWriter(&conn.readBuf)
		io.WriteString(cw, body)
		cw.Close()
		conn.readBuf.WriteString("\r\n")
	} else {
		conn.readBuf.Write([]byte(fmt.Sprintf(
			"POST / HTTP/1.1\r\n"+
				"Host: test\r\n"+
				tt.connectionHeader()+
				"Content-Length: %d\r\n"+
				"\r\n", len(body))))
		conn.readBuf.Write([]byte(body))
	}
	if !tt.reqConnClose {
		conn.readBuf.WriteString("GET / HTTP/1.1\r\nHost: test\r\n\r\n")
	}
	conn.closec = make(chan bool, 1)

	readBufLen := func() int {
		conn.readMu.Lock()
		defer conn.readMu.Unlock()
		return conn.readBuf.Len()
	}

	ls := &oneConnListener{conn}
	var numReqs int
	var size0, size1 int
	go Serve(ls, HandlerFunc(func(rw ResponseWriter, req *Request) {
		numReqs++
		if numReqs == 1 {
			size0 = readBufLen()
			req.Body.Close()
			size1 = readBufLen()
		}
	}))
	<-conn.closec
	if numReqs < 1 || numReqs > 2 {
		t.Fatalf("%d. bug in test. unexpected number of requests = %d", i, numReqs)
	}
	didSearch := size0 != size1
	if didSearch != tt.wantEOFSearch {
		t.Errorf("%d. did EOF search = %v; want %v (size went from %d to %d)", i, didSearch, !didSearch, size0, size1)
	}
	if tt.wantNextReq && numReqs != 2 {
		t.Errorf("%d. numReq = %d; want 2", i, numReqs)
	}
}

// testHandlerBodyConsumer represents a function injected into a test handler to
// vary work done on a request Body.
type testHandlerBodyConsumer struct {
	name string
	f    func(io.ReadCloser)
}

var testHandlerBodyConsumers = []testHandlerBodyConsumer{
	{"nil", func(io.ReadCloser) {}},
	{"close", func(r io.ReadCloser) { r.Close() }},
	{"discard", func(r io.ReadCloser) { io.Copy(ioutil.Discard, r) }},
}

func TestRequestBodyReadErrorClosesConnection(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	for _, handler := range testHandlerBodyConsumers {
		conn := new(testConn)
		conn.readBuf.WriteString("POST /public HTTP/1.1\r\n" +
			"Host: test\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n" +
			"hax\r\n" + // Invalid chunked encoding
			"GET /secret HTTP/1.1\r\n" +
			"Host: test\r\n" +
			"\r\n")

		conn.closec = make(chan bool, 1)
		ls := &oneConnListener{conn}
		var numReqs int
		go Serve(ls, HandlerFunc(func(_ ResponseWriter, req *Request) {
			numReqs++
			if strings.Contains(req.URL.Path, "secret") {
				t.Error("Request for /secret encountered, should not have happened.")
			}
			handler.f(req.Body)
		}))
		<-conn.closec
		if numReqs != 1 {
			t.Errorf("Handler %v: got %d reqs; want 1", handler.name, numReqs)
		}
	}
}

func TestInvalidTrailerClosesConnection(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	for _, handler := range testHandlerBodyConsumers {
		conn := new(testConn)
		conn.readBuf.WriteString("POST /public HTTP/1.1\r\n" +
			"Host: test\r\n" +
			"Trailer: hack\r\n" +
			"Transfer-Encoding: chunked\r\n" +
			"\r\n" +
			"3\r\n" +
			"hax\r\n" +
			"0\r\n" +
			"I'm not a valid trailer\r\n" +
			"GET /secret HTTP/1.1\r\n" +
			"Host: test\r\n" +
			"\r\n")

		conn.closec = make(chan bool, 1)
		ln := &oneConnListener{conn}
		var numReqs int
		go Serve(ln, HandlerFunc(func(_ ResponseWriter, req *Request) {
			numReqs++
			if strings.Contains(req.URL.Path, "secret") {
				t.Errorf("Handler %s, Request for /secret encountered, should not have happened.", handler.name)
			}
			handler.f(req.Body)
		}))
		<-conn.closec
		if numReqs != 1 {
			t.Errorf("Handler %s: got %d reqs; want 1", handler.name, numReqs)
		}
	}
}

// slowTestConn is a net.Conn that provides a means to simulate parts of a
// request being received piecemeal. Deadlines can be set and enforced in both
// Read and Write.
type slowTestConn struct {
	// over multiple calls to Read, time.Durations are slept, strings are read.
	script []interface{}
	closec chan bool

	mu     sync.Mutex // guards rd/wd
	rd, wd time.Time  // read, write deadline
	noopConn
}

func (c *slowTestConn) SetDeadline(t time.Time) error {
	c.SetReadDeadline(t)
	c.SetWriteDeadline(t)
	return nil
}

func (c *slowTestConn) SetReadDeadline(t time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.rd = t
	return nil
}

func (c *slowTestConn) SetWriteDeadline(t time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.wd = t
	return nil
}

func (c *slowTestConn) Read(b []byte) (n int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
restart:
	if !c.rd.IsZero() && time.Now().After(c.rd) {
		return 0, syscall.ETIMEDOUT
	}
	if len(c.script) == 0 {
		return 0, io.EOF
	}

	switch cue := c.script[0].(type) {
	case time.Duration:
		if !c.rd.IsZero() {
			// If the deadline falls in the middle of our sleep window, deduct
			// part of the sleep, then return a timeout.
			if remaining := time.Until(c.rd); remaining < cue {
				c.script[0] = cue - remaining
				time.Sleep(remaining)
				return 0, syscall.ETIMEDOUT
			}
		}
		c.script = c.script[1:]
		time.Sleep(cue)
		goto restart

	case string:
		n = copy(b, cue)
		// If cue is too big for the buffer, leave the end for the next Read.
		if len(cue) > n {
			c.script[0] = cue[n:]
		} else {
			c.script = c.script[1:]
		}

	default:
		panic("unknown cue in slowTestConn script")
	}

	return
}

func (c *slowTestConn) Close() error {
	select {
	case c.closec <- true:
	default:
	}
	return nil
}

func (c *slowTestConn) Write(b []byte) (int, error) {
	if !c.wd.IsZero() && time.Now().After(c.wd) {
		return 0, syscall.ETIMEDOUT
	}
	return len(b), nil
}

func TestRequestBodyTimeoutClosesConnection(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	defer afterTest(t)
	for _, handler := range testHandlerBodyConsumers {
		conn := &slowTestConn{
			script: []interface{}{
				"POST /public HTTP/1.1\r\n" +
					"Host: test\r\n" +
					"Content-Length: 10000\r\n" +
					"\r\n",
				"foo bar baz",
				600 * time.Millisecond, // Request deadline should hit here
				"GET /secret HTTP/1.1\r\n" +
					"Host: test\r\n" +
					"\r\n",
			},
			closec: make(chan bool, 1),
		}
		ls := &oneConnListener{conn}

		var numReqs int
		s := Server{
			Handler: HandlerFunc(func(_ ResponseWriter, req *Request) {
				numReqs++
				if strings.Contains(req.URL.Path, "secret") {
					t.Error("Request for /secret encountered, should not have happened.")
				}
				handler.f(req.Body)
			}),
			ReadTimeout: 400 * time.Millisecond,
		}
		go s.Serve(ls)
		<-conn.closec

		if numReqs != 1 {
			t.Errorf("Handler %v: got %d reqs; want 1", handler.name, numReqs)
		}
	}
}

func TestTimeoutHandler_h1(t *testing.T) { testTimeoutHandler(t, h1Mode) }
func TestTimeoutHandler_h2(t *testing.T) { testTimeoutHandler(t, h2Mode) }
func testTimeoutHandler(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	sendHi := make(chan bool, 1)
	writeErrors := make(chan error, 1)
	sayHi := HandlerFunc(func(w ResponseWriter, r *Request) {
		<-sendHi
		_, werr := w.Write([]byte("hi"))
		writeErrors <- werr
	})
	timeout := make(chan time.Time, 1) // write to this to force timeouts
	cst := newClientServerTest(t, h2, NewTestTimeoutHandler(sayHi, timeout))
	defer cst.close()

	// Succeed without timing out:
	sendHi <- true
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusOK; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ := ioutil.ReadAll(res.Body)
	if g, e := string(body), "hi"; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
	if g := <-writeErrors; g != nil {
		t.Errorf("got unexpected Write error on first request: %v", g)
	}

	// Times out:
	timeout <- time.Time{}
	res, err = cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusServiceUnavailable; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ = ioutil.ReadAll(res.Body)
	if !strings.Contains(string(body), "<title>Timeout</title>") {
		t.Errorf("expected timeout body; got %q", string(body))
	}
	if g, w := res.Header.Get("Content-Type"), "text/html; charset=utf-8"; g != w {
		t.Errorf("response content-type = %q; want %q", g, w)
	}

	// Now make the previously-timed out handler speak again,
	// which verifies the panic is handled:
	sendHi <- true
	if g, e := <-writeErrors, ErrHandlerTimeout; g != e {
		t.Errorf("expected Write error of %v; got %v", e, g)
	}
}

// See issues 8209 and 8414.
func TestTimeoutHandlerRace(t *testing.T) {
	setParallel(t)
	defer afterTest(t)

	delayHi := HandlerFunc(func(w ResponseWriter, r *Request) {
		ms, _ := strconv.Atoi(r.URL.Path[1:])
		if ms == 0 {
			ms = 1
		}
		for i := 0; i < ms; i++ {
			w.Write([]byte("hi"))
			time.Sleep(time.Millisecond)
		}
	})

	ts := httptest.NewServer(TimeoutHandler(delayHi, 20*time.Millisecond, ""))
	defer ts.Close()

	c := ts.Client()

	var wg sync.WaitGroup
	gate := make(chan bool, 10)
	n := 50
	if testing.Short() {
		n = 10
		gate = make(chan bool, 3)
	}
	for i := 0; i < n; i++ {
		gate <- true
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-gate }()
			res, err := c.Get(fmt.Sprintf("%s/%d", ts.URL, rand.Intn(50)))
			if err == nil {
				io.Copy(ioutil.Discard, res.Body)
				res.Body.Close()
			}
		}()
	}
	wg.Wait()
}

// See issues 8209 and 8414.
func TestTimeoutHandlerRaceHeader(t *testing.T) {
	setParallel(t)
	defer afterTest(t)

	delay204 := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(204)
	})

	ts := httptest.NewServer(TimeoutHandler(delay204, time.Nanosecond, ""))
	defer ts.Close()

	var wg sync.WaitGroup
	gate := make(chan bool, 50)
	n := 500
	if testing.Short() {
		n = 10
	}

	c := ts.Client()
	for i := 0; i < n; i++ {
		gate <- true
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-gate }()
			res, err := c.Get(ts.URL)
			if err != nil {
				t.Error(err)
				return
			}
			defer res.Body.Close()
			io.Copy(ioutil.Discard, res.Body)
		}()
	}
	wg.Wait()
}

// Issue 9162
func TestTimeoutHandlerRaceHeaderTimeout(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	sendHi := make(chan bool, 1)
	writeErrors := make(chan error, 1)
	sayHi := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "text/plain")
		<-sendHi
		_, werr := w.Write([]byte("hi"))
		writeErrors <- werr
	})
	timeout := make(chan time.Time, 1) // write to this to force timeouts
	cst := newClientServerTest(t, h1Mode, NewTestTimeoutHandler(sayHi, timeout))
	defer cst.close()

	// Succeed without timing out:
	sendHi <- true
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusOK; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ := ioutil.ReadAll(res.Body)
	if g, e := string(body), "hi"; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
	if g := <-writeErrors; g != nil {
		t.Errorf("got unexpected Write error on first request: %v", g)
	}

	// Times out:
	timeout <- time.Time{}
	res, err = cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusServiceUnavailable; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ = ioutil.ReadAll(res.Body)
	if !strings.Contains(string(body), "<title>Timeout</title>") {
		t.Errorf("expected timeout body; got %q", string(body))
	}

	// Now make the previously-timed out handler speak again,
	// which verifies the panic is handled:
	sendHi <- true
	if g, e := <-writeErrors, ErrHandlerTimeout; g != e {
		t.Errorf("expected Write error of %v; got %v", e, g)
	}
}

// Issue 14568.
func TestTimeoutHandlerStartTimerWhenServing(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping sleeping test in -short mode")
	}
	defer afterTest(t)
	var handler HandlerFunc = func(w ResponseWriter, _ *Request) {
		w.WriteHeader(StatusNoContent)
	}
	timeout := 300 * time.Millisecond
	ts := httptest.NewServer(TimeoutHandler(handler, timeout, ""))
	defer ts.Close()

	c := ts.Client()

	// Issue was caused by the timeout handler starting the timer when
	// was created, not when the request. So wait for more than the timeout
	// to ensure that's not the case.
	time.Sleep(2 * timeout)
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != StatusNoContent {
		t.Errorf("got res.StatusCode %d, want %v", res.StatusCode, StatusNoContent)
	}
}

// https://golang.org/issue/15948
func TestTimeoutHandlerEmptyResponse(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	var handler HandlerFunc = func(w ResponseWriter, _ *Request) {
		// No response.
	}
	timeout := 300 * time.Millisecond
	ts := httptest.NewServer(TimeoutHandler(handler, timeout, ""))
	defer ts.Close()

	c := ts.Client()

	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != StatusOK {
		t.Errorf("got res.StatusCode %d, want %v", res.StatusCode, StatusOK)
	}
}

// https://golang.org/issues/22084
func TestTimeoutHandlerPanicRecovery(t *testing.T) {
	wrapper := func(h Handler) Handler {
		return TimeoutHandler(h, time.Second, "")
	}
	testHandlerPanic(t, false, false, wrapper, "intentional death for testing")
}

func TestRedirectBadPath(t *testing.T) {
	// This used to crash. It's not valid input (bad path), but it
	// shouldn't crash.
	rr := httptest.NewRecorder()
	req := &Request{
		Method: "GET",
		URL: &url.URL{
			Scheme: "http",
			Path:   "not-empty-but-no-leading-slash", // bogus
		},
	}
	Redirect(rr, req, "", 304)
	if rr.Code != 304 {
		t.Errorf("Code = %d; want 304", rr.Code)
	}
}

// Test different URL formats and schemes
func TestRedirect(t *testing.T) {
	req, _ := NewRequest("GET", "http://example.com/qux/", nil)

	var tests = []struct {
		in   string
		want string
	}{
		// normal http
		{"http://foobar.com/baz", "http://foobar.com/baz"},
		// normal https
		{"https://foobar.com/baz", "https://foobar.com/baz"},
		// custom scheme
		{"test://foobar.com/baz", "test://foobar.com/baz"},
		// schemeless
		{"//foobar.com/baz", "//foobar.com/baz"},
		// relative to the root
		{"/foobar.com/baz", "/foobar.com/baz"},
		// relative to the current path
		{"foobar.com/baz", "/qux/foobar.com/baz"},
		// relative to the current path (+ going upwards)
		{"../quux/foobar.com/baz", "/quux/foobar.com/baz"},
		// incorrect number of slashes
		{"///foobar.com/baz", "/foobar.com/baz"},

		// Verifies we don't path.Clean() on the wrong parts in redirects:
		{"/foo?next=http://bar.com/", "/foo?next=http://bar.com/"},
		{"http://localhost:8080/_ah/login?continue=http://localhost:8080/",
			"http://localhost:8080/_ah/login?continue=http://localhost:8080/"},

		{"/фубар", "/%d1%84%d1%83%d0%b1%d0%b0%d1%80"},
		{"http://foo.com/фубар", "http://foo.com/%d1%84%d1%83%d0%b1%d0%b0%d1%80"},
	}

	for _, tt := range tests {
		rec := httptest.NewRecorder()
		Redirect(rec, req, tt.in, 302)
		if got, want := rec.Code, 302; got != want {
			t.Errorf("Redirect(%q) generated status code %v; want %v", tt.in, got, want)
		}
		if got := rec.Header().Get("Location"); got != tt.want {
			t.Errorf("Redirect(%q) generated Location header %q; want %q", tt.in, got, tt.want)
		}
	}
}

// Test that Redirect sets Content-Type header for GET and HEAD requests
// and writes a short HTML body, unless the request already has a Content-Type header.
func TestRedirect_contentTypeAndBody(t *testing.T) {
	type ctHeader struct {
		Values []string
	}

	var tests = []struct {
		method   string
		ct       *ctHeader // Optional Content-Type header to set.
		wantCT   string
		wantBody string
	}{
		{MethodGet, nil, "text/html; charset=utf-8", "<a href=\"/foo\">Found</a>.\n\n"},
		{MethodHead, nil, "text/html; charset=utf-8", ""},
		{MethodPost, nil, "", ""},
		{MethodDelete, nil, "", ""},
		{"foo", nil, "", ""},
		{MethodGet, &ctHeader{[]string{"application/test"}}, "application/test", ""},
		{MethodGet, &ctHeader{[]string{}}, "", ""},
		{MethodGet, &ctHeader{nil}, "", ""},
	}
	for _, tt := range tests {
		req := httptest.NewRequest(tt.method, "http://example.com/qux/", nil)
		rec := httptest.NewRecorder()
		if tt.ct != nil {
			rec.Header()["Content-Type"] = tt.ct.Values
		}
		Redirect(rec, req, "/foo", 302)
		if got, want := rec.Code, 302; got != want {
			t.Errorf("Redirect(%q, %#v) generated status code %v; want %v", tt.method, tt.ct, got, want)
		}
		if got, want := rec.Header().Get("Content-Type"), tt.wantCT; got != want {
			t.Errorf("Redirect(%q, %#v) generated Content-Type header %q; want %q", tt.method, tt.ct, got, want)
		}
		resp := rec.Result()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := string(body), tt.wantBody; got != want {
			t.Errorf("Redirect(%q, %#v) generated Body %q; want %q", tt.method, tt.ct, got, want)
		}
	}
}

// TestZeroLengthPostAndResponse exercises an optimization done by the Transport:
// when there is no body (either because the method doesn't permit a body, or an
// explicit Content-Length of zero is present), then the transport can re-use the
// connection immediately. But when it re-uses the connection, it typically closes
// the previous request's body, which is not optimal for zero-lengthed bodies,
// as the client would then see http.ErrBodyReadAfterClose and not 0, io.EOF.
func TestZeroLengthPostAndResponse_h1(t *testing.T) {
	testZeroLengthPostAndResponse(t, h1Mode)
}
func TestZeroLengthPostAndResponse_h2(t *testing.T) {
	testZeroLengthPostAndResponse(t, h2Mode)
}

func testZeroLengthPostAndResponse(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(rw ResponseWriter, r *Request) {
		all, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("handler ReadAll: %v", err)
		}
		if len(all) != 0 {
			t.Errorf("handler got %d bytes; expected 0", len(all))
		}
		rw.Header().Set("Content-Length", "0")
	}))
	defer cst.close()

	req, err := NewRequest("POST", cst.ts.URL, strings.NewReader(""))
	if err != nil {
		t.Fatal(err)
	}
	req.ContentLength = 0

	var resp [5]*Response
	for i := range resp {
		resp[i], err = cst.c.Do(req)
		if err != nil {
			t.Fatalf("client post #%d: %v", i, err)
		}
	}

	for i := range resp {
		all, err := ioutil.ReadAll(resp[i].Body)
		if err != nil {
			t.Fatalf("req #%d: client ReadAll: %v", i, err)
		}
		if len(all) != 0 {
			t.Errorf("req #%d: client got %d bytes; expected 0", i, len(all))
		}
	}
}

func TestHandlerPanicNil_h1(t *testing.T) { testHandlerPanic(t, false, h1Mode, nil, nil) }
func TestHandlerPanicNil_h2(t *testing.T) { testHandlerPanic(t, false, h2Mode, nil, nil) }

func TestHandlerPanic_h1(t *testing.T) {
	testHandlerPanic(t, false, h1Mode, nil, "intentional death for testing")
}
func TestHandlerPanic_h2(t *testing.T) {
	testHandlerPanic(t, false, h2Mode, nil, "intentional death for testing")
}

func TestHandlerPanicWithHijack(t *testing.T) {
	// Only testing HTTP/1, and our http2 server doesn't support hijacking.
	testHandlerPanic(t, true, h1Mode, nil, "intentional death for testing")
}

func testHandlerPanic(t *testing.T, withHijack, h2 bool, wrapper func(Handler) Handler, panicValue interface{}) {
	defer afterTest(t)
	// Unlike the other tests that set the log output to ioutil.Discard
	// to quiet the output, this test uses a pipe. The pipe serves three
	// purposes:
	//
	//   1) The log.Print from the http server (generated by the caught
	//      panic) will go to the pipe instead of stderr, making the
	//      output quiet.
	//
	//   2) We read from the pipe to verify that the handler
	//      actually caught the panic and logged something.
	//
	//   3) The blocking Read call prevents this TestHandlerPanic
	//      function from exiting before the HTTP server handler
	//      finishes crashing. If this text function exited too
	//      early (and its defer log.SetOutput(os.Stderr) ran),
	//      then the crash output could spill into the next test.
	pr, pw := io.Pipe()
	log.SetOutput(pw)
	defer log.SetOutput(os.Stderr)
	defer pw.Close()

	var handler Handler = HandlerFunc(func(w ResponseWriter, r *Request) {
		if withHijack {
			rwc, _, err := w.(Hijacker).Hijack()
			if err != nil {
				t.Logf("unexpected error: %v", err)
			}
			defer rwc.Close()
		}
		panic(panicValue)
	})
	if wrapper != nil {
		handler = wrapper(handler)
	}
	cst := newClientServerTest(t, h2, handler)
	defer cst.close()

	// Do a blocking read on the log output pipe so its logging
	// doesn't bleed into the next test. But wait only 5 seconds
	// for it.
	done := make(chan bool, 1)
	go func() {
		buf := make([]byte, 4<<10)
		_, err := pr.Read(buf)
		pr.Close()
		if err != nil && err != io.EOF {
			t.Error(err)
		}
		done <- true
	}()

	_, err := cst.c.Get(cst.ts.URL)
	if err == nil {
		t.Logf("expected an error")
	}

	if panicValue == nil {
		return
	}

	select {
	case <-done:
		return
	case <-time.After(5 * time.Second):
		t.Fatal("expected server handler to log an error")
	}
}

type terrorWriter struct{ t *testing.T }

func (w terrorWriter) Write(p []byte) (int, error) {
	w.t.Errorf("%s", p)
	return len(p), nil
}

// Issue 16456: allow writing 0 bytes on hijacked conn to test hijack
// without any log spam.
func TestServerWriteHijackZeroBytes(t *testing.T) {
	defer afterTest(t)
	done := make(chan struct{})
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(done)
		w.(Flusher).Flush()
		conn, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Errorf("Hijack: %v", err)
			return
		}
		defer conn.Close()
		_, err = w.Write(nil)
		if err != ErrHijacked {
			t.Errorf("Write error = %v; want ErrHijacked", err)
		}
	}))
	ts.Config.ErrorLog = log.New(terrorWriter{t}, "Unexpected write: ", 0)
	ts.Start()
	defer ts.Close()

	c := ts.Client()
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout")
	}
}

func TestServerNoDate_h1(t *testing.T)        { testServerNoHeader(t, h1Mode, "Date") }
func TestServerNoDate_h2(t *testing.T)        { testServerNoHeader(t, h2Mode, "Date") }
func TestServerNoContentType_h1(t *testing.T) { testServerNoHeader(t, h1Mode, "Content-Type") }
func TestServerNoContentType_h2(t *testing.T) { testServerNoHeader(t, h2Mode, "Content-Type") }

func testServerNoHeader(t *testing.T, h2 bool, header string) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header()[header] = nil
		io.WriteString(w, "<html>foo</html>") // non-empty
	}))
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if got, ok := res.Header[header]; ok {
		t.Fatalf("Expected no %s header; got %q", header, got)
	}
}

func TestStripPrefix(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Path", r.URL.Path)
	})
	ts := httptest.NewServer(StripPrefix("/foo", h))
	defer ts.Close()

	c := ts.Client()

	res, err := c.Get(ts.URL + "/foo/bar")
	if err != nil {
		t.Fatal(err)
	}
	if g, e := res.Header.Get("X-Path"), "/bar"; g != e {
		t.Errorf("test 1: got %s, want %s", g, e)
	}
	res.Body.Close()

	res, err = Get(ts.URL + "/bar")
	if err != nil {
		t.Fatal(err)
	}
	if g, e := res.StatusCode, 404; g != e {
		t.Errorf("test 2: got status %v, want %v", g, e)
	}
	res.Body.Close()
}

// https://golang.org/issue/18952.
func TestStripPrefix_notModifyRequest(t *testing.T) {
	h := StripPrefix("/foo", NotFoundHandler())
	req := httptest.NewRequest("GET", "/foo/bar", nil)
	h.ServeHTTP(httptest.NewRecorder(), req)
	if req.URL.Path != "/foo/bar" {
		t.Errorf("StripPrefix should not modify the provided Request, but it did")
	}
}

func TestRequestLimit_h1(t *testing.T) { testRequestLimit(t, h1Mode) }
func TestRequestLimit_h2(t *testing.T) { testRequestLimit(t, h2Mode) }
func testRequestLimit(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Fatalf("didn't expect to get request in Handler")
	}), optQuietLog)
	defer cst.close()
	req, _ := NewRequest("GET", cst.ts.URL, nil)
	var bytesPerHeader = len("header12345: val12345\r\n")
	for i := 0; i < ((DefaultMaxHeaderBytes+4096)/bytesPerHeader)+1; i++ {
		req.Header.Set(fmt.Sprintf("header%05d", i), fmt.Sprintf("val%05d", i))
	}
	res, err := cst.c.Do(req)
	if res != nil {
		defer res.Body.Close()
	}
	if h2 {
		// In HTTP/2, the result depends on a race. If the client has received the
		// server's SETTINGS before RoundTrip starts sending the request, then RoundTrip
		// will fail with an error. Otherwise, the client should receive a 431 from the
		// server.
		if err == nil && res.StatusCode != 431 {
			t.Fatalf("expected 431 response status; got: %d %s", res.StatusCode, res.Status)
		}
	} else {
		// In HTTP/1, we expect a 431 from the server.
		// Some HTTP clients may fail on this undefined behavior (server replying and
		// closing the connection while the request is still being written), but
		// we do support it (at least currently), so we expect a response below.
		if err != nil {
			t.Fatalf("Do: %v", err)
		}
		if res.StatusCode != 431 {
			t.Fatalf("expected 431 response status; got: %d %s", res.StatusCode, res.Status)
		}
	}
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

type countReader struct {
	r io.Reader
	n *int64
}

func (cr countReader) Read(p []byte) (n int, err error) {
	n, err = cr.r.Read(p)
	atomic.AddInt64(cr.n, int64(n))
	return
}

func TestRequestBodyLimit_h1(t *testing.T) { testRequestBodyLimit(t, h1Mode) }
func TestRequestBodyLimit_h2(t *testing.T) { testRequestBodyLimit(t, h2Mode) }
func testRequestBodyLimit(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	const limit = 1 << 20
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		r.Body = MaxBytesReader(w, r.Body, limit)
		n, err := io.Copy(ioutil.Discard, r.Body)
		if err == nil {
			t.Errorf("expected error from io.Copy")
		}
		if n != limit {
			t.Errorf("io.Copy = %d, want %d", n, limit)
		}
	}))
	defer cst.close()

	nWritten := new(int64)
	req, _ := NewRequest("POST", cst.ts.URL, io.LimitReader(countReader{neverEnding('a'), nWritten}, limit*200))

	// Send the POST, but don't care it succeeds or not. The
	// remote side is going to reply and then close the TCP
	// connection, and HTTP doesn't really define if that's
	// allowed or not. Some HTTP clients will get the response
	// and some (like ours, currently) will complain that the
	// request write failed, without reading the response.
	//
	// But that's okay, since what we're really testing is that
	// the remote side hung up on us before we wrote too much.
	_, _ = cst.c.Do(req)

	if atomic.LoadInt64(nWritten) > limit*100 {
		t.Errorf("handler restricted the request body to %d bytes, but client managed to write %d",
			limit, nWritten)
	}
}

// TestClientWriteShutdown tests that if the client shuts down the write
// side of their TCP connection, the server doesn't send a 400 Bad Request.
func TestClientWriteShutdown(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/17906")
	}
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {}))
	defer ts.Close()
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	err = conn.(*net.TCPConn).CloseWrite()
	if err != nil {
		t.Fatalf("CloseWrite: %v", err)
	}
	donec := make(chan bool)
	go func() {
		defer close(donec)
		bs, err := ioutil.ReadAll(conn)
		if err != nil {
			t.Errorf("ReadAll: %v", err)
		}
		got := string(bs)
		if got != "" {
			t.Errorf("read %q from server; want nothing", got)
		}
	}()
	select {
	case <-donec:
	case <-time.After(10 * time.Second):
		t.Fatalf("timeout")
	}
}

// Tests that chunked server responses that write 1 byte at a time are
// buffered before chunk headers are added, not after chunk headers.
func TestServerBufferedChunking(t *testing.T) {
	conn := new(testConn)
	conn.readBuf.Write([]byte("GET / HTTP/1.1\r\nHost: foo\r\n\r\n"))
	conn.closec = make(chan bool, 1)
	ls := &oneConnListener{conn}
	go Serve(ls, HandlerFunc(func(rw ResponseWriter, req *Request) {
		rw.(Flusher).Flush() // force the Header to be sent, in chunking mode, not counting the length
		rw.Write([]byte{'x'})
		rw.Write([]byte{'y'})
		rw.Write([]byte{'z'})
	}))
	<-conn.closec
	if !bytes.HasSuffix(conn.writeBuf.Bytes(), []byte("\r\n\r\n3\r\nxyz\r\n0\r\n\r\n")) {
		t.Errorf("response didn't end with a single 3 byte 'xyz' chunk; got:\n%q",
			conn.writeBuf.Bytes())
	}
}

// Tests that the server flushes its response headers out when it's
// ignoring the response body and waits a bit before forcefully
// closing the TCP connection, causing the client to get a RST.
// See https://golang.org/issue/3595
func TestServerGracefulClose(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		Error(w, "bye", StatusUnauthorized)
	}))
	defer ts.Close()

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	const bodySize = 5 << 20
	req := []byte(fmt.Sprintf("POST / HTTP/1.1\r\nHost: foo.com\r\nContent-Length: %d\r\n\r\n", bodySize))
	for i := 0; i < bodySize; i++ {
		req = append(req, 'x')
	}
	writeErr := make(chan error)
	go func() {
		_, err := conn.Write(req)
		writeErr <- err
	}()
	br := bufio.NewReader(conn)
	lineNum := 0
	for {
		line, err := br.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("ReadLine: %v", err)
		}
		lineNum++
		if lineNum == 1 && !strings.Contains(line, "401 Unauthorized") {
			t.Errorf("Response line = %q; want a 401", line)
		}
	}
	// Wait for write to finish. This is a broken pipe on both
	// Darwin and Linux, but checking this isn't the point of
	// the test.
	<-writeErr
}

func TestCaseSensitiveMethod_h1(t *testing.T) { testCaseSensitiveMethod(t, h1Mode) }
func TestCaseSensitiveMethod_h2(t *testing.T) { testCaseSensitiveMethod(t, h2Mode) }
func testCaseSensitiveMethod(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.Method != "get" {
			t.Errorf(`Got method %q; want "get"`, r.Method)
		}
	}))
	defer cst.close()
	req, _ := NewRequest("get", cst.ts.URL, nil)
	res, err := cst.c.Do(req)
	if err != nil {
		t.Error(err)
		return
	}

	res.Body.Close()
}

// TestContentLengthZero tests that for both an HTTP/1.0 and HTTP/1.1
// request (both keep-alive), when a Handler never writes any
// response, the net/http package adds a "Content-Length: 0" response
// header.
func TestContentLengthZero(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, req *Request) {}))
	defer ts.Close()

	for _, version := range []string{"HTTP/1.0", "HTTP/1.1"} {
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatalf("error dialing: %v", err)
		}
		_, err = fmt.Fprintf(conn, "GET / %v\r\nConnection: keep-alive\r\nHost: foo\r\n\r\n", version)
		if err != nil {
			t.Fatalf("error writing: %v", err)
		}
		req, _ := NewRequest("GET", "/", nil)
		res, err := ReadResponse(bufio.NewReader(conn), req)
		if err != nil {
			t.Fatalf("error reading response: %v", err)
		}
		if te := res.TransferEncoding; len(te) > 0 {
			t.Errorf("For version %q, Transfer-Encoding = %q; want none", version, te)
		}
		if cl := res.ContentLength; cl != 0 {
			t.Errorf("For version %q, Content-Length = %v; want 0", version, cl)
		}
		conn.Close()
	}
}

func TestCloseNotifier(t *testing.T) {
	defer afterTest(t)
	gotReq := make(chan bool, 1)
	sawClose := make(chan bool, 1)
	ts := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, req *Request) {
		gotReq <- true
		cc := rw.(CloseNotifier).CloseNotify()
		<-cc
		sawClose <- true
	}))
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("error dialing: %v", err)
	}
	diec := make(chan bool)
	go func() {
		_, err = fmt.Fprintf(conn, "GET / HTTP/1.1\r\nConnection: keep-alive\r\nHost: foo\r\n\r\n")
		if err != nil {
			t.Error(err)
			return
		}
		<-diec
		conn.Close()
	}()
For:
	for {
		select {
		case <-gotReq:
			diec <- true
		case <-sawClose:
			break For
		case <-time.After(5 * time.Second):
			t.Fatal("timeout")
		}
	}
	ts.Close()
}

// Tests that a pipelined request does not cause the first request's
// Handler's CloseNotify channel to fire.
//
// Issue 13165 (where it used to deadlock), but behavior changed in Issue 23921.
func TestCloseNotifierPipelined(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	gotReq := make(chan bool, 2)
	sawClose := make(chan bool, 2)
	ts := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, req *Request) {
		gotReq <- true
		cc := rw.(CloseNotifier).CloseNotify()
		select {
		case <-cc:
			t.Error("unexpected CloseNotify")
		case <-time.After(100 * time.Millisecond):
		}
		sawClose <- true
	}))
	defer ts.Close()
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("error dialing: %v", err)
	}
	diec := make(chan bool, 1)
	defer close(diec)
	go func() {
		const req = "GET / HTTP/1.1\r\nConnection: keep-alive\r\nHost: foo\r\n\r\n"
		_, err = io.WriteString(conn, req+req) // two requests
		if err != nil {
			t.Error(err)
			return
		}
		<-diec
		conn.Close()
	}()
	reqs := 0
	closes := 0
	for {
		select {
		case <-gotReq:
			reqs++
			if reqs > 2 {
				t.Fatal("too many requests")
			}
		case <-sawClose:
			closes++
			if closes > 1 {
				return
			}
		case <-time.After(5 * time.Second):
			ts.CloseClientConnections()
			t.Fatal("timeout")
		}
	}
}

func TestCloseNotifierChanLeak(t *testing.T) {
	defer afterTest(t)
	req := reqBytes("GET / HTTP/1.0\nHost: golang.org")
	for i := 0; i < 20; i++ {
		var output bytes.Buffer
		conn := &rwTestConn{
			Reader: bytes.NewReader(req),
			Writer: &output,
			closec: make(chan bool, 1),
		}
		ln := &oneConnListener{conn: conn}
		handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
			// Ignore the return value and never read from
			// it, testing that we don't leak goroutines
			// on the sending side:
			_ = rw.(CloseNotifier).CloseNotify()
		})
		go Serve(ln, handler)
		<-conn.closec
	}
}

// Tests that we can use CloseNotifier in one request, and later call Hijack
// on a second request on the same connection.
//
// It also tests that the connReader stitches together its background
// 1-byte read for CloseNotifier when CloseNotifier doesn't fire with
// the rest of the second HTTP later.
//
// Issue 9763.
// HTTP/1-only test. (http2 doesn't have Hijack)
func TestHijackAfterCloseNotifier(t *testing.T) {
	defer afterTest(t)
	script := make(chan string, 2)
	script <- "closenotify"
	script <- "hijack"
	close(script)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		plan := <-script
		switch plan {
		default:
			panic("bogus plan; too many requests")
		case "closenotify":
			w.(CloseNotifier).CloseNotify() // discard result
			w.Header().Set("X-Addr", r.RemoteAddr)
		case "hijack":
			c, _, err := w.(Hijacker).Hijack()
			if err != nil {
				t.Errorf("Hijack in Handler: %v", err)
				return
			}
			if _, ok := c.(*net.TCPConn); !ok {
				// Verify it's not wrapped in some type.
				// Not strictly a go1 compat issue, but in practice it probably is.
				t.Errorf("type of hijacked conn is %T; want *net.TCPConn", c)
			}
			fmt.Fprintf(c, "HTTP/1.0 200 OK\r\nX-Addr: %v\r\nContent-Length: 0\r\n\r\n", r.RemoteAddr)
			c.Close()
			return
		}
	}))
	defer ts.Close()
	res1, err := Get(ts.URL)
	if err != nil {
		log.Fatal(err)
	}
	res2, err := Get(ts.URL)
	if err != nil {
		log.Fatal(err)
	}
	addr1 := res1.Header.Get("X-Addr")
	addr2 := res2.Header.Get("X-Addr")
	if addr1 == "" || addr1 != addr2 {
		t.Errorf("addr1, addr2 = %q, %q; want same", addr1, addr2)
	}
}

func TestHijackBeforeRequestBodyRead(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	var requestBody = bytes.Repeat([]byte("a"), 1<<20)
	bodyOkay := make(chan bool, 1)
	gotCloseNotify := make(chan bool, 1)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(bodyOkay) // caller will read false if nothing else

		reqBody := r.Body
		r.Body = nil // to test that server.go doesn't use this value.

		gone := w.(CloseNotifier).CloseNotify()
		slurp, err := ioutil.ReadAll(reqBody)
		if err != nil {
			t.Errorf("Body read: %v", err)
			return
		}
		if len(slurp) != len(requestBody) {
			t.Errorf("Backend read %d request body bytes; want %d", len(slurp), len(requestBody))
			return
		}
		if !bytes.Equal(slurp, requestBody) {
			t.Error("Backend read wrong request body.") // 1MB; omitting details
			return
		}
		bodyOkay <- true
		select {
		case <-gone:
			gotCloseNotify <- true
		case <-time.After(5 * time.Second):
			gotCloseNotify <- false
		}
	}))
	defer ts.Close()

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	fmt.Fprintf(conn, "POST / HTTP/1.1\r\nHost: foo\r\nContent-Length: %d\r\n\r\n%s",
		len(requestBody), requestBody)
	if !<-bodyOkay {
		// already failed.
		return
	}
	conn.Close()
	if !<-gotCloseNotify {
		t.Error("timeout waiting for CloseNotify")
	}
}

func TestOptions(t *testing.T) {
	uric := make(chan string, 2) // only expect 1, but leave space for 2
	mux := NewServeMux()
	mux.HandleFunc("/", func(w ResponseWriter, r *Request) {
		uric <- r.RequestURI
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	// An OPTIONS * request should succeed.
	_, err = conn.Write([]byte("OPTIONS * HTTP/1.1\r\nHost: foo.com\r\n\r\n"))
	if err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(conn)
	res, err := ReadResponse(br, &Request{Method: "OPTIONS"})
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 200 {
		t.Errorf("Got non-200 response to OPTIONS *: %#v", res)
	}

	// A GET * request on a ServeMux should fail.
	_, err = conn.Write([]byte("GET * HTTP/1.1\r\nHost: foo.com\r\n\r\n"))
	if err != nil {
		t.Fatal(err)
	}
	res, err = ReadResponse(br, &Request{Method: "GET"})
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 400 {
		t.Errorf("Got non-400 response to GET *: %#v", res)
	}

	res, err = Get(ts.URL + "/second")
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if got := <-uric; got != "/second" {
		t.Errorf("Handler saw request for %q; want /second", got)
	}
}

// Tests regarding the ordering of Write, WriteHeader, Header, and
// Flush calls. In Go 1.0, rw.WriteHeader immediately flushed the
// (*response).header to the wire. In Go 1.1, the actual wire flush is
// delayed, so we could maybe tack on a Content-Length and better
// Content-Type after we see more (or all) of the output. To preserve
// compatibility with Go 1, we need to be careful to track which
// headers were live at the time of WriteHeader, so we write the same
// ones, even if the handler modifies them (~erroneously) after the
// first Write.
func TestHeaderToWire(t *testing.T) {
	tests := []struct {
		name    string
		handler func(ResponseWriter, *Request)
		check   func(got, logs string) error
	}{
		{
			name: "write without Header",
			handler: func(rw ResponseWriter, r *Request) {
				rw.Write([]byte("hello world"))
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Content-Length:") {
					return errors.New("no content-length")
				}
				if !strings.Contains(got, "Content-Type: text/plain") {
					return errors.New("no content-type")
				}
				return nil
			},
		},
		{
			name: "Header mutation before write",
			handler: func(rw ResponseWriter, r *Request) {
				h := rw.Header()
				h.Set("Content-Type", "some/type")
				rw.Write([]byte("hello world"))
				h.Set("Too-Late", "bogus")
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Content-Length:") {
					return errors.New("no content-length")
				}
				if !strings.Contains(got, "Content-Type: some/type") {
					return errors.New("wrong content-type")
				}
				if strings.Contains(got, "Too-Late") {
					return errors.New("don't want too-late header")
				}
				return nil
			},
		},
		{
			name: "write then useless Header mutation",
			handler: func(rw ResponseWriter, r *Request) {
				rw.Write([]byte("hello world"))
				rw.Header().Set("Too-Late", "Write already wrote headers")
			},
			check: func(got, logs string) error {
				if strings.Contains(got, "Too-Late") {
					return errors.New("header appeared from after WriteHeader")
				}
				return nil
			},
		},
		{
			name: "flush then write",
			handler: func(rw ResponseWriter, r *Request) {
				rw.(Flusher).Flush()
				rw.Write([]byte("post-flush"))
				rw.Header().Set("Too-Late", "Write already wrote headers")
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Transfer-Encoding: chunked") {
					return errors.New("not chunked")
				}
				if strings.Contains(got, "Too-Late") {
					return errors.New("header appeared from after WriteHeader")
				}
				return nil
			},
		},
		{
			name: "header then flush",
			handler: func(rw ResponseWriter, r *Request) {
				rw.Header().Set("Content-Type", "some/type")
				rw.(Flusher).Flush()
				rw.Write([]byte("post-flush"))
				rw.Header().Set("Too-Late", "Write already wrote headers")
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Transfer-Encoding: chunked") {
					return errors.New("not chunked")
				}
				if strings.Contains(got, "Too-Late") {
					return errors.New("header appeared from after WriteHeader")
				}
				if !strings.Contains(got, "Content-Type: some/type") {
					return errors.New("wrong content-type")
				}
				return nil
			},
		},
		{
			name: "sniff-on-first-write content-type",
			handler: func(rw ResponseWriter, r *Request) {
				rw.Write([]byte("<html><head></head><body>some html</body></html>"))
				rw.Header().Set("Content-Type", "x/wrong")
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Content-Type: text/html") {
					return errors.New("wrong content-type; want html")
				}
				return nil
			},
		},
		{
			name: "explicit content-type wins",
			handler: func(rw ResponseWriter, r *Request) {
				rw.Header().Set("Content-Type", "some/type")
				rw.Write([]byte("<html><head></head><body>some html</body></html>"))
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Content-Type: some/type") {
					return errors.New("wrong content-type; want html")
				}
				return nil
			},
		},
		{
			name: "empty handler",
			handler: func(rw ResponseWriter, r *Request) {
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Content-Length: 0") {
					return errors.New("want 0 content-length")
				}
				return nil
			},
		},
		{
			name: "only Header, no write",
			handler: func(rw ResponseWriter, r *Request) {
				rw.Header().Set("Some-Header", "some-value")
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "Some-Header") {
					return errors.New("didn't get header")
				}
				return nil
			},
		},
		{
			name: "WriteHeader call",
			handler: func(rw ResponseWriter, r *Request) {
				rw.WriteHeader(404)
				rw.Header().Set("Too-Late", "some-value")
			},
			check: func(got, logs string) error {
				if !strings.Contains(got, "404") {
					return errors.New("wrong status")
				}
				if strings.Contains(got, "Too-Late") {
					return errors.New("shouldn't have seen Too-Late")
				}
				return nil
			},
		},
	}
	for _, tc := range tests {
		ht := newHandlerTest(HandlerFunc(tc.handler))
		got := ht.rawResponse("GET / HTTP/1.1\nHost: golang.org")
		logs := ht.logbuf.String()
		if err := tc.check(got, logs); err != nil {
			t.Errorf("%s: %v\nGot response:\n%s\n\n%s", tc.name, err, got, logs)
		}
	}
}

// goTimeout runs f, failing t if f takes more than ns to complete.
func goTimeout(t *testing.T, d time.Duration, f func()) {
	ch := make(chan bool, 2)
	timer := time.AfterFunc(d, func() {
		t.Errorf("Timeout expired after %v", d)
		ch <- true
	})
	defer timer.Stop()
	go func() {
		defer func() { ch <- true }()
		f()
	}()
	<-ch
}

type errorListener struct {
	errs []error
}

func (l *errorListener) Accept() (c net.Conn, err error) {
	if len(l.errs) == 0 {
		return nil, io.EOF
	}
	err = l.errs[0]
	l.errs = l.errs[1:]
	return
}

func (l *errorListener) Close() error {
	return nil
}

func (l *errorListener) Addr() net.Addr {
	return dummyAddr("test-address")
}

func TestAcceptMaxFds(t *testing.T) {
	setParallel(t)

	ln := &errorListener{[]error{
		&net.OpError{
			Op:  "accept",
			Err: syscall.EMFILE,
		}}}
	server := &Server{
		Handler:  HandlerFunc(HandlerFunc(func(ResponseWriter, *Request) {})),
		ErrorLog: log.New(ioutil.Discard, "", 0), // noisy otherwise
	}
	err := server.Serve(ln)
	if err != io.EOF {
		t.Errorf("got error %v, want EOF", err)
	}
}

func TestWriteAfterHijack(t *testing.T) {
	req := reqBytes("GET / HTTP/1.1\nHost: golang.org")
	var buf bytes.Buffer
	wrotec := make(chan bool, 1)
	conn := &rwTestConn{
		Reader: bytes.NewReader(req),
		Writer: &buf,
		closec: make(chan bool, 1),
	}
	handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
		conn, bufrw, err := rw.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		go func() {
			bufrw.Write([]byte("[hijack-to-bufw]"))
			bufrw.Flush()
			conn.Write([]byte("[hijack-to-conn]"))
			conn.Close()
			wrotec <- true
		}()
	})
	ln := &oneConnListener{conn: conn}
	go Serve(ln, handler)
	<-conn.closec
	<-wrotec
	if g, w := buf.String(), "[hijack-to-bufw][hijack-to-conn]"; g != w {
		t.Errorf("wrote %q; want %q", g, w)
	}
}

func TestDoubleHijack(t *testing.T) {
	req := reqBytes("GET / HTTP/1.1\nHost: golang.org")
	var buf bytes.Buffer
	conn := &rwTestConn{
		Reader: bytes.NewReader(req),
		Writer: &buf,
		closec: make(chan bool, 1),
	}
	handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
		conn, _, err := rw.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		_, _, err = rw.(Hijacker).Hijack()
		if err == nil {
			t.Errorf("got err = nil;  want err != nil")
		}
		conn.Close()
	})
	ln := &oneConnListener{conn: conn}
	go Serve(ln, handler)
	<-conn.closec
}

// https://golang.org/issue/5955
// Note that this does not test the "request too large"
// exit path from the http server. This is intentional;
// not sending Connection: close is just a minor wire
// optimization and is pointless if dealing with a
// badly behaved client.
func TestHTTP10ConnectionHeader(t *testing.T) {
	defer afterTest(t)

	mux := NewServeMux()
	mux.Handle("/", HandlerFunc(func(ResponseWriter, *Request) {}))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// net/http uses HTTP/1.1 for requests, so write requests manually
	tests := []struct {
		req    string   // raw http request
		expect []string // expected Connection header(s)
	}{
		{
			req:    "GET / HTTP/1.0\r\n\r\n",
			expect: nil,
		},
		{
			req:    "OPTIONS * HTTP/1.0\r\n\r\n",
			expect: nil,
		},
		{
			req:    "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n",
			expect: []string{"keep-alive"},
		},
	}

	for _, tt := range tests {
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal("dial err:", err)
		}

		_, err = fmt.Fprint(conn, tt.req)
		if err != nil {
			t.Fatal("conn write err:", err)
		}

		resp, err := ReadResponse(bufio.NewReader(conn), &Request{Method: "GET"})
		if err != nil {
			t.Fatal("ReadResponse err:", err)
		}
		conn.Close()
		resp.Body.Close()

		got := resp.Header["Connection"]
		if !reflect.DeepEqual(got, tt.expect) {
			t.Errorf("wrong Connection headers for request %q. Got %q expect %q", tt.req, got, tt.expect)
		}
	}
}

// See golang.org/issue/5660
func TestServerReaderFromOrder_h1(t *testing.T) { testServerReaderFromOrder(t, h1Mode) }
func TestServerReaderFromOrder_h2(t *testing.T) { testServerReaderFromOrder(t, h2Mode) }
func testServerReaderFromOrder(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	pr, pw := io.Pipe()
	const size = 3 << 20
	cst := newClientServerTest(t, h2, HandlerFunc(func(rw ResponseWriter, req *Request) {
		rw.Header().Set("Content-Type", "text/plain") // prevent sniffing path
		done := make(chan bool)
		go func() {
			io.Copy(rw, pr)
			close(done)
		}()
		time.Sleep(25 * time.Millisecond) // give Copy a chance to break things
		n, err := io.Copy(ioutil.Discard, req.Body)
		if err != nil {
			t.Errorf("handler Copy: %v", err)
			return
		}
		if n != size {
			t.Errorf("handler Copy = %d; want %d", n, size)
		}
		pw.Write([]byte("hi"))
		pw.Close()
		<-done
	}))
	defer cst.close()

	req, err := NewRequest("POST", cst.ts.URL, io.LimitReader(neverEnding('a'), size))
	if err != nil {
		t.Fatal(err)
	}
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	all, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if string(all) != "hi" {
		t.Errorf("Body = %q; want hi", all)
	}
}

// Issue 6157, Issue 6685
func TestCodesPreventingContentTypeAndBody(t *testing.T) {
	for _, code := range []int{StatusNotModified, StatusNoContent, StatusContinue} {
		ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
			if r.URL.Path == "/header" {
				w.Header().Set("Content-Length", "123")
			}
			w.WriteHeader(code)
			if r.URL.Path == "/more" {
				w.Write([]byte("stuff"))
			}
		}))
		for _, req := range []string{
			"GET / HTTP/1.0",
			"GET /header HTTP/1.0",
			"GET /more HTTP/1.0",
			"GET / HTTP/1.1\nHost: foo",
			"GET /header HTTP/1.1\nHost: foo",
			"GET /more HTTP/1.1\nHost: foo",
		} {
			got := ht.rawResponse(req)
			wantStatus := fmt.Sprintf("%d %s", code, StatusText(code))
			if !strings.Contains(got, wantStatus) {
				t.Errorf("Code %d: Wanted %q Modified for %q: %s", code, wantStatus, req, got)
			} else if strings.Contains(got, "Content-Length") {
				t.Errorf("Code %d: Got a Content-Length from %q: %s", code, req, got)
			} else if strings.Contains(got, "stuff") {
				t.Errorf("Code %d: Response contains a body from %q: %s", code, req, got)
			}
		}
	}
}

func TestContentTypeOkayOn204(t *testing.T) {
	ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "123") // suppressed
		w.Header().Set("Content-Type", "foo/bar")
		w.WriteHeader(204)
	}))
	got := ht.rawResponse("GET / HTTP/1.1\nHost: foo")
	if !strings.Contains(got, "Content-Type: foo/bar") {
		t.Errorf("Response = %q; want Content-Type: foo/bar", got)
	}
	if strings.Contains(got, "Content-Length: 123") {
		t.Errorf("Response = %q; don't want a Content-Length", got)
	}
}

// Issue 6995
// A server Handler can receive a Request, and then turn around and
// give a copy of that Request.Body out to the Transport (e.g. any
// proxy).  So then two people own that Request.Body (both the server
// and the http client), and both think they can close it on failure.
// Therefore, all incoming server requests Bodies need to be thread-safe.
func TestTransportAndServerSharedBodyRace_h1(t *testing.T) {
	testTransportAndServerSharedBodyRace(t, h1Mode)
}
func TestTransportAndServerSharedBodyRace_h2(t *testing.T) {
	testTransportAndServerSharedBodyRace(t, h2Mode)
}
func testTransportAndServerSharedBodyRace(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)

	const bodySize = 1 << 20

	// errorf is like t.Errorf, but also writes to println. When
	// this test fails, it hangs. This helps debugging and I've
	// added this enough times "temporarily".  It now gets added
	// full time.
	errorf := func(format string, args ...interface{}) {
		v := fmt.Sprintf(format, args...)
		println(v)
		t.Error(v)
	}

	unblockBackend := make(chan bool)
	backend := newClientServerTest(t, h2, HandlerFunc(func(rw ResponseWriter, req *Request) {
		gone := rw.(CloseNotifier).CloseNotify()
		didCopy := make(chan interface{})
		go func() {
			n, err := io.CopyN(rw, req.Body, bodySize)
			didCopy <- []interface{}{n, err}
		}()
		isGone := false
	Loop:
		for {
			select {
			case <-didCopy:
				break Loop
			case <-gone:
				isGone = true
			case <-time.After(time.Second):
				println("1 second passes in backend, proxygone=", isGone)
			}
		}
		<-unblockBackend
	}))
	var quitTimer *time.Timer
	defer func() { quitTimer.Stop() }()
	defer backend.close()

	backendRespc := make(chan *Response, 1)
	var proxy *clientServerTest
	proxy = newClientServerTest(t, h2, HandlerFunc(func(rw ResponseWriter, req *Request) {
		req2, _ := NewRequest("POST", backend.ts.URL, req.Body)
		req2.ContentLength = bodySize
		cancel := make(chan struct{})
		req2.Cancel = cancel

		bresp, err := proxy.c.Do(req2)
		if err != nil {
			errorf("Proxy outbound request: %v", err)
			return
		}
		_, err = io.CopyN(ioutil.Discard, bresp.Body, bodySize/2)
		if err != nil {
			errorf("Proxy copy error: %v", err)
			return
		}
		backendRespc <- bresp // to close later

		// Try to cause a race: Both the Transport and the proxy handler's Server
		// will try to read/close req.Body (aka req2.Body)
		if h2 {
			close(cancel)
		} else {
			proxy.c.Transport.(*Transport).CancelRequest(req2)
		}
		rw.Write([]byte("OK"))
	}))
	defer proxy.close()
	defer func() {
		// Before we shut down our two httptest.Servers, start a timer.
		// We choose 7 seconds because httptest.Server starts logging
		// warnings to stderr at 5 seconds. If we don't disarm this bomb
		// in 7 seconds (after the two httptest.Server.Close calls above),
		// then we explode with stacks.
		quitTimer = time.AfterFunc(7*time.Second, func() {
			debug.SetTraceback("ALL")
			stacks := make([]byte, 1<<20)
			stacks = stacks[:runtime.Stack(stacks, true)]
			fmt.Fprintf(os.Stderr, "%s", stacks)
			log.Fatalf("Timeout.")
		})
	}()

	defer close(unblockBackend)
	req, _ := NewRequest("POST", proxy.ts.URL, io.LimitReader(neverEnding('a'), bodySize))
	res, err := proxy.c.Do(req)
	if err != nil {
		t.Fatalf("Original request: %v", err)
	}

	// Cleanup, so we don't leak goroutines.
	res.Body.Close()
	select {
	case res := <-backendRespc:
		res.Body.Close()
	default:
		// We failed earlier. (e.g. on proxy.c.Do(req2))
	}
}

// Test that a hanging Request.Body.Read from another goroutine can't
// cause the Handler goroutine's Request.Body.Close to block.
// See issue 7121.
func TestRequestBodyCloseDoesntBlock(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	defer afterTest(t)

	readErrCh := make(chan error, 1)
	errCh := make(chan error, 2)

	server := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, req *Request) {
		go func(body io.Reader) {
			_, err := body.Read(make([]byte, 100))
			readErrCh <- err
		}(req.Body)
		time.Sleep(500 * time.Millisecond)
	}))
	defer server.Close()

	closeConn := make(chan bool)
	defer close(closeConn)
	go func() {
		conn, err := net.Dial("tcp", server.Listener.Addr().String())
		if err != nil {
			errCh <- err
			return
		}
		defer conn.Close()
		_, err = conn.Write([]byte("POST / HTTP/1.1\r\nConnection: close\r\nHost: foo\r\nContent-Length: 100000\r\n\r\n"))
		if err != nil {
			errCh <- err
			return
		}
		// And now just block, making the server block on our
		// 100000 bytes of body that will never arrive.
		<-closeConn
	}()
	select {
	case err := <-readErrCh:
		if err == nil {
			t.Error("Read was nil. Expected error.")
		}
	case err := <-errCh:
		t.Error(err)
	case <-time.After(5 * time.Second):
		t.Error("timeout")
	}
}

// test that ResponseWriter implements io.StringWriter.
func TestResponseWriterWriteString(t *testing.T) {
	okc := make(chan bool, 1)
	ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
		_, ok := w.(io.StringWriter)
		okc <- ok
	}))
	ht.rawResponse("GET / HTTP/1.0")
	select {
	case ok := <-okc:
		if !ok {
			t.Error("ResponseWriter did not implement io.StringWriter")
		}
	default:
		t.Error("handler was never called")
	}
}

func TestAppendTime(t *testing.T) {
	var b [len(TimeFormat)]byte
	t1 := time.Date(2013, 9, 21, 15, 41, 0, 0, time.FixedZone("CEST", 2*60*60))
	res := ExportAppendTime(b[:0], t1)
	t2, err := ParseTime(string(res))
	if err != nil {
		t.Fatalf("Error parsing time: %s", err)
	}
	if !t1.Equal(t2) {
		t.Fatalf("Times differ; expected: %v, got %v (%s)", t1, t2, string(res))
	}
}

func TestServerConnState(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	handler := map[string]func(w ResponseWriter, r *Request){
		"/": func(w ResponseWriter, r *Request) {
			fmt.Fprintf(w, "Hello.")
		},
		"/close": func(w ResponseWriter, r *Request) {
			w.Header().Set("Connection", "close")
			fmt.Fprintf(w, "Hello.")
		},
		"/hijack": func(w ResponseWriter, r *Request) {
			c, _, _ := w.(Hijacker).Hijack()
			c.Write([]byte("HTTP/1.0 200 OK\r\nConnection: close\r\n\r\nHello."))
			c.Close()
		},
		"/hijack-panic": func(w ResponseWriter, r *Request) {
			c, _, _ := w.(Hijacker).Hijack()
			c.Write([]byte("HTTP/1.0 200 OK\r\nConnection: close\r\n\r\nHello."))
			c.Close()
			panic("intentional panic")
		},
	}
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		handler[r.URL.Path](w, r)
	}))
	defer ts.Close()

	var mu sync.Mutex // guard stateLog and connID
	var stateLog = map[int][]ConnState{}
	var connID = map[net.Conn]int{}

	ts.Config.ErrorLog = log.New(ioutil.Discard, "", 0)
	ts.Config.ConnState = func(c net.Conn, state ConnState) {
		if c == nil {
			t.Errorf("nil conn seen in state %s", state)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		id, ok := connID[c]
		if !ok {
			id = len(connID) + 1
			connID[c] = id
		}
		stateLog[id] = append(stateLog[id], state)
	}
	ts.Start()

	c := ts.Client()

	mustGet := func(url string, headers ...string) {
		req, err := NewRequest("GET", url, nil)
		if err != nil {
			t.Fatal(err)
		}
		for len(headers) > 0 {
			req.Header.Add(headers[0], headers[1])
			headers = headers[2:]
		}
		res, err := c.Do(req)
		if err != nil {
			t.Errorf("Error fetching %s: %v", url, err)
			return
		}
		_, err = ioutil.ReadAll(res.Body)
		defer res.Body.Close()
		if err != nil {
			t.Errorf("Error reading %s: %v", url, err)
		}
	}

	mustGet(ts.URL + "/")
	mustGet(ts.URL + "/close")

	mustGet(ts.URL + "/")
	mustGet(ts.URL+"/", "Connection", "close")

	mustGet(ts.URL + "/hijack")
	mustGet(ts.URL + "/hijack-panic")

	// New->Closed
	{
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		c.Close()
	}

	// New->Active->Closed
	{
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(c, "BOGUS REQUEST\r\n\r\n"); err != nil {
			t.Fatal(err)
		}
		c.Read(make([]byte, 1)) // block until server hangs up on us
		c.Close()
	}

	// New->Idle->Closed
	{
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(c, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n"); err != nil {
			t.Fatal(err)
		}
		res, err := ReadResponse(bufio.NewReader(c), nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(ioutil.Discard, res.Body); err != nil {
			t.Fatal(err)
		}
		c.Close()
	}

	want := map[int][]ConnState{
		1: {StateNew, StateActive, StateIdle, StateActive, StateClosed},
		2: {StateNew, StateActive, StateIdle, StateActive, StateClosed},
		3: {StateNew, StateActive, StateHijacked},
		4: {StateNew, StateActive, StateHijacked},
		5: {StateNew, StateClosed},
		6: {StateNew, StateActive, StateClosed},
		7: {StateNew, StateActive, StateIdle, StateClosed},
	}
	logString := func(m map[int][]ConnState) string {
		var b bytes.Buffer
		var keys []int
		for id := range m {
			keys = append(keys, id)
		}
		sort.Ints(keys)
		for _, id := range keys {
			fmt.Fprintf(&b, "Conn %d: ", id)
			for _, s := range m[id] {
				fmt.Fprintf(&b, "%s ", s)
			}
			b.WriteString("\n")
		}
		return b.String()
	}

	for i := 0; i < 5; i++ {
		time.Sleep(time.Duration(i) * 50 * time.Millisecond)
		mu.Lock()
		match := reflect.DeepEqual(stateLog, want)
		mu.Unlock()
		if match {
			return
		}
	}

	mu.Lock()
	t.Errorf("Unexpected events.\nGot log:\n%s\n   Want:\n%s\n", logString(stateLog), logString(want))
	mu.Unlock()
}

func TestServerKeepAlivesEnabled(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {}))
	ts.Config.SetKeepAlivesEnabled(false)
	ts.Start()
	defer ts.Close()
	res, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if !res.Close {
		t.Errorf("Body.Close == false; want true")
	}
}

// golang.org/issue/7856
func TestServerEmptyBodyRace_h1(t *testing.T) { testServerEmptyBodyRace(t, h1Mode) }
func TestServerEmptyBodyRace_h2(t *testing.T) { testServerEmptyBodyRace(t, h2Mode) }
func testServerEmptyBodyRace(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	var n int32
	cst := newClientServerTest(t, h2, HandlerFunc(func(rw ResponseWriter, req *Request) {
		atomic.AddInt32(&n, 1)
	}))
	defer cst.close()
	var wg sync.WaitGroup
	const reqs = 20
	for i := 0; i < reqs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := cst.c.Get(cst.ts.URL)
			if err != nil {
				t.Error(err)
				return
			}
			defer res.Body.Close()
			_, err = io.Copy(ioutil.Discard, res.Body)
			if err != nil {
				t.Error(err)
				return
			}
		}()
	}
	wg.Wait()
	if got := atomic.LoadInt32(&n); got != reqs {
		t.Errorf("handler ran %d times; want %d", got, reqs)
	}
}

func TestServerConnStateNew(t *testing.T) {
	sawNew := false // if the test is buggy, we'll race on this variable.
	srv := &Server{
		ConnState: func(c net.Conn, state ConnState) {
			if state == StateNew {
				sawNew = true // testing that this write isn't racy
			}
		},
		Handler: HandlerFunc(func(w ResponseWriter, r *Request) {}), // irrelevant
	}
	srv.Serve(&oneConnListener{
		conn: &rwTestConn{
			Reader: strings.NewReader("GET / HTTP/1.1\r\nHost: foo\r\n\r\n"),
			Writer: ioutil.Discard,
		},
	})
	if !sawNew { // testing that this read isn't racy
		t.Error("StateNew not seen")
	}
}

type closeWriteTestConn struct {
	rwTestConn
	didCloseWrite bool
}

func (c *closeWriteTestConn) CloseWrite() error {
	c.didCloseWrite = true
	return nil
}

func TestCloseWrite(t *testing.T) {
	setParallel(t)
	var srv Server
	var testConn closeWriteTestConn
	c := ExportServerNewConn(&srv, &testConn)
	ExportCloseWriteAndWait(c)
	if !testConn.didCloseWrite {
		t.Error("didn't see CloseWrite call")
	}
}

// This verifies that a handler can Flush and then Hijack.
//
// An similar test crashed once during development, but it was only
// testing this tangentially and temporarily until another TODO was
// fixed.
//
// So add an explicit test for this.
func TestServerFlushAndHijack(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, "Hello, ")
		w.(Flusher).Flush()
		conn, buf, _ := w.(Hijacker).Hijack()
		buf.WriteString("6\r\nworld!\r\n0\r\n\r\n")
		if err := buf.Flush(); err != nil {
			t.Error(err)
		}
		if err := conn.Close(); err != nil {
			t.Error(err)
		}
	}))
	defer ts.Close()
	res, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	all, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if want := "Hello, world!"; string(all) != want {
		t.Errorf("Got %q; want %q", all, want)
	}
}

// golang.org/issue/8534 -- the Server shouldn't reuse a connection
// for keep-alive after it's seen any Write error (e.g. a timeout) on
// that net.Conn.
//
// To test, verify we don't timeout or see fewer unique client
// addresses (== unique connections) than requests.
func TestServerKeepAliveAfterWriteError(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	defer afterTest(t)
	const numReq = 3
	addrc := make(chan string, numReq)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		addrc <- r.RemoteAddr
		time.Sleep(500 * time.Millisecond)
		w.(Flusher).Flush()
	}))
	ts.Config.WriteTimeout = 250 * time.Millisecond
	ts.Start()
	defer ts.Close()

	errc := make(chan error, numReq)
	go func() {
		defer close(errc)
		for i := 0; i < numReq; i++ {
			res, err := Get(ts.URL)
			if res != nil {
				res.Body.Close()
			}
			errc <- err
		}
	}()

	timeout := time.NewTimer(numReq * 2 * time.Second) // 4x overkill
	defer timeout.Stop()
	addrSeen := map[string]bool{}
	numOkay := 0
	for {
		select {
		case v := <-addrc:
			addrSeen[v] = true
		case err, ok := <-errc:
			if !ok {
				if len(addrSeen) != numReq {
					t.Errorf("saw %d unique client addresses; want %d", len(addrSeen), numReq)
				}
				if numOkay != 0 {
					t.Errorf("got %d successful client requests; want 0", numOkay)
				}
				return
			}
			if err == nil {
				numOkay++
			}
		case <-timeout.C:
			t.Fatal("timeout waiting for requests to complete")
		}
	}
}

// Issue 9987: shouldn't add automatic Content-Length (or
// Content-Type) if a Transfer-Encoding was set by the handler.
func TestNoContentLengthIfTransferEncoding(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Transfer-Encoding", "foo")
		io.WriteString(w, "<html>")
	}))
	defer ts.Close()
	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	if _, err := io.WriteString(c, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	bs := bufio.NewScanner(c)
	var got bytes.Buffer
	for bs.Scan() {
		if strings.TrimSpace(bs.Text()) == "" {
			break
		}
		got.WriteString(bs.Text())
		got.WriteByte('\n')
	}
	if err := bs.Err(); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got.String(), "Content-Length") {
		t.Errorf("Unexpected Content-Length in response headers: %s", got.String())
	}
	if strings.Contains(got.String(), "Content-Type") {
		t.Errorf("Unexpected Content-Type in response headers: %s", got.String())
	}
}

// tolerate extra CRLF(s) before Request-Line on subsequent requests on a conn
// Issue 10876.
func TestTolerateCRLFBeforeRequestLine(t *testing.T) {
	req := []byte("POST / HTTP/1.1\r\nHost: golang.org\r\nContent-Length: 3\r\n\r\nABC" +
		"\r\n\r\n" + // <-- this stuff is bogus, but we'll ignore it
		"GET / HTTP/1.1\r\nHost: golang.org\r\n\r\n")
	var buf bytes.Buffer
	conn := &rwTestConn{
		Reader: bytes.NewReader(req),
		Writer: &buf,
		closec: make(chan bool, 1),
	}
	ln := &oneConnListener{conn: conn}
	numReq := 0
	go Serve(ln, HandlerFunc(func(rw ResponseWriter, r *Request) {
		numReq++
	}))
	<-conn.closec
	if numReq != 2 {
		t.Errorf("num requests = %d; want 2", numReq)
		t.Logf("Res: %s", buf.Bytes())
	}
}

func TestIssue13893_Expect100(t *testing.T) {
	// test that the Server doesn't filter out Expect headers.
	req := reqBytes(`PUT /readbody HTTP/1.1
User-Agent: PycURL/7.22.0
Host: 127.0.0.1:9000
Accept: */*
Expect: 100-continue
Content-Length: 10

HelloWorld

`)
	var buf bytes.Buffer
	conn := &rwTestConn{
		Reader: bytes.NewReader(req),
		Writer: &buf,
		closec: make(chan bool, 1),
	}
	ln := &oneConnListener{conn: conn}
	go Serve(ln, HandlerFunc(func(w ResponseWriter, r *Request) {
		if _, ok := r.Header["Expect"]; !ok {
			t.Error("Expect header should not be filtered out")
		}
	}))
	<-conn.closec
}

func TestIssue11549_Expect100(t *testing.T) {
	req := reqBytes(`PUT /readbody HTTP/1.1
User-Agent: PycURL/7.22.0
Host: 127.0.0.1:9000
Accept: */*
Expect: 100-continue
Content-Length: 10

HelloWorldPUT /noreadbody HTTP/1.1
User-Agent: PycURL/7.22.0
Host: 127.0.0.1:9000
Accept: */*
Expect: 100-continue
Content-Length: 10

GET /should-be-ignored HTTP/1.1
Host: foo

`)
	var buf bytes.Buffer
	conn := &rwTestConn{
		Reader: bytes.NewReader(req),
		Writer: &buf,
		closec: make(chan bool, 1),
	}
	ln := &oneConnListener{conn: conn}
	numReq := 0
	go Serve(ln, HandlerFunc(func(w ResponseWriter, r *Request) {
		numReq++
		if r.URL.Path == "/readbody" {
			ioutil.ReadAll(r.Body)
		}
		io.WriteString(w, "Hello world!")
	}))
	<-conn.closec
	if numReq != 2 {
		t.Errorf("num requests = %d; want 2", numReq)
	}
	if !strings.Contains(buf.String(), "Connection: close\r\n") {
		t.Errorf("expected 'Connection: close' in response; got: %s", buf.String())
	}
}

// If a Handler finishes and there's an unread request body,
// verify the server try to do implicit read on it before replying.
func TestHandlerFinishSkipBigContentLengthRead(t *testing.T) {
	setParallel(t)
	conn := &testConn{closec: make(chan bool)}
	conn.readBuf.Write([]byte(fmt.Sprintf(
		"POST / HTTP/1.1\r\n" +
			"Host: test\r\n" +
			"Content-Length: 9999999999\r\n" +
			"\r\n" + strings.Repeat("a", 1<<20))))

	ls := &oneConnListener{conn}
	var inHandlerLen int
	go Serve(ls, HandlerFunc(func(rw ResponseWriter, req *Request) {
		inHandlerLen = conn.readBuf.Len()
		rw.WriteHeader(404)
	}))
	<-conn.closec
	afterHandlerLen := conn.readBuf.Len()

	if afterHandlerLen != inHandlerLen {
		t.Errorf("unexpected implicit read. Read buffer went from %d -> %d", inHandlerLen, afterHandlerLen)
	}
}

func TestHandlerSetsBodyNil_h1(t *testing.T) { testHandlerSetsBodyNil(t, h1Mode) }
func TestHandlerSetsBodyNil_h2(t *testing.T) { testHandlerSetsBodyNil(t, h2Mode) }
func testHandlerSetsBodyNil(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		r.Body = nil
		fmt.Fprintf(w, "%v", r.RemoteAddr)
	}))
	defer cst.close()
	get := func() string {
		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		return string(slurp)
	}
	a, b := get(), get()
	if a != b {
		t.Errorf("Failed to reuse connections between requests: %v vs %v", a, b)
	}
}

// Test that we validate the Host header.
// Issue 11206 (invalid bytes in Host) and 13624 (Host present in HTTP/1.1)
func TestServerValidatesHostHeader(t *testing.T) {
	tests := []struct {
		proto string
		host  string
		want  int
	}{
		{"HTTP/0.9", "", 400},

		{"HTTP/1.1", "", 400},
		{"HTTP/1.1", "Host: \r\n", 200},
		{"HTTP/1.1", "Host: 1.2.3.4\r\n", 200},
		{"HTTP/1.1", "Host: foo.com\r\n", 200},
		{"HTTP/1.1", "Host: foo-bar_baz.com\r\n", 200},
		{"HTTP/1.1", "Host: foo.com:80\r\n", 200},
		{"HTTP/1.1", "Host: ::1\r\n", 200},
		{"HTTP/1.1", "Host: [::1]\r\n", 200}, // questionable without port, but accept it
		{"HTTP/1.1", "Host: [::1]:80\r\n", 200},
		{"HTTP/1.1", "Host: [::1%25en0]:80\r\n", 200},
		{"HTTP/1.1", "Host: 1.2.3.4\r\n", 200},
		{"HTTP/1.1", "Host: \x06\r\n", 400},
		{"HTTP/1.1", "Host: \xff\r\n", 400},
		{"HTTP/1.1", "Host: {\r\n", 400},
		{"HTTP/1.1", "Host: }\r\n", 400},
		{"HTTP/1.1", "Host: first\r\nHost: second\r\n", 400},

		// HTTP/1.0 can lack a host header, but if present
		// must play by the rules too:
		{"HTTP/1.0", "", 200},
		{"HTTP/1.0", "Host: first\r\nHost: second\r\n", 400},
		{"HTTP/1.0", "Host: \xff\r\n", 400},

		// Make an exception for HTTP upgrade requests:
		{"PRI * HTTP/2.0", "", 200},

		// Also an exception for CONNECT requests: (Issue 18215)
		{"CONNECT golang.org:443 HTTP/1.1", "", 200},

		// But not other HTTP/2 stuff:
		{"PRI / HTTP/2.0", "", 400},
		{"GET / HTTP/2.0", "", 400},
		{"GET / HTTP/3.0", "", 400},
	}
	for _, tt := range tests {
		conn := &testConn{closec: make(chan bool, 1)}
		methodTarget := "GET / "
		if !strings.HasPrefix(tt.proto, "HTTP/") {
			methodTarget = ""
		}
		io.WriteString(&conn.readBuf, methodTarget+tt.proto+"\r\n"+tt.host+"\r\n")

		ln := &oneConnListener{conn}
		srv := Server{
			ErrorLog: quietLog,
			Handler:  HandlerFunc(func(ResponseWriter, *Request) {}),
		}
		go srv.Serve(ln)
		<-conn.closec
		res, err := ReadResponse(bufio.NewReader(&conn.writeBuf), nil)
		if err != nil {
			t.Errorf("For %s %q, ReadResponse: %v", tt.proto, tt.host, res)
			continue
		}
		if res.StatusCode != tt.want {
			t.Errorf("For %s %q, Status = %d; want %d", tt.proto, tt.host, res.StatusCode, tt.want)
		}
	}
}

func TestServerHandlersCanHandleH2PRI(t *testing.T) {
	const upgradeResponse = "upgrade here"
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, br, err := w.(Hijacker).Hijack()
		defer conn.Close()
		if r.Method != "PRI" || r.RequestURI != "*" {
			t.Errorf("Got method/target %q %q; want PRI *", r.Method, r.RequestURI)
			return
		}
		if !r.Close {
			t.Errorf("Request.Close = true; want false")
		}
		const want = "SM\r\n\r\n"
		buf := make([]byte, len(want))
		n, err := io.ReadFull(br, buf)
		if err != nil || string(buf[:n]) != want {
			t.Errorf("Read = %v, %v (%q), want %q", n, err, buf[:n], want)
			return
		}
		io.WriteString(conn, upgradeResponse)
	}))
	defer ts.Close()

	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	io.WriteString(c, "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
	slurp, err := ioutil.ReadAll(c)
	if err != nil {
		t.Fatal(err)
	}
	if string(slurp) != upgradeResponse {
		t.Errorf("Handler response = %q; want %q", slurp, upgradeResponse)
	}
}

// Test that we validate the valid bytes in HTTP/1 headers.
// Issue 11207.
func TestServerValidatesHeaders(t *testing.T) {
	setParallel(t)
	tests := []struct {
		header string
		want   int
	}{
		{"", 200},
		{"Foo: bar\r\n", 200},
		{"X-Foo: bar\r\n", 200},
		{"Foo: a space\r\n", 200},

		{"A space: foo\r\n", 400},                            // space in header
		{"foo\xffbar: foo\r\n", 400},                         // binary in header
		{"foo\x00bar: foo\r\n", 400},                         // binary in header
		{"Foo: " + strings.Repeat("x", 1<<21) + "\r\n", 431}, // header too large

		{"foo: foo foo\r\n", 200},    // LWS space is okay
		{"foo: foo\tfoo\r\n", 200},   // LWS tab is okay
		{"foo: foo\x00foo\r\n", 400}, // CTL 0x00 in value is bad
		{"foo: foo\x7ffoo\r\n", 400}, // CTL 0x7f in value is bad
		{"foo: foo\xfffoo\r\n", 200}, // non-ASCII high octets in value are fine
	}
	for _, tt := range tests {
		conn := &testConn{closec: make(chan bool, 1)}
		io.WriteString(&conn.readBuf, "GET / HTTP/1.1\r\nHost: foo\r\n"+tt.header+"\r\n")

		ln := &oneConnListener{conn}
		srv := Server{
			ErrorLog: quietLog,
			Handler:  HandlerFunc(func(ResponseWriter, *Request) {}),
		}
		go srv.Serve(ln)
		<-conn.closec
		res, err := ReadResponse(bufio.NewReader(&conn.writeBuf), nil)
		if err != nil {
			t.Errorf("For %q, ReadResponse: %v", tt.header, res)
			continue
		}
		if res.StatusCode != tt.want {
			t.Errorf("For %q, Status = %d; want %d", tt.header, res.StatusCode, tt.want)
		}
	}
}

func TestServerRequestContextCancel_ServeHTTPDone_h1(t *testing.T) {
	testServerRequestContextCancel_ServeHTTPDone(t, h1Mode)
}
func TestServerRequestContextCancel_ServeHTTPDone_h2(t *testing.T) {
	testServerRequestContextCancel_ServeHTTPDone(t, h2Mode)
}
func testServerRequestContextCancel_ServeHTTPDone(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	ctxc := make(chan context.Context, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		ctx := r.Context()
		select {
		case <-ctx.Done():
			t.Error("should not be Done in ServeHTTP")
		default:
		}
		ctxc <- ctx
	}))
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	ctx := <-ctxc
	select {
	case <-ctx.Done():
	default:
		t.Error("context should be done after ServeHTTP completes")
	}
}

// Tests that the Request.Context available to the Handler is canceled
// if the peer closes their TCP connection. This requires that the server
// is always blocked in a Read call so it notices the EOF from the client.
// See issues 15927 and 15224.
func TestServerRequestContextCancel_ConnClose(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	inHandler := make(chan struct{})
	handlerDone := make(chan struct{})
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		close(inHandler)
		select {
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
			t.Errorf("timeout waiting for context to be done")
		}
		close(handlerDone)
	}))
	defer ts.Close()
	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	io.WriteString(c, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n")
	select {
	case <-inHandler:
	case <-time.After(3 * time.Second):
		t.Fatalf("timeout waiting to see ServeHTTP get called")
	}
	c.Close() // this should trigger the context being done

	select {
	case <-handlerDone:
	case <-time.After(4 * time.Second):
		t.Fatalf("timeout waiting to see ServeHTTP exit")
	}
}

func TestServerContext_ServerContextKey_h1(t *testing.T) {
	testServerContext_ServerContextKey(t, h1Mode)
}
func TestServerContext_ServerContextKey_h2(t *testing.T) {
	testServerContext_ServerContextKey(t, h2Mode)
}
func testServerContext_ServerContextKey(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		ctx := r.Context()
		got := ctx.Value(ServerContextKey)
		if _, ok := got.(*Server); !ok {
			t.Errorf("context value = %T; want *http.Server", got)
		}
	}))
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

func TestServerContext_LocalAddrContextKey_h1(t *testing.T) {
	testServerContext_LocalAddrContextKey(t, h1Mode)
}
func TestServerContext_LocalAddrContextKey_h2(t *testing.T) {
	testServerContext_LocalAddrContextKey(t, h2Mode)
}
func testServerContext_LocalAddrContextKey(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	ch := make(chan interface{}, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		ch <- r.Context().Value(LocalAddrContextKey)
	}))
	defer cst.close()
	if _, err := cst.c.Head(cst.ts.URL); err != nil {
		t.Fatal(err)
	}

	host := cst.ts.Listener.Addr().String()
	select {
	case got := <-ch:
		if addr, ok := got.(net.Addr); !ok {
			t.Errorf("local addr value = %T; want net.Addr", got)
		} else if fmt.Sprint(addr) != host {
			t.Errorf("local addr = %v; want %v", addr, host)
		}
	case <-time.After(5 * time.Second):
		t.Error("timed out")
	}
}

// https://golang.org/issue/15960
func TestHandlerSetTransferEncodingChunked(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Transfer-Encoding", "chunked")
		w.Write([]byte("hello"))
	}))
	resp := ht.rawResponse("GET / HTTP/1.1\nHost: foo")
	const hdr = "Transfer-Encoding: chunked"
	if n := strings.Count(resp, hdr); n != 1 {
		t.Errorf("want 1 occurrence of %q in response, got %v\nresponse: %v", hdr, n, resp)
	}
}

// https://golang.org/issue/16063
func TestHandlerSetTransferEncodingGzip(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Transfer-Encoding", "gzip")
		gz := gzip.NewWriter(w)
		gz.Write([]byte("hello"))
		gz.Close()
	}))
	resp := ht.rawResponse("GET / HTTP/1.1\nHost: foo")
	for _, v := range []string{"gzip", "chunked"} {
		hdr := "Transfer-Encoding: " + v
		if n := strings.Count(resp, hdr); n != 1 {
			t.Errorf("want 1 occurrence of %q in response, got %v\nresponse: %v", hdr, n, resp)
		}
	}
}

func BenchmarkClientServer(b *testing.B) {
	b.ReportAllocs()
	b.StopTimer()
	ts := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, r *Request) {
		fmt.Fprintf(rw, "Hello world.\n")
	}))
	defer ts.Close()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		res, err := Get(ts.URL)
		if err != nil {
			b.Fatal("Get:", err)
		}
		all, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			b.Fatal("ReadAll:", err)
		}
		body := string(all)
		if body != "Hello world.\n" {
			b.Fatal("Got body:", body)
		}
	}

	b.StopTimer()
}

func BenchmarkClientServerParallel4(b *testing.B) {
	benchmarkClientServerParallel(b, 4, false)
}

func BenchmarkClientServerParallel64(b *testing.B) {
	benchmarkClientServerParallel(b, 64, false)
}

func BenchmarkClientServerParallelTLS4(b *testing.B) {
	benchmarkClientServerParallel(b, 4, true)
}

func BenchmarkClientServerParallelTLS64(b *testing.B) {
	benchmarkClientServerParallel(b, 64, true)
}

func benchmarkClientServerParallel(b *testing.B, parallelism int, useTLS bool) {
	b.ReportAllocs()
	ts := httptest.NewUnstartedServer(HandlerFunc(func(rw ResponseWriter, r *Request) {
		fmt.Fprintf(rw, "Hello world.\n")
	}))
	if useTLS {
		ts.StartTLS()
	} else {
		ts.Start()
	}
	defer ts.Close()
	b.ResetTimer()
	b.SetParallelism(parallelism)
	b.RunParallel(func(pb *testing.PB) {
		c := ts.Client()
		for pb.Next() {
			res, err := c.Get(ts.URL)
			if err != nil {
				b.Logf("Get: %v", err)
				continue
			}
			all, err := ioutil.ReadAll(res.Body)
			res.Body.Close()
			if err != nil {
				b.Logf("ReadAll: %v", err)
				continue
			}
			body := string(all)
			if body != "Hello world.\n" {
				panic("Got body: " + body)
			}
		}
	})
}

// A benchmark for profiling the server without the HTTP client code.
// The client code runs in a subprocess.
//
// For use like:
//   $ go test -c
//   $ ./http.test -test.run=XX -test.bench=BenchmarkServer -test.benchtime=15s -test.cpuprofile=http.prof
//   $ go tool pprof http.test http.prof
//   (pprof) web
func BenchmarkServer(b *testing.B) {
	b.ReportAllocs()
	// Child process mode;
	if url := os.Getenv("TEST_BENCH_SERVER_URL"); url != "" {
		n, err := strconv.Atoi(os.Getenv("TEST_BENCH_CLIENT_N"))
		if err != nil {
			panic(err)
		}
		for i := 0; i < n; i++ {
			res, err := Get(url)
			if err != nil {
				log.Panicf("Get: %v", err)
			}
			all, err := ioutil.ReadAll(res.Body)
			res.Body.Close()
			if err != nil {
				log.Panicf("ReadAll: %v", err)
			}
			body := string(all)
			if body != "Hello world.\n" {
				log.Panicf("Got body: %q", body)
			}
		}
		os.Exit(0)
		return
	}

	var res = []byte("Hello world.\n")
	b.StopTimer()
	ts := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, r *Request) {
		rw.Header().Set("Content-Type", "text/html; charset=utf-8")
		rw.Write(res)
	}))
	defer ts.Close()
	b.StartTimer()

	cmd := exec.Command(os.Args[0], "-test.run=XXXX", "-test.bench=BenchmarkServer$")
	cmd.Env = append([]string{
		fmt.Sprintf("TEST_BENCH_CLIENT_N=%d", b.N),
		fmt.Sprintf("TEST_BENCH_SERVER_URL=%s", ts.URL),
	}, os.Environ()...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		b.Errorf("Test failure: %v, with output: %s", err, out)
	}
}

// getNoBody wraps Get but closes any Response.Body before returning the response.
func getNoBody(urlStr string) (*Response, error) {
	res, err := Get(urlStr)
	if err != nil {
		return nil, err
	}
	res.Body.Close()
	return res, nil
}

// A benchmark for profiling the client without the HTTP server code.
// The server code runs in a subprocess.
func BenchmarkClient(b *testing.B) {
	b.ReportAllocs()
	b.StopTimer()
	defer afterTest(b)

	var data = []byte("Hello world.\n")
	if server := os.Getenv("TEST_BENCH_SERVER"); server != "" {
		// Server process mode.
		port := os.Getenv("TEST_BENCH_SERVER_PORT") // can be set by user
		if port == "" {
			port = "0"
		}
		ln, err := net.Listen("tcp", "localhost:"+port)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
		fmt.Println(ln.Addr().String())
		HandleFunc("/", func(w ResponseWriter, r *Request) {
			r.ParseForm()
			if r.Form.Get("stop") != "" {
				os.Exit(0)
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write(data)
		})
		var srv Server
		log.Fatal(srv.Serve(ln))
	}

	// Start server process.
	cmd := exec.Command(os.Args[0], "-test.run=XXXX", "-test.bench=BenchmarkClient$")
	cmd.Env = append(os.Environ(), "TEST_BENCH_SERVER=yes")
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		b.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		b.Fatalf("subprocess failed to start: %v", err)
	}
	defer cmd.Process.Kill()

	// Wait for the server in the child process to respond and tell us
	// its listening address, once it's started listening:
	timer := time.AfterFunc(10*time.Second, func() {
		cmd.Process.Kill()
	})
	defer timer.Stop()
	bs := bufio.NewScanner(stdout)
	if !bs.Scan() {
		b.Fatalf("failed to read listening URL from child: %v", bs.Err())
	}
	url := "http://" + strings.TrimSpace(bs.Text()) + "/"
	timer.Stop()
	if _, err := getNoBody(url); err != nil {
		b.Fatalf("initial probe of child process failed: %v", err)
	}

	done := make(chan error)
	go func() {
		done <- cmd.Wait()
	}()

	// Do b.N requests to the server.
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		res, err := Get(url)
		if err != nil {
			b.Fatalf("Get: %v", err)
		}
		body, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			b.Fatalf("ReadAll: %v", err)
		}
		if !bytes.Equal(body, data) {
			b.Fatalf("Got body: %q", body)
		}
	}
	b.StopTimer()

	// Instruct server process to stop.
	getNoBody(url + "?stop=yes")
	select {
	case err := <-done:
		if err != nil {
			b.Fatalf("subprocess failed: %v", err)
		}
	case <-time.After(5 * time.Second):
		b.Fatalf("subprocess did not stop")
	}
}

func BenchmarkServerFakeConnNoKeepAlive(b *testing.B) {
	b.ReportAllocs()
	req := reqBytes(`GET / HTTP/1.0
Host: golang.org
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.52 Safari/537.17
Accept-Encoding: gzip,deflate,sdch
Accept-Language: en-US,en;q=0.8
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.3
`)
	res := []byte("Hello world!\n")

	conn := &testConn{
		// testConn.Close will not push into the channel
		// if it's full.
		closec: make(chan bool, 1),
	}
	handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
		rw.Header().Set("Content-Type", "text/html; charset=utf-8")
		rw.Write(res)
	})
	ln := new(oneConnListener)
	for i := 0; i < b.N; i++ {
		conn.readBuf.Reset()
		conn.writeBuf.Reset()
		conn.readBuf.Write(req)
		ln.conn = conn
		Serve(ln, handler)
		<-conn.closec
	}
}

// repeatReader reads content count times, then EOFs.
type repeatReader struct {
	content []byte
	count   int
	off     int
}

func (r *repeatReader) Read(p []byte) (n int, err error) {
	if r.count <= 0 {
		return 0, io.EOF
	}
	n = copy(p, r.content[r.off:])
	r.off += n
	if r.off == len(r.content) {
		r.count--
		r.off = 0
	}
	return
}

func BenchmarkServerFakeConnWithKeepAlive(b *testing.B) {
	b.ReportAllocs()

	req := reqBytes(`GET / HTTP/1.1
Host: golang.org
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.52 Safari/537.17
Accept-Encoding: gzip,deflate,sdch
Accept-Language: en-US,en;q=0.8
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.3
`)
	res := []byte("Hello world!\n")

	conn := &rwTestConn{
		Reader: &repeatReader{content: req, count: b.N},
		Writer: ioutil.Discard,
		closec: make(chan bool, 1),
	}
	handled := 0
	handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
		handled++
		rw.Header().Set("Content-Type", "text/html; charset=utf-8")
		rw.Write(res)
	})
	ln := &oneConnListener{conn: conn}
	go Serve(ln, handler)
	<-conn.closec
	if b.N != handled {
		b.Errorf("b.N=%d but handled %d", b.N, handled)
	}
}

// same as above, but representing the most simple possible request
// and handler. Notably: the handler does not call rw.Header().
func BenchmarkServerFakeConnWithKeepAliveLite(b *testing.B) {
	b.ReportAllocs()

	req := reqBytes(`GET / HTTP/1.1
Host: golang.org
`)
	res := []byte("Hello world!\n")

	conn := &rwTestConn{
		Reader: &repeatReader{content: req, count: b.N},
		Writer: ioutil.Discard,
		closec: make(chan bool, 1),
	}
	handled := 0
	handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
		handled++
		rw.Write(res)
	})
	ln := &oneConnListener{conn: conn}
	go Serve(ln, handler)
	<-conn.closec
	if b.N != handled {
		b.Errorf("b.N=%d but handled %d", b.N, handled)
	}
}

const someResponse = "<html>some response</html>"

// A Response that's just no bigger than 2KB, the buffer-before-chunking threshold.
var response = bytes.Repeat([]byte(someResponse), 2<<10/len(someResponse))

// Both Content-Type and Content-Length set. Should be no buffering.
func BenchmarkServerHandlerTypeLen(b *testing.B) {
	benchmarkHandler(b, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Header().Set("Content-Length", strconv.Itoa(len(response)))
		w.Write(response)
	}))
}

// A Content-Type is set, but no length. No sniffing, but will count the Content-Length.
func BenchmarkServerHandlerNoLen(b *testing.B) {
	benchmarkHandler(b, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write(response)
	}))
}

// A Content-Length is set, but the Content-Type will be sniffed.
func BenchmarkServerHandlerNoType(b *testing.B) {
	benchmarkHandler(b, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(response)))
		w.Write(response)
	}))
}

// Neither a Content-Type or Content-Length, so sniffed and counted.
func BenchmarkServerHandlerNoHeader(b *testing.B) {
	benchmarkHandler(b, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write(response)
	}))
}

func benchmarkHandler(b *testing.B, h Handler) {
	b.ReportAllocs()
	req := reqBytes(`GET / HTTP/1.1
Host: golang.org
`)
	conn := &rwTestConn{
		Reader: &repeatReader{content: req, count: b.N},
		Writer: ioutil.Discard,
		closec: make(chan bool, 1),
	}
	handled := 0
	handler := HandlerFunc(func(rw ResponseWriter, r *Request) {
		handled++
		h.ServeHTTP(rw, r)
	})
	ln := &oneConnListener{conn: conn}
	go Serve(ln, handler)
	<-conn.closec
	if b.N != handled {
		b.Errorf("b.N=%d but handled %d", b.N, handled)
	}
}

func BenchmarkServerHijack(b *testing.B) {
	b.ReportAllocs()
	req := reqBytes(`GET / HTTP/1.1
Host: golang.org
`)
	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, _, err := w.(Hijacker).Hijack()
		if err != nil {
			panic(err)
		}
		conn.Close()
	})
	conn := &rwTestConn{
		Writer: ioutil.Discard,
		closec: make(chan bool, 1),
	}
	ln := &oneConnListener{conn: conn}
	for i := 0; i < b.N; i++ {
		conn.Reader = bytes.NewReader(req)
		ln.conn = conn
		Serve(ln, h)
		<-conn.closec
	}
}

func BenchmarkCloseNotifier(b *testing.B) {
	b.ReportAllocs()
	b.StopTimer()
	sawClose := make(chan bool)
	ts := httptest.NewServer(HandlerFunc(func(rw ResponseWriter, req *Request) {
		<-rw.(CloseNotifier).CloseNotify()
		sawClose <- true
	}))
	defer ts.Close()
	tot := time.NewTimer(5 * time.Second)
	defer tot.Stop()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			b.Fatalf("error dialing: %v", err)
		}
		_, err = fmt.Fprintf(conn, "GET / HTTP/1.1\r\nConnection: keep-alive\r\nHost: foo\r\n\r\n")
		if err != nil {
			b.Fatal(err)
		}
		conn.Close()
		tot.Reset(5 * time.Second)
		select {
		case <-sawClose:
		case <-tot.C:
			b.Fatal("timeout")
		}
	}
	b.StopTimer()
}

// Verify this doesn't race (Issue 16505)
func TestConcurrentServerServe(t *testing.T) {
	setParallel(t)
	for i := 0; i < 100; i++ {
		ln1 := &oneConnListener{conn: nil}
		ln2 := &oneConnListener{conn: nil}
		srv := Server{}
		go func() { srv.Serve(ln1) }()
		go func() { srv.Serve(ln2) }()
	}
}

func TestServerIdleTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		io.Copy(ioutil.Discard, r.Body)
		io.WriteString(w, r.RemoteAddr)
	}))
	ts.Config.ReadHeaderTimeout = 1 * time.Second
	ts.Config.IdleTimeout = 2 * time.Second
	ts.Start()
	defer ts.Close()
	c := ts.Client()

	get := func() string {
		res, err := c.Get(ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		slurp, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		return string(slurp)
	}

	a1, a2 := get(), get()
	if a1 != a2 {
		t.Fatalf("did requests on different connections")
	}
	time.Sleep(3 * time.Second)
	a3 := get()
	if a2 == a3 {
		t.Fatal("request three unexpectedly on same connection")
	}

	// And test that ReadHeaderTimeout still works:
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	conn.Write([]byte("GET / HTTP/1.1\r\nHost: foo.com\r\n"))
	time.Sleep(2 * time.Second)
	if _, err := io.CopyN(ioutil.Discard, conn, 1); err == nil {
		t.Fatal("copy byte succeeded; want err")
	}
}

func get(t *testing.T, c *Client, url string) string {
	res, err := c.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(slurp)
}

// Tests that calls to Server.SetKeepAlivesEnabled(false) closes any
// currently-open connections.
func TestServerSetKeepAlivesEnabledClosesConns(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, r.RemoteAddr)
	}))
	defer ts.Close()

	c := ts.Client()
	tr := c.Transport.(*Transport)

	get := func() string { return get(t, c, ts.URL) }

	a1, a2 := get(), get()
	if a1 != a2 {
		t.Fatal("expected first two requests on same connection")
	}
	var idle0 int
	if !waitCondition(2*time.Second, 10*time.Millisecond, func() bool {
		idle0 = tr.IdleConnKeyCountForTesting()
		return idle0 == 1
	}) {
		t.Fatalf("idle count before SetKeepAlivesEnabled called = %v; want 1", idle0)
	}

	ts.Config.SetKeepAlivesEnabled(false)

	var idle1 int
	if !waitCondition(2*time.Second, 10*time.Millisecond, func() bool {
		idle1 = tr.IdleConnKeyCountForTesting()
		return idle1 == 0
	}) {
		t.Fatalf("idle count after SetKeepAlivesEnabled called = %v; want 0", idle1)
	}

	a3 := get()
	if a3 == a2 {
		t.Fatal("expected third request on new connection")
	}
}

func TestServerShutdown_h1(t *testing.T) { testServerShutdown(t, h1Mode) }
func TestServerShutdown_h2(t *testing.T) { testServerShutdown(t, h2Mode) }

func testServerShutdown(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	var doShutdown func() // set later
	var shutdownRes = make(chan error, 1)
	var gotOnShutdown = make(chan struct{}, 1)
	handler := HandlerFunc(func(w ResponseWriter, r *Request) {
		go doShutdown()
		// Shutdown is graceful, so it should not interrupt
		// this in-flight response. Add a tiny sleep here to
		// increase the odds of a failure if shutdown has
		// bugs.
		time.Sleep(20 * time.Millisecond)
		io.WriteString(w, r.RemoteAddr)
	})
	cst := newClientServerTest(t, h2, handler, func(srv *httptest.Server) {
		srv.Config.RegisterOnShutdown(func() { gotOnShutdown <- struct{}{} })
	})
	defer cst.close()

	doShutdown = func() {
		shutdownRes <- cst.ts.Config.Shutdown(context.Background())
	}
	get(t, cst.c, cst.ts.URL) // calls t.Fail on failure

	if err := <-shutdownRes; err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	select {
	case <-gotOnShutdown:
	case <-time.After(5 * time.Second):
		t.Errorf("onShutdown callback not called, RegisterOnShutdown broken?")
	}

	res, err := cst.c.Get(cst.ts.URL)
	if err == nil {
		res.Body.Close()
		t.Fatal("second request should fail. server should be shut down")
	}
}

func TestServerShutdownStateNew(t *testing.T) {
	if testing.Short() {
		t.Skip("test takes 5-6 seconds; skipping in short mode")
	}
	setParallel(t)
	defer afterTest(t)

	ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		// nothing.
	}))
	var connAccepted sync.WaitGroup
	ts.Config.ConnState = func(conn net.Conn, state ConnState) {
		if state == StateNew {
			connAccepted.Done()
		}
	}
	ts.Start()
	defer ts.Close()

	// Start a connection but never write to it.
	connAccepted.Add(1)
	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	// Wait for the connection to be accepted by the server. Otherwise, if
	// Shutdown happens to run first, the server will be closed when
	// encountering the connection, in which case it will be rejected
	// immediately.
	connAccepted.Wait()

	shutdownRes := make(chan error, 1)
	go func() {
		shutdownRes <- ts.Config.Shutdown(context.Background())
	}()
	readRes := make(chan error, 1)
	go func() {
		_, err := c.Read([]byte{0})
		readRes <- err
	}()

	const expectTimeout = 5 * time.Second
	t0 := time.Now()
	select {
	case got := <-shutdownRes:
		d := time.Since(t0)
		if got != nil {
			t.Fatalf("shutdown error after %v: %v", d, err)
		}
		if d < expectTimeout/2 {
			t.Errorf("shutdown too soon after %v", d)
		}
	case <-time.After(expectTimeout * 3 / 2):
		t.Fatalf("timeout waiting for shutdown")
	}

	// Wait for c.Read to unblock; should be already done at this point,
	// or within a few milliseconds.
	select {
	case err := <-readRes:
		if err == nil {
			t.Error("expected error from Read")
		}
	case <-time.After(2 * time.Second):
		t.Errorf("timeout waiting for Read to unblock")
	}
}

// Issue 17878: tests that we can call Close twice.
func TestServerCloseDeadlock(t *testing.T) {
	var s Server
	s.Close()
	s.Close()
}

// Issue 17717: tests that Server.SetKeepAlivesEnabled is respected by
// both HTTP/1 and HTTP/2.
func TestServerKeepAlivesEnabled_h1(t *testing.T) { testServerKeepAlivesEnabled(t, h1Mode) }
func TestServerKeepAlivesEnabled_h2(t *testing.T) { testServerKeepAlivesEnabled(t, h2Mode) }
func testServerKeepAlivesEnabled(t *testing.T, h2 bool) {
	if h2 {
		restore := ExportSetH2GoawayTimeout(10 * time.Millisecond)
		defer restore()
	}
	// Not parallel: messes with global variable. (http2goAwayTimeout)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%v", r.RemoteAddr)
	}))
	defer cst.close()
	srv := cst.ts.Config
	srv.SetKeepAlivesEnabled(false)
	a := cst.getURL(cst.ts.URL)
	if !waitCondition(2*time.Second, 10*time.Millisecond, srv.ExportAllConnsIdle) {
		t.Fatalf("test server has active conns")
	}
	b := cst.getURL(cst.ts.URL)
	if a == b {
		t.Errorf("got same connection between first and second requests")
	}
	if !waitCondition(2*time.Second, 10*time.Millisecond, srv.ExportAllConnsIdle) {
		t.Fatalf("test server has active conns")
	}
}

// Issue 18447: test that the Server's ReadTimeout is stopped while
// the server's doing its 1-byte background read between requests,
// waiting for the connection to maybe close.
func TestServerCancelsReadTimeoutWhenIdle(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	runTimeSensitiveTest(t, []time.Duration{
		10 * time.Millisecond,
		50 * time.Millisecond,
		250 * time.Millisecond,
		time.Second,
		2 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		ts := httptest.NewUnstartedServer(HandlerFunc(func(w ResponseWriter, r *Request) {
			select {
			case <-time.After(2 * timeout):
				fmt.Fprint(w, "ok")
			case <-r.Context().Done():
				fmt.Fprint(w, r.Context().Err())
			}
		}))
		ts.Config.ReadTimeout = timeout
		ts.Start()
		defer ts.Close()

		c := ts.Client()

		res, err := c.Get(ts.URL)
		if err != nil {
			return fmt.Errorf("Get: %v", err)
		}
		slurp, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			return fmt.Errorf("Body ReadAll: %v", err)
		}
		if string(slurp) != "ok" {
			return fmt.Errorf("got: %q, want ok", slurp)
		}
		return nil
	})
}

// runTimeSensitiveTest runs test with the provided durations until one passes.
// If they all fail, t.Fatal is called with the last one's duration and error value.
func runTimeSensitiveTest(t *testing.T, durations []time.Duration, test func(t *testing.T, d time.Duration) error) {
	for i, d := range durations {
		err := test(t, d)
		if err == nil {
			return
		}
		if i == len(durations)-1 {
			t.Fatalf("failed with duration %v: %v", d, err)
		}
	}
}

// Issue 18535: test that the Server doesn't try to do a background
// read if it's already done one.
func TestServerDuplicateBackgroundRead(t *testing.T) {
	if runtime.GOOS == "netbsd" && runtime.GOARCH == "arm" {
		testenv.SkipFlaky(t, 24826)
	}

	setParallel(t)
	defer afterTest(t)

	const goroutines = 5
	const requests = 2000

	hts := httptest.NewServer(HandlerFunc(NotFound))
	defer hts.Close()

	reqBytes := []byte("GET / HTTP/1.1\r\nHost: e.com\r\n\r\n")

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cn, err := net.Dial("tcp", hts.Listener.Addr().String())
			if err != nil {
				t.Error(err)
				return
			}
			defer cn.Close()

			wg.Add(1)
			go func() {
				defer wg.Done()
				io.Copy(ioutil.Discard, cn)
			}()

			for j := 0; j < requests; j++ {
				if t.Failed() {
					return
				}
				_, err := cn.Write(reqBytes)
				if err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// Test that the bufio.Reader returned by Hijack includes any buffered
// byte (from the Server's backgroundRead) in its buffer. We want the
// Handler code to be able to tell that a byte is available via
// bufio.Reader.Buffered(), without resorting to Reading it
// (potentially blocking) to get at it.
func TestServerHijackGetsBackgroundByte(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/18657")
	}
	setParallel(t)
	defer afterTest(t)
	done := make(chan struct{})
	inHandler := make(chan bool, 1)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(done)

		// Tell the client to send more data after the GET request.
		inHandler <- true

		conn, buf, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()

		peek, err := buf.Reader.Peek(3)
		if string(peek) != "foo" || err != nil {
			t.Errorf("Peek = %q, %v; want foo, nil", peek, err)
		}

		select {
		case <-r.Context().Done():
			t.Error("context unexpectedly canceled")
		default:
		}
	}))
	defer ts.Close()

	cn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer cn.Close()
	if _, err := cn.Write([]byte("GET / HTTP/1.1\r\nHost: e.com\r\n\r\n")); err != nil {
		t.Fatal(err)
	}
	<-inHandler
	if _, err := cn.Write([]byte("foo")); err != nil {
		t.Fatal(err)
	}

	if err := cn.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("timeout")
	}
}

// Like TestServerHijackGetsBackgroundByte above but sending a
// immediate 1MB of data to the server to fill up the server's 4KB
// buffer.
func TestServerHijackGetsBackgroundByte_big(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/18657")
	}
	setParallel(t)
	defer afterTest(t)
	done := make(chan struct{})
	const size = 8 << 10
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(done)

		conn, buf, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		slurp, err := ioutil.ReadAll(buf.Reader)
		if err != nil {
			t.Errorf("Copy: %v", err)
		}
		allX := true
		for _, v := range slurp {
			if v != 'x' {
				allX = false
			}
		}
		if len(slurp) != size {
			t.Errorf("read %d; want %d", len(slurp), size)
		} else if !allX {
			t.Errorf("read %q; want %d 'x'", slurp, size)
		}
	}))
	defer ts.Close()

	cn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer cn.Close()
	if _, err := fmt.Fprintf(cn, "GET / HTTP/1.1\r\nHost: e.com\r\n\r\n%s",
		strings.Repeat("x", size)); err != nil {
		t.Fatal(err)
	}
	if err := cn.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("timeout")
	}
}

// Issue 18319: test that the Server validates the request method.
func TestServerValidatesMethod(t *testing.T) {
	tests := []struct {
		method string
		want   int
	}{
		{"GET", 200},
		{"GE(T", 400},
	}
	for _, tt := range tests {
		conn := &testConn{closec: make(chan bool, 1)}
		io.WriteString(&conn.readBuf, tt.method+" / HTTP/1.1\r\nHost: foo.example\r\n\r\n")

		ln := &oneConnListener{conn}
		go Serve(ln, serve(200))
		<-conn.closec
		res, err := ReadResponse(bufio.NewReader(&conn.writeBuf), nil)
		if err != nil {
			t.Errorf("For %s, ReadResponse: %v", tt.method, res)
			continue
		}
		if res.StatusCode != tt.want {
			t.Errorf("For %s, Status = %d; want %d", tt.method, res.StatusCode, tt.want)
		}
	}
}

// Listener for TestServerListenNotComparableListener.
type eofListenerNotComparable []int

func (eofListenerNotComparable) Accept() (net.Conn, error) { return nil, io.EOF }
func (eofListenerNotComparable) Addr() net.Addr            { return nil }
func (eofListenerNotComparable) Close() error              { return nil }

// Issue 24812: don't crash on non-comparable Listener
func TestServerListenNotComparableListener(t *testing.T) {
	var s Server
	s.Serve(make(eofListenerNotComparable, 1)) // used to panic
}

// countCloseListener is a Listener wrapper that counts the number of Close calls.
type countCloseListener struct {
	net.Listener
	closes int32 // atomic
}

func (p *countCloseListener) Close() error {
	atomic.AddInt32(&p.closes, 1)
	return nil
}

// Issue 24803: don't call Listener.Close on Server.Shutdown.
func TestServerCloseListenerOnce(t *testing.T) {
	setParallel(t)
	defer afterTest(t)

	ln := newLocalListener(t)
	defer ln.Close()

	cl := &countCloseListener{Listener: ln}
	server := &Server{}
	sdone := make(chan bool, 1)

	go func() {
		server.Serve(cl)
		sdone <- true
	}()
	time.Sleep(10 * time.Millisecond)
	server.Shutdown(context.Background())
	ln.Close()
	<-sdone

	nclose := atomic.LoadInt32(&cl.closes)
	if nclose != 1 {
		t.Errorf("Close calls = %v; want 1", nclose)
	}
}

// Issue 20239: don't block in Serve if Shutdown is called first.
func TestServerShutdownThenServe(t *testing.T) {
	var srv Server
	cl := &countCloseListener{Listener: nil}
	srv.Shutdown(context.Background())
	got := srv.Serve(cl)
	if got != ErrServerClosed {
		t.Errorf("Serve err = %v; want ErrServerClosed", got)
	}
	nclose := atomic.LoadInt32(&cl.closes)
	if nclose != 1 {
		t.Errorf("Close calls = %v; want 1", nclose)
	}
}

// Issue 23351: document and test behavior of ServeMux with ports
func TestStripPortFromHost(t *testing.T) {
	mux := NewServeMux()

	mux.HandleFunc("example.com/", func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "OK")
	})
	mux.HandleFunc("example.com:9000/", func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "uh-oh!")
	})

	req := httptest.NewRequest("GET", "http://example.com:9000/", nil)
	rw := httptest.NewRecorder()

	mux.ServeHTTP(rw, req)

	response := rw.Body.String()
	if response != "OK" {
		t.Errorf("Response gotten was %q", response)
	}
}

func BenchmarkResponseStatusLine(b *testing.B) {
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		bw := bufio.NewWriter(ioutil.Discard)
		var buf3 [3]byte
		for pb.Next() {
			Export_writeStatusLine(bw, true, 200, buf3[:])
		}
	})
}
