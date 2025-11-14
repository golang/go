// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// End-to-end serving tests

package http_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"context"
	crand "crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"internal/synctest"
	"internal/testenv"
	"io"
	"log"
	"math/rand"
	"mime/multipart"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/http/httputil"
	"net/http/internal"
	"net/http/internal/testcert"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"slices"
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
	closec   chan bool // 1-buffered; receives true when Close is called
	noopConn
}

func newTestConn() *testConn {
	return &testConn{closec: make(chan bool, 1)}
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
	var output strings.Builder
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

func TestHostHandlers(t *testing.T) { run(t, testHostHandlers, []testMode{http1Mode}) }
func testHostHandlers(t *testing.T, mode testMode) {
	mux := NewServeMux()
	for _, h := range handlers {
		mux.Handle(h.pattern, stringHandler(h.msg))
	}
	ts := newClientServerTest(t, mode, mux).ts

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

// Issue 73688
func TestServeMuxHandlerTrailingSlash(t *testing.T) {
	setParallel(t)
	mux := NewServeMux()
	const original = "/{x}/"
	mux.Handle(original, NotFoundHandler())
	r, _ := NewRequest("POST", "/foo", nil)
	_, p := mux.Handler(r)
	if p != original {
		t.Errorf("got %q, want %q", p, original)
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
	run(t, testServeWithSlashRedirectKeepsQueryString, []testMode{http1Mode})
}
func testServeWithSlashRedirectKeepsQueryString(t *testing.T, mode testMode) {
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

	ts := newClientServerTest(t, mode, mux).ts

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
		slurp, _ := io.ReadAll(res.Body)
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

// Test that we don't attempt trailing-slash redirect on a path that already has
// a trailing slash.
// See issue #65624.
func TestMuxNoSlashRedirectWithTrailingSlash(t *testing.T) {
	mux := NewServeMux()
	mux.HandleFunc("/{x}/", func(w ResponseWriter, r *Request) {
		fmt.Fprintln(w, "ok")
	})
	w := httptest.NewRecorder()
	req, _ := NewRequest("GET", "/", nil)
	mux.ServeHTTP(w, req)
	if g, w := w.Code, 404; g != w {
		t.Errorf("got %d, want %d", g, w)
	}
}

// Test that we don't attempt trailing-slash response 405 on a path that already has
// a trailing slash.
// See issue #67657.
func TestMuxNoSlash405WithTrailingSlash(t *testing.T) {
	mux := NewServeMux()
	mux.HandleFunc("GET /{x}/", func(w ResponseWriter, r *Request) {
		fmt.Fprintln(w, "ok")
	})
	w := httptest.NewRecorder()
	req, _ := NewRequest("GET", "/", nil)
	mux.ServeHTTP(w, req)
	if g, w := w.Code, 404; g != w {
		t.Errorf("got %d, want %d", g, w)
	}
}

func TestShouldRedirectConcurrency(t *testing.T) { run(t, testShouldRedirectConcurrency) }
func testShouldRedirectConcurrency(t *testing.T, mode testMode) {
	mux := NewServeMux()
	newClientServerTest(t, mode, mux)
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

func TestServerTimeouts(t *testing.T) { run(t, testServerTimeouts, []testMode{http1Mode}) }
func testServerTimeouts(t *testing.T, mode testMode) {
	runTimeSensitiveTest(t, []time.Duration{
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		500 * time.Millisecond,
		1 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		return testServerTimeoutsWithTimeout(t, timeout, mode)
	})
}

func testServerTimeoutsWithTimeout(t *testing.T, timeout time.Duration, mode testMode) error {
	var reqNum atomic.Int32
	cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
		fmt.Fprintf(res, "req=%d", reqNum.Add(1))
	}), func(ts *httptest.Server) {
		ts.Config.ReadTimeout = timeout
		ts.Config.WriteTimeout = timeout
	})
	defer cst.close()
	ts := cst.ts

	// Hit the HTTP server successfully.
	c := ts.Client()
	r, err := c.Get(ts.URL)
	if err != nil {
		return fmt.Errorf("http Get #1: %v", err)
	}
	got, err := io.ReadAll(r.Body)
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
	got, err = io.ReadAll(r.Body)
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
		go io.Copy(io.Discard, conn)
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

func TestServerReadTimeout(t *testing.T) { run(t, testServerReadTimeout) }
func testServerReadTimeout(t *testing.T, mode testMode) {
	respBody := "response body"
	for timeout := 5 * time.Millisecond; ; timeout *= 2 {
		cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
			_, err := io.Copy(io.Discard, req.Body)
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("server timed out reading request body: got err %v; want os.ErrDeadlineExceeded", err)
			}
			res.Write([]byte(respBody))
		}), func(ts *httptest.Server) {
			ts.Config.ReadHeaderTimeout = -1 // don't time out while reading headers
			ts.Config.ReadTimeout = timeout
			t.Logf("Server.Config.ReadTimeout = %v", timeout)
		})

		var retries atomic.Int32
		cst.c.Transport.(*Transport).Proxy = func(*Request) (*url.URL, error) {
			if retries.Add(1) != 1 {
				return nil, errors.New("too many retries")
			}
			return nil, nil
		}

		pr, pw := io.Pipe()
		res, err := cst.c.Post(cst.ts.URL, "text/apocryphal", pr)
		if err != nil {
			t.Logf("Get error, retrying: %v", err)
			cst.close()
			continue
		}
		defer res.Body.Close()
		got, err := io.ReadAll(res.Body)
		if string(got) != respBody || err != nil {
			t.Errorf("client read response body: %q, %v; want %q, nil", string(got), err, respBody)
		}
		pw.Close()
		break
	}
}

func TestServerNoReadTimeout(t *testing.T) { run(t, testServerNoReadTimeout) }
func testServerNoReadTimeout(t *testing.T, mode testMode) {
	reqBody := "Hello, Gophers!"
	resBody := "Hi, Gophers!"
	for _, timeout := range []time.Duration{0, -1} {
		cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
			ctl := NewResponseController(res)
			ctl.EnableFullDuplex()
			res.WriteHeader(StatusOK)
			// Flush the headers before processing the request body
			// to unblock the client from the RoundTrip.
			if err := ctl.Flush(); err != nil {
				t.Errorf("server flush response: %v", err)
				return
			}
			got, err := io.ReadAll(req.Body)
			if string(got) != reqBody || err != nil {
				t.Errorf("server read request body: %v; got %q, want %q", err, got, reqBody)
			}
			res.Write([]byte(resBody))
		}), func(ts *httptest.Server) {
			ts.Config.ReadTimeout = timeout
			t.Logf("Server.Config.ReadTimeout = %d", timeout)
		})

		pr, pw := io.Pipe()
		res, err := cst.c.Post(cst.ts.URL, "text/plain", pr)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()

		// TODO(panjf2000): sleep is not so robust, maybe find a better way to test this?
		time.Sleep(10 * time.Millisecond) // stall sending body to server to test server doesn't time out
		pw.Write([]byte(reqBody))
		pw.Close()

		got, err := io.ReadAll(res.Body)
		if string(got) != resBody || err != nil {
			t.Errorf("client read response body: %v; got %v, want %q", err, got, resBody)
		}
	}
}

func TestServerWriteTimeout(t *testing.T) { run(t, testServerWriteTimeout) }
func testServerWriteTimeout(t *testing.T, mode testMode) {
	for timeout := 5 * time.Millisecond; ; timeout *= 2 {
		errc := make(chan error, 2)
		cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
			errc <- nil
			_, err := io.Copy(res, neverEnding('a'))
			errc <- err
		}), func(ts *httptest.Server) {
			ts.Config.WriteTimeout = timeout
			t.Logf("Server.Config.WriteTimeout = %v", timeout)
		})

		// The server's WriteTimeout parameter also applies to reads during the TLS
		// handshake. The client makes the last write during the handshake, and if
		// the server happens to time out during the read of that write, the client
		// may think that the connection was accepted even though the server thinks
		// it timed out.
		//
		// The client only notices that the server connection is gone when it goes
		// to actually write the request — and when that fails, it retries
		// internally (the same as if the server had closed the connection due to a
		// racing idle-timeout).
		//
		// With unlucky and very stable scheduling (as may be the case with the fake wasm
		// net stack), this can result in an infinite retry loop that doesn't
		// propagate the error up far enough for us to adjust the WriteTimeout.
		//
		// To avoid that problem, we explicitly forbid internal retries by rejecting
		// them in a Proxy hook in the transport.
		var retries atomic.Int32
		cst.c.Transport.(*Transport).Proxy = func(*Request) (*url.URL, error) {
			if retries.Add(1) != 1 {
				return nil, errors.New("too many retries")
			}
			return nil, nil
		}

		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			// Probably caused by the write timeout expiring before the handler runs.
			t.Logf("Get error, retrying: %v", err)
			cst.close()
			continue
		}
		defer res.Body.Close()
		_, err = io.Copy(io.Discard, res.Body)
		if err == nil {
			t.Errorf("client reading from truncated request body: got nil error, want non-nil")
		}
		select {
		case <-errc:
			err = <-errc // io.Copy error
			if !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("server timed out writing request body: got err %v; want os.ErrDeadlineExceeded", err)
			}
			return
		default:
			// The write timeout expired before the handler started.
			t.Logf("handler didn't run, retrying")
			cst.close()
		}
	}
}

func TestServerNoWriteTimeout(t *testing.T) { run(t, testServerNoWriteTimeout) }
func testServerNoWriteTimeout(t *testing.T, mode testMode) {
	for _, timeout := range []time.Duration{0, -1} {
		cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
			_, err := io.Copy(res, neverEnding('a'))
			t.Logf("server write response: %v", err)
		}), func(ts *httptest.Server) {
			ts.Config.WriteTimeout = timeout
			t.Logf("Server.Config.WriteTimeout = %d", timeout)
		})

		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		n, err := io.CopyN(io.Discard, res.Body, 1<<20) // 1MB should be sufficient to prove the point
		if n != 1<<20 || err != nil {
			t.Errorf("client read response body: %d, %v", n, err)
		}
		// This shutdown really should be automatic, but it isn't right now.
		// Shutdown (rather than Close) ensures the handler is done before we return.
		res.Body.Close()
		cst.ts.Config.Shutdown(context.Background())
	}
}

// Test that the HTTP/2 server handles Server.WriteTimeout (Issue 18437)
func TestWriteDeadlineExtendedOnNewRequest(t *testing.T) {
	run(t, testWriteDeadlineExtendedOnNewRequest)
}
func testWriteDeadlineExtendedOnNewRequest(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	ts := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {}),
		func(ts *httptest.Server) {
			ts.Config.WriteTimeout = 250 * time.Millisecond
		},
	).ts

	c := ts.Client()

	for i := 1; i <= 3; i++ {
		req, err := NewRequest("GET", ts.URL, nil)
		if err != nil {
			t.Fatal(err)
		}

		r, err := c.Do(req)
		if err != nil {
			t.Fatalf("http2 Get #%d: %v", i, err)
		}
		r.Body.Close()
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
func TestWriteDeadlineEnforcedPerStream(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	setParallel(t)
	run(t, func(t *testing.T, mode testMode) {
		tryTimeouts(t, func(timeout time.Duration) error {
			return testWriteDeadlineEnforcedPerStream(t, mode, timeout)
		})
	})
}

func testWriteDeadlineEnforcedPerStream(t *testing.T, mode testMode, timeout time.Duration) error {
	firstRequest := make(chan bool, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
		select {
		case firstRequest <- true:
			// first request succeeds
		default:
			// second request times out
			time.Sleep(timeout)
		}
	}), func(ts *httptest.Server) {
		ts.Config.WriteTimeout = timeout / 2
	})
	defer cst.close()
	ts := cst.ts

	c := ts.Client()

	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		return fmt.Errorf("NewRequest: %v", err)
	}
	r, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("Get #1: %v", err)
	}
	r.Body.Close()

	req, err = NewRequest("GET", ts.URL, nil)
	if err != nil {
		return fmt.Errorf("NewRequest: %v", err)
	}
	r, err = c.Do(req)
	if err == nil {
		r.Body.Close()
		return fmt.Errorf("Get #2 expected error, got nil")
	}
	if mode == http2Mode {
		expected := "stream ID 3; INTERNAL_ERROR" // client IDs are odd, second stream should be 3
		if !strings.Contains(err.Error(), expected) {
			return fmt.Errorf("http2 Get #2: expected error to contain %q, got %q", expected, err)
		}
	}
	return nil
}

// Test that the HTTP/2 server does not send RST when WriteDeadline not set.
func TestNoWriteDeadline(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	setParallel(t)
	defer afterTest(t)
	run(t, func(t *testing.T, mode testMode) {
		tryTimeouts(t, func(timeout time.Duration) error {
			return testNoWriteDeadline(t, mode, timeout)
		})
	})
}

func testNoWriteDeadline(t *testing.T, mode testMode, timeout time.Duration) error {
	firstRequest := make(chan bool, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(res ResponseWriter, req *Request) {
		select {
		case firstRequest <- true:
			// first request succeeds
		default:
			// second request times out
			time.Sleep(timeout)
		}
	}))
	defer cst.close()
	ts := cst.ts

	c := ts.Client()

	for i := 0; i < 2; i++ {
		req, err := NewRequest("GET", ts.URL, nil)
		if err != nil {
			return fmt.Errorf("NewRequest: %v", err)
		}
		r, err := c.Do(req)
		if err != nil {
			return fmt.Errorf("Get #%d: %v", i, err)
		}
		r.Body.Close()
	}
	return nil
}

// golang.org/issue/4741 -- setting only a write timeout that triggers
// shouldn't cause a handler to block forever on reads (next HTTP
// request) that will never happen.
func TestOnlyWriteTimeout(t *testing.T) { run(t, testOnlyWriteTimeout, []testMode{http1Mode}) }
func testOnlyWriteTimeout(t *testing.T, mode testMode) {
	var (
		mu   sync.RWMutex
		conn net.Conn
	)
	var afterTimeoutErrc = make(chan error, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, req *Request) {
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
	}), func(ts *httptest.Server) {
		ts.Listener = trackLastConnListener{ts.Listener, &mu, &conn}
	}).ts

	c := ts.Client()

	err := func() error {
		res, err := c.Get(ts.URL)
		if err != nil {
			return err
		}
		_, err = io.Copy(io.Discard, res.Body)
		res.Body.Close()
		return err
	}()
	if err == nil {
		t.Errorf("expected an error copying body from Get request")
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
func TestIdentityResponse(t *testing.T) { run(t, testIdentityResponse) }
func testIdentityResponse(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("https://go.dev/issue/56019")
	}

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

	ts := newClientServerTest(t, mode, handler).ts
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

	if mode != http1Mode {
		return
	}

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

	// The ReadAll will hang for a failing test.
	got, _ := io.ReadAll(conn)
	expectedSuffix := "\r\n\r\ntoo short"
	if !strings.HasSuffix(string(got), expectedSuffix) {
		t.Errorf("Expected output to end with %q; got response body %q",
			expectedSuffix, string(got))
	}
}

func testTCPConnectionCloses(t *testing.T, req string, h Handler) {
	setParallel(t)
	s := newClientServerTest(t, http1Mode, h).ts

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

	_, err = io.ReadAll(r)
	if err != nil {
		t.Fatal("read error:", err)
	}

	if !res.Close {
		t.Errorf("Response.Close = false; want true")
	}
}

func testTCPConnectionStaysOpen(t *testing.T, req string, handler Handler) {
	setParallel(t)
	ts := newClientServerTest(t, http1Mode, handler).ts
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
		if _, err := io.Copy(io.Discard, res.Body); err != nil {
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
func TestKeepAliveFinalChunkWithEOF(t *testing.T) { run(t, testKeepAliveFinalChunkWithEOF) }
func testKeepAliveFinalChunkWithEOF(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.(Flusher).Flush() // force chunked encoding
		w.Write([]byte("{\"Addr\": \"" + r.RemoteAddr + "\"}"))
	}))
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

func TestSetsRemoteAddr(t *testing.T) { run(t, testSetsRemoteAddr) }
func testSetsRemoteAddr(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%s", r.RemoteAddr)
	}))

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	body, err := io.ReadAll(res.Body)
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
	run(t, testServerAllowsBlockingRemoteAddr, []testMode{http1Mode})
}
func testServerAllowsBlockingRemoteAddr(t *testing.T, mode testMode) {
	conns := make(chan net.Conn)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "RA:%s", r.RemoteAddr)
	}), func(ts *httptest.Server) {
		ts.Listener = &blockingRemoteAddrListener{
			Listener: ts.Listener,
			conns:    conns,
		}
	}).ts

	c := ts.Client()
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
		body, err := io.ReadAll(resp.Body)
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
	conn2 := <-conns

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

// TestHeadResponses verifies that all MIME type sniffing and Content-Length
// counting of GET requests also happens on HEAD requests.
func TestHeadResponses(t *testing.T) { run(t, testHeadResponses) }
func testHeadResponses(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("<html>"))
		if err != nil {
			t.Errorf("ResponseWriter.Write: %v", err)
		}

		// Also exercise the ReaderFrom path
		_, err = io.Copy(w, struct{ io.Reader }{strings.NewReader("789a")})
		if err != nil {
			t.Errorf("Copy(ResponseWriter, ...): %v", err)
		}
	}))
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
	body, err := io.ReadAll(res.Body)
	if err != nil {
		t.Error(err)
	}
	if len(body) > 0 {
		t.Errorf("got unexpected body %q", string(body))
	}
}

// Ensure ResponseWriter.ReadFrom doesn't write a body in response to a HEAD request.
// https://go.dev/issue/68609
func TestHeadReaderFrom(t *testing.T) { run(t, testHeadReaderFrom, []testMode{http1Mode}) }
func testHeadReaderFrom(t *testing.T, mode testMode) {
	// Body is large enough to exceed the content-sniffing length.
	wantBody := strings.Repeat("a", 4096)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.(io.ReaderFrom).ReadFrom(strings.NewReader(wantBody))
	}))
	res, err := cst.c.Head(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	res, err = cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	gotBody, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(gotBody) != wantBody {
		t.Errorf("got unexpected body len=%v, want %v", len(gotBody), len(wantBody))
	}
}

func TestTLSHandshakeTimeout(t *testing.T) {
	run(t, testTLSHandshakeTimeout, []testMode{https1Mode, http2Mode})
}
func testTLSHandshakeTimeout(t *testing.T, mode testMode) {
	errLog := new(strings.Builder)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {}),
		func(ts *httptest.Server) {
			ts.Config.ReadTimeout = 250 * time.Millisecond
			ts.Config.ErrorLog = log.New(errLog, "", 0)
		},
	)
	ts := cst.ts

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	var buf [1]byte
	n, err := conn.Read(buf[:])
	if err == nil || n != 0 {
		t.Errorf("Read = %d, %v; want an error and no bytes", n, err)
	}
	conn.Close()

	cst.close()
	if v := errLog.String(); !strings.Contains(v, "timeout") && !strings.Contains(v, "TLS handshake") {
		t.Errorf("expected a TLS handshake timeout error; got %q", v)
	}
}

func TestTLSServer(t *testing.T) { run(t, testTLSServer, []testMode{https1Mode, http2Mode}) }
func testTLSServer(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.TLS != nil {
			w.Header().Set("X-TLS-Set", "true")
			if r.TLS.HandshakeComplete {
				w.Header().Set("X-TLS-HandshakeComplete", "true")
			}
		}
	}), func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(io.Discard, "", 0)
	}).ts

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
}

type fakeConnectionStateConn struct {
	net.Conn
}

func (fcsc *fakeConnectionStateConn) ConnectionState() tls.ConnectionState {
	return tls.ConnectionState{
		ServerName: "example.com",
	}
}

func TestTLSServerWithoutTLSConn(t *testing.T) {
	//set up
	pr, pw := net.Pipe()
	c := make(chan int)
	listener := &oneConnListener{&fakeConnectionStateConn{pr}}
	server := &Server{
		Handler: HandlerFunc(func(writer ResponseWriter, request *Request) {
			if request.TLS == nil {
				t.Fatal("request.TLS is nil, expected not nil")
			}
			if request.TLS.ServerName != "example.com" {
				t.Fatalf("request.TLS.ServerName is %s, expected %s", request.TLS.ServerName, "example.com")
			}
			writer.Header().Set("X-TLS-ServerName", "example.com")
		}),
	}

	// write request and read response
	go func() {
		req, _ := NewRequest(MethodGet, "https://example.com", nil)
		req.Write(pw)

		resp, _ := ReadResponse(bufio.NewReader(pw), req)
		if hdr := resp.Header.Get("X-TLS-ServerName"); hdr != "example.com" {
			t.Errorf("response header X-TLS-ServerName is %s, expected %s", hdr, "example.com")
		}
		close(c)
		pw.Close()
	}()

	server.Serve(listener)

	// oneConnListener returns error after one accept, wait util response is read
	<-c
	pr.Close()
}

func TestServeTLS(t *testing.T) {
	CondSkipHTTP2(t)
	// Not parallel: uses global test hooks.
	defer afterTest(t)
	defer SetTestHookServerServe(nil)

	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
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
	run(t, testTLSServerRejectHTTPRequests, []testMode{https1Mode, http2Mode})
}
func testTLSServerRejectHTTPRequests(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Error("unexpected HTTPS request")
	}), func(ts *httptest.Server) {
		var errBuf bytes.Buffer
		ts.Config.ErrorLog = log.New(&errBuf, "", 0)
	}).ts
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	io.WriteString(conn, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n")
	slurp, err := io.ReadAll(conn)
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
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	testAutomaticHTTP2_ListenAndServe(t, &tls.Config{
		Certificates: []tls.Certificate{cert},
	})
}

func TestAutomaticHTTP2_ListenAndServe_GetCertificate(t *testing.T) {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	testAutomaticHTTP2_ListenAndServe(t, &tls.Config{
		GetCertificate: func(clientHello *tls.ClientHelloInfo) (*tls.Certificate, error) {
			return &cert, nil
		},
	})
}

func TestAutomaticHTTP2_ListenAndServe_GetConfigForClient(t *testing.T) {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	conf := &tls.Config{
		// GetConfigForClient requires specifying a full tls.Config so we must set
		// NextProtos ourselves.
		NextProtos:   []string{"h2"},
		Certificates: []tls.Certificate{cert},
	}
	testAutomaticHTTP2_ListenAndServe(t, &tls.Config{
		GetConfigForClient: func(clientHello *tls.ClientHelloInfo) (*tls.Config, error) {
			return conf, nil
		},
	})
}

func testAutomaticHTTP2_ListenAndServe(t *testing.T, tlsConf *tls.Config) {
	CondSkipHTTP2(t)
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
func TestServerExpect(t *testing.T) { run(t, testServerExpect, []testMode{http1Mode}) }
func testServerExpect(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// Note using r.FormValue("readbody") because for POST
		// requests that would read from r.Body, which we only
		// conditionally want to do.
		if strings.Contains(r.URL.RawQuery, "readbody=true") {
			io.ReadAll(r.Body)
			w.Write([]byte("Hi"))
		} else {
			w.WriteHeader(StatusUnauthorized)
		}
	})).ts

	runTest := func(test serverExpectTest) {
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatalf("Dial: %v", err)
		}
		defer conn.Close()

		// Only send the body immediately if we're acting like an HTTP client
		// that doesn't send 100-continue expectations.
		writeBody := test.contentLength != 0 && strings.ToLower(test.expectation) != "100-continue"

		wg := sync.WaitGroup{}
		wg.Add(1)
		defer wg.Wait()

		go func() {
			defer wg.Done()

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
					io.NopCloser(nil),
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

type bodyDiscardTest struct {
	bodySize     int
	bodyChunked  bool
	reqConnClose bool

	shouldDiscardBody bool // should the handler discard body after it exits?
}

func (t bodyDiscardTest) connectionHeader() string {
	if t.reqConnClose {
		return "Connection: close\r\n"
	}
	return ""
}

var bodyDiscardTests = [...]bodyDiscardTest{
	// Have:
	// - Small body.
	// - Content-Length defined.
	// Should:
	// - Discard remaining body.
	0: {
		bodySize:          20 << 10,
		bodyChunked:       false,
		reqConnClose:      false,
		shouldDiscardBody: true,
	},

	// Have:
	// - Small body.
	// - Chunked (no Content-Length defined).
	// Should:
	// - Discard remaining body.
	1: {
		bodySize:          20 << 10,
		bodyChunked:       true,
		reqConnClose:      false,
		shouldDiscardBody: true,
	},

	// Have:
	// - Small body.
	// - Content-Length defined.
	// - Connection: close.
	// Should:
	// - Not discard remaining body (no point as Connection: close).
	2: {
		bodySize:          20 << 10,
		bodyChunked:       false,
		reqConnClose:      true,
		shouldDiscardBody: false,
	},

	// Have:
	// - Small body.
	// - Chunked (no Content-Length defined).
	// - Connection: close.
	// Should:
	// - Discard remaining body (chunked, so it might have trailers).
	//
	// TODO: maybe skip this if no trailers were declared in the headers.
	3: {
		bodySize:          20 << 10,
		bodyChunked:       true,
		reqConnClose:      true,
		shouldDiscardBody: true,
	},

	// Have:
	// - Large body.
	// - Content-Length defined.
	// Should:
	// - Not discard remaining body (we know it is too large from Content-Length).
	4: {
		bodySize:          1 << 20,
		bodyChunked:       false,
		reqConnClose:      false,
		shouldDiscardBody: false,
	},

	// Have:
	// - Large body.
	// - Chunked (no Content-Length defined).
	// Should:
	// - Discard remaining body (chunked, so we try up to a limit before giving up).
	5: {
		bodySize:          1 << 20,
		bodyChunked:       true,
		reqConnClose:      false,
		shouldDiscardBody: true,
	},

	// Have:
	// - Large body.
	// - Content-Length defined.
	// - Connection: close.
	// Should:
	// - Not discard remaining body (Connection: Close, and Content-Length is too large).
	6: {
		bodySize:          1 << 20,
		bodyChunked:       false,
		reqConnClose:      true,
		shouldDiscardBody: false,
	},
	// Have:
	// - Large body.
	// - Chunked (no Content-Length defined).
	// - Connection: close.
	// Should:
	// - Discard remaining body (chunked, so it might have trailers).
	//
	// TODO: maybe skip this if no trailers were declared in the headers.
	7: {
		bodySize:          1 << 20,
		bodyChunked:       true,
		reqConnClose:      true,
		shouldDiscardBody: true,
	},
}

func TestBodyDiscard(t *testing.T) {
	setParallel(t)
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in -short mode")
	}
	for i, tt := range bodyDiscardTests {
		testBodyDiscard(t, i, tt)
	}
}

func testBodyDiscard(t *testing.T, i int, tt bodyDiscardTest) {
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
		conn.readBuf.Write(fmt.Appendf(nil,
			"POST / HTTP/1.1\r\n"+
				"Host: test\r\n"+
				tt.connectionHeader()+
				"Content-Length: %d\r\n"+
				"\r\n", len(body)))
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
	var initialSize, closedSize, exitSize int
	go Serve(ls, HandlerFunc(func(rw ResponseWriter, req *Request) {
		initialSize = readBufLen()
		req.Body.Close()
		closedSize = readBufLen()
	}))
	<-conn.closec
	exitSize = readBufLen()

	if initialSize != closedSize {
		t.Errorf("%d. Close() within request handler should be a no-op, but body size went from %d to %d", i, initialSize, closedSize)
	}
	if tt.shouldDiscardBody && closedSize <= exitSize {
		t.Errorf("%d. want body content to be discarded upon request handler exit, but size went from %d to %d", i, closedSize, exitSize)
	}
	if !tt.shouldDiscardBody && closedSize != exitSize {
		t.Errorf("%d. want body content to not be discarded upon request handler exit, but size went from %d to %d", i, closedSize, exitSize)
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
	{"discard", func(r io.ReadCloser) { io.Copy(io.Discard, r) }},
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
	script []any
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
			script: []any{
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

// cancelableTimeoutContext overwrites the error message to DeadlineExceeded
type cancelableTimeoutContext struct {
	context.Context
}

func (c cancelableTimeoutContext) Err() error {
	if c.Context.Err() != nil {
		return context.DeadlineExceeded
	}
	return nil
}

func TestTimeoutHandler(t *testing.T) { run(t, testTimeoutHandler) }
func testTimeoutHandler(t *testing.T, mode testMode) {
	sendHi := make(chan bool, 1)
	writeErrors := make(chan error, 1)
	sayHi := HandlerFunc(func(w ResponseWriter, r *Request) {
		<-sendHi
		_, werr := w.Write([]byte("hi"))
		writeErrors <- werr
	})
	ctx, cancel := context.WithCancel(context.Background())
	h := NewTestTimeoutHandler(sayHi, cancelableTimeoutContext{ctx})
	cst := newClientServerTest(t, mode, h)

	// Succeed without timing out:
	sendHi <- true
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusOK; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ := io.ReadAll(res.Body)
	if g, e := string(body), "hi"; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
	if g := <-writeErrors; g != nil {
		t.Errorf("got unexpected Write error on first request: %v", g)
	}

	// Times out:
	cancel()

	res, err = cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusServiceUnavailable; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ = io.ReadAll(res.Body)
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
func TestTimeoutHandlerRace(t *testing.T) { run(t, testTimeoutHandlerRace) }
func testTimeoutHandlerRace(t *testing.T, mode testMode) {
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

	ts := newClientServerTest(t, mode, TimeoutHandler(delayHi, 20*time.Millisecond, "")).ts

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
				io.Copy(io.Discard, res.Body)
				res.Body.Close()
			}
		}()
	}
	wg.Wait()
}

// See issues 8209 and 8414.
// Both issues involved panics in the implementation of TimeoutHandler.
func TestTimeoutHandlerRaceHeader(t *testing.T) { run(t, testTimeoutHandlerRaceHeader) }
func testTimeoutHandlerRaceHeader(t *testing.T, mode testMode) {
	delay204 := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(204)
	})

	ts := newClientServerTest(t, mode, TimeoutHandler(delay204, time.Nanosecond, "")).ts

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
				// We see ECONNRESET from the connection occasionally,
				// and that's OK: this test is checking that the server does not panic.
				t.Log(err)
				return
			}
			defer res.Body.Close()
			io.Copy(io.Discard, res.Body)
		}()
	}
	wg.Wait()
}

// Issue 9162
func TestTimeoutHandlerRaceHeaderTimeout(t *testing.T) { run(t, testTimeoutHandlerRaceHeaderTimeout) }
func testTimeoutHandlerRaceHeaderTimeout(t *testing.T, mode testMode) {
	sendHi := make(chan bool, 1)
	writeErrors := make(chan error, 1)
	sayHi := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "text/plain")
		<-sendHi
		_, werr := w.Write([]byte("hi"))
		writeErrors <- werr
	})
	ctx, cancel := context.WithCancel(context.Background())
	h := NewTestTimeoutHandler(sayHi, cancelableTimeoutContext{ctx})
	cst := newClientServerTest(t, mode, h)

	// Succeed without timing out:
	sendHi <- true
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusOK; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ := io.ReadAll(res.Body)
	if g, e := string(body), "hi"; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
	if g := <-writeErrors; g != nil {
		t.Errorf("got unexpected Write error on first request: %v", g)
	}

	// Times out:
	cancel()

	res, err = cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusServiceUnavailable; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ = io.ReadAll(res.Body)
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
	run(t, testTimeoutHandlerStartTimerWhenServing)
}
func testTimeoutHandlerStartTimerWhenServing(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping sleeping test in -short mode")
	}
	var handler HandlerFunc = func(w ResponseWriter, _ *Request) {
		w.WriteHeader(StatusNoContent)
	}
	timeout := 300 * time.Millisecond
	ts := newClientServerTest(t, mode, TimeoutHandler(handler, timeout, "")).ts
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

func TestTimeoutHandlerContextCanceled(t *testing.T) { run(t, testTimeoutHandlerContextCanceled) }
func testTimeoutHandlerContextCanceled(t *testing.T, mode testMode) {
	writeErrors := make(chan error, 1)
	sayHi := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "text/plain")
		var err error
		// The request context has already been canceled, but
		// retry the write for a while to give the timeout handler
		// a chance to notice.
		for i := 0; i < 100; i++ {
			_, err = w.Write([]byte("a"))
			if err != nil {
				break
			}
			time.Sleep(1 * time.Millisecond)
		}
		writeErrors <- err
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	h := NewTestTimeoutHandler(sayHi, ctx)
	cst := newClientServerTest(t, mode, h)
	defer cst.close()

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Error(err)
	}
	if g, e := res.StatusCode, StatusServiceUnavailable; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	body, _ := io.ReadAll(res.Body)
	if g, e := string(body), ""; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
	if g, e := <-writeErrors, context.Canceled; g != e {
		t.Errorf("got unexpected Write in handler: %v, want %g", g, e)
	}
}

// https://golang.org/issue/15948
func TestTimeoutHandlerEmptyResponse(t *testing.T) { run(t, testTimeoutHandlerEmptyResponse) }
func testTimeoutHandlerEmptyResponse(t *testing.T, mode testMode) {
	var handler HandlerFunc = func(w ResponseWriter, _ *Request) {
		// No response.
	}
	timeout := 300 * time.Millisecond
	ts := newClientServerTest(t, mode, TimeoutHandler(handler, timeout, "")).ts

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
	run(t, func(t *testing.T, mode testMode) {
		testHandlerPanic(t, false, mode, wrapper, "intentional death for testing")
	}, testNotParallel)
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
func TestRedirectContentTypeAndBody(t *testing.T) {
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
		body, err := io.ReadAll(resp.Body)
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
func TestZeroLengthPostAndResponse(t *testing.T) { run(t, testZeroLengthPostAndResponse) }

func testZeroLengthPostAndResponse(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, r *Request) {
		all, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("handler ReadAll: %v", err)
		}
		if len(all) != 0 {
			t.Errorf("handler got %d bytes; expected 0", len(all))
		}
		rw.Header().Set("Content-Length", "0")
	}))

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
		all, err := io.ReadAll(resp[i].Body)
		if err != nil {
			t.Fatalf("req #%d: client ReadAll: %v", i, err)
		}
		if len(all) != 0 {
			t.Errorf("req #%d: client got %d bytes; expected 0", i, len(all))
		}
	}
}

func TestHandlerPanicNil(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testHandlerPanic(t, false, mode, nil, nil)
	}, testNotParallel)
}

func TestHandlerPanic(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testHandlerPanic(t, false, mode, nil, "intentional death for testing")
	}, testNotParallel)
}

func TestHandlerPanicWithHijack(t *testing.T) {
	// Only testing HTTP/1, and our http2 server doesn't support hijacking.
	run(t, func(t *testing.T, mode testMode) {
		testHandlerPanic(t, true, mode, nil, "intentional death for testing")
	}, []testMode{http1Mode})
}

func testHandlerPanic(t *testing.T, withHijack bool, mode testMode, wrapper func(Handler) Handler, panicValue any) {
	// Direct log output to a pipe.
	//
	// We read from the pipe to verify that the handler actually caught the panic
	// and logged something.
	//
	// We use a pipe rather than a buffer, because when testing connection hijacking
	// server shutdown doesn't wait for the hijacking handler to return, so the
	// log may occur after the server has shut down.
	pr, pw := io.Pipe()
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
	cst := newClientServerTest(t, mode, handler, func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(pw, "", 0)
	})

	// Do a blocking read on the log output pipe.
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

	<-done
}

type terrorWriter struct{ t *testing.T }

func (w terrorWriter) Write(p []byte) (int, error) {
	w.t.Errorf("%s", p)
	return len(p), nil
}

// Issue 16456: allow writing 0 bytes on hijacked conn to test hijack
// without any log spam.
func TestServerWriteHijackZeroBytes(t *testing.T) {
	run(t, testServerWriteHijackZeroBytes, []testMode{http1Mode})
}
func testServerWriteHijackZeroBytes(t *testing.T, mode testMode) {
	done := make(chan struct{})
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	}), func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(terrorWriter{t}, "Unexpected write: ", 0)
	}).ts

	c := ts.Client()
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	<-done
}

func TestServerNoDate(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testServerNoHeader(t, mode, "Date")
	})
}

func TestServerContentType(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testServerNoHeader(t, mode, "Content-Type")
	})
}

func testServerNoHeader(t *testing.T, mode testMode, header string) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header()[header] = nil
		io.WriteString(w, "<html>foo</html>") // non-empty
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if got, ok := res.Header[header]; ok {
		t.Fatalf("Expected no %s header; got %q", header, got)
	}
}

func TestStripPrefix(t *testing.T) { run(t, testStripPrefix) }
func testStripPrefix(t *testing.T, mode testMode) {
	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Path", r.URL.Path)
		w.Header().Set("X-RawPath", r.URL.RawPath)
	})
	ts := newClientServerTest(t, mode, StripPrefix("/foo/bar", h)).ts

	c := ts.Client()

	cases := []struct {
		reqPath string
		path    string // If empty we want a 404.
		rawPath string
	}{
		{"/foo/bar/qux", "/qux", ""},
		{"/foo/bar%2Fqux", "/qux", "%2Fqux"},
		{"/foo%2Fbar/qux", "", ""}, // Escaped prefix does not match.
		{"/bar", "", ""},           // No prefix match.
	}
	for _, tc := range cases {
		t.Run(tc.reqPath, func(t *testing.T) {
			res, err := c.Get(ts.URL + tc.reqPath)
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
			if tc.path == "" {
				if res.StatusCode != StatusNotFound {
					t.Errorf("got %q, want 404 Not Found", res.Status)
				}
				return
			}
			if res.StatusCode != StatusOK {
				t.Fatalf("got %q, want 200 OK", res.Status)
			}
			if g, w := res.Header.Get("X-Path"), tc.path; g != w {
				t.Errorf("got Path %q, want %q", g, w)
			}
			if g, w := res.Header.Get("X-RawPath"), tc.rawPath; g != w {
				t.Errorf("got RawPath %q, want %q", g, w)
			}
		})
	}
}

// https://golang.org/issue/18952.
func TestStripPrefixNotModifyRequest(t *testing.T) {
	h := StripPrefix("/foo", NotFoundHandler())
	req := httptest.NewRequest("GET", "/foo/bar", nil)
	h.ServeHTTP(httptest.NewRecorder(), req)
	if req.URL.Path != "/foo/bar" {
		t.Errorf("StripPrefix should not modify the provided Request, but it did")
	}
}

func TestRequestLimit(t *testing.T) { run(t, testRequestLimit) }
func testRequestLimit(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Fatalf("didn't expect to get request in Handler")
	}), optQuietLog)
	req, _ := NewRequest("GET", cst.ts.URL, nil)
	var bytesPerHeader = len("header12345: val12345\r\n")
	for i := 0; i < ((DefaultMaxHeaderBytes+4096)/bytesPerHeader)+1; i++ {
		req.Header.Set(fmt.Sprintf("header%05d", i), fmt.Sprintf("val%05d", i))
	}
	res, err := cst.c.Do(req)
	if res != nil {
		defer res.Body.Close()
	}
	if mode == http2Mode {
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

type bodyLimitReader struct {
	mu     sync.Mutex
	count  int
	limit  int
	closed chan struct{}
}

func (r *bodyLimitReader) Read(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	select {
	case <-r.closed:
		return 0, errors.New("closed")
	default:
	}
	if r.count > r.limit {
		return 0, errors.New("at limit")
	}
	r.count += len(p)
	for i := range p {
		p[i] = 'a'
	}
	return len(p), nil
}

func (r *bodyLimitReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	close(r.closed)
	return nil
}

func TestRequestBodyLimit(t *testing.T) { run(t, testRequestBodyLimit) }
func testRequestBodyLimit(t *testing.T, mode testMode) {
	const limit = 1 << 20
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		r.Body = MaxBytesReader(w, r.Body, limit)
		n, err := io.Copy(io.Discard, r.Body)
		if err == nil {
			t.Errorf("expected error from io.Copy")
		}
		if n != limit {
			t.Errorf("io.Copy = %d, want %d", n, limit)
		}
		mbErr, ok := err.(*MaxBytesError)
		if !ok {
			t.Errorf("expected MaxBytesError, got %T", err)
		}
		if mbErr.Limit != limit {
			t.Errorf("MaxBytesError.Limit = %d, want %d", mbErr.Limit, limit)
		}
	}))

	body := &bodyLimitReader{
		closed: make(chan struct{}),
		limit:  limit * 200,
	}
	req, _ := NewRequest("POST", cst.ts.URL, body)

	// Send the POST, but don't care it succeeds or not. The
	// remote side is going to reply and then close the TCP
	// connection, and HTTP doesn't really define if that's
	// allowed or not. Some HTTP clients will get the response
	// and some (like ours, currently) will complain that the
	// request write failed, without reading the response.
	//
	// But that's okay, since what we're really testing is that
	// the remote side hung up on us before we wrote too much.
	resp, err := cst.c.Do(req)
	if err == nil {
		resp.Body.Close()
	}
	// Wait for the Transport to finish writing the request body.
	// It will close the body when done.
	<-body.closed

	if body.count > limit*100 {
		t.Errorf("handler restricted the request body to %d bytes, but client managed to write %d",
			limit, body.count)
	}
}

// TestClientWriteShutdown tests that if the client shuts down the write
// side of their TCP connection, the server doesn't send a 400 Bad Request.
func TestClientWriteShutdown(t *testing.T) { run(t, testClientWriteShutdown) }
func testClientWriteShutdown(t *testing.T, mode testMode) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/17906")
	}
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {})).ts
	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	err = conn.(*net.TCPConn).CloseWrite()
	if err != nil {
		t.Fatalf("CloseWrite: %v", err)
	}

	bs, err := io.ReadAll(conn)
	if err != nil {
		t.Errorf("ReadAll: %v", err)
	}
	got := string(bs)
	if got != "" {
		t.Errorf("read %q from server; want nothing", got)
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
	// Not parallel: modifies the global rstAvoidanceDelay.
	run(t, testServerGracefulClose, []testMode{http1Mode}, testNotParallel)
}
func testServerGracefulClose(t *testing.T, mode testMode) {
	runTimeSensitiveTest(t, []time.Duration{
		1 * time.Millisecond,
		5 * time.Millisecond,
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		500 * time.Millisecond,
		time.Second,
		5 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		SetRSTAvoidanceDelay(t, timeout)
		t.Logf("set RST avoidance delay to %v", timeout)

		const bodySize = 5 << 20
		req := []byte(fmt.Sprintf("POST / HTTP/1.1\r\nHost: foo.com\r\nContent-Length: %d\r\n\r\n", bodySize))
		for i := 0; i < bodySize; i++ {
			req = append(req, 'x')
		}

		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			Error(w, "bye", StatusUnauthorized)
		}))
		// We need to close cst explicitly here so that in-flight server
		// requests don't race with the call to SetRSTAvoidanceDelay for a retry.
		defer cst.close()
		ts := cst.ts

		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			return err
		}
		writeErr := make(chan error)
		go func() {
			_, err := conn.Write(req)
			writeErr <- err
		}()
		defer func() {
			conn.Close()
			// Wait for write to finish. This is a broken pipe on both
			// Darwin and Linux, but checking this isn't the point of
			// the test.
			<-writeErr
		}()

		br := bufio.NewReader(conn)
		lineNum := 0
		for {
			line, err := br.ReadString('\n')
			if err == io.EOF {
				break
			}
			if err != nil {
				return fmt.Errorf("ReadLine: %v", err)
			}
			lineNum++
			if lineNum == 1 && !strings.Contains(line, "401 Unauthorized") {
				t.Errorf("Response line = %q; want a 401", line)
			}
		}
		return nil
	})
}

func TestCaseSensitiveMethod(t *testing.T) { run(t, testCaseSensitiveMethod) }
func testCaseSensitiveMethod(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	run(t, testContentLengthZero, []testMode{http1Mode})
}
func testContentLengthZero(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {})).ts

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
	run(t, testCloseNotifier, []testMode{http1Mode})
}
func testCloseNotifier(t *testing.T, mode testMode) {
	gotReq := make(chan bool, 1)
	sawClose := make(chan bool, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		gotReq <- true
		cc := rw.(CloseNotifier).CloseNotify()
		<-cc
		sawClose <- true
	})).ts
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
		}
	}
	ts.Close()
}

// Tests that a pipelined request does not cause the first request's
// Handler's CloseNotify channel to fire.
//
// Issue 13165 (where it used to deadlock), but behavior changed in Issue 23921.
func TestCloseNotifierPipelined(t *testing.T) {
	run(t, testCloseNotifierPipelined, []testMode{http1Mode})
}
func testCloseNotifierPipelined(t *testing.T, mode testMode) {
	gotReq := make(chan bool, 2)
	sawClose := make(chan bool, 2)
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		gotReq <- true
		cc := rw.(CloseNotifier).CloseNotify()
		select {
		case <-cc:
			t.Error("unexpected CloseNotify")
		case <-time.After(100 * time.Millisecond):
		}
		sawClose <- true
	})).ts
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
	run(t, testHijackAfterCloseNotifier, []testMode{http1Mode})
}
func testHijackAfterCloseNotifier(t *testing.T, mode testMode) {
	script := make(chan string, 2)
	script <- "closenotify"
	script <- "hijack"
	close(script)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	})).ts
	res1, err := ts.Client().Get(ts.URL)
	if err != nil {
		log.Fatal(err)
	}
	res2, err := ts.Client().Get(ts.URL)
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
	run(t, testHijackBeforeRequestBodyRead, []testMode{http1Mode})
}
func testHijackBeforeRequestBodyRead(t *testing.T, mode testMode) {
	var requestBody = bytes.Repeat([]byte("a"), 1<<20)
	bodyOkay := make(chan bool, 1)
	gotCloseNotify := make(chan bool, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(bodyOkay) // caller will read false if nothing else

		reqBody := r.Body
		r.Body = nil // to test that server.go doesn't use this value.

		gone := w.(CloseNotifier).CloseNotify()
		slurp, err := io.ReadAll(reqBody)
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
		<-gone
		gotCloseNotify <- true
	})).ts

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
	<-gotCloseNotify
}

func TestOptions(t *testing.T) { run(t, testOptions, []testMode{http1Mode}) }
func testOptions(t *testing.T, mode testMode) {
	uric := make(chan string, 2) // only expect 1, but leave space for 2
	mux := NewServeMux()
	mux.HandleFunc("/", func(w ResponseWriter, r *Request) {
		uric <- r.RequestURI
	})
	ts := newClientServerTest(t, mode, mux).ts

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

func TestOptionsHandler(t *testing.T) { run(t, testOptionsHandler, []testMode{http1Mode}) }
func testOptionsHandler(t *testing.T, mode testMode) {
	rc := make(chan *Request, 1)

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		rc <- r
	}), func(ts *httptest.Server) {
		ts.Config.DisableGeneralOptionsHandler = true
	}).ts

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	_, err = conn.Write([]byte("OPTIONS * HTTP/1.1\r\nHost: foo.com\r\n\r\n"))
	if err != nil {
		t.Fatal(err)
	}

	if got := <-rc; got.Method != "OPTIONS" || got.RequestURI != "*" {
		t.Errorf("Expected OPTIONS * request, got %v", got)
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
		ErrorLog: log.New(io.Discard, "", 0), // noisy otherwise
	}
	err := server.Serve(ln)
	if err != io.EOF {
		t.Errorf("got error %v, want EOF", err)
	}
}

func TestWriteAfterHijack(t *testing.T) {
	req := reqBytes("GET / HTTP/1.1\nHost: golang.org")
	var buf strings.Builder
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
	run(t, testHTTP10ConnectionHeader, []testMode{http1Mode})
}
func testHTTP10ConnectionHeader(t *testing.T, mode testMode) {
	mux := NewServeMux()
	mux.Handle("/", HandlerFunc(func(ResponseWriter, *Request) {}))
	ts := newClientServerTest(t, mode, mux).ts

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
		if !slices.Equal(got, tt.expect) {
			t.Errorf("wrong Connection headers for request %q. Got %q expect %q", tt.req, got, tt.expect)
		}
	}
}

// See golang.org/issue/5660
func TestServerReaderFromOrder(t *testing.T) { run(t, testServerReaderFromOrder) }
func testServerReaderFromOrder(t *testing.T, mode testMode) {
	pr, pw := io.Pipe()
	const size = 3 << 20
	cst := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		rw.Header().Set("Content-Type", "text/plain") // prevent sniffing path
		done := make(chan bool)
		go func() {
			io.Copy(rw, pr)
			close(done)
		}()
		time.Sleep(25 * time.Millisecond) // give Copy a chance to break things
		n, err := io.Copy(io.Discard, req.Body)
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

	req, err := NewRequest("POST", cst.ts.URL, io.LimitReader(neverEnding('a'), size))
	if err != nil {
		t.Fatal(err)
	}
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	all, err := io.ReadAll(res.Body)
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
	for _, code := range []int{StatusNotModified, StatusNoContent} {
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
func TestTransportAndServerSharedBodyRace(t *testing.T) {
	run(t, testTransportAndServerSharedBodyRace, testNotParallel)
}
func testTransportAndServerSharedBodyRace(t *testing.T, mode testMode) {
	// The proxy server in the middle of the stack for this test potentially
	// from its handler after only reading half of the body.
	// That can trigger https://go.dev/issue/3595, which is otherwise
	// irrelevant to this test.
	runTimeSensitiveTest(t, []time.Duration{
		1 * time.Millisecond,
		5 * time.Millisecond,
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		500 * time.Millisecond,
		time.Second,
		5 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		SetRSTAvoidanceDelay(t, timeout)
		t.Logf("set RST avoidance delay to %v", timeout)

		const bodySize = 1 << 20

		var wg sync.WaitGroup
		backend := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
			// Work around https://go.dev/issue/38370: clientServerTest uses
			// an httptest.Server under the hood, and in HTTP/2 mode it does not always
			// “[block] until all outstanding requests on this server have completed”,
			// causing the call to Logf below to race with the end of the test.
			//
			// Since the client doesn't cancel the request until we have copied half
			// the body, this call to add happens before the test is cleaned up,
			// preventing the race.
			wg.Add(1)
			defer wg.Done()

			n, err := io.CopyN(rw, req.Body, bodySize)
			t.Logf("backend CopyN: %v, %v", n, err)
			<-req.Context().Done()
		}))
		// We need to close explicitly here so that in-flight server
		// requests don't race with the call to SetRSTAvoidanceDelay for a retry.
		defer func() {
			wg.Wait()
			backend.close()
		}()

		var proxy *clientServerTest
		proxy = newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
			req2, _ := NewRequest("POST", backend.ts.URL, req.Body)
			req2.ContentLength = bodySize
			cancel := make(chan struct{})
			req2.Cancel = cancel

			bresp, err := proxy.c.Do(req2)
			if err != nil {
				t.Errorf("Proxy outbound request: %v", err)
				return
			}
			_, err = io.CopyN(io.Discard, bresp.Body, bodySize/2)
			if err != nil {
				t.Errorf("Proxy copy error: %v", err)
				return
			}
			t.Cleanup(func() { bresp.Body.Close() })

			// Try to cause a race. Canceling the client request will cause the client
			// transport to close req2.Body. Returning from the server handler will
			// cause the server to close req.Body. Since they are the same underlying
			// ReadCloser, that will result in concurrent calls to Close (and possibly a
			// Read concurrent with a Close).
			if mode == http2Mode {
				close(cancel)
			} else {
				proxy.c.Transport.(*Transport).CancelRequest(req2)
			}
			rw.Write([]byte("OK"))
		}))
		defer proxy.close()

		req, _ := NewRequest("POST", proxy.ts.URL, io.LimitReader(neverEnding('a'), bodySize))
		res, err := proxy.c.Do(req)
		if err != nil {
			return fmt.Errorf("original request: %v", err)
		}
		res.Body.Close()
		return nil
	})
}

// Test that a hanging Request.Body.Read from another goroutine can't
// cause the Handler goroutine's Request.Body.Close to block.
// See issue 7121.
func TestRequestBodyCloseDoesntBlock(t *testing.T) {
	run(t, testRequestBodyCloseDoesntBlock, []testMode{http1Mode})
}
func testRequestBodyCloseDoesntBlock(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}

	readErrCh := make(chan error, 1)
	errCh := make(chan error, 2)

	server := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		go func(body io.Reader) {
			_, err := body.Read(make([]byte, 100))
			readErrCh <- err
		}(req.Body)
		time.Sleep(500 * time.Millisecond)
	})).ts

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

func TestServerConnState(t *testing.T) { run(t, testServerConnState, []testMode{http1Mode}) }
func testServerConnState(t *testing.T, mode testMode) {
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

	// A stateLog is a log of states over the lifetime of a connection.
	type stateLog struct {
		active   net.Conn // The connection for which the log is recorded; set to the first connection seen in StateNew.
		got      []ConnState
		want     []ConnState
		complete chan<- struct{} // If non-nil, closed when either 'got' is equal to 'want', or 'got' is no longer a prefix of 'want'.
	}
	activeLog := make(chan *stateLog, 1)

	// wantLog invokes doRequests, then waits for the resulting connection to
	// either pass through the sequence of states in want or enter a state outside
	// of that sequence.
	wantLog := func(doRequests func(), want ...ConnState) {
		t.Helper()
		complete := make(chan struct{})
		activeLog <- &stateLog{want: want, complete: complete}

		doRequests()

		<-complete
		sl := <-activeLog
		if !slices.Equal(sl.got, sl.want) {
			t.Errorf("Request(s) produced unexpected state sequence.\nGot:  %v\nWant: %v", sl.got, sl.want)
		}
		// Don't return sl to activeLog: we don't expect any further states after
		// this point, and want to keep the ConnState callback blocked until the
		// next call to wantLog.
	}

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		handler[r.URL.Path](w, r)
	}), func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(io.Discard, "", 0)
		ts.Config.ConnState = func(c net.Conn, state ConnState) {
			if c == nil {
				t.Errorf("nil conn seen in state %s", state)
				return
			}
			sl := <-activeLog
			if sl.active == nil && state == StateNew {
				sl.active = c
			} else if sl.active != c {
				t.Errorf("unexpected conn in state %s", state)
				activeLog <- sl
				return
			}
			sl.got = append(sl.got, state)
			if sl.complete != nil && (len(sl.got) >= len(sl.want) || !slices.Equal(sl.got, sl.want[:len(sl.got)])) {
				close(sl.complete)
				sl.complete = nil
			}
			activeLog <- sl
		}
	}).ts
	defer func() {
		activeLog <- &stateLog{} // If the test failed, allow any remaining ConnState callbacks to complete.
		ts.Close()
	}()

	c := ts.Client()

	mustGet := func(url string, headers ...string) {
		t.Helper()
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
		_, err = io.ReadAll(res.Body)
		defer res.Body.Close()
		if err != nil {
			t.Errorf("Error reading %s: %v", url, err)
		}
	}

	wantLog(func() {
		mustGet(ts.URL + "/")
		mustGet(ts.URL + "/close")
	}, StateNew, StateActive, StateIdle, StateActive, StateClosed)

	wantLog(func() {
		mustGet(ts.URL + "/")
		mustGet(ts.URL+"/", "Connection", "close")
	}, StateNew, StateActive, StateIdle, StateActive, StateClosed)

	wantLog(func() {
		mustGet(ts.URL + "/hijack")
	}, StateNew, StateActive, StateHijacked)

	wantLog(func() {
		mustGet(ts.URL + "/hijack-panic")
	}, StateNew, StateActive, StateHijacked)

	wantLog(func() {
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		c.Close()
	}, StateNew, StateClosed)

	wantLog(func() {
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(c, "BOGUS REQUEST\r\n\r\n"); err != nil {
			t.Fatal(err)
		}
		c.Read(make([]byte, 1)) // block until server hangs up on us
		c.Close()
	}, StateNew, StateActive, StateClosed)

	wantLog(func() {
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
		if _, err := io.Copy(io.Discard, res.Body); err != nil {
			t.Fatal(err)
		}
		c.Close()
	}, StateNew, StateActive, StateIdle, StateClosed)
}

func TestServerKeepAlivesEnabledResultClose(t *testing.T) {
	run(t, testServerKeepAlivesEnabledResultClose, []testMode{http1Mode})
}
func testServerKeepAlivesEnabledResultClose(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
	}), func(ts *httptest.Server) {
		ts.Config.SetKeepAlivesEnabled(false)
	}).ts
	res, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if !res.Close {
		t.Errorf("Body.Close == false; want true")
	}
}

// golang.org/issue/7856
func TestServerEmptyBodyRace(t *testing.T) { run(t, testServerEmptyBodyRace) }
func testServerEmptyBodyRace(t *testing.T, mode testMode) {
	var n int32
	cst := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		atomic.AddInt32(&n, 1)
	}), optQuietLog)
	var wg sync.WaitGroup
	const reqs = 20
	for i := 0; i < reqs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := cst.c.Get(cst.ts.URL)
			if err != nil {
				// Try to deflake spurious "connection reset by peer" under load.
				// See golang.org/issue/22540.
				time.Sleep(10 * time.Millisecond)
				res, err = cst.c.Get(cst.ts.URL)
				if err != nil {
					t.Error(err)
					return
				}
			}
			defer res.Body.Close()
			_, err = io.Copy(io.Discard, res.Body)
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
			Writer: io.Discard,
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
	SetRSTAvoidanceDelay(t, 1*time.Millisecond)

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
// A similar test crashed once during development, but it was only
// testing this tangentially and temporarily until another TODO was
// fixed.
//
// So add an explicit test for this.
func TestServerFlushAndHijack(t *testing.T) { run(t, testServerFlushAndHijack, []testMode{http1Mode}) }
func testServerFlushAndHijack(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	})).ts
	res, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	all, err := io.ReadAll(res.Body)
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
	run(t, testServerKeepAliveAfterWriteError, []testMode{http1Mode})
}
func testServerKeepAliveAfterWriteError(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	const numReq = 3
	addrc := make(chan string, numReq)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		addrc <- r.RemoteAddr
		time.Sleep(500 * time.Millisecond)
		w.(Flusher).Flush()
	}), func(ts *httptest.Server) {
		ts.Config.WriteTimeout = 250 * time.Millisecond
	}).ts

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
		}
	}
}

// Issue 9987: shouldn't add automatic Content-Length (or
// Content-Type) if a Transfer-Encoding was set by the handler.
func TestNoContentLengthIfTransferEncoding(t *testing.T) {
	run(t, testNoContentLengthIfTransferEncoding, []testMode{http1Mode})
}
func testNoContentLengthIfTransferEncoding(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Transfer-Encoding", "foo")
		io.WriteString(w, "<html>")
	})).ts
	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	if _, err := io.WriteString(c, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n"); err != nil {
		t.Fatal(err)
	}
	bs := bufio.NewScanner(c)
	var got strings.Builder
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
	var buf strings.Builder
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
			io.ReadAll(r.Body)
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
// verify the server implicitly tries to do a read on it before replying.
func TestHandlerFinishSkipBigContentLengthRead(t *testing.T) {
	setParallel(t)
	conn := newTestConn()
	conn.readBuf.WriteString(
		"POST / HTTP/1.1\r\n" +
			"Host: test\r\n" +
			"Content-Length: 9999999999\r\n" +
			"\r\n" + strings.Repeat("a", 1<<20))

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

func TestHandlerSetsBodyNil(t *testing.T) { run(t, testHandlerSetsBodyNil) }
func testHandlerSetsBodyNil(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		r.Body = nil
		fmt.Fprintf(w, "%v", r.RemoteAddr)
	}))
	get := func() string {
		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		slurp, err := io.ReadAll(res.Body)
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
		{"HTTP/0.9", "", 505},

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
		{"PRI / HTTP/2.0", "", 505},
		{"GET / HTTP/2.0", "", 505},
		{"GET / HTTP/3.0", "", 505},
	}
	for _, tt := range tests {
		conn := newTestConn()
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
	run(t, testServerHandlersCanHandleH2PRI, []testMode{http1Mode})
}
func testServerHandlersCanHandleH2PRI(t *testing.T, mode testMode) {
	const upgradeResponse = "upgrade here"
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		conn, br, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
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
	})).ts

	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()
	io.WriteString(c, "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
	slurp, err := io.ReadAll(c)
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
		// Spaces between the header key and colon are not allowed.
		// See RFC 7230, Section 3.2.4.
		{"Foo : bar\r\n", 400},
		{"Foo\t: bar\r\n", 400},

		// Empty header keys are invalid.
		// See RFC 7230, Section 3.2.
		{": empty key\r\n", 400},

		// Requests with invalid Content-Length headers should be rejected
		// regardless of the presence of a Transfer-Encoding header.
		// Check out RFC 9110, Section 8.6 and RFC 9112, Section 6.3.3.
		{"Content-Length: notdigits\r\n", 400},
		{"Content-Length: notdigits\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 400},

		{"foo: foo foo\r\n", 200},    // LWS space is okay
		{"foo: foo\tfoo\r\n", 200},   // LWS tab is okay
		{"foo: foo\x00foo\r\n", 400}, // CTL 0x00 in value is bad
		{"foo: foo\x7ffoo\r\n", 400}, // CTL 0x7f in value is bad
		{"foo: foo\xfffoo\r\n", 200}, // non-ASCII high octets in value are fine
	}
	for _, tt := range tests {
		conn := newTestConn()
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

func TestServerRequestContextCancel_ServeHTTPDone(t *testing.T) {
	run(t, testServerRequestContextCancel_ServeHTTPDone)
}
func testServerRequestContextCancel_ServeHTTPDone(t *testing.T, mode testMode) {
	ctxc := make(chan context.Context, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ctx := r.Context()
		select {
		case <-ctx.Done():
			t.Error("should not be Done in ServeHTTP")
		default:
		}
		ctxc <- ctx
	}))
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
	run(t, testServerRequestContextCancel_ConnClose, []testMode{http1Mode})
}
func testServerRequestContextCancel_ConnClose(t *testing.T, mode testMode) {
	inHandler := make(chan struct{})
	handlerDone := make(chan struct{})
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		close(inHandler)
		<-r.Context().Done()
		close(handlerDone)
	})).ts
	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	io.WriteString(c, "GET / HTTP/1.1\r\nHost: foo\r\n\r\n")
	<-inHandler
	c.Close() // this should trigger the context being done
	<-handlerDone
}

func TestServerContext_ServerContextKey(t *testing.T) {
	run(t, testServerContext_ServerContextKey)
}
func testServerContext_ServerContextKey(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ctx := r.Context()
		got := ctx.Value(ServerContextKey)
		if _, ok := got.(*Server); !ok {
			t.Errorf("context value = %T; want *http.Server", got)
		}
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

func TestServerContext_LocalAddrContextKey(t *testing.T) {
	run(t, testServerContext_LocalAddrContextKey)
}
func testServerContext_LocalAddrContextKey(t *testing.T, mode testMode) {
	ch := make(chan any, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ch <- r.Context().Value(LocalAddrContextKey)
	}))
	if _, err := cst.c.Head(cst.ts.URL); err != nil {
		t.Fatal(err)
	}

	host := cst.ts.Listener.Addr().String()
	got := <-ch
	if addr, ok := got.(net.Addr); !ok {
		t.Errorf("local addr value = %T; want net.Addr", got)
	} else if fmt.Sprint(addr) != host {
		t.Errorf("local addr = %v; want %v", addr, host)
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
	run(b, benchmarkClientServer, []testMode{http1Mode, https1Mode, http2Mode})
}
func benchmarkClientServer(b *testing.B, mode testMode) {
	b.ReportAllocs()
	b.StopTimer()
	ts := newClientServerTest(b, mode, HandlerFunc(func(rw ResponseWriter, r *Request) {
		fmt.Fprintf(rw, "Hello world.\n")
	})).ts
	b.StartTimer()

	c := ts.Client()
	for i := 0; i < b.N; i++ {
		res, err := c.Get(ts.URL)
		if err != nil {
			b.Fatal("Get:", err)
		}
		all, err := io.ReadAll(res.Body)
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

func BenchmarkClientServerParallel(b *testing.B) {
	for _, parallelism := range []int{4, 64} {
		b.Run(fmt.Sprint(parallelism), func(b *testing.B) {
			run(b, func(b *testing.B, mode testMode) {
				benchmarkClientServerParallel(b, parallelism, mode)
			}, []testMode{http1Mode, https1Mode, http2Mode})
		})
	}
}

func benchmarkClientServerParallel(b *testing.B, parallelism int, mode testMode) {
	b.ReportAllocs()
	ts := newClientServerTest(b, mode, HandlerFunc(func(rw ResponseWriter, r *Request) {
		fmt.Fprintf(rw, "Hello world.\n")
	})).ts
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
			all, err := io.ReadAll(res.Body)
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
//
//	$ go test -c
//	$ ./http.test -test.run='^$' -test.bench='^BenchmarkServer$' -test.benchtime=15s -test.cpuprofile=http.prof
//	$ go tool pprof http.test http.prof
//	(pprof) web
func BenchmarkServer(b *testing.B) {
	b.ReportAllocs()
	// Child process mode;
	if url := os.Getenv("GO_TEST_BENCH_SERVER_URL"); url != "" {
		n, err := strconv.Atoi(os.Getenv("GO_TEST_BENCH_CLIENT_N"))
		if err != nil {
			panic(err)
		}
		for i := 0; i < n; i++ {
			res, err := Get(url)
			if err != nil {
				log.Panicf("Get: %v", err)
			}
			all, err := io.ReadAll(res.Body)
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

	cmd := testenv.Command(b, os.Args[0], "-test.run=^$", "-test.bench=^BenchmarkServer$")
	cmd.Env = append([]string{
		fmt.Sprintf("GO_TEST_BENCH_CLIENT_N=%d", b.N),
		fmt.Sprintf("GO_TEST_BENCH_SERVER_URL=%s", ts.URL),
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
	var data = []byte("Hello world.\n")

	url := startClientBenchmarkServer(b, HandlerFunc(func(w ResponseWriter, _ *Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(data)
	}))

	// Do b.N requests to the server.
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		res, err := Get(url)
		if err != nil {
			b.Fatalf("Get: %v", err)
		}
		body, err := io.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			b.Fatalf("ReadAll: %v", err)
		}
		if !bytes.Equal(body, data) {
			b.Fatalf("Got body: %q", body)
		}
	}
	b.StopTimer()
}

func startClientBenchmarkServer(b *testing.B, handler Handler) string {
	b.ReportAllocs()
	b.StopTimer()

	if server := os.Getenv("GO_TEST_BENCH_SERVER"); server != "" {
		// Server process mode.
		port := os.Getenv("GO_TEST_BENCH_SERVER_PORT") // can be set by user
		if port == "" {
			port = "0"
		}
		ln, err := net.Listen("tcp", "localhost:"+port)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(ln.Addr().String())

		HandleFunc("/", func(w ResponseWriter, r *Request) {
			r.ParseForm()
			if r.Form.Get("stop") != "" {
				os.Exit(0)
			}
			handler.ServeHTTP(w, r)
		})
		var srv Server
		log.Fatal(srv.Serve(ln))
	}

	// Start server process.
	ctx, cancel := context.WithCancel(context.Background())
	cmd := testenv.CommandContext(b, ctx, os.Args[0], "-test.run=^$", "-test.bench=^"+b.Name()+"$")
	cmd.Env = append(cmd.Environ(), "GO_TEST_BENCH_SERVER=yes")
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		b.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		b.Fatalf("subprocess failed to start: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
		close(done)
	}()

	// Wait for the server in the child process to respond and tell us
	// its listening address, once it's started listening:
	bs := bufio.NewScanner(stdout)
	if !bs.Scan() {
		b.Fatalf("failed to read listening URL from child: %v", bs.Err())
	}
	url := "http://" + strings.TrimSpace(bs.Text()) + "/"
	if _, err := getNoBody(url); err != nil {
		b.Fatalf("initial probe of child process failed: %v", err)
	}

	// Instruct server process to stop.
	b.Cleanup(func() {
		getNoBody(url + "?stop=yes")
		if err := <-done; err != nil {
			b.Fatalf("subprocess failed: %v", err)
		}

		cancel()
		<-done

		afterTest(b)
	})

	return url
}

func BenchmarkClientGzip(b *testing.B) {
	const responseSize = 1024 * 1024

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := io.CopyN(gz, crand.Reader, responseSize); err != nil {
		b.Fatal(err)
	}
	gz.Close()

	data := buf.Bytes()

	url := startClientBenchmarkServer(b, HandlerFunc(func(w ResponseWriter, _ *Request) {
		w.Header().Set("Content-Encoding", "gzip")
		w.Write(data)
	}))

	// Do b.N requests to the server.
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		res, err := Get(url)
		if err != nil {
			b.Fatalf("Get: %v", err)
		}
		n, err := io.Copy(io.Discard, res.Body)
		res.Body.Close()
		if err != nil {
			b.Fatalf("ReadAll: %v", err)
		}
		if n != responseSize {
			b.Fatalf("ReadAll: expected %d bytes, got %d", responseSize, n)
		}
	}
	b.StopTimer()
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

	conn := newTestConn()
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
		Writer: io.Discard,
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
		Writer: io.Discard,
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
		Writer: io.Discard,
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
		Writer: io.Discard,
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

func BenchmarkCloseNotifier(b *testing.B) { run(b, benchmarkCloseNotifier, []testMode{http1Mode}) }
func benchmarkCloseNotifier(b *testing.B, mode testMode) {
	b.ReportAllocs()
	b.StopTimer()
	sawClose := make(chan bool)
	ts := newClientServerTest(b, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		<-rw.(CloseNotifier).CloseNotify()
		sawClose <- true
	})).ts
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
		<-sawClose
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

func TestServerIdleTimeout(t *testing.T) { run(t, testServerIdleTimeout, []testMode{http1Mode}) }
func testServerIdleTimeout(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	runTimeSensitiveTest(t, []time.Duration{
		10 * time.Millisecond,
		100 * time.Millisecond,
		1 * time.Second,
		10 * time.Second,
	}, func(t *testing.T, readHeaderTimeout time.Duration) error {
		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			io.Copy(io.Discard, r.Body)
			io.WriteString(w, r.RemoteAddr)
		}), func(ts *httptest.Server) {
			ts.Config.ReadHeaderTimeout = readHeaderTimeout
			ts.Config.IdleTimeout = 2 * readHeaderTimeout
		})
		defer cst.close()
		ts := cst.ts
		t.Logf("ReadHeaderTimeout = %v", ts.Config.ReadHeaderTimeout)
		t.Logf("IdleTimeout = %v", ts.Config.IdleTimeout)
		c := ts.Client()

		get := func() (string, error) {
			res, err := c.Get(ts.URL)
			if err != nil {
				return "", err
			}
			defer res.Body.Close()
			slurp, err := io.ReadAll(res.Body)
			if err != nil {
				// If we're at this point the headers have definitely already been
				// read and the server is not idle, so neither timeout applies:
				// this should never fail.
				t.Fatal(err)
			}
			return string(slurp), nil
		}

		a1, err := get()
		if err != nil {
			return err
		}
		a2, err := get()
		if err != nil {
			return err
		}
		if a1 != a2 {
			return fmt.Errorf("did requests on different connections")
		}
		time.Sleep(ts.Config.IdleTimeout * 3 / 2)
		a3, err := get()
		if err != nil {
			return err
		}
		if a2 == a3 {
			return fmt.Errorf("request three unexpectedly on same connection")
		}

		// And test that ReadHeaderTimeout still works:
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			return err
		}
		defer conn.Close()
		conn.Write([]byte("GET / HTTP/1.1\r\nHost: foo.com\r\n"))
		time.Sleep(ts.Config.ReadHeaderTimeout * 2)
		if _, err := io.CopyN(io.Discard, conn, 1); err == nil {
			return fmt.Errorf("copy byte succeeded; want err")
		}

		return nil
	})
}

func get(t *testing.T, c *Client, url string) string {
	res, err := c.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	slurp, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(slurp)
}

// Tests that calls to Server.SetKeepAlivesEnabled(false) closes any
// currently-open connections.
func TestServerSetKeepAlivesEnabledClosesConns(t *testing.T) {
	run(t, testServerSetKeepAlivesEnabledClosesConns, []testMode{http1Mode})
}
func testServerSetKeepAlivesEnabledClosesConns(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, r.RemoteAddr)
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)

	get := func() string { return get(t, c, ts.URL) }

	a1, a2 := get(), get()
	if a1 == a2 {
		t.Logf("made two requests from a single conn %q (as expected)", a1)
	} else {
		t.Errorf("server reported requests from %q and %q; expected same connection", a1, a2)
	}

	// The two requests should have used the same connection,
	// and there should not have been a second connection that
	// was created by racing dial against reuse.
	// (The first get was completed when the second get started.)
	if conns := tr.IdleConnStrsForTesting(); len(conns) != 1 {
		t.Errorf("found %d idle conns (%q); want 1", len(conns), conns)
	}

	// SetKeepAlivesEnabled should discard idle conns.
	ts.Config.SetKeepAlivesEnabled(false)

	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		if conns := tr.IdleConnStrsForTesting(); len(conns) > 0 {
			if d > 0 {
				t.Logf("idle conns %v after SetKeepAlivesEnabled called = %q; waiting for empty", d, conns)
			}
			return false
		}
		return true
	})

	// If we make a third request it should use a new connection, but in general
	// we have no way to verify that: the new connection could happen to reuse the
	// exact same ports from the previous connection.
}

func TestServerShutdown(t *testing.T) { run(t, testServerShutdown) }
func testServerShutdown(t *testing.T, mode testMode) {
	var cst *clientServerTest

	var once sync.Once
	statesRes := make(chan map[ConnState]int, 1)
	shutdownRes := make(chan error, 1)
	gotOnShutdown := make(chan struct{})
	handler := HandlerFunc(func(w ResponseWriter, r *Request) {
		first := false
		once.Do(func() {
			statesRes <- cst.ts.Config.ExportAllConnsByState()
			go func() {
				shutdownRes <- cst.ts.Config.Shutdown(context.Background())
			}()
			first = true
		})

		if first {
			// Shutdown is graceful, so it should not interrupt this in-flight response
			// but should reject new requests. (Since this request is still in flight,
			// the server's port should not be reused for another server yet.)
			<-gotOnShutdown
			// TODO(#59038): The HTTP/2 server empirically does not always reject new
			// requests. As a workaround, loop until we see a failure.
			for !t.Failed() {
				res, err := cst.c.Get(cst.ts.URL)
				if err != nil {
					break
				}
				out, _ := io.ReadAll(res.Body)
				res.Body.Close()
				if mode == http2Mode {
					t.Logf("%v: unexpected success (%q). Listener should be closed before OnShutdown is called.", cst.ts.URL, out)
					t.Logf("Retrying to work around https://go.dev/issue/59038.")
					continue
				}
				t.Errorf("%v: unexpected success (%q). Listener should be closed before OnShutdown is called.", cst.ts.URL, out)
			}
		}

		io.WriteString(w, r.RemoteAddr)
	})

	cst = newClientServerTest(t, mode, handler, func(srv *httptest.Server) {
		srv.Config.RegisterOnShutdown(func() { close(gotOnShutdown) })
	})

	out := get(t, cst.c, cst.ts.URL) // calls t.Fail on failure
	t.Logf("%v: %q", cst.ts.URL, out)

	if err := <-shutdownRes; err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	<-gotOnShutdown // Will hang if RegisterOnShutdown is broken.

	if states := <-statesRes; states[StateActive] != 1 {
		t.Errorf("connection in wrong state, %v", states)
	}
}

func TestServerShutdownStateNew(t *testing.T) { runSynctest(t, testServerShutdownStateNew) }
func testServerShutdownStateNew(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("test takes 5-6 seconds; skipping in short mode")
	}

	listener := fakeNetListen()
	defer listener.Close()

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		// nothing.
	}), func(ts *httptest.Server) {
		ts.Listener.Close()
		ts.Listener = listener
		// Ignore irrelevant error about TLS handshake failure.
		ts.Config.ErrorLog = log.New(io.Discard, "", 0)
	}).ts

	// Start a connection but never write to it.
	c := listener.connect()
	defer c.Close()
	synctest.Wait()

	shutdownRes := runAsync(func() (struct{}, error) {
		return struct{}{}, ts.Config.Shutdown(context.Background())
	})

	// TODO(#59037): This timeout is hard-coded in closeIdleConnections.
	// It is undocumented, and some users may find it surprising.
	// Either document it, or switch to a less surprising behavior.
	const expectTimeout = 5 * time.Second

	// Wait until just before the expected timeout.
	time.Sleep(expectTimeout - 1)
	synctest.Wait()
	if shutdownRes.done() {
		t.Fatal("shutdown too soon")
	}
	if c.IsClosedByPeer() {
		t.Fatal("connection was closed by server too soon")
	}

	// closeIdleConnections isn't precise about its actual shutdown time.
	// Wait long enough for it to definitely have shut down.
	//
	// (It would be good to make closeIdleConnections less sloppy.)
	time.Sleep(2 * time.Second)
	synctest.Wait()
	if _, err := shutdownRes.result(); err != nil {
		t.Fatalf("Shutdown() = %v, want complete", err)
	}
	if !c.IsClosedByPeer() {
		t.Fatalf("connection was not closed by server after shutdown")
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
func TestServerKeepAlivesEnabled(t *testing.T) { run(t, testServerKeepAlivesEnabled, testNotParallel) }
func testServerKeepAlivesEnabled(t *testing.T, mode testMode) {
	if mode == http2Mode {
		restore := ExportSetH2GoawayTimeout(10 * time.Millisecond)
		defer restore()
	}
	// Not parallel: messes with global variable. (http2goAwayTimeout)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {}))
	defer cst.close()
	srv := cst.ts.Config
	srv.SetKeepAlivesEnabled(false)
	for try := 0; try < 2; try++ {
		waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
			if !srv.ExportAllConnsIdle() {
				if d > 0 {
					t.Logf("test server still has active conns after %v", d)
				}
				return false
			}
			return true
		})
		conns := 0
		var info httptrace.GotConnInfo
		ctx := httptrace.WithClientTrace(context.Background(), &httptrace.ClientTrace{
			GotConn: func(v httptrace.GotConnInfo) {
				conns++
				info = v
			},
		})
		req, err := NewRequestWithContext(ctx, "GET", cst.ts.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		res, err := cst.c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if conns != 1 {
			t.Fatalf("request %v: got %v conns, want 1", try, conns)
		}
		if info.Reused || info.WasIdle {
			t.Fatalf("request %v: Reused=%v (want false), WasIdle=%v (want false)", try, info.Reused, info.WasIdle)
		}
	}
}

// Issue 18447: test that the Server's ReadTimeout is stopped while
// the server's doing its 1-byte background read between requests,
// waiting for the connection to maybe close.
func TestServerCancelsReadTimeoutWhenIdle(t *testing.T) { run(t, testServerCancelsReadTimeoutWhenIdle) }
func testServerCancelsReadTimeoutWhenIdle(t *testing.T, mode testMode) {
	runTimeSensitiveTest(t, []time.Duration{
		10 * time.Millisecond,
		50 * time.Millisecond,
		250 * time.Millisecond,
		time.Second,
		2 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
			select {
			case <-time.After(2 * timeout):
				fmt.Fprint(w, "ok")
			case <-r.Context().Done():
				fmt.Fprint(w, r.Context().Err())
			}
		}), func(ts *httptest.Server) {
			ts.Config.ReadTimeout = timeout
			t.Logf("Server.Config.ReadTimeout = %v", timeout)
		})
		defer cst.close()
		ts := cst.ts

		var retries atomic.Int32
		cst.c.Transport.(*Transport).Proxy = func(*Request) (*url.URL, error) {
			if retries.Add(1) != 1 {
				return nil, errors.New("too many retries")
			}
			return nil, nil
		}

		c := ts.Client()

		res, err := c.Get(ts.URL)
		if err != nil {
			return fmt.Errorf("Get: %v", err)
		}
		slurp, err := io.ReadAll(res.Body)
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

// Issue 54784: test that the Server's ReadHeaderTimeout only starts once the
// beginning of a request has been received, rather than including time the
// connection spent idle.
func TestServerCancelsReadHeaderTimeoutWhenIdle(t *testing.T) {
	run(t, testServerCancelsReadHeaderTimeoutWhenIdle, []testMode{http1Mode})
}
func testServerCancelsReadHeaderTimeoutWhenIdle(t *testing.T, mode testMode) {
	runTimeSensitiveTest(t, []time.Duration{
		10 * time.Millisecond,
		50 * time.Millisecond,
		250 * time.Millisecond,
		time.Second,
		2 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		cst := newClientServerTest(t, mode, serve(200), func(ts *httptest.Server) {
			ts.Config.ReadHeaderTimeout = timeout
			ts.Config.IdleTimeout = 0 // disable idle timeout
		})
		defer cst.close()
		ts := cst.ts

		// rather than using an http.Client, create a single connection, so that
		// we can ensure this connection is not closed.
		conn, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatalf("dial failed: %v", err)
		}
		br := bufio.NewReader(conn)
		defer conn.Close()

		if _, err := conn.Write([]byte("GET / HTTP/1.1\r\nHost: e.com\r\n\r\n")); err != nil {
			return fmt.Errorf("writing first request failed: %v", err)
		}

		if _, err := ReadResponse(br, nil); err != nil {
			return fmt.Errorf("first response (before timeout) failed: %v", err)
		}

		// wait for longer than the server's ReadHeaderTimeout, and then send
		// another request
		time.Sleep(timeout * 3 / 2)

		if _, err := conn.Write([]byte("GET / HTTP/1.1\r\nHost: e.com\r\n\r\n")); err != nil {
			return fmt.Errorf("writing second request failed: %v", err)
		}

		if _, err := ReadResponse(br, nil); err != nil {
			return fmt.Errorf("second response (after timeout) failed: %v", err)
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
		if i == len(durations)-1 || t.Failed() {
			t.Fatalf("failed with duration %v: %v", d, err)
		}
		t.Logf("retrying after error with duration %v: %v", d, err)
	}
}

// Issue 18535: test that the Server doesn't try to do a background
// read if it's already done one.
func TestServerDuplicateBackgroundRead(t *testing.T) {
	run(t, testServerDuplicateBackgroundRead, []testMode{http1Mode})
}
func testServerDuplicateBackgroundRead(t *testing.T, mode testMode) {
	if runtime.GOOS == "netbsd" && runtime.GOARCH == "arm" {
		testenv.SkipFlaky(t, 24826)
	}

	goroutines := 5
	requests := 2000
	if testing.Short() {
		goroutines = 3
		requests = 100
	}

	hts := newClientServerTest(t, mode, HandlerFunc(NotFound)).ts

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
				io.Copy(io.Discard, cn)
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
	run(t, testServerHijackGetsBackgroundByte, []testMode{http1Mode})
}
func testServerHijackGetsBackgroundByte(t *testing.T, mode testMode) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/18657")
	}
	done := make(chan struct{})
	inHandler := make(chan bool, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	})).ts

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
	<-done
}

// Test that the bufio.Reader returned by Hijack yields the entire body.
func TestServerHijackGetsFullBody(t *testing.T) {
	run(t, testServerHijackGetsFullBody, []testMode{http1Mode})
}
func testServerHijackGetsFullBody(t *testing.T, mode testMode) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/18657")
	}
	done := make(chan struct{})
	needle := strings.Repeat("x", 100*1024) // assume: larger than net/http bufio size
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(done)

		conn, buf, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()

		got := make([]byte, len(needle))
		n, err := io.ReadFull(buf.Reader, got)
		if n != len(needle) || string(got) != needle || err != nil {
			t.Errorf("Peek = %q, %v; want 'x'*4096, nil", got, err)
		}
	})).ts

	cn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer cn.Close()
	buf := []byte("GET / HTTP/1.1\r\nHost: e.com\r\n\r\n")
	buf = append(buf, []byte(needle)...)
	if _, err := cn.Write(buf); err != nil {
		t.Fatal(err)
	}

	if err := cn.(*net.TCPConn).CloseWrite(); err != nil {
		t.Fatal(err)
	}
	<-done
}

// Like TestServerHijackGetsBackgroundByte above but sending a
// immediate 1MB of data to the server to fill up the server's 4KB
// buffer.
func TestServerHijackGetsBackgroundByte_big(t *testing.T) {
	run(t, testServerHijackGetsBackgroundByte_big, []testMode{http1Mode})
}
func testServerHijackGetsBackgroundByte_big(t *testing.T, mode testMode) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see https://golang.org/issue/18657")
	}
	done := make(chan struct{})
	const size = 8 << 10
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(done)

		conn, buf, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		slurp, err := io.ReadAll(buf.Reader)
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
	})).ts

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

	<-done
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
		conn := newTestConn()
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
	var err error
	if n := atomic.AddInt32(&p.closes, 1); n == 1 && p.Listener != nil {
		err = p.Listener.Close()
	}
	return err
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

func TestServerContexts(t *testing.T) { run(t, testServerContexts) }
func testServerContexts(t *testing.T, mode testMode) {
	type baseKey struct{}
	type connKey struct{}
	ch := make(chan context.Context, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, r *Request) {
		ch <- r.Context()
	}), func(ts *httptest.Server) {
		ts.Config.BaseContext = func(ln net.Listener) context.Context {
			if strings.Contains(reflect.TypeOf(ln).String(), "onceClose") {
				t.Errorf("unexpected onceClose listener type %T", ln)
			}
			return context.WithValue(context.Background(), baseKey{}, "base")
		}
		ts.Config.ConnContext = func(ctx context.Context, c net.Conn) context.Context {
			if got, want := ctx.Value(baseKey{}), "base"; got != want {
				t.Errorf("in ConnContext, base context key = %#v; want %q", got, want)
			}
			return context.WithValue(ctx, connKey{}, "conn")
		}
	}).ts
	res, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	ctx := <-ch
	if got, want := ctx.Value(baseKey{}), "base"; got != want {
		t.Errorf("base context key = %#v; want %q", got, want)
	}
	if got, want := ctx.Value(connKey{}), "conn"; got != want {
		t.Errorf("conn context key = %#v; want %q", got, want)
	}
}

// Issue 35750: check ConnContext not modifying context for other connections
func TestConnContextNotModifyingAllContexts(t *testing.T) {
	run(t, testConnContextNotModifyingAllContexts)
}
func testConnContextNotModifyingAllContexts(t *testing.T, mode testMode) {
	type connKey struct{}
	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, r *Request) {
		rw.Header().Set("Connection", "close")
	}), func(ts *httptest.Server) {
		ts.Config.ConnContext = func(ctx context.Context, c net.Conn) context.Context {
			if got := ctx.Value(connKey{}); got != nil {
				t.Errorf("in ConnContext, unexpected context key = %#v", got)
			}
			return context.WithValue(ctx, connKey{}, "conn")
		}
	}).ts

	var res *Response
	var err error

	res, err = ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()

	res, err = ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

// Issue 30710: ensure that as per the spec, a server responds
// with 501 Not Implemented for unsupported transfer-encodings.
func TestUnsupportedTransferEncodingsReturn501(t *testing.T) {
	run(t, testUnsupportedTransferEncodingsReturn501, []testMode{http1Mode})
}
func testUnsupportedTransferEncodingsReturn501(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello, World!"))
	})).ts

	serverURL, err := url.Parse(cst.URL)
	if err != nil {
		t.Fatalf("Failed to parse server URL: %v", err)
	}

	unsupportedTEs := []string{
		"fugazi",
		"foo-bar",
		"unknown",
		`" chunked"`,
	}

	for _, badTE := range unsupportedTEs {
		http1ReqBody := fmt.Sprintf(""+
			"POST / HTTP/1.1\r\nConnection: close\r\n"+
			"Host: localhost\r\nTransfer-Encoding: %s\r\n\r\n", badTE)

		gotBody, err := fetchWireResponse(serverURL.Host, []byte(http1ReqBody))
		if err != nil {
			t.Errorf("%q. unexpected error: %v", badTE, err)
			continue
		}

		wantBody := fmt.Sprintf("" +
			"HTTP/1.1 501 Not Implemented\r\nContent-Type: text/plain; charset=utf-8\r\n" +
			"Connection: close\r\n\r\nUnsupported transfer encoding")

		if string(gotBody) != wantBody {
			t.Errorf("%q. body\ngot\n%q\nwant\n%q", badTE, gotBody, wantBody)
		}
	}
}

// Issue 31753: don't sniff when Content-Encoding is set
func TestContentEncodingNoSniffing(t *testing.T) { run(t, testContentEncodingNoSniffing) }
func testContentEncodingNoSniffing(t *testing.T, mode testMode) {
	type setting struct {
		name string
		body []byte

		// setting contentEncoding as an interface instead of a string
		// directly, so as to differentiate between 3 states:
		//    unset, empty string "" and set string "foo/bar".
		contentEncoding any
		wantContentType string
	}

	settings := []*setting{
		{
			name:            "gzip content-encoding, gzipped", // don't sniff.
			contentEncoding: "application/gzip",
			wantContentType: "",
			body: func() []byte {
				buf := new(bytes.Buffer)
				gzw := gzip.NewWriter(buf)
				gzw.Write([]byte("doctype html><p>Hello</p>"))
				gzw.Close()
				return buf.Bytes()
			}(),
		},
		{
			name:            "zlib content-encoding, zlibbed", // don't sniff.
			contentEncoding: "application/zlib",
			wantContentType: "",
			body: func() []byte {
				buf := new(bytes.Buffer)
				zw := zlib.NewWriter(buf)
				zw.Write([]byte("doctype html><p>Hello</p>"))
				zw.Close()
				return buf.Bytes()
			}(),
		},
		{
			name:            "no content-encoding", // must sniff.
			wantContentType: "application/x-gzip",
			body: func() []byte {
				buf := new(bytes.Buffer)
				gzw := gzip.NewWriter(buf)
				gzw.Write([]byte("doctype html><p>Hello</p>"))
				gzw.Close()
				return buf.Bytes()
			}(),
		},
		{
			name:            "phony content-encoding", // don't sniff.
			contentEncoding: "foo/bar",
			body:            []byte("doctype html><p>Hello</p>"),
		},
		{
			name:            "empty but set content-encoding",
			contentEncoding: "",
			wantContentType: "audio/mpeg",
			body:            []byte("ID3"),
		},
	}

	for _, tt := range settings {
		t.Run(tt.name, func(t *testing.T) {
			cst := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, r *Request) {
				if tt.contentEncoding != nil {
					rw.Header().Set("Content-Encoding", tt.contentEncoding.(string))
				}
				rw.Write(tt.body)
			}))

			res, err := cst.c.Get(cst.ts.URL)
			if err != nil {
				t.Fatalf("Failed to fetch URL: %v", err)
			}
			defer res.Body.Close()

			if g, w := res.Header.Get("Content-Encoding"), tt.contentEncoding; g != w {
				if w != nil { // The case where contentEncoding was set explicitly.
					t.Errorf("Content-Encoding mismatch\n\tgot:  %q\n\twant: %q", g, w)
				} else if g != "" { // "" should be the equivalent when the contentEncoding is unset.
					t.Errorf("Unexpected Content-Encoding %q", g)
				}
			}

			if g, w := res.Header.Get("Content-Type"), tt.wantContentType; g != w {
				t.Errorf("Content-Type mismatch\n\tgot:  %q\n\twant: %q", g, w)
			}
		})
	}
}

// Issue 30803: ensure that TimeoutHandler logs spurious
// WriteHeader calls, for consistency with other Handlers.
func TestTimeoutHandlerSuperfluousLogs(t *testing.T) {
	run(t, testTimeoutHandlerSuperfluousLogs, []testMode{http1Mode})
}
func testTimeoutHandlerSuperfluousLogs(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	pc, curFile, _, _ := runtime.Caller(0)
	curFileBaseName := filepath.Base(curFile)
	testFuncName := runtime.FuncForPC(pc).Name()

	timeoutMsg := "timed out here!"

	tests := []struct {
		name        string
		mustTimeout bool
		wantResp    string
	}{
		{
			name:     "return before timeout",
			wantResp: "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
		},
		{
			name:        "return after timeout",
			mustTimeout: true,
			wantResp: fmt.Sprintf("HTTP/1.1 503 Service Unavailable\r\nContent-Length: %d\r\n\r\n%s",
				len(timeoutMsg), timeoutMsg),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			exitHandler := make(chan bool, 1)
			defer close(exitHandler)
			lastLine := make(chan int, 1)

			sh := HandlerFunc(func(w ResponseWriter, r *Request) {
				w.WriteHeader(404)
				w.WriteHeader(404)
				w.WriteHeader(404)
				w.WriteHeader(404)
				_, _, line, _ := runtime.Caller(0)
				lastLine <- line
				<-exitHandler
			})

			if !tt.mustTimeout {
				exitHandler <- true
			}

			logBuf := new(strings.Builder)
			srvLog := log.New(logBuf, "", 0)
			// When expecting to timeout, we'll keep the duration short.
			dur := 20 * time.Millisecond
			if !tt.mustTimeout {
				// Otherwise, make it arbitrarily long to reduce the risk of flakes.
				dur = 10 * time.Second
			}
			th := TimeoutHandler(sh, dur, timeoutMsg)
			cst := newClientServerTest(t, mode, th, optWithServerLog(srvLog))
			defer cst.close()

			res, err := cst.c.Get(cst.ts.URL)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			// Deliberately removing the "Date" header since it is highly ephemeral
			// and will cause failure if we try to match it exactly.
			res.Header.Del("Date")
			res.Header.Del("Content-Type")

			// Match the response.
			blob, _ := httputil.DumpResponse(res, true)
			if g, w := string(blob), tt.wantResp; g != w {
				t.Errorf("Response mismatch\nGot\n%q\n\nWant\n%q", g, w)
			}

			// Given 4 w.WriteHeader calls, only the first one is valid
			// and the rest should be reported as the 3 spurious logs.
			logEntries := strings.Split(strings.TrimSpace(logBuf.String()), "\n")
			if g, w := len(logEntries), 3; g != w {
				blob, _ := json.MarshalIndent(logEntries, "", "  ")
				t.Fatalf("Server logs count mismatch\ngot %d, want %d\n\nGot\n%s\n", g, w, blob)
			}

			lastSpuriousLine := <-lastLine
			firstSpuriousLine := lastSpuriousLine - 3
			// Now ensure that the regexes match exactly.
			//      "http: superfluous response.WriteHeader call from <fn>.func\d.\d (<curFile>:lastSpuriousLine-[1, 3]"
			for i, logEntry := range logEntries {
				wantLine := firstSpuriousLine + i
				pat := fmt.Sprintf("^http: superfluous response.WriteHeader call from %s.func\\d+.\\d+ \\(%s:%d\\)$",
					testFuncName, curFileBaseName, wantLine)
				re := regexp.MustCompile(pat)
				if !re.MatchString(logEntry) {
					t.Errorf("Log entry mismatch\n\t%s\ndoes not match\n\t%s", logEntry, pat)
				}
			}
		})
	}
}

// fetchWireResponse is a helper for dialing to host,
// sending http1ReqBody as the payload and retrieving
// the response as it was sent on the wire.
func fetchWireResponse(host string, http1ReqBody []byte) ([]byte, error) {
	conn, err := net.Dial("tcp", host)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	if _, err := conn.Write(http1ReqBody); err != nil {
		return nil, err
	}
	return io.ReadAll(conn)
}

func BenchmarkResponseStatusLine(b *testing.B) {
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		bw := bufio.NewWriter(io.Discard)
		var buf3 [3]byte
		for pb.Next() {
			Export_writeStatusLine(bw, true, 200, buf3[:])
		}
	})
}

func TestDisableKeepAliveUpgrade(t *testing.T) {
	run(t, testDisableKeepAliveUpgrade, []testMode{http1Mode})
}
func testDisableKeepAliveUpgrade(t *testing.T, mode testMode) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	s := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Connection", "Upgrade")
		w.Header().Set("Upgrade", "someProto")
		w.WriteHeader(StatusSwitchingProtocols)
		c, buf, err := w.(Hijacker).Hijack()
		if err != nil {
			return
		}
		defer c.Close()

		// Copy from the *bufio.ReadWriter, which may contain buffered data.
		// Copy to the net.Conn, to avoid buffering the output.
		io.Copy(c, buf)
	}), func(ts *httptest.Server) {
		ts.Config.SetKeepAlivesEnabled(false)
	}).ts

	cl := s.Client()
	cl.Transport.(*Transport).DisableKeepAlives = true

	resp, err := cl.Get(s.URL)
	if err != nil {
		t.Fatalf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != StatusSwitchingProtocols {
		t.Fatalf("unexpected status code: %v", resp.StatusCode)
	}

	rwc, ok := resp.Body.(io.ReadWriteCloser)
	if !ok {
		t.Fatalf("Response.Body is not an io.ReadWriteCloser: %T", resp.Body)
	}

	_, err = rwc.Write([]byte("hello"))
	if err != nil {
		t.Fatalf("failed to write to body: %v", err)
	}

	b := make([]byte, 5)
	_, err = io.ReadFull(rwc, b)
	if err != nil {
		t.Fatalf("failed to read from body: %v", err)
	}

	if string(b) != "hello" {
		t.Fatalf("unexpected value read from body:\ngot: %q\nwant: %q", b, "hello")
	}
}

type tlogWriter struct{ t *testing.T }

func (w tlogWriter) Write(p []byte) (int, error) {
	w.t.Log(string(p))
	return len(p), nil
}

func TestWriteHeaderSwitchingProtocols(t *testing.T) {
	run(t, testWriteHeaderSwitchingProtocols, []testMode{http1Mode})
}
func testWriteHeaderSwitchingProtocols(t *testing.T, mode testMode) {
	const wantBody = "want"
	const wantUpgrade = "someProto"
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Connection", "Upgrade")
		w.Header().Set("Upgrade", wantUpgrade)
		w.WriteHeader(StatusSwitchingProtocols)
		NewResponseController(w).Flush()

		// Writing headers or the body after sending a 101 header should fail.
		w.WriteHeader(200)
		if _, err := w.Write([]byte("x")); err == nil {
			t.Errorf("Write to body after 101 Switching Protocols unexpectedly succeeded")
		}

		c, _, err := NewResponseController(w).Hijack()
		if err != nil {
			t.Errorf("Hijack: %v", err)
			return
		}
		defer c.Close()
		if _, err := c.Write([]byte(wantBody)); err != nil {
			t.Errorf("Write to hijacked body: %v", err)
		}
	}), func(ts *httptest.Server) {
		// Don't spam log with warning about superfluous WriteHeader call.
		ts.Config.ErrorLog = log.New(tlogWriter{t}, "log: ", 0)
	}).ts

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatalf("net.Dial: %v", err)
	}
	_, err = conn.Write([]byte("GET / HTTP/1.1\r\nHost: foo\r\n\r\n"))
	if err != nil {
		t.Fatalf("conn.Write: %v", err)
	}
	defer conn.Close()

	r := bufio.NewReader(conn)
	res, err := ReadResponse(r, &Request{Method: "GET"})
	if err != nil {
		t.Fatal("ReadResponse error:", err)
	}
	if res.StatusCode != StatusSwitchingProtocols {
		t.Errorf("Response StatusCode=%v, want 101", res.StatusCode)
	}
	if got := res.Header.Get("Upgrade"); got != wantUpgrade {
		t.Errorf("Response Upgrade header = %q, want %q", got, wantUpgrade)
	}
	body, err := io.ReadAll(r)
	if err != nil {
		t.Error(err)
	}
	if string(body) != wantBody {
		t.Errorf("Response body = %q, want %q", string(body), wantBody)
	}
}

func TestMuxRedirectRelative(t *testing.T) {
	setParallel(t)
	req, err := ReadRequest(bufio.NewReader(strings.NewReader("GET http://example.com HTTP/1.1\r\nHost: test\r\n\r\n")))
	if err != nil {
		t.Errorf("%s", err)
	}
	mux := NewServeMux()
	resp := httptest.NewRecorder()
	mux.ServeHTTP(resp, req)
	if got, want := resp.Header().Get("Location"), "/"; got != want {
		t.Errorf("Location header expected %q; got %q", want, got)
	}
	if got, want := resp.Code, StatusMovedPermanently; got != want {
		t.Errorf("Expected response code %d; got %d", want, got)
	}
}

// TestQuerySemicolon tests the behavior of semicolons in queries. See Issue 25192.
func TestQuerySemicolon(t *testing.T) {
	t.Cleanup(func() { afterTest(t) })

	tests := []struct {
		query              string
		xNoSemicolons      string
		xWithSemicolons    string
		expectParseFormErr bool
	}{
		{"?a=1;x=bad&x=good", "good", "bad", true},
		{"?a=1;b=bad&x=good", "good", "good", true},
		{"?a=1%3Bx=bad&x=good%3B", "good;", "good;", false},
		{"?a=1;x=good;x=bad", "", "good", true},
	}

	run(t, func(t *testing.T, mode testMode) {
		for _, tt := range tests {
			t.Run(tt.query+"/allow=false", func(t *testing.T) {
				allowSemicolons := false
				testQuerySemicolon(t, mode, tt.query, tt.xNoSemicolons, allowSemicolons, tt.expectParseFormErr)
			})
			t.Run(tt.query+"/allow=true", func(t *testing.T) {
				allowSemicolons, expectParseFormErr := true, false
				testQuerySemicolon(t, mode, tt.query, tt.xWithSemicolons, allowSemicolons, expectParseFormErr)
			})
		}
	})
}

func testQuerySemicolon(t *testing.T, mode testMode, query string, wantX string, allowSemicolons, expectParseFormErr bool) {
	writeBackX := func(w ResponseWriter, r *Request) {
		x := r.URL.Query().Get("x")
		if expectParseFormErr {
			if err := r.ParseForm(); err == nil || !strings.Contains(err.Error(), "semicolon") {
				t.Errorf("expected error mentioning semicolons from ParseForm, got %v", err)
			}
		} else {
			if err := r.ParseForm(); err != nil {
				t.Errorf("expected no error from ParseForm, got %v", err)
			}
		}
		if got := r.FormValue("x"); x != got {
			t.Errorf("got %q from FormValue, want %q", got, x)
		}
		fmt.Fprintf(w, "%s", x)
	}

	h := Handler(HandlerFunc(writeBackX))
	if allowSemicolons {
		h = AllowQuerySemicolons(h)
	}

	logBuf := &strings.Builder{}
	ts := newClientServerTest(t, mode, h, func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(logBuf, "", 0)
	}).ts

	req, _ := NewRequest("GET", ts.URL+query, nil)
	res, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	slurp, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if got, want := res.StatusCode, 200; got != want {
		t.Errorf("Status = %d; want = %d", got, want)
	}
	if got, want := string(slurp), wantX; got != want {
		t.Errorf("Body = %q; want = %q", got, want)
	}
}

func TestMaxBytesHandler(t *testing.T) {
	// Not parallel: modifies the global rstAvoidanceDelay.
	defer afterTest(t)

	for _, maxSize := range []int64{100, 1_000, 1_000_000} {
		for _, requestSize := range []int64{100, 1_000, 1_000_000} {
			t.Run(fmt.Sprintf("max size %d request size %d", maxSize, requestSize),
				func(t *testing.T) {
					run(t, func(t *testing.T, mode testMode) {
						testMaxBytesHandler(t, mode, maxSize, requestSize)
					}, testNotParallel)
				})
		}
	}
}

func testMaxBytesHandler(t *testing.T, mode testMode, maxSize, requestSize int64) {
	runTimeSensitiveTest(t, []time.Duration{
		1 * time.Millisecond,
		5 * time.Millisecond,
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		500 * time.Millisecond,
		time.Second,
		5 * time.Second,
	}, func(t *testing.T, timeout time.Duration) error {
		SetRSTAvoidanceDelay(t, timeout)
		t.Logf("set RST avoidance delay to %v", timeout)

		var (
			handlerN   int64
			handlerErr error
		)
		echo := HandlerFunc(func(w ResponseWriter, r *Request) {
			var buf bytes.Buffer
			handlerN, handlerErr = io.Copy(&buf, r.Body)
			io.Copy(w, &buf)
		})

		cst := newClientServerTest(t, mode, MaxBytesHandler(echo, maxSize))
		// We need to close cst explicitly here so that in-flight server
		// requests don't race with the call to SetRSTAvoidanceDelay for a retry.
		defer cst.close()
		ts := cst.ts
		c := ts.Client()

		body := strings.Repeat("a", int(requestSize))
		var wg sync.WaitGroup
		defer wg.Wait()
		getBody := func() (io.ReadCloser, error) {
			wg.Add(1)
			body := &wgReadCloser{
				Reader: strings.NewReader(body),
				wg:     &wg,
			}
			return body, nil
		}
		reqBody, _ := getBody()
		req, err := NewRequest("POST", ts.URL, reqBody)
		if err != nil {
			reqBody.Close()
			t.Fatal(err)
		}
		req.ContentLength = int64(len(body))
		req.GetBody = getBody
		req.Header.Set("Content-Type", "text/plain")

		var buf strings.Builder
		res, err := c.Do(req)
		if err != nil {
			return fmt.Errorf("unexpected connection error: %v", err)
		} else {
			_, err = io.Copy(&buf, res.Body)
			res.Body.Close()
			if err != nil {
				return fmt.Errorf("unexpected read error: %v", err)
			}
		}
		// We don't expect any of the errors after this point to occur due
		// to rstAvoidanceDelay being too short, so we use t.Errorf for those
		// instead of returning a (retriable) error.

		if handlerN > maxSize {
			t.Errorf("expected max request body %d; got %d", maxSize, handlerN)
		}
		if requestSize > maxSize && handlerErr == nil {
			t.Error("expected error on handler side; got nil")
		}
		if requestSize <= maxSize {
			if handlerErr != nil {
				t.Errorf("%d expected nil error on handler side; got %v", requestSize, handlerErr)
			}
			if handlerN != requestSize {
				t.Errorf("expected request of size %d; got %d", requestSize, handlerN)
			}
		}
		if buf.Len() != int(handlerN) {
			t.Errorf("expected echo of size %d; got %d", handlerN, buf.Len())
		}

		return nil
	})
}

func TestEarlyHints(t *testing.T) {
	ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
		h := w.Header()
		h.Add("Link", "</style.css>; rel=preload; as=style")
		h.Add("Link", "</script.js>; rel=preload; as=script")
		w.WriteHeader(StatusEarlyHints)

		h.Add("Link", "</foo.js>; rel=preload; as=script")
		w.WriteHeader(StatusEarlyHints)

		w.Write([]byte("stuff"))
	}))

	got := ht.rawResponse("GET / HTTP/1.1\nHost: golang.org")
	expected := "HTTP/1.1 103 Early Hints\r\nLink: </style.css>; rel=preload; as=style\r\nLink: </script.js>; rel=preload; as=script\r\n\r\nHTTP/1.1 103 Early Hints\r\nLink: </style.css>; rel=preload; as=style\r\nLink: </script.js>; rel=preload; as=script\r\nLink: </foo.js>; rel=preload; as=script\r\n\r\nHTTP/1.1 200 OK\r\nLink: </style.css>; rel=preload; as=style\r\nLink: </script.js>; rel=preload; as=script\r\nLink: </foo.js>; rel=preload; as=script\r\nDate: " // dynamic content expected
	if !strings.Contains(got, expected) {
		t.Errorf("unexpected response; got %q; should start by %q", got, expected)
	}
}
func TestProcessing(t *testing.T) {
	ht := newHandlerTest(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusProcessing)
		w.Write([]byte("stuff"))
	}))

	got := ht.rawResponse("GET / HTTP/1.1\nHost: golang.org")
	expected := "HTTP/1.1 102 Processing\r\n\r\nHTTP/1.1 200 OK\r\nDate: " // dynamic content expected
	if !strings.Contains(got, expected) {
		t.Errorf("unexpected response; got %q; should start by %q", got, expected)
	}
}

func TestParseFormCleanup(t *testing.T) { run(t, testParseFormCleanup) }
func testParseFormCleanup(t *testing.T, mode testMode) {
	if mode == http2Mode {
		t.Skip("https://go.dev/issue/20253")
	}

	const maxMemory = 1024
	const key = "file"

	if runtime.GOOS == "windows" {
		// Windows sometimes refuses to remove a file that was just closed.
		t.Skip("https://go.dev/issue/25965")
	}

	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		r.ParseMultipartForm(maxMemory)
		f, _, err := r.FormFile(key)
		if err != nil {
			t.Errorf("r.FormFile(%q) = %v", key, err)
			return
		}
		of, ok := f.(*os.File)
		if !ok {
			t.Errorf("r.FormFile(%q) returned type %T, want *os.File", key, f)
			return
		}
		w.Write([]byte(of.Name()))
	}))

	fBuf := new(bytes.Buffer)
	mw := multipart.NewWriter(fBuf)
	mf, err := mw.CreateFormFile(key, "myfile.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mf.Write(bytes.Repeat([]byte("A"), maxMemory*2)); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	req, err := NewRequest("POST", cst.ts.URL, fBuf)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	fname, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	cst.close()
	if _, err := os.Stat(string(fname)); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("file %q exists after HTTP handler returned", string(fname))
	}
}

func TestHeadBody(t *testing.T) {
	const identityMode = false
	const chunkedMode = true
	run(t, func(t *testing.T, mode testMode) {
		t.Run("identity", func(t *testing.T) { testHeadBody(t, mode, identityMode, "HEAD") })
		t.Run("chunked", func(t *testing.T) { testHeadBody(t, mode, chunkedMode, "HEAD") })
	})
}

func TestGetBody(t *testing.T) {
	const identityMode = false
	const chunkedMode = true
	run(t, func(t *testing.T, mode testMode) {
		t.Run("identity", func(t *testing.T) { testHeadBody(t, mode, identityMode, "GET") })
		t.Run("chunked", func(t *testing.T) { testHeadBody(t, mode, chunkedMode, "GET") })
	})
}

func testHeadBody(t *testing.T, mode testMode, chunked bool, method string) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("server reading body: %v", err)
			return
		}
		w.Header().Set("X-Request-Body", string(b))
		w.Header().Set("Content-Length", "0")
	}))
	defer cst.close()
	for _, reqBody := range []string{
		"",
		"",
		"request_body",
		"",
	} {
		var bodyReader io.Reader
		if reqBody != "" {
			bodyReader = strings.NewReader(reqBody)
			if chunked {
				bodyReader = bufio.NewReader(bodyReader)
			}
		}
		req, err := NewRequest(method, cst.ts.URL, bodyReader)
		if err != nil {
			t.Fatal(err)
		}
		res, err := cst.c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if got, want := res.StatusCode, 200; got != want {
			t.Errorf("%v request with %d-byte body: StatusCode = %v, want %v", method, len(reqBody), got, want)
		}
		if got, want := res.Header.Get("X-Request-Body"), reqBody; got != want {
			t.Errorf("%v request with %d-byte body: handler read body %q, want %q", method, len(reqBody), got, want)
		}
	}
}

// TestDisableContentLength verifies that the Content-Length is set by default
// or disabled when the header is set to nil.
func TestDisableContentLength(t *testing.T) { run(t, testDisableContentLength) }
func testDisableContentLength(t *testing.T, mode testMode) {
	noCL := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header()["Content-Length"] = nil // disable the default Content-Length response
		fmt.Fprintf(w, "OK")
	}))

	res, err := noCL.c.Get(noCL.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if got, haveCL := res.Header["Content-Length"]; haveCL {
		t.Errorf("Unexpected Content-Length: %q", got)
	}
	if err := res.Body.Close(); err != nil {
		t.Fatal(err)
	}

	withCL := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "OK")
	}))

	res, err = withCL.c.Get(withCL.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.Header.Get("Content-Length"); got != "2" {
		t.Errorf("Content-Length: %q; want 2", got)
	}
	if err := res.Body.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestErrorContentLength(t *testing.T) { run(t, testErrorContentLength) }
func testErrorContentLength(t *testing.T, mode testMode) {
	const errorBody = "an error occurred"
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "1000")
		Error(w, errorBody, 400)
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("Get(%q) = %v", cst.ts.URL, err)
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("io.ReadAll(res.Body) = %v", err)
	}
	if string(body) != errorBody+"\n" {
		t.Fatalf("read body: %q, want %q", string(body), errorBody)
	}
}

func TestError(t *testing.T) {
	w := httptest.NewRecorder()
	w.Header().Set("Content-Length", "1")
	w.Header().Set("X-Content-Type-Options", "scratch and sniff")
	w.Header().Set("Other", "foo")
	Error(w, "oops", 432)

	h := w.Header()
	for _, hdr := range []string{"Content-Length"} {
		if v, ok := h[hdr]; ok {
			t.Errorf("%s: %q, want not present", hdr, v)
		}
	}
	if v := h.Get("Content-Type"); v != "text/plain; charset=utf-8" {
		t.Errorf("Content-Type: %q, want %q", v, "text/plain; charset=utf-8")
	}
	if v := h.Get("X-Content-Type-Options"); v != "nosniff" {
		t.Errorf("X-Content-Type-Options: %q, want %q", v, "nosniff")
	}
}

func TestServerReadAfterWriteHeader100Continue(t *testing.T) {
	run(t, testServerReadAfterWriteHeader100Continue)
}
func testServerReadAfterWriteHeader100Continue(t *testing.T, mode testMode) {
	t.Skip("https://go.dev/issue/67555")
	body := []byte("body")
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(200)
		NewResponseController(w).Flush()
		io.ReadAll(r.Body)
		w.Write(body)
	}), func(tr *Transport) {
		tr.ExpectContinueTimeout = 24 * time.Hour // forever
	})

	req, _ := NewRequest("GET", cst.ts.URL, strings.NewReader("body"))
	req.Header.Set("Expect", "100-continue")
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatalf("Get(%q) = %v", cst.ts.URL, err)
	}
	defer res.Body.Close()
	got, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("io.ReadAll(res.Body) = %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("response body = %q, want %q", got, body)
	}
}

func TestServerReadAfterHandlerDone100Continue(t *testing.T) {
	run(t, testServerReadAfterHandlerDone100Continue)
}
func testServerReadAfterHandlerDone100Continue(t *testing.T, mode testMode) {
	t.Skip("https://go.dev/issue/67555")
	readyc := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		go func() {
			<-readyc
			io.ReadAll(r.Body)
			<-readyc
		}()
	}), func(tr *Transport) {
		tr.ExpectContinueTimeout = 24 * time.Hour // forever
	})

	req, _ := NewRequest("GET", cst.ts.URL, strings.NewReader("body"))
	req.Header.Set("Expect", "100-continue")
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatalf("Get(%q) = %v", cst.ts.URL, err)
	}
	res.Body.Close()
	readyc <- struct{}{} // server starts reading from the request body
	readyc <- struct{}{} // server finishes reading from the request body
}

func TestServerReadAfterHandlerAbort100Continue(t *testing.T) {
	run(t, testServerReadAfterHandlerAbort100Continue)
}
func testServerReadAfterHandlerAbort100Continue(t *testing.T, mode testMode) {
	t.Skip("https://go.dev/issue/67555")
	readyc := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		go func() {
			<-readyc
			io.ReadAll(r.Body)
			<-readyc
		}()
		panic(ErrAbortHandler)
	}), func(tr *Transport) {
		tr.ExpectContinueTimeout = 24 * time.Hour // forever
	})

	req, _ := NewRequest("GET", cst.ts.URL, strings.NewReader("body"))
	req.Header.Set("Expect", "100-continue")
	res, err := cst.c.Do(req)
	if err == nil {
		res.Body.Close()
	}
	readyc <- struct{}{} // server starts reading from the request body
	readyc <- struct{}{} // server finishes reading from the request body
}

func TestInvalidChunkedBodies(t *testing.T) {
	for _, test := range []struct {
		name string
		b    string
	}{{
		name: "bare LF in chunk size",
		b:    "1\na\r\n0\r\n\r\n",
	}, {
		name: "bare LF at body end",
		b:    "1\r\na\r\n0\r\n\n",
	}} {
		t.Run(test.name, func(t *testing.T) {
			reqc := make(chan error)
			ts := newClientServerTest(t, http1Mode, HandlerFunc(func(w ResponseWriter, r *Request) {
				got, err := io.ReadAll(r.Body)
				if err == nil {
					t.Logf("read body: %q", got)
				}
				reqc <- err
			})).ts

			serverURL, err := url.Parse(ts.URL)
			if err != nil {
				t.Fatal(err)
			}

			conn, err := net.Dial("tcp", serverURL.Host)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := conn.Write([]byte(
				"POST / HTTP/1.1\r\n" +
					"Host: localhost\r\n" +
					"Transfer-Encoding: chunked\r\n" +
					"Connection: close\r\n" +
					"\r\n" +
					test.b)); err != nil {
				t.Fatal(err)
			}
			conn.(*net.TCPConn).CloseWrite()

			if err := <-reqc; err == nil {
				t.Errorf("server handler: io.ReadAll(r.Body) succeeded, want error")
			}
		})
	}
}

// Issue #72100: Verify that we don't modify the caller's TLS.Config.NextProtos slice.
func TestServerTLSNextProtos(t *testing.T) {
	run(t, testServerTLSNextProtos, []testMode{https1Mode, http2Mode})
}
func testServerTLSNextProtos(t *testing.T, mode testMode) {
	CondSkipHTTP2(t)

	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	leafCert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	certpool := x509.NewCertPool()
	certpool.AddCert(leafCert)

	protos := new(Protocols)
	switch mode {
	case https1Mode:
		protos.SetHTTP1(true)
	case http2Mode:
		protos.SetHTTP2(true)
	}

	wantNextProtos := []string{"http/1.1", "h2", "other"}
	nextProtos := slices.Clone(wantNextProtos)

	// We don't use httptest here because it overrides the tls.Config.
	srv := &Server{
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{cert},
			NextProtos:   nextProtos,
		},
		Handler:   HandlerFunc(func(w ResponseWriter, req *Request) {}),
		Protocols: protos,
	}
	tr := &Transport{
		TLSClientConfig: &tls.Config{
			RootCAs:    certpool,
			NextProtos: nextProtos,
		},
		Protocols: protos,
	}

	listener := newLocalListener(t)
	srvc := make(chan error, 1)
	go func() {
		srvc <- srv.ServeTLS(listener, "", "")
	}()
	t.Cleanup(func() {
		srv.Close()
		<-srvc
	})

	client := &Client{Transport: tr}
	resp, err := client.Get("https://" + listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	if !slices.Equal(nextProtos, wantNextProtos) {
		t.Fatalf("after running test: original NextProtos slice = %v, want %v", nextProtos, wantNextProtos)
	}
}
