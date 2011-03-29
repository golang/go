// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// End-to-end serving tests

package http_test

import (
	"bufio"
	"bytes"
	"fmt"
	. "http"
	"http/httptest"
	"io/ioutil"
	"os"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"
)

type dummyAddr string
type oneConnListener struct {
	conn net.Conn
}

func (l *oneConnListener) Accept() (c net.Conn, err os.Error) {
	c = l.conn
	if c == nil {
		err = os.EOF
		return
	}
	err = nil
	l.conn = nil
	return
}

func (l *oneConnListener) Close() os.Error {
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

type testConn struct {
	readBuf  bytes.Buffer
	writeBuf bytes.Buffer
}

func (c *testConn) Read(b []byte) (int, os.Error) {
	return c.readBuf.Read(b)
}

func (c *testConn) Write(b []byte) (int, os.Error) {
	return c.writeBuf.Write(b)
}

func (c *testConn) Close() os.Error {
	return nil
}

func (c *testConn) LocalAddr() net.Addr {
	return dummyAddr("local-addr")
}

func (c *testConn) RemoteAddr() net.Addr {
	return dummyAddr("remote-addr")
}

func (c *testConn) SetTimeout(nsec int64) os.Error {
	return nil
}

func (c *testConn) SetReadTimeout(nsec int64) os.Error {
	return nil
}

func (c *testConn) SetWriteTimeout(nsec int64) os.Error {
	return nil
}

func TestConsumingBodyOnNextConn(t *testing.T) {
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
	servech := make(chan os.Error)
	listener := &oneConnListener{conn}
	handler := func(res ResponseWriter, req *Request) {
		reqNum++
		t.Logf("Got request #%d: %v", reqNum, req)
		ch <- req
	}

	go func() {
		servech <- Serve(listener, HandlerFunc(handler))
	}()

	var req *Request
	t.Log("Waiting for first request.")
	req = <-ch
	if req == nil {
		t.Fatal("Got nil first request.")
	}
	if req.Method != "POST" {
		t.Errorf("For request #1's method, got %q; expected %q",
			req.Method, "POST")
	}

	t.Log("Waiting for second request.")
	req = <-ch
	if req == nil {
		t.Fatal("Got nil first request.")
	}
	if req.Method != "POST" {
		t.Errorf("For request #2's method, got %q; expected %q",
			req.Method, "POST")
	}

	t.Log("Waiting for EOF.")
	if serveerr := <-servech; serveerr != os.EOF {
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
	{"someHost.com/someDir/", "someHost.com/someDir"},
}

var vtests = []struct {
	url      string
	expected string
}{
	{"http://localhost/someDir/apage", "someDir"},
	{"http://localhost/otherDir/apage", "Default"},
	{"http://someHost.com/someDir/apage", "someHost.com/someDir"},
	{"http://otherHost.com/someDir/apage", "someDir"},
	{"http://otherHost.com/aDir/apage", "Default"},
}

func TestHostHandlers(t *testing.T) {
	for _, h := range handlers {
		Handle(h.pattern, stringHandler(h.msg))
	}
	ts := httptest.NewServer(nil)
	defer ts.Close()

	conn, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	cc := NewClientConn(conn, nil)
	for _, vt := range vtests {
		var r *Response
		var req Request
		if req.URL, err = ParseURL(vt.url); err != nil {
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
		s := r.Header.Get("Result")
		if s != vt.expected {
			t.Errorf("Get(%q) = %q, want %q", vt.url, s, vt.expected)
		}
	}
}

// Tests for http://code.google.com/p/go/issues/detail?id=900
func TestMuxRedirectLeadingSlashes(t *testing.T) {
	paths := []string{"//foo.txt", "///foo.txt", "/../../foo.txt"}
	for _, path := range paths {
		req, err := ReadRequest(bufio.NewReader(bytes.NewBufferString("GET " + path + " HTTP/1.1\r\nHost: test\r\n\r\n")))
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

func TestServerTimeouts(t *testing.T) {
	// TODO(bradfitz): convert this to use httptest.Server
	l, err := net.ListenTCP("tcp", &net.TCPAddr{Port: 0})
	if err != nil {
		t.Fatalf("listen error: %v", err)
	}
	addr, _ := l.Addr().(*net.TCPAddr)

	reqNum := 0
	handler := HandlerFunc(func(res ResponseWriter, req *Request) {
		reqNum++
		fmt.Fprintf(res, "req=%d", reqNum)
	})

	const second = 1000000000 /* nanos */
	server := &Server{Handler: handler, ReadTimeout: 0.25 * second, WriteTimeout: 0.25 * second}
	go server.Serve(l)

	url := fmt.Sprintf("http://localhost:%d/", addr.Port)

	// Hit the HTTP server successfully.
	tr := &Transport{DisableKeepAlives: true} // they interfere with this test
	c := &Client{Transport: tr}
	r, _, err := c.Get(url)
	if err != nil {
		t.Fatalf("http Get #1: %v", err)
	}
	got, _ := ioutil.ReadAll(r.Body)
	expected := "req=1"
	if string(got) != expected {
		t.Errorf("Unexpected response for request #1; got %q; expected %q",
			string(got), expected)
	}

	// Slow client that should timeout.
	t1 := time.Nanoseconds()
	conn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", addr.Port))
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	buf := make([]byte, 1)
	n, err := conn.Read(buf)
	latency := time.Nanoseconds() - t1
	if n != 0 || err != os.EOF {
		t.Errorf("Read = %v, %v, wanted %v, %v", n, err, 0, os.EOF)
	}
	if latency < second*0.20 /* fudge from 0.25 above */ {
		t.Errorf("got EOF after %d ns, want >= %d", latency, second*0.20)
	}

	// Hit the HTTP server successfully again, verifying that the
	// previous slow connection didn't run our handler.  (that we
	// get "req=2", not "req=3")
	r, _, err = Get(url)
	if err != nil {
		t.Fatalf("http Get #2: %v", err)
	}
	got, _ = ioutil.ReadAll(r.Body)
	expected = "req=2"
	if string(got) != expected {
		t.Errorf("Get #2 got %q, want %q", string(got), expected)
	}

	l.Close()
}

// TestIdentityResponse verifies that a handler can unset 
func TestIdentityResponse(t *testing.T) {
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

	// Note: this relies on the assumption (which is true) that
	// Get sends HTTP/1.1 or greater requests.  Otherwise the
	// server wouldn't have the choice to send back chunked
	// responses.
	for _, te := range []string{"", "identity"} {
		url := ts.URL + "/?te=" + te
		res, _, err := Get(url)
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
	_, _, err := Get(url)
	if err != nil {
		t.Fatalf("error with Get of %s: %v", url, err)
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
	// The next ReadAll will hang for a failing test, so use a Timer instead
	// to fail more traditionally
	timer := time.AfterFunc(2e9, func() {
		t.Fatalf("Timeout expired in ReadAll.")
	})
	defer timer.Stop()
	got, _ := ioutil.ReadAll(conn)
	expectedSuffix := "\r\n\r\ntoo short"
	if !strings.HasSuffix(string(got), expectedSuffix) {
		t.Fatalf("Expected output to end with %q; got response body %q",
			expectedSuffix, string(got))
	}
}

// TestServeHTTP10Close verifies that HTTP/1.0 requests won't be kept alive.
func TestServeHTTP10Close(t *testing.T) {
	s := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/file")
	}))
	defer s.Close()

	conn, err := net.Dial("tcp", s.Listener.Addr().String())
	if err != nil {
		t.Fatal("dial error:", err)
	}
	defer conn.Close()

	_, err = fmt.Fprint(conn, "GET / HTTP/1.0\r\n\r\n")
	if err != nil {
		t.Fatal("print error:", err)
	}

	r := bufio.NewReader(conn)
	_, err = ReadResponse(r, "GET")
	if err != nil {
		t.Fatal("ReadResponse error:", err)
	}

	success := make(chan bool)
	go func() {
		select {
		case <-time.After(5e9):
			t.Fatal("body not closed after 5s")
		case <-success:
		}
	}()

	_, err = ioutil.ReadAll(r)
	if err != nil {
		t.Fatal("read error:", err)
	}

	success <- true
}

func TestSetsRemoteAddr(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%s", r.RemoteAddr)
	}))
	defer ts.Close()

	res, _, err := Get(ts.URL)
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

func TestChunkedResponseHeaders(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "intentional gibberish") // we check that this is deleted
		fmt.Fprintf(w, "I am a chunked response.")
	}))
	defer ts.Close()

	res, _, err := Get(ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	if g, e := res.ContentLength, int64(-1); g != e {
		t.Errorf("expected ContentLength of %d; got %d", e, g)
	}
	if g, e := res.TransferEncoding, []string{"chunked"}; !reflect.DeepEqual(g, e) {
		t.Errorf("expected TransferEncoding of %v; got %v", e, g)
	}
	if _, haveCL := res.Header["Content-Length"]; haveCL {
		t.Errorf("Unexpected Content-Length")
	}
}

// Test304Responses verifies that 304s don't declare that they're
// chunking in their response headers and aren't allowed to produce
// output.
func Test304Responses(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusNotModified)
		_, err := w.Write([]byte("illegal body"))
		if err != ErrBodyNotAllowed {
			t.Errorf("on Write, expected ErrBodyNotAllowed, got %v", err)
		}
	}))
	defer ts.Close()
	res, _, err := Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if len(res.TransferEncoding) > 0 {
		t.Errorf("expected no TransferEncoding; got %v", res.TransferEncoding)
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Error(err)
	}
	if len(body) > 0 {
		t.Errorf("got unexpected body %q", string(body))
	}
}

// TestHeadResponses verifies that responses to HEAD requests don't
// declare that they're chunking in their response headers and aren't
// allowed to produce output.
func TestHeadResponses(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := w.Write([]byte("Ignored body"))
		if err != ErrBodyNotAllowed {
			t.Errorf("on Write, expected ErrBodyNotAllowed, got %v", err)
		}
	}))
	defer ts.Close()
	res, err := Head(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if len(res.TransferEncoding) > 0 {
		t.Errorf("expected no TransferEncoding; got %v", res.TransferEncoding)
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Error(err)
	}
	if len(body) > 0 {
		t.Errorf("got unexpected body %q", string(body))
	}
}
