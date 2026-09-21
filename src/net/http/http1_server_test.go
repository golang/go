// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"bytes"
	"errors"
	"internal/nettest"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
)

func TestHTTP1ServerInvalidTrailers(t *testing.T) {
	for _, test := range []struct {
		name    string
		request string
	}{{
		name: "invalid trailer",
		request: joinCRLF(
			"POST / HTTP/1.1",
			"Host: example.tld",
			"Trailer: Park",
			"Transfer-Encoding: chunked",
			"",
			"3",
			"xxx",
			"0",
			"I'm not a valid trailer",
			"GET /smuggled HTTP/1.1",
			"Host: example.tld",
			"Content-Length: 0",
			"",
		),
	}, {
		name: "trailer section ends with bare LF",
		request: joinCRLF(
			"POST / HTTP/1.1",
			"Host: example.tld",
			"Transfer-Encoding: chunked",
			"",
			"3",
			"xxx",
			"0",
			"\nGET /smuggled HTTP/1.1",
			"Host: example.tld",
			"Content-Length: 0",
			"",
		),
	}, {
		name: "trailer line ends with bare LF",
		request: joinCRLF(
			"POST / HTTP/1.1",
			"Host: example.tld",
			"Transfer-Encoding: chunked",
			"",
			"3",
			"xxx",
			"0",
			"A: 1\nB: 2",
			"",
		),
	}, {
		name: "bare CR before end of trailers",
		request: joinCRLF(
			"POST / HTTP/1.1",
			"Host: example.tld",
			"Transfer-Encoding: chunked",
			"",
			"3",
			"xxx",
			"0",
			"Foo: bar\r\r\n\r\n",
		),
	}} {
		synctest.Subtest(t, test.name, func(t *testing.T) {
			handler := newTestHandler(t)
			st := newHTTP1ServerTest(t, handler.ServeHTTP)
			defer handler.Close()

			conn := st.dial()
			conn.writeMessage(test.request)

			call := handler.nextCall()
			http.NewResponseController(call.w).EnableFullDuplex()
			n, err := io.Copy(io.Discard, call.req.Body)
			if err == nil {
				t.Errorf("read %v request data bytes without error; want error", n)
			}
			call.exit()

			// We should close the connection after sending the response.
			conn.wantResponse("HTTP/1.1 200 OK", nil)
			conn.wantClosed()
		})
	}
}

// An http1ServerTest tests an HTTP/1 server using a fake network.
// It must be used in a synctest bubble.
type http1ServerTest struct {
	t  *testing.T
	ts *httptest.Server
}

func newHTTP1ServerTest(t *testing.T, h http.HandlerFunc) *http1ServerTest {
	if h == nil {
		h = func(w http.ResponseWriter, req *http.Request) {}
	}
	st := &http1ServerTest{
		t:  t,
		ts: httptest.NewTestServer(t, h),
	}
	return st
}

// client returns a Client that sends requests to the server.
func (st *http1ServerTest) client() *http.Client {
	return st.ts.Client()
}

// transport returns a Transport that sends requests to the server.
func (st *http1ServerTest) transport() *http.Transport {
	return st.ts.Client().Transport.(*http.Transport)
}

// dial returns a connection to the server.
func (st *http1ServerTest) dial() *http1TestConn {
	t := st.t
	t.Helper()
	nc, err := st.transport().DialContext(st.t.Context(), "tcp", "example.tld")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		nc.Close()
	})
	conn := nc.(*nettest.Conn)
	conn.SetReadError(errWouldBlock) // effectively make reads non-blocking
	return &http1TestConn{
		t:    st.t,
		conn: conn,
		bufr: bufio.NewReader(conn),
	}
}

var errWouldBlock = errors.New("would block")

type http1TestConn struct {
	t    *testing.T
	conn *nettest.Conn
	bufr *bufio.Reader
}

// writeMessage writes a number of CRLF-terminated lines to the connection.
func (tc *http1TestConn) writeMessage(lines ...string) {
	t := tc.t
	t.Helper()
	if _, err := tc.conn.Write([]byte(strings.Join(lines, "\r\n") + "\r\n")); err != nil {
		t.Fatalf("conn write: %v", err)
	}
}

// readRequest reads a request from the connection (not including the request body).
func (tc *http1TestConn) readRequest() *http.Request {
	t := tc.t
	t.Helper()
	synctest.Wait()
	req, err := http.ReadRequest(tc.bufr)
	if err != nil {
		t.Fatalf("ReadRequest: %v", err)
	}
	return req
}

// readResponse reads a response from the connection (not including the response body).
func (tc *http1TestConn) readResponse() *http.Response {
	t := tc.t
	t.Helper()
	synctest.Wait()
	resp, err := http.ReadResponse(tc.bufr, nil)
	if err != nil {
		t.Fatalf("ReadResponse: %v", err)
	}
	return resp
}

func (tc *http1TestConn) wantResponse(wantStart string, wantHeaders http.Header) {
	t := tc.t
	t.Helper()
	synctest.Wait()
	gotStart, err := tc.bufr.ReadString('\n')
	if err != nil {
		t.Fatalf("read from conn: %q, %v; want start line %q", gotStart, err, wantStart)
	}
	if got, want := gotStart, wantStart+"\r\n"; got != want {
		t.Fatalf("read start line:\n%q\nwant:\n%q", got, want)
	}
	gotHeaders := make(http.Header)
	for {
		line, err := tc.bufr.ReadString('\n')
		if err != nil {
			t.Fatalf("read from conn: %v (want header)", err)
		}
		line, ok := strings.CutSuffix(line, "\r\n")
		if !ok {
			t.Fatalf("header line has no CRLF suffix: %q", line)
		}
		if line == "" {
			break
		}
		k, v, ok := strings.Cut(line, ": ")
		if !ok {
			t.Fatalf("invalid header line: %q", line)
		}
		gotHeaders[k] = append(gotHeaders[k], v)
	}
	for k, wantv := range wantHeaders {
		gotv := gotHeaders[k]
		if !slices.Equal(gotv, wantv) {
			t.Errorf("header %v = %q, want %q", k, gotv, wantv)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
}

// wantBytes asserts that the given bytes can be read from the connection.
func (tc *http1TestConn) wantBytes(want []byte) {
	t := tc.t
	t.Helper()
	synctest.Wait()
	got := make([]byte, len(want))
	n, err := io.ReadFull(tc.bufr, got)
	got = got[:n]
	if err != nil || !bytes.Equal(want, got) {
		t.Fatalf("want bytes %q, got %q and error %v", want, got, err)
	}
}

// wantIdle asserts that the connection is not closed and has no pending data to read.
func (tc *http1TestConn) wantIdle() {
	t := tc.t
	t.Helper()
	synctest.Wait()
	if got, err := tc.bufr.Peek(32); len(got) != 0 || !errors.Is(err, errWouldBlock) {
		t.Fatalf("read from conn: %q, %v; expect conn to be idle", got, err)
	}
}

// wantClosed asserts that the connection is read-closed and has no pending data to read.
func (tc *http1TestConn) wantClosed() {
	t := tc.t
	t.Helper()
	synctest.Wait()
	if got, err := tc.bufr.Peek(32); len(got) != 0 || err != io.EOF {
		t.Fatalf("read from conn: %q; expect conn to be closed", got)
	}
}

type testHandler struct {
	t      *testing.T
	mu     sync.Mutex
	calls  []*testHandlerCall
	closed bool
}

func newTestHandler(t *testing.T) *testHandler {
	h := &testHandler{t: t}
	t.Cleanup(func() {
		// testHandler.Close should be called before the server shuts down.
		// Catch the case where we forgot to do this.
		if !h.closed {
			t.Errorf("testHandler.Close not called")
		}
	})
	return h
}

func (h *testHandler) Close() {
	h.t.Helper()
	synctest.Wait()
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.calls) > 0 {
		h.t.Errorf("test finished with %v handler calls unhandled", len(h.calls))
	}
	for _, call := range h.calls {
		call.exit()
	}
	h.calls = nil
	h.closed = true
}

func (h *testHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	call := &testHandlerCall{
		w:   w,
		req: req,
		ch:  make(chan func()),
	}
	h.mu.Lock()
	if h.closed {
		h.t.Errorf("test handler called after close")
	}
	h.calls = append(h.calls, call)
	h.mu.Unlock()
	for f := range call.ch {
		f()
	}
}

func (h *testHandler) nextCall() *testHandlerCall {
	h.t.Helper()
	synctest.Wait()
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.calls) == 0 {
		h.t.Fatal("expected server handler call, got none")
	}
	call := h.calls[0]
	h.calls = h.calls[1:]
	h.t.Cleanup(call.exit)
	return call
}

// testHandlerCall is a call to the server handler's ServeHTTP method.
type testHandlerCall struct {
	w         http.ResponseWriter
	req       *http.Request
	closeOnce sync.Once
	ch        chan func()
}

// do executes f in the handler's goroutine.
func (call *testHandlerCall) do(f func(http.ResponseWriter, *http.Request)) {
	donec := make(chan struct{})
	call.ch <- func() {
		defer close(donec)
		f(call.w, call.req)
	}
	<-donec
}

// exit causes the handler to return.
func (call *testHandlerCall) exit() {
	call.closeOnce.Do(func() {
		close(call.ch)
	})
}

func joinCRLF(s ...string) string {
	return strings.Join(s, "\r\n")
}
