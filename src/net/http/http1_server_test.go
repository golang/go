// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"errors"
	"internal/nettest"
	"internal/synctest"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
