// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// End-to-end serving tests

package http

import (
	"bytes"
	"os"
	"net"
	"testing"
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
