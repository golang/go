// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// End-to-end serving tests

package http

import (
	"bufio"
	"bytes"
	"io"
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

type responseWriterMethodCall struct {
	method                 string
	headerKey, headerValue string // if method == "SetHeader"
	bytesWritten           []byte // if method == "Write"
	responseCode           int    // if method == "WriteHeader"
}

type recordingResponseWriter struct {
	log []*responseWriterMethodCall
}

func (rw *recordingResponseWriter) RemoteAddr() string {
	return "1.2.3.4"
}

func (rw *recordingResponseWriter) UsingTLS() bool {
	return false
}

func (rw *recordingResponseWriter) SetHeader(k, v string) {
	rw.log = append(rw.log, &responseWriterMethodCall{method: "SetHeader", headerKey: k, headerValue: v})
}

func (rw *recordingResponseWriter) Write(buf []byte) (int, os.Error) {
	rw.log = append(rw.log, &responseWriterMethodCall{method: "Write", bytesWritten: buf})
	return len(buf), nil
}

func (rw *recordingResponseWriter) WriteHeader(code int) {
	rw.log = append(rw.log, &responseWriterMethodCall{method: "WriteHeader", responseCode: code})
}

func (rw *recordingResponseWriter) Flush() {
	rw.log = append(rw.log, &responseWriterMethodCall{method: "Flush"})
}

func (rw *recordingResponseWriter) Hijack() (io.ReadWriteCloser, *bufio.ReadWriter, os.Error) {
	panic("Not supported")
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
		resp := new(recordingResponseWriter)
		resp.log = make([]*responseWriterMethodCall, 0)

		mux.ServeHTTP(resp, req)

		dumpLog := func() {
			t.Logf("For path %q:", path)
			for _, call := range resp.log {
				t.Logf("Got call: %s, header=%s, value=%s, buf=%q, code=%d", call.method,
					call.headerKey, call.headerValue, call.bytesWritten, call.responseCode)
			}
		}

		if len(resp.log) != 2 {
			dumpLog()
			t.Errorf("expected 2 calls to response writer; got %d", len(resp.log))
			return
		}

		if resp.log[0].method != "SetHeader" ||
			resp.log[0].headerKey != "Location" || resp.log[0].headerValue != "/foo.txt" {
			dumpLog()
			t.Errorf("Expected SetHeader of Location to /foo.txt")
			return
		}

		if resp.log[1].method != "WriteHeader" || resp.log[1].responseCode != StatusMovedPermanently {
			dumpLog()
			t.Errorf("Expected WriteHeader of StatusMovedPermanently")
			return
		}
	}
}
