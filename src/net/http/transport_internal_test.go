// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// White-box tests for transport.go (in package http instead of http_test).

package http

import (
	"bytes"
	"crypto/tls"
	"errors"
	"io"
	"io/ioutil"
	"net"
	"net/http/internal"
	"strings"
	"testing"
)

// Issue 15446: incorrect wrapping of errors when server closes an idle connection.
func TestTransportPersistConnReadLoopEOF(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	connc := make(chan net.Conn, 1)
	go func() {
		defer close(connc)
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		connc <- c
	}()

	tr := new(Transport)
	req, _ := NewRequest("GET", "http://"+ln.Addr().String(), nil)
	req = req.WithT(t)
	treq := &transportRequest{Request: req}
	cm := connectMethod{targetScheme: "http", targetAddr: ln.Addr().String()}
	pc, err := tr.getConn(treq, cm)
	if err != nil {
		t.Fatal(err)
	}
	defer pc.close(errors.New("test over"))

	conn := <-connc
	if conn == nil {
		// Already called t.Error in the accept goroutine.
		return
	}
	conn.Close() // simulate the server hanging up on the client

	_, err = pc.roundTrip(treq)
	if !isTransportReadFromServerError(err) && err != errServerClosedIdle {
		t.Errorf("roundTrip = %#v, %v; want errServerClosedIdle or transportReadFromServerError", err, err)
	}

	<-pc.closech
	err = pc.closed
	if !isTransportReadFromServerError(err) && err != errServerClosedIdle {
		t.Errorf("pc.closed = %#v, %v; want errServerClosedIdle or transportReadFromServerError", err, err)
	}
}

func isTransportReadFromServerError(err error) bool {
	_, ok := err.(transportReadFromServerError)
	return ok
}

func newLocalListener(t *testing.T) net.Listener {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

func dummyRequest(method string) *Request {
	req, err := NewRequest(method, "http://fake.tld/", nil)
	if err != nil {
		panic(err)
	}
	return req
}
func dummyRequestWithBody(method string) *Request {
	req, err := NewRequest(method, "http://fake.tld/", strings.NewReader("foo"))
	if err != nil {
		panic(err)
	}
	return req
}

func dummyRequestWithBodyNoGetBody(method string) *Request {
	req := dummyRequestWithBody(method)
	req.GetBody = nil
	return req
}

// issue22091Error acts like a golang.org/x/net/http2.ErrNoCachedConn.
type issue22091Error struct{}

func (issue22091Error) IsHTTP2NoCachedConnError() {}
func (issue22091Error) Error() string             { return "issue22091Error" }

func TestTransportShouldRetryRequest(t *testing.T) {
	tests := []struct {
		pc  *persistConn
		req *Request

		err  error
		want bool
	}{
		0: {
			pc:   &persistConn{reused: false},
			req:  dummyRequest("POST"),
			err:  nothingWrittenError{},
			want: false,
		},
		1: {
			pc:   &persistConn{reused: true},
			req:  dummyRequest("POST"),
			err:  nothingWrittenError{},
			want: true,
		},
		2: {
			pc:   &persistConn{reused: true},
			req:  dummyRequest("POST"),
			err:  http2ErrNoCachedConn,
			want: true,
		},
		3: {
			pc:   nil,
			req:  nil,
			err:  issue22091Error{}, // like an external http2ErrNoCachedConn
			want: true,
		},
		4: {
			pc:   &persistConn{reused: true},
			req:  dummyRequest("POST"),
			err:  errMissingHost,
			want: false,
		},
		5: {
			pc:   &persistConn{reused: true},
			req:  dummyRequest("POST"),
			err:  transportReadFromServerError{},
			want: false,
		},
		6: {
			pc:   &persistConn{reused: true},
			req:  dummyRequest("GET"),
			err:  transportReadFromServerError{},
			want: true,
		},
		7: {
			pc:   &persistConn{reused: true},
			req:  dummyRequest("GET"),
			err:  errServerClosedIdle,
			want: true,
		},
		8: {
			pc:   &persistConn{reused: true},
			req:  dummyRequestWithBody("POST"),
			err:  nothingWrittenError{},
			want: true,
		},
		9: {
			pc:   &persistConn{reused: true},
			req:  dummyRequestWithBodyNoGetBody("POST"),
			err:  nothingWrittenError{},
			want: false,
		},
	}
	for i, tt := range tests {
		got := tt.pc.shouldRetryRequest(tt.req, tt.err)
		if got != tt.want {
			t.Errorf("%d. shouldRetryRequest = %v; want %v", i, got, tt.want)
		}
	}
}

type roundTripFunc func(r *Request) (*Response, error)

func (f roundTripFunc) RoundTrip(r *Request) (*Response, error) {
	return f(r)
}

// Issue 25009
func TestTransportBodyAltRewind(t *testing.T) {
	cert, err := tls.X509KeyPair(internal.LocalhostCert, internal.LocalhostKey)
	if err != nil {
		t.Fatal(err)
	}
	ln := newLocalListener(t)
	defer ln.Close()

	go func() {
		tln := tls.NewListener(ln, &tls.Config{
			NextProtos:   []string{"foo"},
			Certificates: []tls.Certificate{cert},
		})
		for i := 0; i < 2; i++ {
			sc, err := tln.Accept()
			if err != nil {
				t.Error(err)
				return
			}
			if err := sc.(*tls.Conn).Handshake(); err != nil {
				t.Error(err)
				return
			}
			sc.Close()
		}
	}()

	addr := ln.Addr().String()
	req, _ := NewRequest("POST", "https://example.org/", bytes.NewBufferString("request"))
	roundTripped := false
	tr := &Transport{
		DisableKeepAlives: true,
		TLSNextProto: map[string]func(string, *tls.Conn) RoundTripper{
			"foo": func(authority string, c *tls.Conn) RoundTripper {
				return roundTripFunc(func(r *Request) (*Response, error) {
					n, _ := io.Copy(ioutil.Discard, r.Body)
					if n == 0 {
						t.Error("body length is zero")
					}
					if roundTripped {
						return &Response{
							Body:       NoBody,
							StatusCode: 200,
						}, nil
					}
					roundTripped = true
					return nil, http2noCachedConnError{}
				})
			},
		},
		DialTLS: func(_, _ string) (net.Conn, error) {
			tc, err := tls.Dial("tcp", addr, &tls.Config{
				InsecureSkipVerify: true,
				NextProtos:         []string{"foo"},
			})
			if err != nil {
				return nil, err
			}
			if err := tc.Handshake(); err != nil {
				return nil, err
			}
			return tc, nil
		},
	}
	c := &Client{Transport: tr}
	_, err = c.Do(req)
	if err != nil {
		t.Error(err)
	}
}
