// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// White-box tests for transport.go (in package http instead of http_test).

package http

import (
	"errors"
	"net"
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
	if err != errServerClosedConn && err != errServerClosedIdle {
		t.Fatalf("roundTrip = %#v, %v; want errServerClosedConn or errServerClosedIdle", err, err)
	}

	<-pc.closech
	err = pc.closed
	if err != errServerClosedConn && err != errServerClosedIdle {
		t.Fatalf("pc.closed = %#v, %v; want errServerClosedConn or errServerClosedIdle", err, err)
	}
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
