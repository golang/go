// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements API tests across platforms and will never have a build
// tag.

package net

import (
	"testing"
	"time"
)

// someTimeout is used just to test that net.Conn implementations
// don't explode when their SetFooDeadline methods are called.
// It isn't actually used for testing timeouts.
const someTimeout = 10 * time.Second

func TestConnAndListener(t *testing.T) {
	handler := func(ls *localServer, ln Listener) { transponder(t, ln) }
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		ls, err := newLocalServer(network)
		if err != nil {
			t.Fatalf("Listen failed: %v", err)
		}
		defer ls.teardown()
		if err := ls.buildup(handler); err != nil {
			t.Fatal(err)
		}
		if ls.Listener.Addr().Network() != network {
			t.Fatalf("got %s; want %s", ls.Listener.Addr().Network(), network)
		}

		c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		defer c.Close()
		if c.LocalAddr().Network() != network || c.LocalAddr().Network() != network {
			t.Fatalf("got %v->%v; want %v->%v", c.LocalAddr().Network(), c.RemoteAddr().Network(), network, network)
		}
		c.SetDeadline(time.Now().Add(someTimeout))
		c.SetReadDeadline(time.Now().Add(someTimeout))
		c.SetWriteDeadline(time.Now().Add(someTimeout))

		if _, err := c.Write([]byte("CONN TEST")); err != nil {
			t.Fatalf("Conn.Write failed: %v", err)
		}
		rb := make([]byte, 128)
		if _, err := c.Read(rb); err != nil {
			t.Fatalf("Conn.Read failed: %v", err)
		}
	}
}

func transponder(t *testing.T, ln Listener) {
	switch ln := ln.(type) {
	case *TCPListener:
		ln.SetDeadline(time.Now().Add(someTimeout))
	case *UnixListener:
		ln.SetDeadline(time.Now().Add(someTimeout))
	}
	c, err := ln.Accept()
	if err != nil {
		t.Errorf("Listener.Accept failed: %v", err)
		return
	}
	defer c.Close()

	network := ln.Addr().Network()
	if c.LocalAddr().Network() != network || c.LocalAddr().Network() != network {
		t.Errorf("got %v->%v; expected %v->%v", c.LocalAddr().Network(), c.RemoteAddr().Network(), network, network)
		return
	}
	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))

	b := make([]byte, 128)
	n, err := c.Read(b)
	if err != nil {
		t.Errorf("Conn.Read failed: %v", err)
		return
	}
	if _, err := c.Write(b[:n]); err != nil {
		t.Errorf("Conn.Write failed: %v", err)
		return
	}
}
