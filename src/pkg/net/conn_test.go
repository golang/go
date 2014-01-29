// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements API tests across platforms and will never have a build
// tag.

package net

import (
	"os"
	"runtime"
	"testing"
	"time"
)

var connTests = []struct {
	net  string
	addr func() string
}{
	{"tcp", func() string { return "127.0.0.1:0" }},
	{"unix", testUnixAddr},
	{"unixpacket", testUnixAddr},
}

// someTimeout is used just to test that net.Conn implementations
// don't explode when their SetFooDeadline methods are called.
// It isn't actually used for testing timeouts.
const someTimeout = 10 * time.Second

func TestConnAndListener(t *testing.T) {
	for _, tt := range connTests {
		switch tt.net {
		case "unix", "unixpacket":
			switch runtime.GOOS {
			case "plan9", "windows":
				continue
			}
			if tt.net == "unixpacket" && runtime.GOOS != "linux" {
				continue
			}
		}

		addr := tt.addr()
		ln, err := Listen(tt.net, addr)
		if err != nil {
			t.Fatalf("Listen failed: %v", err)
		}
		defer func(ln Listener, net, addr string) {
			ln.Close()
			switch net {
			case "unix", "unixpacket":
				os.Remove(addr)
			}
		}(ln, tt.net, addr)
		if ln.Addr().Network() != tt.net {
			t.Fatalf("got %v; expected %v", ln.Addr().Network(), tt.net)
		}

		done := make(chan int)
		go transponder(t, ln, done)

		c, err := Dial(tt.net, ln.Addr().String())
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		defer c.Close()
		if c.LocalAddr().Network() != tt.net || c.LocalAddr().Network() != tt.net {
			t.Fatalf("got %v->%v; expected %v->%v", c.LocalAddr().Network(), c.RemoteAddr().Network(), tt.net, tt.net)
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

		<-done
	}
}

func transponder(t *testing.T, ln Listener, done chan<- int) {
	defer func() { done <- 1 }()

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
