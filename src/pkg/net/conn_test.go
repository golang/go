// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net_test

import (
	"net"
	"os"
	"runtime"
	"testing"
	"time"
)

var connTests = []struct {
	net  string
	addr string
}{
	{"tcp", "127.0.0.1:0"},
	{"unix", "/tmp/gotest.net1"},
	{"unixpacket", "/tmp/gotest.net2"},
}

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
			os.Remove(tt.addr)
		}

		ln, err := net.Listen(tt.net, tt.addr)
		if err != nil {
			t.Errorf("net.Listen failed: %v", err)
			return
		}
		ln.Addr()
		defer func(ln net.Listener, net, addr string) {
			ln.Close()
			switch net {
			case "unix", "unixpacket":
				os.Remove(addr)
			}
		}(ln, tt.net, tt.addr)

		done := make(chan int)
		go transponder(t, ln, done)

		c, err := net.Dial(tt.net, ln.Addr().String())
		if err != nil {
			t.Errorf("net.Dial failed: %v", err)
			return
		}
		c.LocalAddr()
		c.RemoteAddr()
		c.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
		defer c.Close()

		if _, err := c.Write([]byte("CONN TEST")); err != nil {
			t.Errorf("net.Conn.Write failed: %v", err)
			return
		}
		rb := make([]byte, 128)
		if _, err := c.Read(rb); err != nil {
			t.Errorf("net.Conn.Read failed: %v", err)
		}

		<-done
	}
}

func transponder(t *testing.T, ln net.Listener, done chan<- int) {
	defer func() { done <- 1 }()

	c, err := ln.Accept()
	if err != nil {
		t.Errorf("net.Listener.Accept failed: %v", err)
		return
	}
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	defer c.Close()

	b := make([]byte, 128)
	n, err := c.Read(b)
	if err != nil {
		t.Errorf("net.Conn.Read failed: %v", err)
		return
	}
	if _, err := c.Write(b[:n]); err != nil {
		t.Errorf("net.Conn.Write failed: %v", err)
		return
	}
}
