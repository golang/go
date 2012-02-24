// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"runtime"
	"testing"
	"time"
)

func TestShutdown(t *testing.T) {
	if runtime.GOOS == "plan9" {
		return
	}
	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		if l, err = Listen("tcp6", "[::1]:0"); err != nil {
			t.Fatalf("ListenTCP on :0: %v", err)
		}
	}

	go func() {
		c, err := l.Accept()
		if err != nil {
			t.Fatalf("Accept: %v", err)
		}
		var buf [10]byte
		n, err := c.Read(buf[:])
		if n != 0 || err != io.EOF {
			t.Fatalf("server Read = %d, %v; want 0, io.EOF", n, err)
		}
		c.Write([]byte("response"))
		c.Close()
	}()

	c, err := Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer c.Close()

	err = c.(*TCPConn).CloseWrite()
	if err != nil {
		t.Fatalf("CloseWrite: %v", err)
	}
	var buf [10]byte
	n, err := c.Read(buf[:])
	if err != nil {
		t.Fatalf("client Read: %d, %v", n, err)
	}
	got := string(buf[:n])
	if got != "response" {
		t.Errorf("read = %q, want \"response\"", got)
	}
}

func TestTCPListenClose(t *testing.T) {
	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}

	done := make(chan bool, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		l.Close()
	}()
	go func() {
		_, err = l.Accept()
		if err == nil {
			t.Error("Accept succeeded")
		} else {
			t.Logf("Accept timeout error: %s (any error is fine)", err)
		}
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for TCP close")
	}
}

func TestUDPListenClose(t *testing.T) {
	l, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}

	buf := make([]byte, 1000)
	done := make(chan bool, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		l.Close()
	}()
	go func() {
		_, _, err = l.ReadFrom(buf)
		if err == nil {
			t.Error("ReadFrom succeeded")
		} else {
			t.Logf("ReadFrom timeout error: %s (any error is fine)", err)
		}
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for UDP close")
	}
}
