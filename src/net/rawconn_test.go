// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"bytes"
	"syscall"
	"testing"
)

func TestRawConn(t *testing.T) {
	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		defer c.Close()
		var b [32]byte
		n, err := c.Read(b[:])
		if err != nil {
			t.Error(err)
			return
		}
		if _, err := c.Write(b[:n]); err != nil {
			t.Error(err)
			return
		}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	cc, err := c.(*TCPConn).SyscallConn()
	if err != nil {
		t.Fatal(err)
	}

	var operr error
	data := []byte("HELLO-R-U-THERE")
	err = cc.Write(func(s uintptr) bool {
		_, operr = syscall.Write(int(s), data)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	})
	if err != nil || operr != nil {
		t.Fatal(err, operr)
	}

	var nr int
	var b [32]byte
	err = cc.Read(func(s uintptr) bool {
		nr, operr = syscall.Read(int(s), b[:])
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	})
	if err != nil || operr != nil {
		t.Fatal(err, operr)
	}
	if bytes.Compare(b[:nr], data) != 0 {
		t.Fatalf("got %#v; want %#v", b[:nr], data)
	}

	fn := func(s uintptr) {
		operr = syscall.SetsockoptInt(int(s), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)
	}
	err = cc.Control(fn)
	if err != nil || operr != nil {
		t.Fatal(err, operr)
	}
	c.Close()
	err = cc.Control(fn)
	if err == nil {
		t.Fatal("should fail")
	}
}
