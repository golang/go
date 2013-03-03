// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package net

import (
	"bytes"
	"os"
	"reflect"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func TestReadUnixgramWithUnnamedSocket(t *testing.T) {
	addr := testUnixAddr()
	la, err := ResolveUnixAddr("unixgram", addr)
	if err != nil {
		t.Fatalf("ResolveUnixAddr failed: %v", err)
	}
	c, err := ListenUnixgram("unixgram", la)
	if err != nil {
		t.Fatalf("ListenUnixgram failed: %v", err)
	}
	defer func() {
		c.Close()
		os.Remove(addr)
	}()

	off := make(chan bool)
	data := [5]byte{1, 2, 3, 4, 5}

	go func() {
		defer func() { off <- true }()
		s, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_DGRAM, 0)
		if err != nil {
			t.Errorf("syscall.Socket failed: %v", err)
			return
		}
		defer syscall.Close(s)
		rsa := &syscall.SockaddrUnix{Name: addr}
		if err := syscall.Sendto(s, data[:], 0, rsa); err != nil {
			t.Errorf("syscall.Sendto failed: %v", err)
			return
		}
	}()

	<-off
	b := make([]byte, 64)
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	n, from, err := c.ReadFrom(b)
	if err != nil {
		t.Errorf("UnixConn.ReadFrom failed: %v", err)
		return
	}
	if from != nil {
		t.Errorf("neighbor address is %v", from)
	}
	if !bytes.Equal(b[:n], data[:]) {
		t.Errorf("got %v, want %v", b[:n], data[:])
		return
	}
}

func TestReadUnixgramWithZeroBytesBuffer(t *testing.T) {
	// issue 4352: Recvfrom failed with "address family not
	// supported by protocol family" if zero-length buffer provided

	addr := testUnixAddr()
	la, err := ResolveUnixAddr("unixgram", addr)
	if err != nil {
		t.Fatalf("ResolveUnixAddr failed: %v", err)
	}
	c, err := ListenUnixgram("unixgram", la)
	if err != nil {
		t.Fatalf("ListenUnixgram failed: %v", err)
	}
	defer func() {
		c.Close()
		os.Remove(addr)
	}()

	off := make(chan bool)
	go func() {
		defer func() { off <- true }()
		c, err := DialUnix("unixgram", nil, la)
		if err != nil {
			t.Errorf("DialUnix failed: %v", err)
			return
		}
		defer c.Close()
		if _, err := c.Write([]byte{1, 2, 3, 4, 5}); err != nil {
			t.Errorf("UnixConn.Write failed: %v", err)
			return
		}
	}()

	<-off
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	var peer Addr
	if _, peer, err = c.ReadFrom(nil); err != nil {
		t.Errorf("UnixConn.ReadFrom failed: %v", err)
		return
	}
	if peer != nil {
		t.Errorf("peer adddress is %v", peer)
	}
}

func TestUnixAutobind(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("skipping: autobind is linux only")
	}

	laddr := &UnixAddr{Name: "", Net: "unixgram"}
	c1, err := ListenUnixgram("unixgram", laddr)
	if err != nil {
		t.Fatalf("ListenUnixgram failed: %v", err)
	}
	defer c1.Close()

	// retrieve the autobind address
	autoAddr := c1.LocalAddr().(*UnixAddr)
	if len(autoAddr.Name) <= 1 {
		t.Fatalf("Invalid autobind address: %v", autoAddr)
	}
	if autoAddr.Name[0] != '@' {
		t.Fatalf("Invalid autobind address: %v", autoAddr)
	}

	c2, err := DialUnix("unixgram", nil, autoAddr)
	if err != nil {
		t.Fatalf("DialUnix failed: %v", err)
	}
	defer c2.Close()

	if !reflect.DeepEqual(c1.LocalAddr(), c2.RemoteAddr()) {
		t.Fatalf("Expected autobind address %v, got %v", c1.LocalAddr(), c2.RemoteAddr())
	}
}
