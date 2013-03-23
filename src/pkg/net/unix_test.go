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
		t.Fatalf("UnixConn.ReadFrom failed: %v", err)
	}
	if from != nil {
		t.Fatalf("neighbor address is %v", from)
	}
	if !bytes.Equal(b[:n], data[:]) {
		t.Fatalf("got %v, want %v", b[:n], data[:])
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
	_, from, err := c.ReadFrom(nil)
	if err != nil {
		t.Fatalf("UnixConn.ReadFrom failed: %v", err)
	}
	if from != nil {
		t.Fatalf("neighbor address is %v", from)
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
		t.Fatalf("invalid autobind address: %v", autoAddr)
	}
	if autoAddr.Name[0] != '@' {
		t.Fatalf("invalid autobind address: %v", autoAddr)
	}

	c2, err := DialUnix("unixgram", nil, autoAddr)
	if err != nil {
		t.Fatalf("DialUnix failed: %v", err)
	}
	defer c2.Close()

	if !reflect.DeepEqual(c1.LocalAddr(), c2.RemoteAddr()) {
		t.Fatalf("expected autobind address %v, got %v", c1.LocalAddr(), c2.RemoteAddr())
	}
}

func TestUnixConnLocalAndRemoteNames(t *testing.T) {
	for _, laddr := range []string{"", testUnixAddr()} {
		taddr := testUnixAddr()
		ta, err := ResolveUnixAddr("unix", taddr)
		if err != nil {
			t.Fatalf("ResolveUnixAddr failed: %v", err)
		}
		ln, err := ListenUnix("unix", ta)
		if err != nil {
			t.Fatalf("ListenUnix failed: %v", err)
		}
		defer func() {
			ln.Close()
			os.Remove(taddr)
		}()

		done := make(chan int)
		go transponder(t, ln, done)

		la, err := ResolveUnixAddr("unix", laddr)
		if err != nil {
			t.Fatalf("ResolveUnixAddr failed: %v", err)
		}
		c, err := DialUnix("unix", la, ta)
		if err != nil {
			t.Fatalf("DialUnix failed: %v", err)
		}
		defer func() {
			c.Close()
			if la != nil {
				defer os.Remove(laddr)
			}
		}()
		if _, err := c.Write([]byte("UNIXCONN LOCAL AND REMOTE NAME TEST")); err != nil {
			t.Fatalf("UnixConn.Write failed: %v", err)
		}

		if runtime.GOOS == "linux" && laddr == "" {
			laddr = "@" // autobind feature
		}
		var connAddrs = [3]struct{ got, want Addr }{
			{ln.Addr(), ta},
			{c.LocalAddr(), &UnixAddr{Name: laddr, Net: "unix"}},
			{c.RemoteAddr(), ta},
		}
		for _, ca := range connAddrs {
			if !reflect.DeepEqual(ca.got, ca.want) {
				t.Fatalf("got %#v, expected %#v", ca.got, ca.want)
			}
		}

		<-done
	}
}

func TestUnixgramConnLocalAndRemoteNames(t *testing.T) {
	for _, laddr := range []string{"", testUnixAddr()} {
		taddr := testUnixAddr()
		ta, err := ResolveUnixAddr("unixgram", taddr)
		if err != nil {
			t.Fatalf("ResolveUnixAddr failed: %v", err)
		}
		c1, err := ListenUnixgram("unixgram", ta)
		if err != nil {
			t.Fatalf("ListenUnixgram failed: %v", err)
		}
		defer func() {
			c1.Close()
			os.Remove(taddr)
		}()

		var la *UnixAddr
		if laddr != "" {
			var err error
			if la, err = ResolveUnixAddr("unixgram", laddr); err != nil {
				t.Fatalf("ResolveUnixAddr failed: %v", err)
			}
		}
		c2, err := DialUnix("unixgram", la, ta)
		if err != nil {
			t.Fatalf("DialUnix failed: %v", err)
		}
		defer func() {
			c2.Close()
			if la != nil {
				defer os.Remove(laddr)
			}
		}()

		if runtime.GOOS == "linux" && laddr == "" {
			laddr = "@" // autobind feature
		}
		var connAddrs = [4]struct{ got, want Addr }{
			{c1.LocalAddr(), ta},
			{c1.RemoteAddr(), nil},
			{c2.LocalAddr(), &UnixAddr{Name: laddr, Net: "unixgram"}},
			{c2.RemoteAddr(), ta},
		}
		for _, ca := range connAddrs {
			if !reflect.DeepEqual(ca.got, ca.want) {
				t.Fatalf("got %#v, expected %#v", ca.got, ca.want)
			}
		}
	}
}
