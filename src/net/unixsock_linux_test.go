// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"reflect"
	"syscall"
	"testing"
	"time"
)

func TestUnixgramAutobind(t *testing.T) {
	laddr := &UnixAddr{Name: "", Net: "unixgram"}
	c1, err := ListenUnixgram("unixgram", laddr)
	if err != nil {
		t.Fatal(err)
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
		t.Fatal(err)
	}
	defer c2.Close()

	if !reflect.DeepEqual(c1.LocalAddr(), c2.RemoteAddr()) {
		t.Fatalf("expected autobind address %v, got %v", c1.LocalAddr(), c2.RemoteAddr())
	}
}

func TestUnixAutobindClose(t *testing.T) {
	laddr := &UnixAddr{Name: "", Net: "unix"}
	ln, err := ListenUnix("unix", laddr)
	if err != nil {
		t.Fatal(err)
	}
	ln.Close()
}

func TestUnixAbstractLongNameeNullStart(t *testing.T) {
	addr := "\x00abstract_test"
	rsu := syscall.RawSockaddrUnix{}
	paddedAddr := make([]byte, len(rsu.Path))
	copy(paddedAddr, "\x00abstract_test")
	addr = string(paddedAddr)

	la, err := ResolveUnixAddr("unix", addr)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenUnix("unix", la)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
}

func TestUnixgramLinuxAbstractLongName(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("abstract unix socket long name test")
	}

	// Create an abstract socket name whose length is exactly
	// the maximum RawSockkaddrUnix Path len
	rsu := syscall.RawSockaddrUnix{}
	addrBytes := make([]byte, len(rsu.Path))
	copy(addrBytes, "@abstract_test")
	addr := string(addrBytes)

	la, err := ResolveUnixAddr("unixgram", addr)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenUnixgram("unixgram", la)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	off := make(chan bool)
	data := [5]byte{1, 2, 3, 4, 5}
	go func() {
		defer func() { off <- true }()
		s, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_DGRAM, 0)
		if err != nil {
			t.Error(err)
			return
		}
		defer syscall.Close(s)
		rsa := &syscall.SockaddrUnix{Name: addr}
		if err := syscall.Sendto(s, data[:], 0, rsa); err != nil {
			t.Error(err)
			return
		}
	}()

	<-off
	b := make([]byte, 64)
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	n, from, err := c.ReadFrom(b)
	if err != nil {
		t.Fatal(err)
	}
	if from != nil {
		t.Fatalf("unexpected peer address: %v", from)
	}
	if !bytes.Equal(b[:n], data[:]) {
		t.Fatalf("got %v; want %v", b[:n], data[:])
	}
}
