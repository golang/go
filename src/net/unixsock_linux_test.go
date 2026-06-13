// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"fmt"
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

func TestUnixAbstractLongNameNulStart(t *testing.T) {
	// Create an abstract socket name that starts with a null byte ("\x00")
	// whose length is the maximum of RawSockaddrUnix Path len
	paddedAddr := make([]byte, len(syscall.RawSockaddrUnix{}.Path))
	copy(paddedAddr, "\x00abstract_test")

	la, err := ResolveUnixAddr("unix", string(paddedAddr))
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenUnix("unix", la)
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
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

func TestUnixgramLinuxWriteToConnectedSockDgram(t *testing.T) {
	fmt.Println("TestUnixgramLinuxToConnectedSockDgram")
	if !testableNetwork("unixgram") {
		t.Skip("abstract unix socket long name test")
	}

	srv, err := ListenUnixgram("unixgram", &UnixAddr{Name: "", Net: "unixgram"})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	err = srv.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}

	msg := "hello connected unixgram"
	rcv := make(chan []byte)
	go func() {
		b := make([]byte, 1024)
		n, _, _, _, err := srv.ReadMsgUnix(b, nil)
		if err != nil {
			t.Error(err)
			return
		}
		if n != len(msg) {
			t.Errorf("got %d; want %d", n, len(msg))
			return
		}
		rcv <- b[:n]
	}()

	uc, err := DialUnix("unixgram", nil, srv.LocalAddr().(*UnixAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer uc.Close()

	_, _, err = uc.WriteMsgUnix([]byte(msg), nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	b := <-rcv
	if string(b) != msg {
		t.Fatalf("got %q; want %q", b, msg)
	}
}
