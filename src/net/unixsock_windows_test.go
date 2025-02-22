// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package net

import (
	"internal/syscall/windows"
	"os"
	"reflect"
	"syscall"
	"testing"
)

func TestUnixConnLocalWindows(t *testing.T) {
	if !windows.SupportUnixSocket() {
		t.Skip("unix test")
	}
	handler := func(ls *localServer, ln Listener) {}
	for _, laddr := range []string{"", testUnixAddr(t)} {
		laddr := laddr
		taddr := testUnixAddr(t)
		ta, err := ResolveUnixAddr("unix", taddr)
		if err != nil {
			t.Fatal(err)
		}
		ln, err := ListenUnix("unix", ta)
		if err != nil {
			t.Fatal(err)
		}
		ls := (&streamListener{Listener: ln}).newLocalServer()
		defer ls.teardown()
		if err := ls.buildup(handler); err != nil {
			t.Fatal(err)
		}

		la, err := ResolveUnixAddr("unix", laddr)
		if err != nil {
			t.Fatal(err)
		}
		c, err := DialUnix("unix", la, ta)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			c.Close()
			if la != nil {
				defer os.Remove(laddr)
			}
		}()
		if _, err := c.Write([]byte("UNIXCONN LOCAL AND REMOTE NAME TEST")); err != nil {
			t.Fatal(err)
		}

		if laddr == "" {
			laddr = "@"
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
	}
}

func TestUnixAbstractLongNameNulStart(t *testing.T) {
	if !windows.SupportUnixSocket() {
		t.Skip("unix test")
	}

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

func TestModeSocket(t *testing.T) {
	if !windows.SupportUnixSocket() {
		t.Skip("unix test")
	}

	addr := testUnixAddr(t)

	l, err := Listen("unix", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	stat, err := os.Stat(addr)
	if err != nil {
		t.Fatal(err)
	}

	mode := stat.Mode()
	if mode&os.ModeSocket == 0 {
		t.Fatalf("%v should have ModeSocket", mode)
	}
}
