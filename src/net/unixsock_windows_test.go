// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package net

import (
	"internal/syscall/windows/registry"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"testing"
)

func isBuild17063() bool {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion`, registry.READ)
	if err != nil {
		return false
	}
	defer k.Close()

	s, _, err := k.GetStringValue("CurrentBuild")
	if err != nil {
		return false
	}
	ver, err := strconv.Atoi(s)
	if err != nil {
		return false
	}
	return ver >= 17063
}

func skipIfUnixSocketNotSupported(t *testing.T) {
	// TODO: the isBuild17063 check should be enough, investigate why 386 and arm
	// can't run these tests on newer Windows.
	switch runtime.GOARCH {
	case "386":
		t.Skip("not supported on windows/386, see golang.org/issue/27943")
	case "arm":
		t.Skip("not supported on windows/arm, see golang.org/issue/28061")
	}
	if !isBuild17063() {
		t.Skip("unix test")
	}
}

func TestUnixConnLocalWindows(t *testing.T) {
	skipIfUnixSocketNotSupported(t)
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

func TestModeSocket(t *testing.T) {
	skipIfUnixSocketNotSupported(t)
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
