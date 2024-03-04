// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || freebsd || linux || netbsd || dragonfly || darwin || solaris || windows

package net

import (
	"runtime"
	"testing"
)

func TestTCPConnDialerKeepAliveConfig(t *testing.T) {
	// TODO(panjf2000): stop skipping this test on Solaris
	//  when https://go.dev/issue/64251 is fixed.
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris for now")
	}

	t.Cleanup(func() {
		testPreHookSetKeepAlive = func(*netFD) {}
	})
	var (
		errHook error
		oldCfg  KeepAliveConfig
	)
	testPreHookSetKeepAlive = func(nfd *netFD) {
		oldCfg, errHook = getCurrentKeepAliveSettings(int(nfd.pfd.Sysfd))
	}

	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}
	ln := newLocalListener(t, "tcp", &ListenConfig{
		KeepAlive: -1, // prevent calling hook from accepting
	})
	ls := (&streamListener{Listener: ln}).newLocalServer()
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	for _, cfg := range testConfigs {
		d := Dialer{
			KeepAlive:       defaultTCPKeepAliveIdle, // should be ignored
			KeepAliveConfig: cfg}
		c, err := d.Dial("tcp", ls.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		if errHook != nil {
			t.Fatal(errHook)
		}

		sc, err := c.(*TCPConn).SyscallConn()
		if err != nil {
			t.Fatal(err)
		}
		if err := sc.Control(func(fd uintptr) {
			verifyKeepAliveSettings(t, int(fd), oldCfg, cfg)
		}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTCPConnListenerKeepAliveConfig(t *testing.T) {
	// TODO(panjf2000): stop skipping this test on Solaris
	//  when https://go.dev/issue/64251 is fixed.
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris for now")
	}

	t.Cleanup(func() {
		testPreHookSetKeepAlive = func(*netFD) {}
	})
	var (
		errHook error
		oldCfg  KeepAliveConfig
	)
	testPreHookSetKeepAlive = func(nfd *netFD) {
		oldCfg, errHook = getCurrentKeepAliveSettings(int(nfd.pfd.Sysfd))
	}

	ch := make(chan Conn, 1)
	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		ch <- c
	}
	for _, cfg := range testConfigs {
		ln := newLocalListener(t, "tcp", &ListenConfig{
			KeepAlive:       defaultTCPKeepAliveIdle, // should be ignored
			KeepAliveConfig: cfg})
		ls := (&streamListener{Listener: ln}).newLocalServer()
		defer ls.teardown()
		if err := ls.buildup(handler); err != nil {
			t.Fatal(err)
		}
		d := Dialer{KeepAlive: -1} // prevent calling hook from dialing
		c, err := d.Dial("tcp", ls.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		cc := <-ch
		defer cc.Close()
		if errHook != nil {
			t.Fatal(errHook)
		}
		sc, err := cc.(*TCPConn).SyscallConn()
		if err != nil {
			t.Fatal(err)
		}
		if err := sc.Control(func(fd uintptr) {
			verifyKeepAliveSettings(t, int(fd), oldCfg, cfg)
		}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTCPConnSetKeepAliveConfig(t *testing.T) {
	// TODO(panjf2000): stop skipping this test on Solaris
	//  when https://go.dev/issue/64251 is fixed.
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris for now")
	}

	handler := func(ls *localServer, ln Listener) {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}
	ls := newLocalServer(t, "tcp")
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}
	ra, err := ResolveTCPAddr("tcp", ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	for _, cfg := range testConfigs {
		c, err := DialTCP("tcp", nil, ra)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		sc, err := c.SyscallConn()
		if err != nil {
			t.Fatal(err)
		}

		var (
			errHook error
			oldCfg  KeepAliveConfig
		)
		if err := sc.Control(func(fd uintptr) {
			oldCfg, errHook = getCurrentKeepAliveSettings(int(fd))
		}); err != nil {
			t.Fatal(err)
		}
		if errHook != nil {
			t.Fatal(errHook)
		}

		if err := c.SetKeepAliveConfig(cfg); err != nil {
			t.Fatal(err)
		}

		if err := sc.Control(func(fd uintptr) {
			verifyKeepAliveSettings(t, int(fd), oldCfg, cfg)
		}); err != nil {
			t.Fatal(err)
		}
	}
}
