// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || solaris || windows

package net

import "testing"

func TestTCPConnKeepAliveConfigDialer(t *testing.T) {
	maybeSkipKeepAliveTest(t)

	t.Cleanup(func() {
		testPreHookSetKeepAlive = func(*netFD) {}
	})
	var (
		errHook error
		oldCfg  KeepAliveConfig
	)
	testPreHookSetKeepAlive = func(nfd *netFD) {
		oldCfg, errHook = getCurrentKeepAliveSettings(fdType(nfd.pfd.Sysfd))
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
			verifyKeepAliveSettings(t, fdType(fd), oldCfg, cfg)
		}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTCPConnKeepAliveConfigListener(t *testing.T) {
	maybeSkipKeepAliveTest(t)

	t.Cleanup(func() {
		testPreHookSetKeepAlive = func(*netFD) {}
	})
	var (
		errHook error
		oldCfg  KeepAliveConfig
	)
	testPreHookSetKeepAlive = func(nfd *netFD) {
		oldCfg, errHook = getCurrentKeepAliveSettings(fdType(nfd.pfd.Sysfd))
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
			verifyKeepAliveSettings(t, fdType(fd), oldCfg, cfg)
		}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTCPConnKeepAliveConfig(t *testing.T) {
	maybeSkipKeepAliveTest(t)

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
	for _, cfg := range testConfigs {
		d := Dialer{KeepAlive: -1} // avoid setting default values before the test
		c, err := d.Dial("tcp", ls.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		sc, err := c.(*TCPConn).SyscallConn()
		if err != nil {
			t.Fatal(err)
		}

		var (
			errHook error
			oldCfg  KeepAliveConfig
		)
		if err := sc.Control(func(fd uintptr) {
			oldCfg, errHook = getCurrentKeepAliveSettings(fdType(fd))
		}); err != nil {
			t.Fatal(err)
		}
		if errHook != nil {
			t.Fatal(errHook)
		}

		if err := c.(*TCPConn).SetKeepAliveConfig(cfg); err != nil {
			t.Fatal(err)
		}

		if err := sc.Control(func(fd uintptr) {
			verifyKeepAliveSettings(t, fdType(fd), oldCfg, cfg)
		}); err != nil {
			t.Fatal(err)
		}
	}
}
