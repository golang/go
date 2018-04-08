// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"runtime"
	"testing"
)

func TestRawConnReadWrite(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	t.Run("TCP", func(t *testing.T) {
		handler := func(ls *localServer, ln Listener) {
			c, err := ln.Accept()
			if err != nil {
				t.Error(err)
				return
			}
			defer c.Close()

			cc, err := ln.(*TCPListener).SyscallConn()
			if err != nil {
				t.Fatal(err)
			}
			called := false
			op := func(uintptr) bool {
				called = true
				return true
			}
			err = cc.Write(op)
			if err == nil {
				t.Error("Write should return an error")
			}
			if called {
				t.Error("Write shouldn't call op")
			}
			called = false
			err = cc.Read(op)
			if err == nil {
				t.Error("Read should return an error")
			}
			if called {
				t.Error("Read shouldn't call op")
			}

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
		data := []byte("HELLO-R-U-THERE")
		if err := writeRawConn(cc, data); err != nil {
			t.Fatal(err)
		}
		var b [32]byte
		n, err := readRawConn(cc, b[:])
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Compare(b[:n], data) != 0 {
			t.Fatalf("got %q; want %q", b[:n], data)
		}
	})
}

func TestRawConnControl(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	t.Run("TCP", func(t *testing.T) {
		ln, err := newLocalListener("tcp")
		if err != nil {
			t.Fatal(err)
		}
		defer ln.Close()

		cc1, err := ln.(*TCPListener).SyscallConn()
		if err != nil {
			t.Fatal(err)
		}
		if err := controlRawConn(cc1, ln.Addr()); err != nil {
			t.Fatal(err)
		}

		c, err := Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		cc2, err := c.(*TCPConn).SyscallConn()
		if err != nil {
			t.Fatal(err)
		}
		if err := controlRawConn(cc2, c.LocalAddr()); err != nil {
			t.Fatal(err)
		}

		ln.Close()
		if err := controlRawConn(cc1, ln.Addr()); err == nil {
			t.Fatal("Control after Close should fail")
		}
		c.Close()
		if err := controlRawConn(cc2, c.LocalAddr()); err == nil {
			t.Fatal("Control after Close should fail")
		}
	})
}
