// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"runtime"
	"testing"
)

func TestCloseRead(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		ln, err := newLocalListener(network)
		if err != nil {
			t.Fatal(err)
		}
		switch network {
		case "unix", "unixpacket":
			defer os.Remove(ln.Addr().String())
		}
		defer ln.Close()

		c, err := Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		switch network {
		case "unix", "unixpacket":
			defer os.Remove(c.LocalAddr().String())
		}
		defer c.Close()

		switch c := c.(type) {
		case *TCPConn:
			err = c.CloseRead()
		case *UnixConn:
			err = c.CloseRead()
		}
		if err != nil {
			if perr := parseCloseError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		var b [1]byte
		n, err := c.Read(b[:])
		if n != 0 || err == nil {
			t.Fatalf("got (%d, %v); want (0, error)", n, err)
		}
	}
}

func TestCloseWrite(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		defer c.Close()

		var b [1]byte
		n, err := c.Read(b[:])
		if n != 0 || err != io.EOF {
			t.Errorf("got (%d, %v); want (0, io.EOF)", n, err)
			return
		}
		switch c := c.(type) {
		case *TCPConn:
			err = c.CloseWrite()
		case *UnixConn:
			err = c.CloseWrite()
		}
		if err != nil {
			if perr := parseCloseError(err); perr != nil {
				t.Error(perr)
			}
			t.Error(err)
			return
		}
		n, err = c.Write(b[:])
		if err == nil {
			t.Errorf("got (%d, %v); want (any, error)", n, err)
			return
		}
	}

	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		ls, err := newLocalServer(network)
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
		switch network {
		case "unix", "unixpacket":
			defer os.Remove(c.LocalAddr().String())
		}
		defer c.Close()

		switch c := c.(type) {
		case *TCPConn:
			err = c.CloseWrite()
		case *UnixConn:
			err = c.CloseWrite()
		}
		if err != nil {
			if perr := parseCloseError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		var b [1]byte
		n, err := c.Read(b[:])
		if n != 0 || err != io.EOF {
			t.Fatalf("got (%d, %v); want (0, io.EOF)", n, err)
		}
		n, err = c.Write(b[:])
		if err == nil {
			t.Fatalf("got (%d, %v); want (any, error)", n, err)
		}
	}
}

func TestConnClose(t *testing.T) {
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		ln, err := newLocalListener(network)
		if err != nil {
			t.Fatal(err)
		}
		switch network {
		case "unix", "unixpacket":
			defer os.Remove(ln.Addr().String())
		}
		defer ln.Close()

		c, err := Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		switch network {
		case "unix", "unixpacket":
			defer os.Remove(c.LocalAddr().String())
		}
		defer c.Close()

		if err := c.Close(); err != nil {
			if perr := parseCloseError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		var b [1]byte
		n, err := c.Read(b[:])
		if n != 0 || err == nil {
			t.Fatalf("got (%d, %v); want (0, error)", n, err)
		}
	}
}

func TestListenerClose(t *testing.T) {
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		ln, err := newLocalListener(network)
		if err != nil {
			t.Fatal(err)
		}
		switch network {
		case "unix", "unixpacket":
			defer os.Remove(ln.Addr().String())
		}
		defer ln.Close()

		if err := ln.Close(); err != nil {
			if perr := parseCloseError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		c, err := ln.Accept()
		if err == nil {
			c.Close()
			t.Fatal("should fail")
		}
	}
}

func TestPacketConnClose(t *testing.T) {
	for _, network := range []string{"udp", "unixgram"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		c, err := newLocalPacketListener(network)
		if err != nil {
			t.Fatal(err)
		}
		switch network {
		case "unixgram":
			defer os.Remove(c.LocalAddr().String())
		}
		defer c.Close()

		if err := c.Close(); err != nil {
			if perr := parseCloseError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		var b [1]byte
		n, _, err := c.ReadFrom(b[:])
		if n != 0 || err == nil {
			t.Fatalf("got (%d, %v); want (0, error)", n, err)
		}
	}
}
