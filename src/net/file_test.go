// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
)

// The full stack test cases for IPConn have been moved to the
// following:
//      golang.org/x/net/ipv4
//      golang.org/x/net/ipv6
//      golang.org/x/net/icmp

var fileConnTests = []struct {
	network string
}{
	{"tcp"},
	{"udp"},
	{"unix"},
	{"unixpacket"},
}

func TestFileConn(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows", "js", "wasip1":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range fileConnTests {
		if !testableNetwork(tt.network) {
			t.Logf("skipping %s test", tt.network)
			continue
		}

		var network, address string
		switch tt.network {
		case "udp":
			c := newLocalPacketListener(t, tt.network)
			defer c.Close()
			network = c.LocalAddr().Network()
			address = c.LocalAddr().String()
		default:
			handler := func(ls *localServer, ln Listener) {
				c, err := ln.Accept()
				if err != nil {
					return
				}
				defer c.Close()
				var b [1]byte
				c.Read(b[:])
			}
			ls := newLocalServer(t, tt.network)
			defer ls.teardown()
			if err := ls.buildup(handler); err != nil {
				t.Fatal(err)
			}
			network = ls.Listener.Addr().Network()
			address = ls.Listener.Addr().String()
		}

		c1, err := Dial(network, address)
		if err != nil {
			if perr := parseDialError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		addr := c1.LocalAddr()

		var f *os.File
		switch c1 := c1.(type) {
		case *TCPConn:
			f, err = c1.File()
		case *UDPConn:
			f, err = c1.File()
		case *UnixConn:
			f, err = c1.File()
		}
		if err := c1.Close(); err != nil {
			if perr := parseCloseError(err, false); perr != nil {
				t.Error(perr)
			}
			t.Error(err)
		}
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}

		c2, err := FileConn(f)
		if err := f.Close(); err != nil {
			t.Error(err)
		}
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		defer c2.Close()

		if _, err := c2.Write([]byte("FILECONN TEST")); err != nil {
			if perr := parseWriteError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		if !reflect.DeepEqual(c2.LocalAddr(), addr) {
			t.Fatalf("got %#v; want %#v", c2.LocalAddr(), addr)
		}
	}
}

var fileListenerTests = []struct {
	network string
}{
	{"tcp"},
	{"unix"},
	{"unixpacket"},
}

func TestFileListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows", "js", "wasip1":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range fileListenerTests {
		if !testableNetwork(tt.network) {
			t.Logf("skipping %s test", tt.network)
			continue
		}

		ln1 := newLocalListener(t, tt.network)
		switch tt.network {
		case "unix", "unixpacket":
			defer os.Remove(ln1.Addr().String())
		}
		addr := ln1.Addr()

		var (
			f   *os.File
			err error
		)
		switch ln1 := ln1.(type) {
		case *TCPListener:
			f, err = ln1.File()
		case *UnixListener:
			f, err = ln1.File()
		}
		switch tt.network {
		case "unix", "unixpacket":
			defer ln1.Close() // UnixListener.Close calls syscall.Unlink internally
		default:
			if err := ln1.Close(); err != nil {
				t.Error(err)
			}
		}
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}

		ln2, err := FileListener(f)
		if err := f.Close(); err != nil {
			t.Error(err)
		}
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		defer ln2.Close()

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := Dial(ln2.Addr().Network(), ln2.Addr().String())
			if err != nil {
				if perr := parseDialError(err); perr != nil {
					t.Error(perr)
				}
				t.Error(err)
				return
			}
			c.Close()
		}()
		c, err := ln2.Accept()
		if err != nil {
			if perr := parseAcceptError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		c.Close()
		wg.Wait()
		if !reflect.DeepEqual(ln2.Addr(), addr) {
			t.Fatalf("got %#v; want %#v", ln2.Addr(), addr)
		}
	}
}

var filePacketConnTests = []struct {
	network string
}{
	{"udp"},
	{"unixgram"},
}

func TestFilePacketConn(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows", "js", "wasip1":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range filePacketConnTests {
		if !testableNetwork(tt.network) {
			t.Logf("skipping %s test", tt.network)
			continue
		}

		c1 := newLocalPacketListener(t, tt.network)
		switch tt.network {
		case "unixgram":
			defer os.Remove(c1.LocalAddr().String())
		}
		addr := c1.LocalAddr()

		var (
			f   *os.File
			err error
		)
		switch c1 := c1.(type) {
		case *UDPConn:
			f, err = c1.File()
		case *UnixConn:
			f, err = c1.File()
		}
		if err := c1.Close(); err != nil {
			if perr := parseCloseError(err, false); perr != nil {
				t.Error(perr)
			}
			t.Error(err)
		}
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}

		c2, err := FilePacketConn(f)
		if err := f.Close(); err != nil {
			t.Error(err)
		}
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		defer c2.Close()

		if _, err := c2.WriteTo([]byte("FILEPACKETCONN TEST"), addr); err != nil {
			if perr := parseWriteError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		if !reflect.DeepEqual(c2.LocalAddr(), addr) {
			t.Fatalf("got %#v; want %#v", c2.LocalAddr(), addr)
		}
	}
}

// Issue 24483.
func TestFileCloseRace(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows", "js", "wasip1":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !testableNetwork("tcp") {
		t.Skip("tcp not supported")
	}

	handler := func(ls *localServer, ln Listener) {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		var b [1]byte
		c.Read(b[:])
	}

	ls := newLocalServer(t, "tcp")
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	const tries = 100
	for i := 0; i < tries; i++ {
		c1, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		tc := c1.(*TCPConn)

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			f, err := tc.File()
			if err == nil {
				f.Close()
			}
		}()
		go func() {
			defer wg.Done()
			c1.Close()
		}()
		wg.Wait()
	}
}
