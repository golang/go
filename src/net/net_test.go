// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !js

package net

import (
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"net/internal/socktest"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestCloseRead(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
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
			if perr := parseCloseError(err, true); perr != nil {
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
			if perr := parseCloseError(err, true); perr != nil {
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
			if perr := parseCloseError(err, true); perr != nil {
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
			if perr := parseCloseError(err, false); perr != nil {
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

		dst := ln.Addr().String()
		if err := ln.Close(); err != nil {
			if perr := parseCloseError(err, false); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		c, err := ln.Accept()
		if err == nil {
			c.Close()
			t.Fatal("should fail")
		}

		if network == "tcp" {
			// We will have two TCP FSMs inside the
			// kernel here. There's no guarantee that a
			// signal comes from the far end FSM will be
			// delivered immediately to the near end FSM,
			// especially on the platforms that allow
			// multiple consumer threads to pull pending
			// established connections at the same time by
			// enabling SO_REUSEPORT option such as Linux,
			// DragonFly BSD. So we need to give some time
			// quantum to the kernel.
			//
			// Note that net.inet.tcp.reuseport_ext=1 by
			// default on DragonFly BSD.
			time.Sleep(time.Millisecond)

			cc, err := Dial("tcp", dst)
			if err == nil {
				t.Error("Dial to closed TCP listener succeeded.")
				cc.Close()
			}
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
			if perr := parseCloseError(err, false); perr != nil {
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

// nacl was previous failing to reuse an address.
func TestListenCloseListen(t *testing.T) {
	const maxTries = 10
	for tries := 0; tries < maxTries; tries++ {
		ln, err := newLocalListener("tcp")
		if err != nil {
			t.Fatal(err)
		}
		addr := ln.Addr().String()
		if err := ln.Close(); err != nil {
			if perr := parseCloseError(err, false); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}
		ln, err = Listen("tcp", addr)
		if err == nil {
			// Success. nacl couldn't do this before.
			ln.Close()
			return
		}
		t.Errorf("failed on try %d/%d: %v", tries+1, maxTries, err)
	}
	t.Fatalf("failed to listen/close/listen on same address after %d tries", maxTries)
}

// See golang.org/issue/6163, golang.org/issue/6987.
func TestAcceptIgnoreAbortedConnRequest(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	}

	syserr := make(chan error)
	go func() {
		defer close(syserr)
		for _, err := range abortedConnRequestErrors {
			syserr <- err
		}
	}()
	sw.Set(socktest.FilterAccept, func(so *socktest.Status) (socktest.AfterFilter, error) {
		if err, ok := <-syserr; ok {
			return nil, err
		}
		return nil, nil
	})
	defer sw.Set(socktest.FilterAccept, nil)

	operr := make(chan error, 1)
	handler := func(ls *localServer, ln Listener) {
		defer close(operr)
		c, err := ln.Accept()
		if err != nil {
			if perr := parseAcceptError(err); perr != nil {
				operr <- perr
			}
			operr <- err
			return
		}
		c.Close()
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
	c.Close()

	for err := range operr {
		t.Error(err)
	}
}

func TestZeroByteRead(t *testing.T) {
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		if !testableNetwork(network) {
			t.Logf("skipping %s test", network)
			continue
		}

		ln, err := newLocalListener(network)
		if err != nil {
			t.Fatal(err)
		}
		connc := make(chan Conn, 1)
		go func() {
			defer ln.Close()
			c, err := ln.Accept()
			if err != nil {
				t.Error(err)
			}
			connc <- c // might be nil
		}()
		c, err := Dial(network, ln.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		sc := <-connc
		if sc == nil {
			continue
		}
		defer sc.Close()

		if runtime.GOOS == "windows" {
			// A zero byte read on Windows caused a wait for readability first.
			// Rather than change that behavior, satisfy it in this test.
			// See Issue 15735.
			go io.WriteString(sc, "a")
		}

		n, err := c.Read(nil)
		if n != 0 || err != nil {
			t.Errorf("%s: zero byte client read = %v, %v; want 0, nil", network, n, err)
		}

		if runtime.GOOS == "windows" {
			// Same as comment above.
			go io.WriteString(c, "a")
		}
		n, err = sc.Read(nil)
		if n != 0 || err != nil {
			t.Errorf("%s: zero byte server read = %v, %v; want 0, nil", network, n, err)
		}
	}
}

// withTCPConnPair sets up a TCP connection between two peers, then
// runs peer1 and peer2 concurrently. withTCPConnPair returns when
// both have completed.
func withTCPConnPair(t *testing.T, peer1, peer2 func(c *TCPConn) error) {
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	errc := make(chan error, 2)
	go func() {
		c1, err := ln.Accept()
		if err != nil {
			errc <- err
			return
		}
		defer c1.Close()
		errc <- peer1(c1.(*TCPConn))
	}()
	go func() {
		c2, err := Dial("tcp", ln.Addr().String())
		if err != nil {
			errc <- err
			return
		}
		defer c2.Close()
		errc <- peer2(c2.(*TCPConn))
	}()
	for i := 0; i < 2; i++ {
		if err := <-errc; err != nil {
			t.Fatal(err)
		}
	}
}

// Tests that a blocked Read is interrupted by a concurrent SetReadDeadline
// modifying that Conn's read deadline to the past.
// See golang.org/cl/30164 which documented this. The net/http package
// depends on this.
func TestReadTimeoutUnblocksRead(t *testing.T) {
	serverDone := make(chan struct{})
	server := func(cs *TCPConn) error {
		defer close(serverDone)
		errc := make(chan error, 1)
		go func() {
			defer close(errc)
			go func() {
				// TODO: find a better way to wait
				// until we're blocked in the cs.Read
				// call below. Sleep is lame.
				time.Sleep(100 * time.Millisecond)

				// Interrupt the upcoming Read, unblocking it:
				cs.SetReadDeadline(time.Unix(123, 0)) // time in the past
			}()
			var buf [1]byte
			n, err := cs.Read(buf[:1])
			if n != 0 || err == nil {
				errc <- fmt.Errorf("Read = %v, %v; want 0, non-nil", n, err)
			}
		}()
		select {
		case err := <-errc:
			return err
		case <-time.After(5 * time.Second):
			buf := make([]byte, 2<<20)
			buf = buf[:runtime.Stack(buf, true)]
			println("Stacks at timeout:\n", string(buf))
			return errors.New("timeout waiting for Read to finish")
		}

	}
	// Do nothing in the client. Never write. Just wait for the
	// server's half to be done.
	client := func(*TCPConn) error {
		<-serverDone
		return nil
	}
	withTCPConnPair(t, client, server)
}

// Issue 17695: verify that a blocked Read is woken up by a Close.
func TestCloseUnblocksRead(t *testing.T) {
	t.Parallel()
	server := func(cs *TCPConn) error {
		// Give the client time to get stuck in a Read:
		time.Sleep(20 * time.Millisecond)
		cs.Close()
		return nil
	}
	client := func(ss *TCPConn) error {
		n, err := ss.Read([]byte{0})
		if n != 0 || err != io.EOF {
			return fmt.Errorf("Read = %v, %v; want 0, EOF", n, err)
		}
		return nil
	}
	withTCPConnPair(t, client, server)
}

// Issue 24808: verify that ECONNRESET is not temporary for read.
func TestNotTemporaryRead(t *testing.T) {
	if runtime.GOOS == "freebsd" {
		testenv.SkipFlaky(t, 25289)
	}
	t.Parallel()
	server := func(cs *TCPConn) error {
		cs.SetLinger(0)
		// Give the client time to get stuck in a Read.
		time.Sleep(50 * time.Millisecond)
		cs.Close()
		return nil
	}
	client := func(ss *TCPConn) error {
		_, err := ss.Read([]byte{0})
		if err == nil {
			return errors.New("Read succeeded unexpectedly")
		} else if err == io.EOF {
			// This happens on NaCl and Plan 9.
			return nil
		} else if ne, ok := err.(Error); !ok {
			return fmt.Errorf("unexpected error %v", err)
		} else if ne.Temporary() {
			return fmt.Errorf("unexpected temporary error %v", err)
		}
		return nil
	}
	withTCPConnPair(t, client, server)
}
