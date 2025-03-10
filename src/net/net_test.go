// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"fmt"
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
	t.Parallel()

	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		network := network
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("network %s is not testable on the current platform", network)
			}
			t.Parallel()

			ln := newLocalListener(t, network)
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
		})
	}
}

func TestCloseWrite(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	t.Parallel()
	deadline, _ := t.Deadline()
	if !deadline.IsZero() {
		// Leave 10% headroom on the deadline to report errors and clean up.
		deadline = deadline.Add(-time.Until(deadline) / 10)
	}

	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		network := network
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("network %s is not testable on the current platform", network)
			}
			t.Parallel()

			handler := func(ls *localServer, ln Listener) {
				c, err := ln.Accept()
				if err != nil {
					t.Error(err)
					return
				}

				// Workaround for https://go.dev/issue/49352.
				// On arm64 macOS (current as of macOS 12.4),
				// reading from a socket at the same time as the client
				// is closing it occasionally hangs for 60 seconds before
				// returning ECONNRESET. Sleep for a bit to give the
				// socket time to close before trying to read from it.
				if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
					time.Sleep(10 * time.Millisecond)
				}

				if !deadline.IsZero() {
					c.SetDeadline(deadline)
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

			ls := newLocalServer(t, network)
			defer ls.teardown()
			if err := ls.buildup(handler); err != nil {
				t.Fatal(err)
			}

			c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
			if err != nil {
				t.Fatal(err)
			}
			if !deadline.IsZero() {
				c.SetDeadline(deadline)
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
		})
	}
}

func TestConnClose(t *testing.T) {
	t.Parallel()
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		network := network
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("network %s is not testable on the current platform", network)
			}
			t.Parallel()

			ln := newLocalListener(t, network)
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
		})
	}
}

func TestListenerClose(t *testing.T) {
	t.Parallel()
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		network := network
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("network %s is not testable on the current platform", network)
			}
			t.Parallel()

			ln := newLocalListener(t, network)
			switch network {
			case "unix", "unixpacket":
				defer os.Remove(ln.Addr().String())
			}

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

			// Note: we cannot ensure that a subsequent Dial does not succeed, because
			// we do not in general have any guarantee that ln.Addr is not immediately
			// reused. (TCP sockets enter a TIME_WAIT state when closed, but that only
			// applies to existing connections for the port — it does not prevent the
			// port itself from being used for entirely new connections in the
			// meantime.)
		})
	}
}

func TestPacketConnClose(t *testing.T) {
	t.Parallel()
	for _, network := range []string{"udp", "unixgram"} {
		network := network
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("network %s is not testable on the current platform", network)
			}
			t.Parallel()

			c := newLocalPacketListener(t, network)
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
		})
	}
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
	ls := newLocalServer(t, "tcp")
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
	t.Parallel()
	for _, network := range []string{"tcp", "unix", "unixpacket"} {
		network := network
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("network %s is not testable on the current platform", network)
			}
			t.Parallel()

			ln := newLocalListener(t, network)
			connc := make(chan Conn, 1)
			defer func() {
				ln.Close()
				for c := range connc {
					if c != nil {
						c.Close()
					}
				}
			}()
			go func() {
				defer close(connc)
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
				return
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
		})
	}
}

// withTCPConnPair sets up a TCP connection between two peers, then
// runs peer1 and peer2 concurrently. withTCPConnPair returns when
// both have completed.
func withTCPConnPair(t *testing.T, peer1, peer2 func(c *TCPConn) error) {
	t.Helper()
	ln := newLocalListener(t, "tcp")
	defer ln.Close()
	errc := make(chan error, 2)
	go func() {
		c1, err := ln.Accept()
		if err != nil {
			errc <- err
			return
		}
		err = peer1(c1.(*TCPConn))
		c1.Close()
		errc <- err
	}()
	go func() {
		c2, err := Dial("tcp", ln.Addr().String())
		if err != nil {
			errc <- err
			return
		}
		err = peer2(c2.(*TCPConn))
		c2.Close()
		errc <- err
	}()
	for i := 0; i < 2; i++ {
		if err := <-errc; err != nil {
			t.Error(err)
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

// Issue 72770: verify that a blocked UDP read is woken up by a Close.
func TestCloseUnblocksReadUDP(t *testing.T) {
	t.Parallel()
	pc, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	time.AfterFunc(250*time.Millisecond, func() {
		t.Logf("closing conn...")
		pc.Close()
	})
	timer := time.AfterFunc(time.Second*10, func() {
		panic("timeout waiting for Close")
	})
	defer timer.Stop()

	n, src, err := pc.(*UDPConn).ReadFromUDPAddrPort([]byte{})

	// Check for n > 0. Checking err == nil alone isn't enough;
	// on macOS, it returns (n=0, src=0.0.0.0:0, err=nil).
	if n > 0 {
		t.Fatalf("unexpected Read success from ReadFromUDPAddrPort; read %d bytes from %v, err=%v", n, src, err)
	}
	t.Logf("got expected UDP read error")
}

// Issue 24808: verify that ECONNRESET is not temporary for read.
func TestNotTemporaryRead(t *testing.T) {
	t.Parallel()

	ln := newLocalListener(t, "tcp")
	serverDone := make(chan struct{})
	dialed := make(chan struct{})
	go func() {
		defer close(serverDone)

		cs, err := ln.Accept()
		if err != nil {
			return
		}
		<-dialed
		cs.(*TCPConn).SetLinger(0)
		cs.Close()
	}()
	defer func() {
		ln.Close()
		<-serverDone
	}()

	ss, err := Dial("tcp", ln.Addr().String())
	close(dialed)
	if err != nil {
		t.Fatal(err)
	}
	defer ss.Close()

	_, err = ss.Read([]byte{0})
	if err == nil {
		t.Fatal("Read succeeded unexpectedly")
	} else if err == io.EOF {
		// This happens on Plan 9, but for some reason (prior to CL 385314) it was
		// accepted everywhere else too.
		if runtime.GOOS == "plan9" {
			return
		}
		t.Fatal("Read unexpectedly returned io.EOF after socket was abruptly closed")
	}
	if ne, ok := err.(Error); !ok {
		t.Errorf("Read error does not implement net.Error: %v", err)
	} else if ne.Temporary() {
		t.Errorf("Read error is unexpectedly temporary: %v", err)
	}
}

// The various errors should implement the Error interface.
func TestErrors(t *testing.T) {
	var (
		_ Error = &OpError{}
		_ Error = &ParseError{}
		_ Error = &AddrError{}
		_ Error = UnknownNetworkError("")
		_ Error = InvalidAddrError("")
		_ Error = &timeoutError{}
		_ Error = &DNSConfigError{}
		_ Error = &DNSError{}
	)

	// ErrClosed was introduced as type error, so we can't check
	// it using a declaration.
	if _, ok := ErrClosed.(Error); !ok {
		t.Fatal("ErrClosed does not implement Error")
	}
}
