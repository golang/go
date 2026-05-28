// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest_test

import (
	"errors"
	"internal/nettest"
	"io"
	"net"
	"net/netip"
	"slices"
	"testing"
	"testing/synctest"
)

func TestListenerNewConn(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		li := nettest.NewListener()
		defer li.Close()

		// Create several connections in parallel.
		want := []string{"a", "b", "c"}
		for i := range len(want) {
			go func() {
				conn := li.NewConn()
				defer conn.Close()
				n, err := conn.Write([]byte(want[i]))
				if n != len(want[i]) || err != nil {
					t.Errorf("conn%v.Write() = %v, %v; want %v, nil", i, n, err, len(want[i]))
				}
			}()
		}

		// Accept the connections in parallel as well.
		got := make([]string, len(want))
		for i := range len(want) {
			go func() {
				conn, err := li.Accept()
				if err != nil {
					t.Errorf("li.Accept() = %v", err)
				}
				b, err := io.ReadAll(conn)
				if err != nil {
					t.Errorf("io.ReadAll(conn%v) = %v", i, err)
				}
				got[i] = string(b)
			}()
		}

		synctest.Wait()
		slices.Sort(got)
		slices.Sort(want)
		if !slices.Equal(got, want) {
			t.Errorf("connections read %v; want %q", got, want)
		}
	})
}

func TestListenerInterruptAccept(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		li := nettest.NewListener()

		var acceptErr error
		go func() {
			_, acceptErr = li.Accept()
		}()

		synctest.Wait()
		if acceptErr != nil {
			t.Fatalf("li.Accept() = %v, want still running before close", acceptErr)
		}

		li.Close()
		synctest.Wait()
		if !errors.Is(acceptErr, net.ErrClosed) {
			t.Fatalf("li.Accept() = %v, want ErrClosed", acceptErr)
		}
	})
}

func TestListenerAddresses(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		srvaddr := netip.MustParseAddrPort("10.0.0.1:80")
		cliaddr := netip.MustParseAddrPort("10.0.0.2:1234")

		li := nettest.NewListener()
		defer li.Close()

		li.SetAddr(net.TCPAddrFromAddrPort(srvaddr))
		if got, want := li.Addr().(*net.TCPAddr).AddrPort(), srvaddr; got != want {
			t.Errorf("li.Addr() = %v, want %v", got, want)
		}

		cli := li.NewConnConfig(func(conn *nettest.Conn) {
			conn.SetLocalAddr(net.TCPAddrFromAddrPort(cliaddr))
		})
		srvc, err := li.Accept()
		if err != nil {
			t.Fatalf("li.Accept() = %v", err)
		}
		srv := srvc.(*nettest.Conn)

		if cli.Peer() != srv {
			t.Errorf("cli.Peer() != srv; should be the same")
		}
		if srv.Peer() != cli {
			t.Errorf("cli.Peer() != srv; should be the same")
		}

		if got, want := cli.LocalAddr().(*net.TCPAddr).AddrPort(), cliaddr; got != want {
			t.Errorf("cli.LocalAddr() = %v, want %v", got, want)
		}
		if got, want := cli.RemoteAddr().(*net.TCPAddr).AddrPort(), srvaddr; got != want {
			t.Errorf("cli.LocalAddr() = %v, want %v", got, want)
		}
		if got, want := srv.LocalAddr().(*net.TCPAddr).AddrPort(), srvaddr; got != want {
			t.Errorf("srv.LocalAddr() = %v, want %v", got, want)
		}
		if got, want := srv.RemoteAddr().(*net.TCPAddr).AddrPort(), cliaddr; got != want {
			t.Errorf("cli.LocalAddr() = %v, want %v", got, want)
		}
	})
}

func wantListenerAccept(t *testing.T, li *nettest.Listener, want *nettest.Conn) {
	t.Helper()
	got, err := li.Accept()
	if err != nil {
		t.Fatalf("li.Accept() = %v, want conn", err)
	}
	if got != want {
		t.Fatalf("li.Accept() returned unexpected conn")
	}
}

func wantListenerAcceptErr(t *testing.T, li *nettest.Listener, want error) {
	t.Helper()
	got, err := li.Accept()
	if got != nil || !isOpError(err, want) {
		t.Fatalf("li.Accept() = %p, %v; want nil, OpError{Err: %q}", got, err, want)
	}
}

func wantListenerAcceptBlocked(t *testing.T, li *nettest.Listener) {
	cancelErr := errors.New("cancel")
	done := false
	go func() {
		got, err := li.Accept()
		if got != nil || !errors.Is(err, cancelErr) {
			t.Errorf("li.Accept = %p, %v; want nil, cancelErr", got, err)
		}
		done = true
	}()
	synctest.Wait()
	if done {
		t.Fatalf("Accept unexpectedly returned before canceling")
	}
	li.SetAcceptError(cancelErr)
	synctest.Wait()
	li.SetAcceptError(nil)
	if !done {
		t.Fatalf("Accept unexpectedly did not return after canceling")
	}
}

func TestListenerSetAcceptError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		acceptErr := errors.New("accept error")
		li := nettest.NewListener()
		defer li.Close()
		li.SetAcceptError(acceptErr)

		// Accept conns from queue before returning error.
		c1 := li.NewConn()
		wantListenerAccept(t, li, c1.Peer())
		wantListenerAcceptErr(t, li, acceptErr)

		// Add a new conn, suppressing error until the queue is empty.
		c2 := li.NewConn()
		wantListenerAccept(t, li, c2.Peer())
		wantListenerAcceptErr(t, li, acceptErr)

		// Error may be cleared.
		li.SetAcceptError(nil)
		wantListenerAcceptBlocked(t, li)

		// ErrClosed takes precedence over accept error.
		li.SetAcceptError(acceptErr)
		li.Close()
		wantListenerAcceptErr(t, li, net.ErrClosed)
	})
}

func TestListenerSetCloseError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		li := nettest.NewListener()
		closeErr := errors.New("close error")
		li.SetCloseError(closeErr)

		// First close uses the user-provided error.
		if err := li.Close(); !isOpError(err, closeErr) {
			t.Fatalf("li.Close() = %v; want OpError wrapping accept error", err)
		}

		// Repeated closes return ErrClosed.
		if err := li.Close(); !isOpError(err, net.ErrClosed) {
			t.Fatalf("li.Close() = %v; want OpError wrapping net.ErrClosed", err)
		}
	})
}
