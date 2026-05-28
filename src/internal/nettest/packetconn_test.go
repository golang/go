// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest_test

import (
	"bytes"
	"errors"
	"internal/nettest"
	"io"
	"net"
	"net/netip"
	"os"
	"testing"
	"testing/synctest"
	"time"
)

func TestPacketConnListenConflict(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		addr := net.UDPAddrFromAddrPort(netip.MustParseAddrPort("10.0.0.1:1000"))
		pnet := nettest.NewPacketNet()
		conn, err := pnet.NewConn(addr)
		if err != nil {
			t.Fatalf("with no existing listener, pnet.NewConn(%v) = %v; want success", addr, err)
		}
		_, err = pnet.NewConn(addr)
		if err == nil {
			t.Fatalf("with existing listener, pnet.NewConn(%v) = nil; want error", addr)
		}
		conn.Close()
		_, err = pnet.NewConn(addr)
		if err != nil {
			t.Fatalf("after closing existing listener, pnet.NewConn(%v) = %v; want success", addr, err)
		}
	})
}

func TestPacketConnReadWrite(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		pnet := nettest.NewPacketNet()
		c1 := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		c2 := mustNewPacketConn(t, pnet, "10.0.0.2:2000")
		c3 := mustNewPacketConn(t, pnet, "10.0.0.3:3000")

		wantPacketConnWriteTo(t, c1, []byte("1->3"), c3.LocalAddr())
		wantPacketConnWriteTo(t, c2, []byte("2->3"), c3.LocalAddr())
		wantPacketConnWriteTo(t, c3, []byte("3->1"), c1.LocalAddr())

		wantPacketConnReadBytes(t, c1, []byte("3->1"), c3.LocalAddr())
		wantPacketConnReadBytes(t, c3, []byte("1->3"), c1.LocalAddr())
		wantPacketConnReadBytes(t, c3, []byte("2->3"), c2.LocalAddr())
		wantPacketConnReadBlocked(t, c1)
		wantPacketConnReadBlocked(t, c2)
		wantPacketConnReadBlocked(t, c3)

		// Write a packet into the void (no listener on this address).
		wantPacketConnWriteTo(t, c1, []byte("1->lost"), net.UDPAddrFromAddrPort(netip.MustParseAddrPort("10.0.0.100:1000")))
	})
}

func TestPacketConnWriteAddressErrors(t *testing.T) {
	t.Skip("TODO: figure out if these should be errors")

	synctest.Test(t, func(t *testing.T) {
		pnet := nettest.NewPacketNet()
		c4 := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		c6 := mustNewPacketConn(t, pnet, "[::1]:1000")

		wantPacketConnWriteErr(t, c4, c6.LocalAddr(), anyError) // IPv4 -> IPv6
		wantPacketConnWriteErr(t, c6, c4.LocalAddr(), anyError) // IPv6 -> IPv4

		// Not a *net.UDPAddr.
		wantPacketConnWriteErr(t, c4, net.UDPAddrFromAddrPort(netip.MustParseAddrPort("10.0.0.1:1000")), anyError)
	})
}

func TestPacketConnClose(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		pnet := nettest.NewPacketNet()
		pconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")

		wantPacketConnWriteTo(t, pconn, []byte("hello"), pconn.LocalAddr())

		if err := pconn.Close(); err != nil {
			t.Errorf("pconn.Close() = %v, want success", err)
		}
		if err := pconn.Close(); !isOpError(err, net.ErrClosed) {
			t.Errorf("pconn.Close() = %v, want ErrClosed", err)
		}

		wantPacketConnReadErr(t, pconn, net.ErrClosed)
		wantPacketConnWriteErr(t, pconn, pconn.LocalAddr(), net.ErrClosed)
	})
}

func TestPacketConnReadDeadline(t *testing.T) {
	for _, setDeadline := range []struct {
		name string
		f    func(*nettest.PacketConn, time.Time) error
	}{{
		name: "SetDeadline",
		f:    (*nettest.PacketConn).SetDeadline,
	}, {
		name: "SetReadDeadline",
		f:    (*nettest.PacketConn).SetReadDeadline,
	}} {
		t.Run(setDeadline.name, func(t *testing.T) {
			testDeadline(t, func() deadlineTest {
				pnet := nettest.NewPacketNet()
				rconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
				wconn := mustNewPacketConn(t, pnet, "10.0.0.2:2000")
				return deadlineTest{
					what: "ReadFrom()",
					block: func() error {
						_, _, err := rconn.ReadFrom(make([]byte, 1))
						return err
					},
					unblock: func() {
						wconn.WriteTo([]byte("x"), rconn.LocalAddr())
					},
					setDeadline: func(d time.Duration) {
						setDeadline.f(rconn, time.Now().Add(d))
					},
				}
			})
		})
	}
}

func TestPacketConnWriteDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		pnet := nettest.NewPacketNet()
		rconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		wconn := mustNewPacketConn(t, pnet, "10.0.0.2:2000")

		// This does nothing, even though the deadline has expired.
		wconn.SetWriteDeadline(time.Now().Add(-1 * time.Second))

		wantPacketConnWriteTo(t, wconn, []byte("data"), rconn.LocalAddr())
		wantPacketConnReadBytes(t, rconn, []byte("data"), wconn.LocalAddr())
	})
}

func TestPacketConnSetReadError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		wantErr := errors.New("error")
		pnet := nettest.NewPacketNet()
		rconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		wconn := mustNewPacketConn(t, pnet, "10.0.0.2:2000")
		rconn.SetReadError(wantErr)

		// Consume buffer before returning error.
		wantPacketConnWriteTo(t, wconn, []byte("one"), rconn.LocalAddr())
		wantPacketConnReadBytes(t, rconn, []byte("one"), wconn.LocalAddr())
		wantPacketConnReadErr(t, rconn, wantErr)

		// Write more, suppressing error until buffer drains again.
		wantPacketConnWriteTo(t, wconn, []byte("two"), rconn.LocalAddr())
		wantPacketConnReadBytes(t, rconn, []byte("two"), wconn.LocalAddr())
		wantPacketConnReadErr(t, rconn, wantErr)

		// Error may be cleared.
		rconn.SetReadError(nil)
		wantPacketConnReadBlocked(t, rconn)

		// ErrClosed takes precedence over read error.
		rconn.SetReadError(wantErr)
		rconn.Close()
		wantPacketConnReadErr(t, rconn, net.ErrClosed)
	})
}

func TestPacketConnSetWriteError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		wantErr := errors.New("error")
		pnet := nettest.NewPacketNet()
		rconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		wconn := mustNewPacketConn(t, pnet, "10.0.0.2:2000")
		wconn.SetWriteError(wantErr)

		// Error blocks writes.
		wantPacketConnWriteErr(t, wconn, rconn.LocalAddr(), wantErr)
		wantPacketConnReadBlocked(t, rconn)

		// Error may be cleared.
		wconn.SetWriteError(nil)
		wantPacketConnWriteTo(t, wconn, []byte("one"), rconn.LocalAddr())

		// Restoring error does not prevent reading buffered data.
		wconn.SetWriteError(wantErr)
		wantPacketConnWriteErr(t, wconn, rconn.LocalAddr(), wantErr)
		wantPacketConnReadBytes(t, rconn, []byte("one"), wconn.LocalAddr())

		// Error does not interfere with closing the conn.
		wconn.Close()
		wantPacketConnWriteErr(t, wconn, rconn.LocalAddr(), net.ErrClosed)
	})
}

func TestPacketConnSetCloseError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		wantErr := errors.New("error")
		pnet := nettest.NewPacketNet()
		rconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		wconn := mustNewPacketConn(t, pnet, "10.0.0.2:2000")
		wconn.SetCloseError(wantErr)

		wantPacketConnWriteTo(t, wconn, []byte("one"), rconn.LocalAddr())
		if err := wconn.Close(); !isOpError(err, wantErr) {
			t.Fatalf("wconn.Close = %v, want OpError{Err: %v}", err, wantErr)
		}
		if err := wconn.Close(); !isOpError(err, net.ErrClosed) {
			t.Fatalf("wconn.Close = %v, want OpError{Err: net.ErrClosed}", err)
		}
		wantPacketConnReadBytes(t, rconn, []byte("one"), wconn.LocalAddr())
	})
}

func TestPacketConnCanRead(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		pnet := nettest.NewPacketNet()
		rconn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		wconn := mustNewPacketConn(t, pnet, "10.0.0.2:2000")
		if got, want := rconn.CanRead(), false; got != want {
			t.Fatalf("before writing data: rconn.CanRead() = %v, want %v", got, want)
		}
		wconn.WriteTo([]byte("a"), rconn.LocalAddr())
		if got, want := rconn.CanRead(), true; got != want {
			t.Fatalf("after writing data: rconn.CanRead() = %v, want %v", got, want)
		}
		rconn.ReadFrom(make([]byte, 1))
		if got, want := rconn.CanRead(), false; got != want {
			t.Fatalf("after reading data: rconn.CanRead() = %v, want %v", got, want)
		}
	})
}

func TestPacketConnIsClosed(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		pnet := nettest.NewPacketNet()
		conn := mustNewPacketConn(t, pnet, "10.0.0.1:1000")
		if got, want := conn.IsClosed(), false; got != want {
			t.Fatalf("before closing: conn.IsClosed() = %v, want %v", got, want)
		}
		conn.Close()
		if got, want := conn.IsClosed(), true; got != want {
			t.Fatalf("after closing: conn.IsClosed() = %v, want %v", got, want)
		}
	})
}

func mustNewPacketConn(t *testing.T, pnet *nettest.PacketNet, addr string) *nettest.PacketConn {
	t.Helper()
	c, err := pnet.NewConn(net.UDPAddrFromAddrPort(netip.MustParseAddrPort(addr)))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func wantPacketConnWriteTo(t *testing.T, c *nettest.PacketConn, b []byte, dst net.Addr) {
	t.Helper()
	if n, err := c.WriteTo(b, dst); n != len(b) || err != nil {
		t.Fatalf("conn.WriteTo(%q, %v) = %v, %v; want %v, nil", b, dst, n, err, len(b))
	}
}

func wantPacketConnWriteErr(t *testing.T, c *nettest.PacketConn, dst net.Addr, want error) {
	t.Helper()
	n, err := c.WriteTo(make([]byte, 1), dst)
	if n != 0 || !isOpError(err, want) {
		t.Fatalf("c.WriteTo() = %v, %v; want 0, OpError{Err: %q}", n, err, want)
	}
}

func wantPacketConnReadBytes(t *testing.T, c *nettest.PacketConn, want []byte, wantAddr net.Addr) {
	t.Helper()
	udpWantAddr, ok := wantAddr.(*net.UDPAddr)
	if !ok {
		t.Fatalf("wantAddr is %T, should be *net.UDPAddr", wantAddr)
	}
	got := make([]byte, len(want)+1)
	n, addr, err := c.ReadFrom(got)
	got = got[:n]
	udpAddr, addrOK := addr.(*net.UDPAddr)
	if n != len(want) || !addrOK || udpAddr.AddrPort() != udpWantAddr.AddrPort() {
		t.Fatalf("conn.ReadFrom() = %v, %v, %v; want %v, %v, nil", n, addr, err, len(want), wantAddr)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("conn.ReadFrom() read %q, want %q", got, want)
	}
}

func wantPacketConnReadErr(t *testing.T, c *nettest.PacketConn, want error) {
	t.Helper()
	n, addr, err := c.ReadFrom(make([]byte, 1))
	if want == io.EOF {
		if n != 0 || err != io.EOF {
			t.Fatalf("c.ReadFrom() = %v, %v, %v; want 0, nil, io.EOF", n, addr, err)
		}
	} else {
		if n != 0 || !isOpError(err, want) {
			t.Fatalf("c.ReadFrom() = %v, %v, %v; want 0, nil, OpError{Err: %q}", n, addr, err, want)
		}
	}
}

func wantPacketConnReadBlocked(t *testing.T, c *nettest.PacketConn) {
	done := false
	go func() {
		n, addr, err := c.ReadFrom(make([]byte, 1))
		if n != 0 || !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("c.Read() = %v, %v, %v; want 0, nil, ErrDeadlineExceeded", n, addr, err)
		}
		done = true
	}()
	synctest.Wait()
	if done {
		t.Fatalf("ReadFrom unexpectedly returned before setting deadline")
	}
	c.SetReadDeadline(time.Now().Add(-1 * time.Second))
	synctest.Wait()
	c.SetReadDeadline(time.Time{})
	if !done {
		t.Fatalf("ReadFrom unexpectedly did not return after setting deadline")
	}
}
