// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/testenv"
	"reflect"
	"runtime"
	"testing"
	"time"
)

func BenchmarkUDP6LinkLocalUnicast(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	if !supportsIPv6 {
		b.Skip("IPv6 is not supported")
	}
	ifi := loopbackInterface()
	if ifi == nil {
		b.Skip("loopback interface not found")
	}
	lla := ipv6LinkLocalUnicastAddr(ifi)
	if lla == "" {
		b.Skip("IPv6 link-local unicast address not found")
	}

	c1, err := ListenPacket("udp6", JoinHostPort(lla+"%"+ifi.Name, "0"))
	if err != nil {
		b.Fatal(err)
	}
	defer c1.Close()
	c2, err := ListenPacket("udp6", JoinHostPort(lla+"%"+ifi.Name, "0"))
	if err != nil {
		b.Fatal(err)
	}
	defer c2.Close()

	var buf [1]byte
	for i := 0; i < b.N; i++ {
		if _, err := c1.WriteTo(buf[:], c2.LocalAddr()); err != nil {
			b.Fatal(err)
		}
		if _, _, err := c2.ReadFrom(buf[:]); err != nil {
			b.Fatal(err)
		}
	}
}

type resolveUDPAddrTest struct {
	network       string
	litAddrOrName string
	addr          *UDPAddr
	err           error
}

var resolveUDPAddrTests = []resolveUDPAddrTest{
	{"udp", "127.0.0.1:0", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil},
	{"udp4", "127.0.0.1:65535", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 65535}, nil},

	{"udp", "[::1]:0", &UDPAddr{IP: ParseIP("::1"), Port: 0}, nil},
	{"udp6", "[::1]:65535", &UDPAddr{IP: ParseIP("::1"), Port: 65535}, nil},

	{"udp", "[::1%en0]:1", &UDPAddr{IP: ParseIP("::1"), Port: 1, Zone: "en0"}, nil},
	{"udp6", "[::1%911]:2", &UDPAddr{IP: ParseIP("::1"), Port: 2, Zone: "911"}, nil},

	{"", "127.0.0.1:0", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil}, // Go 1.0 behavior
	{"", "[::1]:0", &UDPAddr{IP: ParseIP("::1"), Port: 0}, nil},         // Go 1.0 behavior

	{"udp", ":12345", &UDPAddr{Port: 12345}, nil},

	{"http", "127.0.0.1:0", nil, UnknownNetworkError("http")},

	{"udp", "127.0.0.1:domain", &UDPAddr{IP: ParseIP("127.0.0.1"), Port: 53}, nil},
	{"udp", "[::ffff:127.0.0.1]:domain", &UDPAddr{IP: ParseIP("::ffff:127.0.0.1"), Port: 53}, nil},
	{"udp", "[2001:db8::1]:domain", &UDPAddr{IP: ParseIP("2001:db8::1"), Port: 53}, nil},
	{"udp4", "127.0.0.1:domain", &UDPAddr{IP: ParseIP("127.0.0.1"), Port: 53}, nil},
	{"udp4", "[::ffff:127.0.0.1]:domain", &UDPAddr{IP: ParseIP("127.0.0.1"), Port: 53}, nil},
	{"udp6", "[2001:db8::1]:domain", &UDPAddr{IP: ParseIP("2001:db8::1"), Port: 53}, nil},

	{"udp4", "[2001:db8::1]:domain", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "2001:db8::1"}},
	{"udp6", "127.0.0.1:domain", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "127.0.0.1"}},
	{"udp6", "[::ffff:127.0.0.1]:domain", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "::ffff:127.0.0.1"}},
}

func TestResolveUDPAddr(t *testing.T) {
	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupLocalhost

	for _, tt := range resolveUDPAddrTests {
		addr, err := ResolveUDPAddr(tt.network, tt.litAddrOrName)
		if !reflect.DeepEqual(addr, tt.addr) || !reflect.DeepEqual(err, tt.err) {
			t.Errorf("ResolveUDPAddr(%q, %q) = %#v, %v, want %#v, %v", tt.network, tt.litAddrOrName, addr, err, tt.addr, tt.err)
			continue
		}
		if err == nil {
			addr2, err := ResolveUDPAddr(addr.Network(), addr.String())
			if !reflect.DeepEqual(addr2, tt.addr) || err != tt.err {
				t.Errorf("(%q, %q): ResolveUDPAddr(%q, %q) = %#v, %v, want %#v, %v", tt.network, tt.litAddrOrName, addr.Network(), addr.String(), addr2, err, tt.addr, tt.err)
			}
		}
	}
}

func TestWriteToUDP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	c, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	testWriteToConn(t, c.LocalAddr().String())
	testWriteToPacketConn(t, c.LocalAddr().String())
}

func testWriteToConn(t *testing.T, raddr string) {
	c, err := Dial("udp", raddr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	ra, err := ResolveUDPAddr("udp", raddr)
	if err != nil {
		t.Fatal(err)
	}

	b := []byte("CONNECTED-MODE SOCKET")
	_, err = c.(*UDPConn).WriteToUDP(b, ra)
	if err == nil {
		t.Fatal("should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("should fail as ErrWriteToConnected: %v", err)
	}
	_, err = c.(*UDPConn).WriteTo(b, ra)
	if err == nil {
		t.Fatal("should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("should fail as ErrWriteToConnected: %v", err)
	}
	_, err = c.Write(b)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = c.(*UDPConn).WriteMsgUDP(b, nil, ra)
	if err == nil {
		t.Fatal("should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("should fail as ErrWriteToConnected: %v", err)
	}
	_, _, err = c.(*UDPConn).WriteMsgUDP(b, nil, nil)
	switch runtime.GOOS {
	case "nacl", "windows": // see golang.org/issue/9252
		t.Skipf("not implemented yet on %s", runtime.GOOS)
	default:
		if err != nil {
			t.Fatal(err)
		}
	}
}

func testWriteToPacketConn(t *testing.T, raddr string) {
	c, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	ra, err := ResolveUDPAddr("udp", raddr)
	if err != nil {
		t.Fatal(err)
	}

	b := []byte("UNCONNECTED-MODE SOCKET")
	_, err = c.(*UDPConn).WriteToUDP(b, ra)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.WriteTo(b, ra)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.(*UDPConn).Write(b)
	if err == nil {
		t.Fatal("should fail")
	}
	_, _, err = c.(*UDPConn).WriteMsgUDP(b, nil, nil)
	if err == nil {
		t.Fatal("should fail")
	}
	if err != nil && err.(*OpError).Err != errMissingAddress {
		t.Fatalf("should fail as errMissingAddress: %v", err)
	}
	_, _, err = c.(*UDPConn).WriteMsgUDP(b, nil, ra)
	switch runtime.GOOS {
	case "nacl", "windows": // see golang.org/issue/9252
		t.Skipf("not implemented yet on %s", runtime.GOOS)
	default:
		if err != nil {
			t.Fatal(err)
		}
	}
}

var udpConnLocalNameTests = []struct {
	net   string
	laddr *UDPAddr
}{
	{"udp4", &UDPAddr{IP: IPv4(127, 0, 0, 1)}},
	{"udp4", &UDPAddr{}},
	{"udp4", nil},
}

func TestUDPConnLocalName(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	for _, tt := range udpConnLocalNameTests {
		c, err := ListenUDP(tt.net, tt.laddr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		la := c.LocalAddr()
		if a, ok := la.(*UDPAddr); !ok || a.Port == 0 {
			t.Fatalf("got %v; expected a proper address with non-zero port number", la)
		}
	}
}

func TestUDPConnLocalAndRemoteNames(t *testing.T) {
	for _, laddr := range []string{"", "127.0.0.1:0"} {
		c1, err := ListenPacket("udp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer c1.Close()

		var la *UDPAddr
		if laddr != "" {
			var err error
			if la, err = ResolveUDPAddr("udp", laddr); err != nil {
				t.Fatal(err)
			}
		}
		c2, err := DialUDP("udp", la, c1.LocalAddr().(*UDPAddr))
		if err != nil {
			t.Fatal(err)
		}
		defer c2.Close()

		var connAddrs = [4]struct {
			got Addr
			ok  bool
		}{
			{c1.LocalAddr(), true},
			{c1.(*UDPConn).RemoteAddr(), false},
			{c2.LocalAddr(), true},
			{c2.RemoteAddr(), true},
		}
		for _, ca := range connAddrs {
			if a, ok := ca.got.(*UDPAddr); ok != ca.ok || ok && a.Port == 0 {
				t.Fatalf("got %v; expected a proper address with non-zero port number", ca.got)
			}
		}
	}
}

func TestIPv6LinkLocalUnicastUDP(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	if !supportsIPv6 {
		t.Skip("IPv6 is not supported")
	}

	for i, tt := range ipv6LinkLocalUnicastUDPTests {
		c1, err := ListenPacket(tt.network, tt.address)
		if err != nil {
			// It might return "LookupHost returned no
			// suitable address" error on some platforms.
			t.Log(err)
			continue
		}
		ls, err := (&packetListener{PacketConn: c1}).newLocalServer()
		if err != nil {
			t.Fatal(err)
		}
		defer ls.teardown()
		ch := make(chan error, 1)
		handler := func(ls *localPacketServer, c PacketConn) { packetTransponder(c, ch) }
		if err := ls.buildup(handler); err != nil {
			t.Fatal(err)
		}
		if la, ok := c1.LocalAddr().(*UDPAddr); !ok || !tt.nameLookup && la.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", la)
		}

		c2, err := Dial(tt.network, ls.PacketConn.LocalAddr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer c2.Close()
		if la, ok := c2.LocalAddr().(*UDPAddr); !ok || !tt.nameLookup && la.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", la)
		}
		if ra, ok := c2.RemoteAddr().(*UDPAddr); !ok || !tt.nameLookup && ra.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", ra)
		}

		if _, err := c2.Write([]byte("UDP OVER IPV6 LINKLOCAL TEST")); err != nil {
			t.Fatal(err)
		}
		b := make([]byte, 32)
		if _, err := c2.Read(b); err != nil {
			t.Fatal(err)
		}

		for err := range ch {
			t.Errorf("#%d: %v", i, err)
		}
	}
}

func TestUDPZeroBytePayload(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	c, err := newLocalPacketListener("udp")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for _, genericRead := range []bool{false, true} {
		n, err := c.WriteTo(nil, c.LocalAddr())
		if err != nil {
			t.Fatal(err)
		}
		if n != 0 {
			t.Errorf("got %d; want 0", n)
		}
		c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		var b [1]byte
		if genericRead {
			_, err = c.(Conn).Read(b[:])
		} else {
			_, _, err = c.ReadFrom(b[:])
		}
		switch err {
		case nil: // ReadFrom succeeds
		default: // Read may timeout, it depends on the platform
			if nerr, ok := err.(Error); !ok || !nerr.Timeout() {
				t.Fatal(err)
			}
		}
	}
}

func TestUDPZeroByteBuffer(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	c, err := newLocalPacketListener("udp")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	b := []byte("UDP ZERO BYTE BUFFER TEST")
	for _, genericRead := range []bool{false, true} {
		n, err := c.WriteTo(b, c.LocalAddr())
		if err != nil {
			t.Fatal(err)
		}
		if n != len(b) {
			t.Errorf("got %d; want %d", n, len(b))
		}
		c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		if genericRead {
			_, err = c.(Conn).Read(nil)
		} else {
			_, _, err = c.ReadFrom(nil)
		}
		switch err {
		case nil: // ReadFrom succeeds
		default: // Read may timeout, it depends on the platform
			if nerr, ok := err.(Error); (!ok || !nerr.Timeout()) && runtime.GOOS != "windows" { // Windows returns WSAEMSGSIZ
				t.Fatal(err)
			}
		}
	}
}
