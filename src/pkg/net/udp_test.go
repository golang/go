// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
)

type resolveUDPAddrTest struct {
	net     string
	litAddr string
	addr    *UDPAddr
	err     error
}

var resolveUDPAddrTests = []resolveUDPAddrTest{
	{"udp", "127.0.0.1:0", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil},
	{"udp4", "127.0.0.1:65535", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 65535}, nil},

	{"udp", "[::1]:1", &UDPAddr{IP: ParseIP("::1"), Port: 1}, nil},
	{"udp6", "[::1]:65534", &UDPAddr{IP: ParseIP("::1"), Port: 65534}, nil},

	{"udp", "[::1%en0]:1", &UDPAddr{IP: ParseIP("::1"), Port: 1, Zone: "en0"}, nil},
	{"udp6", "[::1%911]:2", &UDPAddr{IP: ParseIP("::1"), Port: 2, Zone: "911"}, nil},

	{"", "127.0.0.1:0", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil}, // Go 1.0 behavior
	{"", "[::1]:0", &UDPAddr{IP: ParseIP("::1"), Port: 0}, nil},         // Go 1.0 behavior

	{"sip", "127.0.0.1:0", nil, UnknownNetworkError("sip")},
}

func init() {
	if ifi := loopbackInterface(); ifi != nil {
		index := fmt.Sprintf("%v", ifi.Index)
		resolveUDPAddrTests = append(resolveUDPAddrTests, []resolveUDPAddrTest{
			{"udp6", "[fe80::1%" + ifi.Name + "]:3", &UDPAddr{IP: ParseIP("fe80::1"), Port: 3, Zone: zoneToString(ifi.Index)}, nil},
			{"udp6", "[fe80::1%" + index + "]:4", &UDPAddr{IP: ParseIP("fe80::1"), Port: 4, Zone: index}, nil},
		}...)
	}
}

func TestResolveUDPAddr(t *testing.T) {
	for _, tt := range resolveUDPAddrTests {
		addr, err := ResolveUDPAddr(tt.net, tt.litAddr)
		if err != tt.err {
			t.Fatalf("ResolveUDPAddr(%v, %v) failed: %v", tt.net, tt.litAddr, err)
		}
		if !reflect.DeepEqual(addr, tt.addr) {
			t.Fatalf("got %#v; expected %#v", addr, tt.addr)
		}
	}
}

func TestWriteToUDP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	l, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer l.Close()

	testWriteToConn(t, l.LocalAddr().String())
	testWriteToPacketConn(t, l.LocalAddr().String())
}

func testWriteToConn(t *testing.T, raddr string) {
	c, err := Dial("udp", raddr)
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	defer c.Close()

	ra, err := ResolveUDPAddr("udp", raddr)
	if err != nil {
		t.Fatalf("ResolveUDPAddr failed: %v", err)
	}

	_, err = c.(*UDPConn).WriteToUDP([]byte("Connection-oriented mode socket"), ra)
	if err == nil {
		t.Fatal("WriteToUDP should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("WriteToUDP should fail as ErrWriteToConnected: %v", err)
	}

	_, err = c.(*UDPConn).WriteTo([]byte("Connection-oriented mode socket"), ra)
	if err == nil {
		t.Fatal("WriteTo should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("WriteTo should fail as ErrWriteToConnected: %v", err)
	}

	_, err = c.Write([]byte("Connection-oriented mode socket"))
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}
}

func testWriteToPacketConn(t *testing.T, raddr string) {
	c, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ListenPacket failed: %v", err)
	}
	defer c.Close()

	ra, err := ResolveUDPAddr("udp", raddr)
	if err != nil {
		t.Fatalf("ResolveUDPAddr failed: %v", err)
	}

	_, err = c.(*UDPConn).WriteToUDP([]byte("Connection-less mode socket"), ra)
	if err != nil {
		t.Fatalf("WriteToUDP failed: %v", err)
	}

	_, err = c.WriteTo([]byte("Connection-less mode socket"), ra)
	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	_, err = c.(*UDPConn).Write([]byte("Connection-less mode socket"))
	if err == nil {
		t.Fatal("Write should fail")
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
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	for _, tt := range udpConnLocalNameTests {
		c, err := ListenUDP(tt.net, tt.laddr)
		if err != nil {
			t.Fatalf("ListenUDP failed: %v", err)
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
			t.Fatalf("ListenUDP failed: %v", err)
		}
		defer c1.Close()

		var la *UDPAddr
		if laddr != "" {
			var err error
			if la, err = ResolveUDPAddr("udp", laddr); err != nil {
				t.Fatalf("ResolveUDPAddr failed: %v", err)
			}
		}
		c2, err := DialUDP("udp", la, c1.LocalAddr().(*UDPAddr))
		if err != nil {
			t.Fatalf("DialUDP failed: %v", err)
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
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}
	ifi := loopbackInterface()
	if ifi == nil {
		t.Skip("loopback interface not found")
	}
	laddr := ipv6LinkLocalUnicastAddr(ifi)
	if laddr == "" {
		t.Skip("ipv6 unicast address on loopback not found")
	}

	type test struct {
		net, addr  string
		nameLookup bool
	}
	var tests = []test{
		{"udp", "[" + laddr + "%" + ifi.Name + "]:0", false},
		{"udp6", "[" + laddr + "%" + ifi.Name + "]:0", false},
	}
	switch runtime.GOOS {
	case "darwin", "freebsd", "openbsd", "netbsd":
		tests = append(tests, []test{
			{"udp", "[localhost%" + ifi.Name + "]:0", true},
			{"udp6", "[localhost%" + ifi.Name + "]:0", true},
		}...)
	case "linux":
		tests = append(tests, []test{
			{"udp", "[ip6-localhost%" + ifi.Name + "]:0", true},
			{"udp6", "[ip6-localhost%" + ifi.Name + "]:0", true},
		}...)
	}
	for _, tt := range tests {
		c1, err := ListenPacket(tt.net, tt.addr)
		if err != nil {
			// It might return "LookupHost returned no
			// suitable address" error on some platforms.
			t.Logf("ListenPacket failed: %v", err)
			continue
		}
		defer c1.Close()
		if la, ok := c1.LocalAddr().(*UDPAddr); !ok || !tt.nameLookup && la.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", la)
		}

		c2, err := Dial(tt.net, c1.LocalAddr().String())
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		defer c2.Close()
		if la, ok := c2.LocalAddr().(*UDPAddr); !ok || !tt.nameLookup && la.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", la)
		}
		if ra, ok := c2.RemoteAddr().(*UDPAddr); !ok || !tt.nameLookup && ra.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", ra)
		}

		if _, err := c2.Write([]byte("UDP OVER IPV6 LINKLOCAL TEST")); err != nil {
			t.Fatalf("Conn.Write failed: %v", err)
		}
		b := make([]byte, 32)
		if _, from, err := c1.ReadFrom(b); err != nil {
			t.Fatalf("PacketConn.ReadFrom failed: %v", err)
		} else {
			if ra, ok := from.(*UDPAddr); !ok || !tt.nameLookup && ra.Zone == "" {
				t.Fatalf("got %v; expected a proper address with zone identifier", ra)
			}
		}
	}
}
