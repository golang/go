// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestResolveUDPAddr(t *testing.T) {
	for _, tt := range resolveTCPAddrTests {
		net := strings.Replace(tt.net, "tcp", "udp", -1)
		addr, err := ResolveUDPAddr(net, tt.litAddrOrName)
		if err != tt.err {
			t.Fatalf("ResolveUDPAddr(%q, %q) failed: %v", net, tt.litAddrOrName, err)
		}
		if !reflect.DeepEqual(addr, (*UDPAddr)(tt.addr)) {
			t.Fatalf("ResolveUDPAddr(%q, %q) = %#v, want %#v", net, tt.litAddrOrName, addr, tt.addr)
		}
		if err == nil {
			str := addr.String()
			addr1, err := ResolveUDPAddr(net, str)
			if err != nil {
				t.Fatalf("ResolveUDPAddr(%q, %q) [from %q]: %v", net, str, tt.litAddrOrName, err)
			}
			if !reflect.DeepEqual(addr1, addr) {
				t.Fatalf("ResolveUDPAddr(%q, %q) [from %q] = %#v, want %#v", net, str, tt.litAddrOrName, addr1, addr)
			}
		}
	}
}

func TestReadFromUDP(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("skipping test on %q, see issue 8916", runtime.GOOS)
	}

	ra, err := ResolveUDPAddr("udp", "127.0.0.1:7")
	if err != nil {
		t.Fatal(err)
	}

	la, err := ResolveUDPAddr("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	c, err := ListenUDP("udp", la)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	_, err = c.WriteToUDP([]byte("a"), ra)
	if err != nil {
		t.Fatal(err)
	}

	err = c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 1)
	_, _, err = c.ReadFromUDP(b)
	if err == nil {
		t.Fatal("ReadFromUDP should fail")
	} else if !isTimeout(err) {
		t.Fatal(err)
	}
}

func TestWriteToUDP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
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
		t.Fatal(err)
	}

	_, _, err = c.(*UDPConn).WriteMsgUDP([]byte("Connection-oriented mode socket"), nil, ra)
	if err == nil {
		t.Fatal("WriteMsgUDP should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("WriteMsgUDP should fail as ErrWriteToConnected: %v", err)
	}
	_, _, err = c.(*UDPConn).WriteMsgUDP([]byte("Connection-oriented mode socket"), nil, nil)
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

	_, err = c.(*UDPConn).WriteToUDP([]byte("Connection-less mode socket"), ra)
	if err != nil {
		t.Fatal(err)
	}

	_, err = c.WriteTo([]byte("Connection-less mode socket"), ra)
	if err != nil {
		t.Fatal(err)
	}

	_, err = c.(*UDPConn).Write([]byte("Connection-less mode socket"))
	if err == nil {
		t.Fatal("Write should fail")
	}

	_, _, err = c.(*UDPConn).WriteMsgUDP([]byte("Connection-less mode socket"), nil, nil)
	if err == nil {
		t.Fatal("WriteMsgUDP should fail")
	}
	if err != nil && err.(*OpError).Err != errMissingAddress {
		t.Fatalf("WriteMsgUDP should fail as errMissingAddress: %v", err)
	}
	_, _, err = c.(*UDPConn).WriteMsgUDP([]byte("Connection-less mode socket"), nil, ra)
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
	// The first udp test fails on DragonFly - see issue 7473.
	if runtime.GOOS == "dragonfly" {
		tests = tests[1:]
	}
	switch runtime.GOOS {
	case "darwin", "dragonfly", "freebsd", "openbsd", "netbsd":
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
