// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"runtime"
	"testing"
)

var resolveUDPAddrTests = []struct {
	net     string
	litAddr string
	addr    *UDPAddr
	err     error
}{
	{"udp", "127.0.0.1:0", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil},
	{"udp4", "127.0.0.1:65535", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 65535}, nil},

	{"udp", "[::1]:1", &UDPAddr{IP: ParseIP("::1"), Port: 1}, nil},
	{"udp6", "[::1]:65534", &UDPAddr{IP: ParseIP("::1"), Port: 65534}, nil},

	{"", "127.0.0.1:0", &UDPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil}, // Go 1.0 behavior
	{"", "[::1]:0", &UDPAddr{IP: ParseIP("::1"), Port: 0}, nil},         // Go 1.0 behavior

	{"sip", "127.0.0.1:0", nil, UnknownNetworkError("sip")},
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
			t.Errorf("ListenUDP failed: %v", err)
			return
		}
		defer c.Close()
		la := c.LocalAddr()
		if a, ok := la.(*UDPAddr); !ok || a.Port == 0 {
			t.Errorf("got %v; expected a proper address with non-zero port number", la)
			return
		}
	}
}
