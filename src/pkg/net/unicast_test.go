// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"testing"
)

var unicastTests = []struct {
	net    string
	laddr  string
	ipv6   bool
	packet bool
}{
	{"tcp4", "127.0.0.1:0", false, false},
	{"tcp6", "[::1]:0", true, false},
	{"udp4", "127.0.0.1:0", false, true},
	{"udp6", "[::1]:0", true, true},
}

func TestUnicastTCPAndUDP(t *testing.T) {
	if runtime.GOOS == "plan9" || runtime.GOOS == "windows" {
		return
	}

	for _, tt := range unicastTests {
		if tt.ipv6 && !supportsIPv6 {
			continue
		}
		var fd *netFD
		if !tt.packet {
			c, err := Listen(tt.net, tt.laddr)
			if err != nil {
				t.Fatalf("Listen failed: %v", err)
			}
			defer c.Close()
			fd = c.(*TCPListener).fd
		} else {
			c, err := ListenPacket(tt.net, tt.laddr)
			if err != nil {
				t.Fatalf("ListenPacket failed: %v", err)
			}
			defer c.Close()
			fd = c.(*UDPConn).fd
		}
		if !tt.ipv6 {
			testIPv4UnicastSocketOptions(t, fd)
		} else {
			testIPv6UnicastSocketOptions(t, fd)
		}
	}
}

func testIPv4UnicastSocketOptions(t *testing.T, fd *netFD) {
	tos, err := ipv4TOS(fd)
	if err != nil {
		t.Fatalf("ipv4TOS failed: %v", err)
	}
	t.Logf("IPv4 TOS: %v", tos)
	err = setIPv4TOS(fd, 1)
	if err != nil {
		t.Fatalf("setIPv4TOS failed: %v", err)
	}

	ttl, err := ipv4TTL(fd)
	if err != nil {
		t.Fatalf("ipv4TTL failed: %v", err)
	}
	t.Logf("IPv4 TTL: %v", ttl)
	err = setIPv4TTL(fd, 1)
	if err != nil {
		t.Fatalf("setIPv4TTL failed: %v", err)
	}
}

func testIPv6UnicastSocketOptions(t *testing.T, fd *netFD) {
	tos, err := ipv6TrafficClass(fd)
	if err != nil {
		t.Fatalf("ipv6TrafficClass failed: %v", err)
	}
	t.Logf("IPv6 TrafficClass: %v", tos)
	err = setIPv6TrafficClass(fd, 1)
	if err != nil {
		t.Fatalf("setIPv6TrafficClass failed: %v", err)
	}

	hoplim, err := ipv6HopLimit(fd)
	if err != nil {
		t.Fatalf("ipv6HopLimit failed: %v", err)
	}
	t.Logf("IPv6 HopLimit: %v", hoplim)
	err = setIPv6HopLimit(fd, 1)
	if err != nil {
		t.Fatalf("setIPv6HopLimit failed: %v", err)
	}
}
