// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"runtime"
	"testing"
)

var unicastTests = []struct {
	net    string
	laddr  string
	ipv6   bool
	packet bool
}{
	{net: "tcp4", laddr: "127.0.0.1:0"},
	{net: "tcp4", laddr: "previous"},
	{net: "tcp6", laddr: "[::1]:0", ipv6: true},
	{net: "tcp6", laddr: "previous", ipv6: true},
	{net: "udp4", laddr: "127.0.0.1:0", packet: true},
	{net: "udp6", laddr: "[::1]:0", ipv6: true, packet: true},
}

func TestUnicastTCPAndUDP(t *testing.T) {
	if runtime.GOOS == "plan9" || runtime.GOOS == "windows" {
		return
	}

	prevladdr := ""
	for _, tt := range unicastTests {
		if tt.ipv6 && !supportsIPv6 {
			continue
		}
		var (
			fd     *netFD
			closer io.Closer
		)
		if !tt.packet {
			if tt.laddr == "previous" {
				tt.laddr = prevladdr
			}
			l, err := Listen(tt.net, tt.laddr)
			if err != nil {
				t.Fatalf("Listen failed: %v", err)
			}
			prevladdr = l.Addr().String()
			closer = l
			fd = l.(*TCPListener).fd
		} else {
			c, err := ListenPacket(tt.net, tt.laddr)
			if err != nil {
				t.Fatalf("ListenPacket failed: %v", err)
			}
			closer = c
			fd = c.(*UDPConn).fd
		}
		if !tt.ipv6 {
			testIPv4UnicastSocketOptions(t, fd)
		} else {
			testIPv6UnicastSocketOptions(t, fd)
		}
		closer.Close()
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
