// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"runtime"
	"testing"
)

var listenMulticastUDPTests = []struct {
	net   string
	gaddr *UDPAddr
	flags Flags
	ipv6  bool
}{
	// cf. RFC 4727: Experimental Values in IPv4, IPv6, ICMPv4, ICMPv6, UDP, and TCP Headers
	{"udp", &UDPAddr{IPv4(224, 0, 0, 254), 12345}, FlagUp | FlagLoopback, false},
	{"udp4", &UDPAddr{IPv4(224, 0, 0, 254), 12345}, FlagUp | FlagLoopback, false},
	{"udp", &UDPAddr{ParseIP("ff0e::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff01::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff02::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff04::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff05::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff08::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff0e::114"), 12345}, FlagUp | FlagLoopback, true},
}

func TestListenMulticastUDP(t *testing.T) {
	switch runtime.GOOS {
	case "netbsd", "openbsd", "plan9", "windows":
		return
	case "linux":
		if runtime.GOARCH == "arm" {
			return
		}
	}

	for _, tt := range listenMulticastUDPTests {
		if tt.ipv6 && (!supportsIPv6 || os.Getuid() != 0) {
			continue
		}
		ift, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
		}
		var ifi *Interface
		for _, x := range ift {
			if x.Flags&tt.flags == tt.flags {
				ifi = &x
				break
			}
		}
		if ifi == nil {
			t.Logf("an appropriate multicast interface not found")
			return
		}
		c, err := ListenMulticastUDP(tt.net, ifi, tt.gaddr)
		if err != nil {
			t.Fatalf("ListenMulticastUDP failed: %v", err)
		}
		defer c.Close() // test to listen concurrently across multiple listeners
		if !tt.ipv6 {
			testIPv4MulticastSocketOptions(t, c.fd, ifi)
		} else {
			testIPv6MulticastSocketOptions(t, c.fd, ifi)
		}
		ifmat, err := ifi.MulticastAddrs()
		if err != nil {
			t.Fatalf("MulticastAddrs failed: %v", err)
		}
		var found bool
		for _, ifma := range ifmat {
			if ifma.(*IPAddr).IP.Equal(tt.gaddr.IP) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("%q not found in RIB", tt.gaddr.String())
		}
	}
}

func TestSimpleListenMulticastUDP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		return
	}

	for _, tt := range listenMulticastUDPTests {
		if tt.ipv6 {
			continue
		}
		tt.flags = FlagUp | FlagMulticast
		ift, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
		}
		var ifi *Interface
		for _, x := range ift {
			if x.Flags&tt.flags == tt.flags {
				ifi = &x
				break
			}
		}
		if ifi == nil {
			t.Logf("an appropriate multicast interface not found")
			return
		}
		c, err := ListenMulticastUDP(tt.net, ifi, tt.gaddr)
		if err != nil {
			t.Fatalf("ListenMulticastUDP failed: %v", err)
		}
		c.Close()
	}
}

func testIPv4MulticastSocketOptions(t *testing.T, fd *netFD, ifi *Interface) {
	ifmc, err := ipv4MulticastInterface(fd)
	if err != nil {
		t.Fatalf("ipv4MulticastInterface failed: %v", err)
	}
	t.Logf("IPv4 multicast interface: %v", ifmc)
	err = setIPv4MulticastInterface(fd, ifi)
	if err != nil {
		t.Fatalf("setIPv4MulticastInterface failed: %v", err)
	}

	ttl, err := ipv4MulticastTTL(fd)
	if err != nil {
		t.Fatalf("ipv4MulticastTTL failed: %v", err)
	}
	t.Logf("IPv4 multicast TTL: %v", ttl)
	err = setIPv4MulticastTTL(fd, 1)
	if err != nil {
		t.Fatalf("setIPv4MulticastTTL failed: %v", err)
	}

	loop, err := ipv4MulticastLoopback(fd)
	if err != nil {
		t.Fatalf("ipv4MulticastLoopback failed: %v", err)
	}
	t.Logf("IPv4 multicast loopback: %v", loop)
	err = setIPv4MulticastLoopback(fd, false)
	if err != nil {
		t.Fatalf("setIPv4MulticastLoopback failed: %v", err)
	}
}

func testIPv6MulticastSocketOptions(t *testing.T, fd *netFD, ifi *Interface) {
	ifmc, err := ipv6MulticastInterface(fd)
	if err != nil {
		t.Fatalf("ipv6MulticastInterface failed: %v", err)
	}
	t.Logf("IPv6 multicast interface: %v", ifmc)
	err = setIPv6MulticastInterface(fd, ifi)
	if err != nil {
		t.Fatalf("setIPv6MulticastInterface failed: %v", err)
	}

	hoplim, err := ipv6MulticastHopLimit(fd)
	if err != nil {
		t.Fatalf("ipv6MulticastHopLimit failed: %v", err)
	}
	t.Logf("IPv6 multicast hop limit: %v", hoplim)
	err = setIPv6MulticastHopLimit(fd, 1)
	if err != nil {
		t.Fatalf("setIPv6MulticastHopLimit failed: %v", err)
	}

	loop, err := ipv6MulticastLoopback(fd)
	if err != nil {
		t.Fatalf("ipv6MulticastLoopback failed: %v", err)
	}
	t.Logf("IPv6 multicast loopback: %v", loop)
	err = setIPv6MulticastLoopback(fd, false)
	if err != nil {
		t.Fatalf("setIPv6MulticastLoopback failed: %v", err)
	}
}
