// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"os"
	"runtime"
	"syscall"
	"testing"
)

var multicastListenerTests = []struct {
	net   string
	gaddr *UDPAddr
	flags Flags
	ipv6  bool // test with underlying AF_INET6 socket
}{
	// cf. RFC 4727: Experimental Values in IPv4, IPv6, ICMPv4, ICMPv6, UDP, and TCP Headers

	{"udp", &UDPAddr{IPv4(224, 0, 0, 254), 12345}, FlagUp | FlagLoopback, false},
	{"udp", &UDPAddr{IPv4(224, 0, 0, 254), 12345}, 0, false},
	{"udp", &UDPAddr{ParseIP("ff0e::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp", &UDPAddr{ParseIP("ff0e::114"), 12345}, 0, true},

	{"udp4", &UDPAddr{IPv4(224, 0, 0, 254), 12345}, FlagUp | FlagLoopback, false},
	{"udp4", &UDPAddr{IPv4(224, 0, 0, 254), 12345}, 0, false},

	{"udp6", &UDPAddr{ParseIP("ff01::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff01::114"), 12345}, 0, true},
	{"udp6", &UDPAddr{ParseIP("ff02::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff02::114"), 12345}, 0, true},
	{"udp6", &UDPAddr{ParseIP("ff04::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff04::114"), 12345}, 0, true},
	{"udp6", &UDPAddr{ParseIP("ff05::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff05::114"), 12345}, 0, true},
	{"udp6", &UDPAddr{ParseIP("ff08::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff08::114"), 12345}, 0, true},
	{"udp6", &UDPAddr{ParseIP("ff0e::114"), 12345}, FlagUp | FlagLoopback, true},
	{"udp6", &UDPAddr{ParseIP("ff0e::114"), 12345}, 0, true},
}

// TestMulticastListener tests both single and double listen to a test
// listener with same address family, same group address and same port.
func TestMulticastListener(t *testing.T) {
	switch runtime.GOOS {
	case "netbsd", "openbsd", "plan9", "windows":
		return
	case "linux":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "alpha" {
			return
		}
	}

	for _, tt := range multicastListenerTests {
		if tt.ipv6 && (!supportsIPv6 || os.Getuid() != 0) {
			continue
		}
		ifi, err := availMulticastInterface(t, tt.flags)
		if err != nil {
			continue
		}
		c1, err := ListenMulticastUDP(tt.net, ifi, tt.gaddr)
		if err != nil {
			t.Fatalf("First ListenMulticastUDP failed: %v", err)
		}
		checkMulticastListener(t, err, c1, tt.gaddr)
		c2, err := ListenMulticastUDP(tt.net, ifi, tt.gaddr)
		if err != nil {
			t.Fatalf("Second ListenMulticastUDP failed: %v", err)
		}
		checkMulticastListener(t, err, c2, tt.gaddr)
		c2.Close()
		switch c1.fd.family {
		case syscall.AF_INET:
			testIPv4MulticastSocketOptions(t, c1.fd, ifi)
		case syscall.AF_INET6:
			testIPv6MulticastSocketOptions(t, c1.fd, ifi)
		}
		c1.Close()
	}
}

func TestSimpleMulticastListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		return
	}

	for _, tt := range multicastListenerTests {
		if tt.ipv6 {
			continue
		}
		tt.flags = FlagUp | FlagMulticast // for windows testing
		ifi, err := availMulticastInterface(t, tt.flags)
		if err != nil {
			continue
		}
		c1, err := ListenMulticastUDP(tt.net, ifi, tt.gaddr)
		if err != nil {
			t.Fatalf("First ListenMulticastUDP failed: %v", err)
		}
		checkSimpleMulticastListener(t, err, c1, tt.gaddr)
		c2, err := ListenMulticastUDP(tt.net, ifi, tt.gaddr)
		if err != nil {
			t.Fatalf("Second ListenMulticastUDP failed: %v", err)
		}
		checkSimpleMulticastListener(t, err, c2, tt.gaddr)
		c2.Close()
		c1.Close()
	}
}

func checkMulticastListener(t *testing.T, err error, c *UDPConn, gaddr *UDPAddr) {
	if !multicastRIBContains(t, gaddr.IP) {
		t.Fatalf("%q not found in RIB", gaddr.String())
	}
	if c.LocalAddr().String() != gaddr.String() {
		t.Fatalf("LocalAddr returns %q, expected %q", c.LocalAddr().String(), gaddr.String())
	}
}

func checkSimpleMulticastListener(t *testing.T, err error, c *UDPConn, gaddr *UDPAddr) {
	if c.LocalAddr().String() != gaddr.String() {
		t.Fatalf("LocalAddr returns %q, expected %q", c.LocalAddr().String(), gaddr.String())
	}
}

func availMulticastInterface(t *testing.T, flags Flags) (*Interface, error) {
	var ifi *Interface
	if flags != Flags(0) {
		ift, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
		}
		for _, x := range ift {
			if x.Flags&flags == flags {
				ifi = &x
				break
			}
		}
		if ifi == nil {
			return nil, errors.New("an appropriate multicast interface not found")
		}
	}
	return ifi, nil
}

func multicastRIBContains(t *testing.T, ip IP) bool {
	ift, err := Interfaces()
	if err != nil {
		t.Fatalf("Interfaces failed: %v", err)
	}
	for _, ifi := range ift {
		ifmat, err := ifi.MulticastAddrs()
		if err != nil {
			t.Fatalf("MulticastAddrs failed: %v", err)
		}
		for _, ifma := range ifmat {
			if ifma.(*IPAddr).IP.Equal(ip) {
				return true
			}
		}
	}
	return false
}

func testIPv4MulticastSocketOptions(t *testing.T, fd *netFD, ifi *Interface) {
	_, err := ipv4MulticastInterface(fd)
	if err != nil {
		t.Fatalf("ipv4MulticastInterface failed: %v", err)
	}
	if ifi != nil {
		err = setIPv4MulticastInterface(fd, ifi)
		if err != nil {
			t.Fatalf("setIPv4MulticastInterface failed: %v", err)
		}
	}
	_, err = ipv4MulticastTTL(fd)
	if err != nil {
		t.Fatalf("ipv4MulticastTTL failed: %v", err)
	}
	err = setIPv4MulticastTTL(fd, 1)
	if err != nil {
		t.Fatalf("setIPv4MulticastTTL failed: %v", err)
	}
	_, err = ipv4MulticastLoopback(fd)
	if err != nil {
		t.Fatalf("ipv4MulticastLoopback failed: %v", err)
	}
	err = setIPv4MulticastLoopback(fd, false)
	if err != nil {
		t.Fatalf("setIPv4MulticastLoopback failed: %v", err)
	}
}

func testIPv6MulticastSocketOptions(t *testing.T, fd *netFD, ifi *Interface) {
	_, err := ipv6MulticastInterface(fd)
	if err != nil {
		t.Fatalf("ipv6MulticastInterface failed: %v", err)
	}
	if ifi != nil {
		err = setIPv6MulticastInterface(fd, ifi)
		if err != nil {
			t.Fatalf("setIPv6MulticastInterface failed: %v", err)
		}
	}
	_, err = ipv6MulticastHopLimit(fd)
	if err != nil {
		t.Fatalf("ipv6MulticastHopLimit failed: %v", err)
	}
	err = setIPv6MulticastHopLimit(fd, 1)
	if err != nil {
		t.Fatalf("setIPv6MulticastHopLimit failed: %v", err)
	}
	_, err = ipv6MulticastLoopback(fd)
	if err != nil {
		t.Fatalf("ipv6MulticastLoopback failed: %v", err)
	}
	err = setIPv6MulticastLoopback(fd, false)
	if err != nil {
		t.Fatalf("setIPv6MulticastLoopback failed: %v", err)
	}
}
