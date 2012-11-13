// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

package net

import (
	"errors"
	"os"
	"runtime"
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
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	case "linux":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "alpha" {
			t.Logf("skipping test on %q/%q", runtime.GOOS, runtime.GOARCH)
			return
		}
	}

	for _, tt := range multicastListenerTests {
		if tt.ipv6 && (!*testIPv6 || !supportsIPv6 || os.Getuid() != 0) {
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
		c1.Close()
	}
}

func TestSimpleMulticastListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	case "windows":
		if testing.Short() || !*testExternal {
			t.Logf("skipping test on windows to avoid firewall")
			return
		}
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
		t.Errorf("%q not found in RIB", gaddr.String())
		return
	}
	la := c.LocalAddr()
	if la == nil {
		t.Error("LocalAddr failed")
		return
	}
	if a, ok := la.(*UDPAddr); !ok || a.Port == 0 {
		t.Errorf("got %v; expected a proper address with non-zero port number", la)
		return
	}
}

func checkSimpleMulticastListener(t *testing.T, err error, c *UDPConn, gaddr *UDPAddr) {
	la := c.LocalAddr()
	if la == nil {
		t.Error("LocalAddr failed")
		return
	}
	if a, ok := la.(*UDPAddr); !ok || a.Port == 0 {
		t.Errorf("got %v; expected a proper address with non-zero port number", la)
		return
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
