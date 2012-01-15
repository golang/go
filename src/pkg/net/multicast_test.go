// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"os"
	"runtime"
	"testing"
)

var multicast = flag.Bool("multicast", false, "enable multicast tests")

var multicastUDPTests = []struct {
	net   string
	laddr IP
	gaddr IP
	flags Flags
	ipv6  bool
}{
	// cf. RFC 4727: Experimental Values in IPv4, IPv6, ICMPv4, ICMPv6, UDP, and TCP Headers
	{"udp", IPv4zero, IPv4(224, 0, 0, 254), (FlagUp | FlagLoopback), false},
	{"udp4", IPv4zero, IPv4(224, 0, 0, 254), (FlagUp | FlagLoopback), false},
	{"udp", IPv6unspecified, ParseIP("ff0e::114"), (FlagUp | FlagLoopback), true},
	{"udp6", IPv6unspecified, ParseIP("ff01::114"), (FlagUp | FlagLoopback), true},
	{"udp6", IPv6unspecified, ParseIP("ff02::114"), (FlagUp | FlagLoopback), true},
	{"udp6", IPv6unspecified, ParseIP("ff04::114"), (FlagUp | FlagLoopback), true},
	{"udp6", IPv6unspecified, ParseIP("ff05::114"), (FlagUp | FlagLoopback), true},
	{"udp6", IPv6unspecified, ParseIP("ff08::114"), (FlagUp | FlagLoopback), true},
	{"udp6", IPv6unspecified, ParseIP("ff0e::114"), (FlagUp | FlagLoopback), true},
}

func TestMulticastUDP(t *testing.T) {
	if runtime.GOOS == "plan9" || runtime.GOOS == "windows" {
		return
	}
	if !*multicast {
		t.Logf("test disabled; use --multicast to enable")
		return
	}

	for _, tt := range multicastUDPTests {
		var (
			ifi   *Interface
			found bool
		)
		if tt.ipv6 && (!supportsIPv6 || os.Getuid() != 0) {
			continue
		}
		ift, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
		}
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
		c, err := ListenUDP(tt.net, &UDPAddr{IP: tt.laddr})
		if err != nil {
			t.Fatalf("ListenUDP failed: %v", err)
		}
		defer c.Close()
		if err := c.JoinGroup(ifi, tt.gaddr); err != nil {
			t.Fatalf("JoinGroup failed: %v", err)
		}
		if !tt.ipv6 {
			testIPv4MulticastSocketOptions(t, c.fd, ifi)
		} else {
			testIPv6MulticastSocketOptions(t, c.fd, ifi)
		}
		ifmat, err := ifi.MulticastAddrs()
		if err != nil {
			t.Fatalf("MulticastAddrs failed: %v", err)
		}
		for _, ifma := range ifmat {
			if ifma.(*IPAddr).IP.Equal(tt.gaddr) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("%q not found in RIB", tt.gaddr.String())
		}
		if err := c.LeaveGroup(ifi, tt.gaddr); err != nil {
			t.Fatalf("LeaveGroup failed: %v", err)
		}
	}
}

func TestSimpleMulticastUDP(t *testing.T) {
	if runtime.GOOS == "plan9" {
		return
	}
	if !*multicast {
		t.Logf("test disabled; use --multicast to enable")
		return
	}

	for _, tt := range multicastUDPTests {
		var ifi *Interface
		if tt.ipv6 {
			continue
		}
		tt.flags = FlagUp | FlagMulticast
		ift, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces failed: %v", err)
		}
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
		c, err := ListenUDP(tt.net, &UDPAddr{IP: tt.laddr})
		if err != nil {
			t.Fatalf("ListenUDP failed: %v", err)
		}
		defer c.Close()
		if err := c.JoinGroup(ifi, tt.gaddr); err != nil {
			t.Fatalf("JoinGroup failed: %v", err)
		}
		if err := c.LeaveGroup(ifi, tt.gaddr); err != nil {
			t.Fatalf("LeaveGroup failed: %v", err)
		}
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
