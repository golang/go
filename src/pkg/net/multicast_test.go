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

var joinAndLeaveGroupUDPTests = []struct {
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

func TestJoinAndLeaveGroupUDP(t *testing.T) {
	if runtime.GOOS == "windows" {
		return
	}
	if !*multicast {
		t.Logf("test disabled; use --multicast to enable")
		return
	}

	for _, tt := range joinAndLeaveGroupUDPTests {
		var (
			ifi   *Interface
			found bool
		)
		if tt.ipv6 && (!supportsIPv6 || os.Getuid() != 0) {
			continue
		}
		ift, err := Interfaces()
		if err != nil {
			t.Fatalf("Interfaces() failed: %v", err)
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
			t.Fatal(err)
		}
		defer c.Close()
		if err := c.JoinGroup(ifi, tt.gaddr); err != nil {
			t.Fatal(err)
		}
		ifmat, err := ifi.MulticastAddrs()
		if err != nil {
			t.Fatalf("MulticastAddrs() failed: %v", err)
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
			t.Fatal(err)
		}
	}
}
