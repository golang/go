// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"os"
	"reflect"
	"runtime"
	"testing"
)

// The full stack test cases for IPConn have been moved to the
// following:
//	golang.org/x/net/ipv4
//	golang.org/x/net/ipv6
//	golang.org/x/net/icmp

type resolveIPAddrTest struct {
	net           string
	litAddrOrName string
	addr          *IPAddr
	err           error
}

var resolveIPAddrTests = []resolveIPAddrTest{
	{"ip", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
	{"ip4", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
	{"ip4:icmp", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},

	{"ip", "::1", &IPAddr{IP: ParseIP("::1")}, nil},
	{"ip6", "::1", &IPAddr{IP: ParseIP("::1")}, nil},
	{"ip6:ipv6-icmp", "::1", &IPAddr{IP: ParseIP("::1")}, nil},
	{"ip6:IPv6-ICMP", "::1", &IPAddr{IP: ParseIP("::1")}, nil},

	{"ip", "::1%en0", &IPAddr{IP: ParseIP("::1"), Zone: "en0"}, nil},
	{"ip6", "::1%911", &IPAddr{IP: ParseIP("::1"), Zone: "911"}, nil},

	{"", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil}, // Go 1.0 behavior
	{"", "::1", &IPAddr{IP: ParseIP("::1")}, nil},           // Go 1.0 behavior

	{"l2tp", "127.0.0.1", nil, UnknownNetworkError("l2tp")},
	{"l2tp:gre", "127.0.0.1", nil, UnknownNetworkError("l2tp:gre")},
	{"tcp", "1.2.3.4:123", nil, UnknownNetworkError("tcp")},
}

func init() {
	if ifi := loopbackInterface(); ifi != nil {
		index := fmt.Sprintf("%v", ifi.Index)
		resolveIPAddrTests = append(resolveIPAddrTests, []resolveIPAddrTest{
			{"ip6", "fe80::1%" + ifi.Name, &IPAddr{IP: ParseIP("fe80::1"), Zone: zoneToString(ifi.Index)}, nil},
			{"ip6", "fe80::1%" + index, &IPAddr{IP: ParseIP("fe80::1"), Zone: index}, nil},
		}...)
	}
	if ips, err := LookupIP("localhost"); err == nil && len(ips) > 1 && supportsIPv4 && supportsIPv6 {
		resolveIPAddrTests = append(resolveIPAddrTests, []resolveIPAddrTest{
			{"ip", "localhost", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
			{"ip4", "localhost", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
			{"ip6", "localhost", &IPAddr{IP: IPv6loopback}, nil},
		}...)
	}
}

func TestResolveIPAddr(t *testing.T) {
	switch runtime.GOOS {
	case "nacl":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	for _, tt := range resolveIPAddrTests {
		addr, err := ResolveIPAddr(tt.net, tt.litAddrOrName)
		if err != tt.err {
			t.Fatalf("ResolveIPAddr(%v, %v) failed: %v", tt.net, tt.litAddrOrName, err)
		} else if !reflect.DeepEqual(addr, tt.addr) {
			t.Fatalf("got %#v; expected %#v", addr, tt.addr)
		}
	}
}

var ipConnLocalNameTests = []struct {
	net   string
	laddr *IPAddr
}{
	{"ip4:icmp", &IPAddr{IP: IPv4(127, 0, 0, 1)}},
	{"ip4:icmp", &IPAddr{}},
	{"ip4:icmp", nil},
}

func TestIPConnLocalName(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	default:
		if os.Getuid() != 0 {
			t.Skip("skipping test; must be root")
		}
	}

	for _, tt := range ipConnLocalNameTests {
		c, err := ListenIP(tt.net, tt.laddr)
		if err != nil {
			t.Fatalf("ListenIP failed: %v", err)
		}
		defer c.Close()
		if la := c.LocalAddr(); la == nil {
			t.Fatal("IPConn.LocalAddr failed")
		}
	}
}

func TestIPConnRemoteName(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	default:
		if os.Getuid() != 0 {
			t.Skip("skipping test; must be root")
		}
	}

	raddr := &IPAddr{IP: IPv4(127, 0, 0, 1).To4()}
	c, err := DialIP("ip:tcp", &IPAddr{IP: IPv4(127, 0, 0, 1)}, raddr)
	if err != nil {
		t.Fatalf("DialIP failed: %v", err)
	}
	defer c.Close()
	if !reflect.DeepEqual(raddr, c.RemoteAddr()) {
		t.Fatalf("got %#v, expected %#v", c.RemoteAddr(), raddr)
	}
}
