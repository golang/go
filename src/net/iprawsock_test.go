// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"testing"
)

// The full stack test cases for IPConn have been moved to the
// following:
//	golang.org/x/net/ipv4
//	golang.org/x/net/ipv6
//	golang.org/x/net/icmp

type resolveIPAddrTest struct {
	network       string
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

	{"ip4:icmp", "", &IPAddr{}, nil},

	{"l2tp", "127.0.0.1", nil, UnknownNetworkError("l2tp")},
	{"l2tp:gre", "127.0.0.1", nil, UnknownNetworkError("l2tp:gre")},
	{"tcp", "1.2.3.4:123", nil, UnknownNetworkError("tcp")},
}

func TestResolveIPAddr(t *testing.T) {
	if !testableNetwork("ip+nopriv") {
		t.Skip("ip+nopriv test")
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupLocalhost

	for i, tt := range resolveIPAddrTests {
		addr, err := ResolveIPAddr(tt.network, tt.litAddrOrName)
		if err != tt.err {
			t.Errorf("#%d: %v", i, err)
		} else if !reflect.DeepEqual(addr, tt.addr) {
			t.Errorf("#%d: got %#v; want %#v", i, addr, tt.addr)
		}
		if err != nil {
			continue
		}
		rtaddr, err := ResolveIPAddr(addr.Network(), addr.String())
		if err != nil {
			t.Errorf("#%d: %v", i, err)
		} else if !reflect.DeepEqual(rtaddr, addr) {
			t.Errorf("#%d: got %#v; want %#v", i, rtaddr, addr)
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
	for _, tt := range ipConnLocalNameTests {
		if !testableNetwork(tt.net) {
			t.Logf("skipping %s test", tt.net)
			continue
		}
		c, err := ListenIP(tt.net, tt.laddr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if la := c.LocalAddr(); la == nil {
			t.Fatal("should not fail")
		}
	}
}

func TestIPConnRemoteName(t *testing.T) {
	if !testableNetwork("ip:tcp") {
		t.Skip("ip:tcp test")
	}

	raddr := &IPAddr{IP: IPv4(127, 0, 0, 1).To4()}
	c, err := DialIP("ip:tcp", &IPAddr{IP: IPv4(127, 0, 0, 1)}, raddr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !reflect.DeepEqual(raddr, c.RemoteAddr()) {
		t.Fatalf("got %#v; want %#v", c.RemoteAddr(), raddr)
	}
}
