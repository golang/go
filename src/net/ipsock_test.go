// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"runtime"
	"testing"
)

var testInetaddr = func(ip IPAddr) Addr { return &TCPAddr{IP: ip.IP, Port: 5682, Zone: ip.Zone} }

var addrListTests = []struct {
	filter    func(IPAddr) bool
	ips       []IPAddr
	inetaddr  func(IPAddr) Addr
	first     Addr
	primaries addrList
	fallbacks addrList
	err       error
}{
	{
		nil,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682}},
		addrList{&TCPAddr{IP: IPv6loopback, Port: 5682}},
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: IPv4(127, 0, 0, 1)},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{&TCPAddr{IP: IPv6loopback, Port: 5682}},
		addrList{&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682}},
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv4(192, 168, 0, 1)},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv4(192, 168, 0, 1), Port: 5682},
		},
		nil,
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: ParseIP("fe80::1"), Zone: "eth0"},
		},
		testInetaddr,
		&TCPAddr{IP: IPv6loopback, Port: 5682},
		addrList{
			&TCPAddr{IP: IPv6loopback, Port: 5682},
			&TCPAddr{IP: ParseIP("fe80::1"), Port: 5682, Zone: "eth0"},
		},
		nil,
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv4(192, 168, 0, 1)},
			{IP: IPv6loopback},
			{IP: ParseIP("fe80::1"), Zone: "eth0"},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv4(192, 168, 0, 1), Port: 5682},
		},
		addrList{
			&TCPAddr{IP: IPv6loopback, Port: 5682},
			&TCPAddr{IP: ParseIP("fe80::1"), Port: 5682, Zone: "eth0"},
		},
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: ParseIP("fe80::1"), Zone: "eth0"},
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv4(192, 168, 0, 1)},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{
			&TCPAddr{IP: IPv6loopback, Port: 5682},
			&TCPAddr{IP: ParseIP("fe80::1"), Port: 5682, Zone: "eth0"},
		},
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv4(192, 168, 0, 1), Port: 5682},
		},
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
			{IP: IPv4(192, 168, 0, 1)},
			{IP: ParseIP("fe80::1"), Zone: "eth0"},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv4(192, 168, 0, 1), Port: 5682},
		},
		addrList{
			&TCPAddr{IP: IPv6loopback, Port: 5682},
			&TCPAddr{IP: ParseIP("fe80::1"), Port: 5682, Zone: "eth0"},
		},
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: IPv4(127, 0, 0, 1)},
			{IP: ParseIP("fe80::1"), Zone: "eth0"},
			{IP: IPv4(192, 168, 0, 1)},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{
			&TCPAddr{IP: IPv6loopback, Port: 5682},
			&TCPAddr{IP: ParseIP("fe80::1"), Port: 5682, Zone: "eth0"},
		},
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv4(192, 168, 0, 1), Port: 5682},
		},
		nil,
	},

	{
		ipv4only,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682}},
		nil,
		nil,
	},
	{
		ipv4only,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: IPv4(127, 0, 0, 1)},
		},
		testInetaddr,
		&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
		addrList{&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682}},
		nil,
		nil,
	},

	{
		ipv6only,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		},
		testInetaddr,
		&TCPAddr{IP: IPv6loopback, Port: 5682},
		addrList{&TCPAddr{IP: IPv6loopback, Port: 5682}},
		nil,
		nil,
	},
	{
		ipv6only,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: IPv4(127, 0, 0, 1)},
		},
		testInetaddr,
		&TCPAddr{IP: IPv6loopback, Port: 5682},
		addrList{&TCPAddr{IP: IPv6loopback, Port: 5682}},
		nil,
		nil,
	},

	{nil, nil, testInetaddr, nil, nil, nil, &AddrError{errNoSuitableAddress.Error(), "ADDR"}},

	{ipv4only, nil, testInetaddr, nil, nil, nil, &AddrError{errNoSuitableAddress.Error(), "ADDR"}},
	{ipv4only, []IPAddr{{IP: IPv6loopback}}, testInetaddr, nil, nil, nil, &AddrError{errNoSuitableAddress.Error(), "ADDR"}},

	{ipv6only, nil, testInetaddr, nil, nil, nil, &AddrError{errNoSuitableAddress.Error(), "ADDR"}},
	{ipv6only, []IPAddr{{IP: IPv4(127, 0, 0, 1)}}, testInetaddr, nil, nil, nil, &AddrError{errNoSuitableAddress.Error(), "ADDR"}},
}

func TestAddrList(t *testing.T) {
	if !supportsIPv4() || !supportsIPv6() {
		t.Skip("both IPv4 and IPv6 are required")
	}

	for i, tt := range addrListTests {
		addrs, err := filterAddrList(tt.filter, tt.ips, tt.inetaddr, "ADDR")
		if !reflect.DeepEqual(err, tt.err) {
			t.Errorf("#%v: got %v; want %v", i, err, tt.err)
		}
		if tt.err != nil {
			if len(addrs) != 0 {
				t.Errorf("#%v: got %v; want 0", i, len(addrs))
			}
			continue
		}
		first := addrs.first(isIPv4)
		if !reflect.DeepEqual(first, tt.first) {
			t.Errorf("#%v: got %v; want %v", i, first, tt.first)
		}
		primaries, fallbacks := addrs.partition(isIPv4)
		if !reflect.DeepEqual(primaries, tt.primaries) {
			t.Errorf("#%v: got %v; want %v", i, primaries, tt.primaries)
		}
		if !reflect.DeepEqual(fallbacks, tt.fallbacks) {
			t.Errorf("#%v: got %v; want %v", i, fallbacks, tt.fallbacks)
		}
		expectedLen := len(primaries) + len(fallbacks)
		if len(addrs) != expectedLen {
			t.Errorf("#%v: got %v; want %v", i, len(addrs), expectedLen)
		}
	}
}

func TestAddrListPartition(t *testing.T) {
	addrs := addrList{
		&IPAddr{IP: ParseIP("fe80::"), Zone: "eth0"},
		&IPAddr{IP: ParseIP("fe80::1"), Zone: "eth0"},
		&IPAddr{IP: ParseIP("fe80::2"), Zone: "eth0"},
	}
	cases := []struct {
		lastByte  byte
		primaries addrList
		fallbacks addrList
	}{
		{0, addrList{addrs[0]}, addrList{addrs[1], addrs[2]}},
		{1, addrList{addrs[0], addrs[2]}, addrList{addrs[1]}},
		{2, addrList{addrs[0], addrs[1]}, addrList{addrs[2]}},
		{3, addrList{addrs[0], addrs[1], addrs[2]}, nil},
	}
	for i, tt := range cases {
		// Inverting the function's output should not affect the outcome.
		for _, invert := range []bool{false, true} {
			primaries, fallbacks := addrs.partition(func(a Addr) bool {
				ip := a.(*IPAddr).IP
				return (ip[len(ip)-1] == tt.lastByte) != invert
			})
			if !reflect.DeepEqual(primaries, tt.primaries) {
				t.Errorf("#%v: got %v; want %v", i, primaries, tt.primaries)
			}
			if !reflect.DeepEqual(fallbacks, tt.fallbacks) {
				t.Errorf("#%v: got %v; want %v", i, fallbacks, tt.fallbacks)
			}
		}
	}
}

func TestListenIPv6WildcardAddr(t *testing.T) {
	switch runtime.GOOS {
	case "js", "wasip1":
		t.Skip("fake networking does not implement [::] wildcard address assertions")
	case "dragonfly", "openbsd":
		// We are assuming that this test will not work on DragonFly BSD and
		// OpenBSD due to their lack of support for IPV6_V6ONLY=0. However,
		// this assumption is still unconfirmed. For context, see:
		// https://go.dev/issue/77945#issuecomment-4008106922
		t.Skip("The latest DragonFly BSD and OpenBSD kernels do not support IPV6_V6ONLY=0")
	}
	if !supportsIPv6() {
		t.Skip("IPv6 not supported")
	}

	ln, err := Listen("tcp", "[::]:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	addr := ln.Addr().(*TCPAddr)
	if addr.IP.To4() != nil {
		t.Errorf("Listen(\"tcp\", \"[::]:0\") bound to %v, want IPv6 address", addr)
	}
}
