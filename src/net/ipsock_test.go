// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"testing"
)

var testInetaddr = func(ip IPAddr) netaddr { return &TCPAddr{IP: ip.IP, Port: 5682, Zone: ip.Zone} }

var firstFavoriteAddrTests = []struct {
	filter   func(IPAddr) bool
	ips      []IPAddr
	inetaddr func(IPAddr) netaddr
	addr     netaddr
	err      error
}{
	{
		nil,
		[]IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		},
		testInetaddr,
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv6loopback, Port: 5682},
		},
		nil,
	},
	{
		nil,
		[]IPAddr{
			{IP: IPv6loopback},
			{IP: IPv4(127, 0, 0, 1)},
		},
		testInetaddr,
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv6loopback, Port: 5682},
		},
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
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv6loopback, Port: 5682},
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
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv6loopback, Port: 5682},
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
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv6loopback, Port: 5682},
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
		addrList{
			&TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5682},
			&TCPAddr{IP: IPv6loopback, Port: 5682},
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
		nil,
	},

	{nil, nil, testInetaddr, nil, errNoSuitableAddress},

	{ipv4only, nil, testInetaddr, nil, errNoSuitableAddress},
	{ipv4only, []IPAddr{{IP: IPv6loopback}}, testInetaddr, nil, errNoSuitableAddress},

	{ipv6only, nil, testInetaddr, nil, errNoSuitableAddress},
	{ipv6only, []IPAddr{{IP: IPv4(127, 0, 0, 1)}}, testInetaddr, nil, errNoSuitableAddress},
}

func TestFirstFavoriteAddr(t *testing.T) {
	if !supportsIPv4 || !supportsIPv6 {
		t.Skip("ipv4 or ipv6 is not supported")
	}

	for i, tt := range firstFavoriteAddrTests {
		addr, err := firstFavoriteAddr(tt.filter, tt.ips, tt.inetaddr)
		if err != tt.err {
			t.Errorf("#%v: got %v; expected %v", i, err, tt.err)
		}
		if !reflect.DeepEqual(addr, tt.addr) {
			t.Errorf("#%v: got %v; expected %v", i, addr, tt.addr)
		}
	}
}
