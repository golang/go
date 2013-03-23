// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"runtime"
	"testing"
)

var parseIPTests = []struct {
	in  string
	out IP
}{
	{"127.0.1.2", IPv4(127, 0, 1, 2)},
	{"127.0.0.1", IPv4(127, 0, 0, 1)},
	{"127.0.0.256", nil},
	{"abc", nil},
	{"123:", nil},
	{"::ffff:127.0.0.1", IPv4(127, 0, 0, 1)},
	{"2001:4860:0:2001::68", IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01, 0, 0, 0, 0, 0, 0, 0x00, 0x68}},
	{"::ffff:4a7d:1363", IPv4(74, 125, 19, 99)},
	{"fe80::1%lo0", nil},
	{"fe80::1%911", nil},
	{"", nil},
}

func TestParseIP(t *testing.T) {
	for _, tt := range parseIPTests {
		if out := ParseIP(tt.in); !reflect.DeepEqual(out, tt.out) {
			t.Errorf("ParseIP(%q) = %v, want %v", tt.in, out, tt.out)
		}
	}
}

var ipStringTests = []struct {
	in  IP
	out string // see RFC 5952
}{
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0x1, 0x23, 0, 0x12, 0, 0x1}, "2001:db8::123:12:1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x1}, "2001:db8::1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0x1, 0, 0, 0, 0x1, 0, 0, 0, 0x1}, "2001:db8:0:1:0:1:0:1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0x1, 0, 0, 0, 0x1, 0, 0, 0, 0x1, 0, 0}, "2001:db8:1:0:1:0:1:0"},
	{IP{0x20, 0x1, 0, 0, 0, 0, 0, 0, 0, 0x1, 0, 0, 0, 0, 0, 0x1}, "2001::1:0:0:1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0x1, 0, 0, 0, 0, 0, 0}, "2001:db8:0:0:1::"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0x1, 0, 0, 0, 0, 0, 0x1}, "2001:db8::1:0:0:1"},
	{IP{0x20, 0x1, 0xD, 0xB8, 0, 0, 0, 0, 0, 0xA, 0, 0xB, 0, 0xC, 0, 0xD}, "2001:db8::a:b:c:d"},
	{nil, "<nil>"},
}

func TestIPString(t *testing.T) {
	for _, tt := range ipStringTests {
		if out := tt.in.String(); out != tt.out {
			t.Errorf("IP.String(%v) = %q, want %q", tt.in, out, tt.out)
		}
	}
}

var ipMaskTests = []struct {
	in   IP
	mask IPMask
	out  IP
}{
	{IPv4(192, 168, 1, 127), IPv4Mask(255, 255, 255, 128), IPv4(192, 168, 1, 0)},
	{IPv4(192, 168, 1, 127), IPMask(ParseIP("255.255.255.192")), IPv4(192, 168, 1, 64)},
	{IPv4(192, 168, 1, 127), IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffe0")), IPv4(192, 168, 1, 96)},
	{IPv4(192, 168, 1, 127), IPv4Mask(255, 0, 255, 0), IPv4(192, 0, 1, 0)},
	{ParseIP("2001:db8::1"), IPMask(ParseIP("ffff:ff80::")), ParseIP("2001:d80::")},
	{ParseIP("2001:db8::1"), IPMask(ParseIP("f0f0:0f0f::")), ParseIP("2000:d08::")},
}

func TestIPMask(t *testing.T) {
	for _, tt := range ipMaskTests {
		if out := tt.in.Mask(tt.mask); out == nil || !tt.out.Equal(out) {
			t.Errorf("IP(%v).Mask(%v) = %v, want %v", tt.in, tt.mask, out, tt.out)
		}
	}
}

var ipMaskStringTests = []struct {
	in  IPMask
	out string
}{
	{IPv4Mask(255, 255, 255, 240), "fffffff0"},
	{IPv4Mask(255, 0, 128, 0), "ff008000"},
	{IPMask(ParseIP("ffff:ff80::")), "ffffff80000000000000000000000000"},
	{IPMask(ParseIP("ef00:ff80::cafe:0")), "ef00ff800000000000000000cafe0000"},
	{nil, "<nil>"},
}

func TestIPMaskString(t *testing.T) {
	for _, tt := range ipMaskStringTests {
		if out := tt.in.String(); out != tt.out {
			t.Errorf("IPMask.String(%v) = %q, want %q", tt.in, out, tt.out)
		}
	}
}

var parseCIDRTests = []struct {
	in  string
	ip  IP
	net *IPNet
	err error
}{
	{"135.104.0.0/32", IPv4(135, 104, 0, 0), &IPNet{IP: IPv4(135, 104, 0, 0), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"0.0.0.0/24", IPv4(0, 0, 0, 0), &IPNet{IP: IPv4(0, 0, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)}, nil},
	{"135.104.0.0/24", IPv4(135, 104, 0, 0), &IPNet{IP: IPv4(135, 104, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)}, nil},
	{"135.104.0.1/32", IPv4(135, 104, 0, 1), &IPNet{IP: IPv4(135, 104, 0, 1), Mask: IPv4Mask(255, 255, 255, 255)}, nil},
	{"135.104.0.1/24", IPv4(135, 104, 0, 1), &IPNet{IP: IPv4(135, 104, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)}, nil},
	{"::1/128", ParseIP("::1"), &IPNet{IP: ParseIP("::1"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"))}, nil},
	{"abcd:2345::/127", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe"))}, nil},
	{"abcd:2345::/65", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:8000::"))}, nil},
	{"abcd:2345::/64", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff::"))}, nil},
	{"abcd:2345::/63", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:fffe::"))}, nil},
	{"abcd:2345::/33", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff:8000::"))}, nil},
	{"abcd:2345::/32", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2345::"), Mask: IPMask(ParseIP("ffff:ffff::"))}, nil},
	{"abcd:2344::/31", ParseIP("abcd:2344::"), &IPNet{IP: ParseIP("abcd:2344::"), Mask: IPMask(ParseIP("ffff:fffe::"))}, nil},
	{"abcd:2300::/24", ParseIP("abcd:2300::"), &IPNet{IP: ParseIP("abcd:2300::"), Mask: IPMask(ParseIP("ffff:ff00::"))}, nil},
	{"abcd:2345::/24", ParseIP("abcd:2345::"), &IPNet{IP: ParseIP("abcd:2300::"), Mask: IPMask(ParseIP("ffff:ff00::"))}, nil},
	{"2001:DB8::/48", ParseIP("2001:DB8::"), &IPNet{IP: ParseIP("2001:DB8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff::"))}, nil},
	{"2001:DB8::1/48", ParseIP("2001:DB8::1"), &IPNet{IP: ParseIP("2001:DB8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff::"))}, nil},
	{"192.168.1.1/255.255.255.0", nil, nil, &ParseError{"CIDR address", "192.168.1.1/255.255.255.0"}},
	{"192.168.1.1/35", nil, nil, &ParseError{"CIDR address", "192.168.1.1/35"}},
	{"2001:db8::1/-1", nil, nil, &ParseError{"CIDR address", "2001:db8::1/-1"}},
	{"", nil, nil, &ParseError{"CIDR address", ""}},
}

func TestParseCIDR(t *testing.T) {
	for _, tt := range parseCIDRTests {
		ip, net, err := ParseCIDR(tt.in)
		if !reflect.DeepEqual(err, tt.err) {
			t.Errorf("ParseCIDR(%q) = %v, %v; want %v, %v", tt.in, ip, net, tt.ip, tt.net)
		}
		if err == nil && (!tt.ip.Equal(ip) || !tt.net.IP.Equal(net.IP) || !reflect.DeepEqual(net.Mask, tt.net.Mask)) {
			t.Errorf("ParseCIDR(%q) = %v, {%v, %v}; want %v, {%v, %v}", tt.in, ip, net.IP, net.Mask, tt.ip, tt.net.IP, tt.net.Mask)
		}
	}
}

var ipNetContainsTests = []struct {
	ip  IP
	net *IPNet
	ok  bool
}{
	{IPv4(172, 16, 1, 1), &IPNet{IP: IPv4(172, 16, 0, 0), Mask: CIDRMask(12, 32)}, true},
	{IPv4(172, 24, 0, 1), &IPNet{IP: IPv4(172, 16, 0, 0), Mask: CIDRMask(13, 32)}, false},
	{IPv4(192, 168, 0, 3), &IPNet{IP: IPv4(192, 168, 0, 0), Mask: IPv4Mask(0, 0, 255, 252)}, true},
	{IPv4(192, 168, 0, 4), &IPNet{IP: IPv4(192, 168, 0, 0), Mask: IPv4Mask(0, 255, 0, 252)}, false},
	{ParseIP("2001:db8:1:2::1"), &IPNet{IP: ParseIP("2001:db8:1::"), Mask: CIDRMask(47, 128)}, true},
	{ParseIP("2001:db8:1:2::1"), &IPNet{IP: ParseIP("2001:db8:2::"), Mask: CIDRMask(47, 128)}, false},
	{ParseIP("2001:db8:1:2::1"), &IPNet{IP: ParseIP("2001:db8:1::"), Mask: IPMask(ParseIP("ffff:0:ffff::"))}, true},
	{ParseIP("2001:db8:1:2::1"), &IPNet{IP: ParseIP("2001:db8:1::"), Mask: IPMask(ParseIP("0:0:0:ffff::"))}, false},
}

func TestIPNetContains(t *testing.T) {
	for _, tt := range ipNetContainsTests {
		if ok := tt.net.Contains(tt.ip); ok != tt.ok {
			t.Errorf("IPNet(%v).Contains(%v) = %v, want %v", tt.net, tt.ip, ok, tt.ok)
		}
	}
}

var ipNetStringTests = []struct {
	in  *IPNet
	out string
}{
	{&IPNet{IP: IPv4(192, 168, 1, 0), Mask: CIDRMask(26, 32)}, "192.168.1.0/26"},
	{&IPNet{IP: IPv4(192, 168, 1, 0), Mask: IPv4Mask(255, 0, 255, 0)}, "192.168.1.0/ff00ff00"},
	{&IPNet{IP: ParseIP("2001:db8::"), Mask: CIDRMask(55, 128)}, "2001:db8::/55"},
	{&IPNet{IP: ParseIP("2001:db8::"), Mask: IPMask(ParseIP("8000:f123:0:cafe::"))}, "2001:db8::/8000f1230000cafe0000000000000000"},
}

func TestIPNetString(t *testing.T) {
	for _, tt := range ipNetStringTests {
		if out := tt.in.String(); out != tt.out {
			t.Errorf("IPNet.String(%v) = %q, want %q", tt.in, out, tt.out)
		}
	}
}

var cidrMaskTests = []struct {
	ones int
	bits int
	out  IPMask
}{
	{0, 32, IPv4Mask(0, 0, 0, 0)},
	{12, 32, IPv4Mask(255, 240, 0, 0)},
	{24, 32, IPv4Mask(255, 255, 255, 0)},
	{32, 32, IPv4Mask(255, 255, 255, 255)},
	{0, 128, IPMask{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{4, 128, IPMask{0xf0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{48, 128, IPMask{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{128, 128, IPMask{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
	{33, 32, nil},
	{32, 33, nil},
	{-1, 128, nil},
	{128, -1, nil},
}

func TestCIDRMask(t *testing.T) {
	for _, tt := range cidrMaskTests {
		if out := CIDRMask(tt.ones, tt.bits); !reflect.DeepEqual(out, tt.out) {
			t.Errorf("CIDRMask(%v, %v) = %v, want %v", tt.ones, tt.bits, out, tt.out)
		}
	}
}

var (
	v4addr         = IP{192, 168, 0, 1}
	v4mappedv6addr = IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 0, 1}
	v6addr         = IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0x1, 0x23, 0, 0x12, 0, 0x1}
	v4mask         = IPMask{255, 255, 255, 0}
	v4mappedv6mask = IPMask{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 255, 255, 255, 0}
	v6mask         = IPMask{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0}
	badaddr        = IP{192, 168, 0}
	badmask        = IPMask{255, 255, 0}
	v4maskzero     = IPMask{0, 0, 0, 0}
)

var networkNumberAndMaskTests = []struct {
	in  IPNet
	out IPNet
}{
	{IPNet{IP: v4addr, Mask: v4mask}, IPNet{IP: v4addr, Mask: v4mask}},
	{IPNet{IP: v4addr, Mask: v4mappedv6mask}, IPNet{IP: v4addr, Mask: v4mask}},
	{IPNet{IP: v4mappedv6addr, Mask: v4mappedv6mask}, IPNet{IP: v4addr, Mask: v4mask}},
	{IPNet{IP: v4mappedv6addr, Mask: v6mask}, IPNet{IP: v4addr, Mask: v4maskzero}},
	{IPNet{IP: v4addr, Mask: v6mask}, IPNet{IP: v4addr, Mask: v4maskzero}},
	{IPNet{IP: v6addr, Mask: v6mask}, IPNet{IP: v6addr, Mask: v6mask}},
	{IPNet{IP: v6addr, Mask: v4mappedv6mask}, IPNet{IP: v6addr, Mask: v4mappedv6mask}},
	{in: IPNet{IP: v6addr, Mask: v4mask}},
	{in: IPNet{IP: v4addr, Mask: badmask}},
	{in: IPNet{IP: v4mappedv6addr, Mask: badmask}},
	{in: IPNet{IP: v6addr, Mask: badmask}},
	{in: IPNet{IP: badaddr, Mask: v4mask}},
	{in: IPNet{IP: badaddr, Mask: v4mappedv6mask}},
	{in: IPNet{IP: badaddr, Mask: v6mask}},
	{in: IPNet{IP: badaddr, Mask: badmask}},
}

func TestNetworkNumberAndMask(t *testing.T) {
	for _, tt := range networkNumberAndMaskTests {
		ip, m := networkNumberAndMask(&tt.in)
		out := &IPNet{IP: ip, Mask: m}
		if !reflect.DeepEqual(&tt.out, out) {
			t.Errorf("networkNumberAndMask(%v) = %v, want %v", tt.in, out, &tt.out)
		}
	}
}

var splitJoinTests = []struct {
	host string
	port string
	join string
}{
	{"www.google.com", "80", "www.google.com:80"},
	{"127.0.0.1", "1234", "127.0.0.1:1234"},
	{"::1", "80", "[::1]:80"},
	{"fe80::1%lo0", "80", "[fe80::1%lo0]:80"},
	{"localhost%lo0", "80", "[localhost%lo0]:80"},
	{"", "0", ":0"},

	{"google.com", "https%foo", "google.com:https%foo"}, // Go 1.0 behavior
	{"127.0.0.1", "", "127.0.0.1:"},                     // Go 1.0 behaviour
	{"www.google.com", "", "www.google.com:"},           // Go 1.0 behaviour
}

var splitFailureTests = []struct {
	hostPort string
	err      string
}{
	{"www.google.com", "missing port in address"},
	{"127.0.0.1", "missing port in address"},
	{"[::1]", "missing port in address"},
	{"[fe80::1%lo0]", "missing port in address"},
	{"[localhost%lo0]", "missing port in address"},
	{"localhost%lo0", "missing port in address"},

	{"::1", "too many colons in address"},
	{"fe80::1%lo0", "too many colons in address"},
	{"fe80::1%lo0:80", "too many colons in address"},

	{"localhost%lo0:80", "missing brackets in address"},

	// Test cases that didn't fail in Go 1.0

	{"[foo:bar]", "missing port in address"},
	{"[foo:bar]baz", "missing port in address"},
	{"[foo]bar:baz", "missing port in address"},

	{"[foo]:[bar]:baz", "too many colons in address"},

	{"[foo]:[bar]baz", "unexpected '[' in address"},
	{"foo[bar]:baz", "unexpected '[' in address"},

	{"foo]bar:baz", "unexpected ']' in address"},
}

func TestSplitHostPort(t *testing.T) {
	for _, tt := range splitJoinTests {
		if host, port, err := SplitHostPort(tt.join); host != tt.host || port != tt.port || err != nil {
			t.Errorf("SplitHostPort(%q) = %q, %q, %v; want %q, %q, nil", tt.join, host, port, err, tt.host, tt.port)
		}
	}
	for _, tt := range splitFailureTests {
		if _, _, err := SplitHostPort(tt.hostPort); err == nil {
			t.Errorf("SplitHostPort(%q) should have failed", tt.hostPort)
		} else {
			e := err.(*AddrError)
			if e.Err != tt.err {
				t.Errorf("SplitHostPort(%q) = _, _, %q; want %q", tt.hostPort, e.Err, tt.err)
			}
		}
	}
}

func TestJoinHostPort(t *testing.T) {
	for _, tt := range splitJoinTests {
		if join := JoinHostPort(tt.host, tt.port); join != tt.join {
			t.Errorf("JoinHostPort(%q, %q) = %q; want %q", tt.host, tt.port, join, tt.join)
		}
	}
}

var ipAddrFamilyTests = []struct {
	in  IP
	af4 bool
	af6 bool
}{
	{IPv4bcast, true, false},
	{IPv4allsys, true, false},
	{IPv4allrouter, true, false},
	{IPv4zero, true, false},
	{IPv4(224, 0, 0, 1), true, false},
	{IPv4(127, 0, 0, 1), true, false},
	{IPv4(240, 0, 0, 1), true, false},
	{IPv6unspecified, false, true},
	{IPv6loopback, false, true},
	{IPv6interfacelocalallnodes, false, true},
	{IPv6linklocalallnodes, false, true},
	{IPv6linklocalallrouters, false, true},
	{ParseIP("ff05::a:b:c:d"), false, true},
	{ParseIP("fe80::1:2:3:4"), false, true},
	{ParseIP("2001:db8::123:12:1"), false, true},
}

func TestIPAddrFamily(t *testing.T) {
	for _, tt := range ipAddrFamilyTests {
		if af := tt.in.To4() != nil; af != tt.af4 {
			t.Errorf("verifying IPv4 address family for %q = %v, want %v", tt.in, af, tt.af4)
		}
		if af := len(tt.in) == IPv6len && tt.in.To4() == nil; af != tt.af6 {
			t.Errorf("verifying IPv6 address family for %q = %v, want %v", tt.in, af, tt.af6)
		}
	}
}

var ipAddrScopeTests = []struct {
	scope func(IP) bool
	in    IP
	ok    bool
}{
	{IP.IsUnspecified, IPv4zero, true},
	{IP.IsUnspecified, IPv4(127, 0, 0, 1), false},
	{IP.IsUnspecified, IPv6unspecified, true},
	{IP.IsUnspecified, IPv6interfacelocalallnodes, false},
	{IP.IsLoopback, IPv4(127, 0, 0, 1), true},
	{IP.IsLoopback, IPv4(127, 255, 255, 254), true},
	{IP.IsLoopback, IPv4(128, 1, 2, 3), false},
	{IP.IsLoopback, IPv6loopback, true},
	{IP.IsLoopback, IPv6linklocalallrouters, false},
	{IP.IsMulticast, IPv4(224, 0, 0, 0), true},
	{IP.IsMulticast, IPv4(239, 0, 0, 0), true},
	{IP.IsMulticast, IPv4(240, 0, 0, 0), false},
	{IP.IsMulticast, IPv6linklocalallnodes, true},
	{IP.IsMulticast, IP{0xff, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true},
	{IP.IsMulticast, IP{0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsLinkLocalMulticast, IPv4(224, 0, 0, 0), true},
	{IP.IsLinkLocalMulticast, IPv4(239, 0, 0, 0), false},
	{IP.IsLinkLocalMulticast, IPv6linklocalallrouters, true},
	{IP.IsLinkLocalMulticast, IP{0xff, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsLinkLocalUnicast, IPv4(169, 254, 0, 0), true},
	{IP.IsLinkLocalUnicast, IPv4(169, 255, 0, 0), false},
	{IP.IsLinkLocalUnicast, IP{0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true},
	{IP.IsLinkLocalUnicast, IP{0xfe, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsGlobalUnicast, IPv4(240, 0, 0, 0), true},
	{IP.IsGlobalUnicast, IPv4(232, 0, 0, 0), false},
	{IP.IsGlobalUnicast, IPv4(169, 254, 0, 0), false},
	{IP.IsGlobalUnicast, IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0x1, 0x23, 0, 0x12, 0, 0x1}, true},
	{IP.IsGlobalUnicast, IP{0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsGlobalUnicast, IP{0xff, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
}

func name(f interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

func TestIPAddrScope(t *testing.T) {
	for _, tt := range ipAddrScopeTests {
		if ok := tt.scope(tt.in); ok != tt.ok {
			t.Errorf("%s(%q) = %v, want %v", name(tt.scope), tt.in, ok, tt.ok)
		}
	}
}
