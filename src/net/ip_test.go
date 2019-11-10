// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !js

package net

import (
	"bytes"
	"math/rand"
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
	{"127.001.002.003", IPv4(127, 1, 2, 3)},
	{"::ffff:127.1.2.3", IPv4(127, 1, 2, 3)},
	{"::ffff:127.001.002.003", IPv4(127, 1, 2, 3)},
	{"::ffff:7f01:0203", IPv4(127, 1, 2, 3)},
	{"0:0:0:0:0000:ffff:127.1.2.3", IPv4(127, 1, 2, 3)},
	{"0:0:0:0:000000:ffff:127.1.2.3", IPv4(127, 1, 2, 3)},
	{"0:0:0:0::ffff:127.1.2.3", IPv4(127, 1, 2, 3)},

	{"2001:4860:0:2001::68", IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01, 0, 0, 0, 0, 0, 0, 0x00, 0x68}},
	{"2001:4860:0000:2001:0000:0000:0000:0068", IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01, 0, 0, 0, 0, 0, 0, 0x00, 0x68}},

	{"-0.0.0.0", nil},
	{"0.-1.0.0", nil},
	{"0.0.-2.0", nil},
	{"0.0.0.-3", nil},
	{"127.0.0.256", nil},
	{"abc", nil},
	{"123:", nil},
	{"fe80::1%lo0", nil},
	{"fe80::1%911", nil},
	{"", nil},
	{"a1:a2:a3:a4::b1:b2:b3:b4", nil}, // Issue 6628
}

func TestParseIP(t *testing.T) {
	for _, tt := range parseIPTests {
		if out := ParseIP(tt.in); !reflect.DeepEqual(out, tt.out) {
			t.Errorf("ParseIP(%q) = %v, want %v", tt.in, out, tt.out)
		}
		if tt.in == "" {
			// Tested in TestMarshalEmptyIP below.
			continue
		}
		var out IP
		if err := out.UnmarshalText([]byte(tt.in)); !reflect.DeepEqual(out, tt.out) || (tt.out == nil) != (err != nil) {
			t.Errorf("IP.UnmarshalText(%q) = %v, %v, want %v", tt.in, out, err, tt.out)
		}
	}
}

func TestLookupWithIP(t *testing.T) {
	_, err := LookupIP("")
	if err == nil {
		t.Errorf(`LookupIP("") succeeded, should fail`)
	}
	_, err = LookupHost("")
	if err == nil {
		t.Errorf(`LookupIP("") succeeded, should fail`)
	}

	// Test that LookupHost and LookupIP, which normally
	// expect host names, work with IP addresses.
	for _, tt := range parseIPTests {
		if tt.out != nil {
			addrs, err := LookupHost(tt.in)
			if len(addrs) != 1 || addrs[0] != tt.in || err != nil {
				t.Errorf("LookupHost(%q) = %v, %v, want %v, nil", tt.in, addrs, err, []string{tt.in})
			}
		} else if !testing.Short() {
			// We can't control what the host resolver does; if it can resolve, say,
			// 127.0.0.256 or fe80::1%911 or a host named 'abc', who are we to judge?
			// Warn about these discrepancies but don't fail the test.
			addrs, err := LookupHost(tt.in)
			if err == nil {
				t.Logf("warning: LookupHost(%q) = %v, want error", tt.in, addrs)
			}
		}

		if tt.out != nil {
			ips, err := LookupIP(tt.in)
			if len(ips) != 1 || !reflect.DeepEqual(ips[0], tt.out) || err != nil {
				t.Errorf("LookupIP(%q) = %v, %v, want %v, nil", tt.in, ips, err, []IP{tt.out})
			}
		} else if !testing.Short() {
			ips, err := LookupIP(tt.in)
			// We can't control what the host resolver does. See above.
			if err == nil {
				t.Logf("warning: LookupIP(%q) = %v, want error", tt.in, ips)
			}
		}
	}
}

func BenchmarkParseIP(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		for _, tt := range parseIPTests {
			ParseIP(tt.in)
		}
	}
}

// Issue 6339
func TestMarshalEmptyIP(t *testing.T) {
	for _, in := range [][]byte{nil, []byte("")} {
		var out = IP{1, 2, 3, 4}
		if err := out.UnmarshalText(in); err != nil || out != nil {
			t.Errorf("UnmarshalText(%v) = %v, %v; want nil, nil", in, out, err)
		}
	}
	var ip IP
	got, err := ip.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []byte("")) {
		t.Errorf(`got %#v, want []byte("")`, got)
	}
}

var ipStringTests = []*struct {
	in  IP     // see RFC 791 and RFC 4291
	str string // see RFC 791, RFC 4291 and RFC 5952
	byt []byte
	error
}{
	// IPv4 address
	{
		IP{192, 0, 2, 1},
		"192.0.2.1",
		[]byte("192.0.2.1"),
		nil,
	},
	{
		IP{0, 0, 0, 0},
		"0.0.0.0",
		[]byte("0.0.0.0"),
		nil,
	},

	// IPv4-mapped IPv6 address
	{
		IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 0, 2, 1},
		"192.0.2.1",
		[]byte("192.0.2.1"),
		nil,
	},
	{
		IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0},
		"0.0.0.0",
		[]byte("0.0.0.0"),
		nil,
	},

	// IPv6 address
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0x1, 0x23, 0, 0x12, 0, 0x1},
		"2001:db8::123:12:1",
		[]byte("2001:db8::123:12:1"),
		nil,
	},
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x1},
		"2001:db8::1",
		[]byte("2001:db8::1"),
		nil,
	},
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0x1, 0, 0, 0, 0x1, 0, 0, 0, 0x1},
		"2001:db8:0:1:0:1:0:1",
		[]byte("2001:db8:0:1:0:1:0:1"),
		nil,
	},
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0x1, 0, 0, 0, 0x1, 0, 0, 0, 0x1, 0, 0},
		"2001:db8:1:0:1:0:1:0",
		[]byte("2001:db8:1:0:1:0:1:0"),
		nil,
	},
	{
		IP{0x20, 0x1, 0, 0, 0, 0, 0, 0, 0, 0x1, 0, 0, 0, 0, 0, 0x1},
		"2001::1:0:0:1",
		[]byte("2001::1:0:0:1"),
		nil,
	},
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0x1, 0, 0, 0, 0, 0, 0},
		"2001:db8:0:0:1::",
		[]byte("2001:db8:0:0:1::"),
		nil,
	},
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0x1, 0, 0, 0, 0, 0, 0x1},
		"2001:db8::1:0:0:1",
		[]byte("2001:db8::1:0:0:1"),
		nil,
	},
	{
		IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0xa, 0, 0xb, 0, 0xc, 0, 0xd},
		"2001:db8::a:b:c:d",
		[]byte("2001:db8::a:b:c:d"),
		nil,
	},
	{
		IPv6unspecified,
		"::",
		[]byte("::"),
		nil,
	},

	// IP wildcard equivalent address in Dial/Listen API
	{
		nil,
		"<nil>",
		nil,
		nil,
	},

	// Opaque byte sequence
	{
		IP{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef},
		"?0123456789abcdef",
		nil,
		&AddrError{Err: "invalid IP address", Addr: "0123456789abcdef"},
	},
}

func TestIPString(t *testing.T) {
	for _, tt := range ipStringTests {
		if out := tt.in.String(); out != tt.str {
			t.Errorf("IP.String(%v) = %q, want %q", tt.in, out, tt.str)
		}
		if out, err := tt.in.MarshalText(); !bytes.Equal(out, tt.byt) || !reflect.DeepEqual(err, tt.error) {
			t.Errorf("IP.MarshalText(%v) = %v, %v, want %v, %v", tt.in, out, err, tt.byt, tt.error)
		}
	}
}

var sink string

func BenchmarkIPString(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	b.Run("IPv4", func(b *testing.B) {
		benchmarkIPString(b, IPv4len)
	})

	b.Run("IPv6", func(b *testing.B) {
		benchmarkIPString(b, IPv6len)
	})
}

func benchmarkIPString(b *testing.B, size int) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, tt := range ipStringTests {
			if tt.in != nil && len(tt.in) == size {
				sink = tt.in.String()
			}
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

func BenchmarkIPMaskString(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		for _, tt := range ipMaskStringTests {
			sink = tt.in.String()
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
	{"192.168.1.1/255.255.255.0", nil, nil, &ParseError{Type: "CIDR address", Text: "192.168.1.1/255.255.255.0"}},
	{"192.168.1.1/35", nil, nil, &ParseError{Type: "CIDR address", Text: "192.168.1.1/35"}},
	{"2001:db8::1/-1", nil, nil, &ParseError{Type: "CIDR address", Text: "2001:db8::1/-1"}},
	{"2001:db8::1/-0", nil, nil, &ParseError{Type: "CIDR address", Text: "2001:db8::1/-0"}},
	{"-0.0.0.0/32", nil, nil, &ParseError{Type: "CIDR address", Text: "-0.0.0.0/32"}},
	{"0.-1.0.0/32", nil, nil, &ParseError{Type: "CIDR address", Text: "0.-1.0.0/32"}},
	{"0.0.-2.0/32", nil, nil, &ParseError{Type: "CIDR address", Text: "0.0.-2.0/32"}},
	{"0.0.0.-3/32", nil, nil, &ParseError{Type: "CIDR address", Text: "0.0.0.-3/32"}},
	{"0.0.0.0/-0", nil, nil, &ParseError{Type: "CIDR address", Text: "0.0.0.0/-0"}},
	{"", nil, nil, &ParseError{Type: "CIDR address", Text: ""}},
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

func TestSplitHostPort(t *testing.T) {
	for _, tt := range []struct {
		hostPort string
		host     string
		port     string
	}{
		// Host name
		{"localhost:http", "localhost", "http"},
		{"localhost:80", "localhost", "80"},

		// Go-specific host name with zone identifier
		{"localhost%lo0:http", "localhost%lo0", "http"},
		{"localhost%lo0:80", "localhost%lo0", "80"},
		{"[localhost%lo0]:http", "localhost%lo0", "http"}, // Go 1 behavior
		{"[localhost%lo0]:80", "localhost%lo0", "80"},     // Go 1 behavior

		// IP literal
		{"127.0.0.1:http", "127.0.0.1", "http"},
		{"127.0.0.1:80", "127.0.0.1", "80"},
		{"[::1]:http", "::1", "http"},
		{"[::1]:80", "::1", "80"},

		// IP literal with zone identifier
		{"[::1%lo0]:http", "::1%lo0", "http"},
		{"[::1%lo0]:80", "::1%lo0", "80"},

		// Go-specific wildcard for host name
		{":http", "", "http"}, // Go 1 behavior
		{":80", "", "80"},     // Go 1 behavior

		// Go-specific wildcard for service name or transport port number
		{"golang.org:", "golang.org", ""}, // Go 1 behavior
		{"127.0.0.1:", "127.0.0.1", ""},   // Go 1 behavior
		{"[::1]:", "::1", ""},             // Go 1 behavior

		// Opaque service name
		{"golang.org:https%foo", "golang.org", "https%foo"}, // Go 1 behavior
	} {
		if host, port, err := SplitHostPort(tt.hostPort); host != tt.host || port != tt.port || err != nil {
			t.Errorf("SplitHostPort(%q) = %q, %q, %v; want %q, %q, nil", tt.hostPort, host, port, err, tt.host, tt.port)
		}
	}

	for _, tt := range []struct {
		hostPort string
		err      string
	}{
		{"golang.org", "missing port in address"},
		{"127.0.0.1", "missing port in address"},
		{"[::1]", "missing port in address"},
		{"[fe80::1%lo0]", "missing port in address"},
		{"[localhost%lo0]", "missing port in address"},
		{"localhost%lo0", "missing port in address"},

		{"::1", "too many colons in address"},
		{"fe80::1%lo0", "too many colons in address"},
		{"fe80::1%lo0:80", "too many colons in address"},

		// Test cases that didn't fail in Go 1

		{"[foo:bar]", "missing port in address"},
		{"[foo:bar]baz", "missing port in address"},
		{"[foo]bar:baz", "missing port in address"},

		{"[foo]:[bar]:baz", "too many colons in address"},

		{"[foo]:[bar]baz", "unexpected '[' in address"},
		{"foo[bar]:baz", "unexpected '[' in address"},

		{"foo]bar:baz", "unexpected ']' in address"},
	} {
		if host, port, err := SplitHostPort(tt.hostPort); err == nil {
			t.Errorf("SplitHostPort(%q) should have failed", tt.hostPort)
		} else {
			e := err.(*AddrError)
			if e.Err != tt.err {
				t.Errorf("SplitHostPort(%q) = _, _, %q; want %q", tt.hostPort, e.Err, tt.err)
			}
			if host != "" || port != "" {
				t.Errorf("SplitHostPort(%q) = %q, %q, err; want %q, %q, err on failure", tt.hostPort, host, port, "", "")
			}
		}
	}
}

func TestJoinHostPort(t *testing.T) {
	for _, tt := range []struct {
		host     string
		port     string
		hostPort string
	}{
		// Host name
		{"localhost", "http", "localhost:http"},
		{"localhost", "80", "localhost:80"},

		// Go-specific host name with zone identifier
		{"localhost%lo0", "http", "localhost%lo0:http"},
		{"localhost%lo0", "80", "localhost%lo0:80"},

		// IP literal
		{"127.0.0.1", "http", "127.0.0.1:http"},
		{"127.0.0.1", "80", "127.0.0.1:80"},
		{"::1", "http", "[::1]:http"},
		{"::1", "80", "[::1]:80"},

		// IP literal with zone identifier
		{"::1%lo0", "http", "[::1%lo0]:http"},
		{"::1%lo0", "80", "[::1%lo0]:80"},

		// Go-specific wildcard for host name
		{"", "http", ":http"}, // Go 1 behavior
		{"", "80", ":80"},     // Go 1 behavior

		// Go-specific wildcard for service name or transport port number
		{"golang.org", "", "golang.org:"}, // Go 1 behavior
		{"127.0.0.1", "", "127.0.0.1:"},   // Go 1 behavior
		{"::1", "", "[::1]:"},             // Go 1 behavior

		// Opaque service name
		{"golang.org", "https%foo", "golang.org:https%foo"}, // Go 1 behavior
	} {
		if hostPort := JoinHostPort(tt.host, tt.port); hostPort != tt.hostPort {
			t.Errorf("JoinHostPort(%q, %q) = %q; want %q", tt.host, tt.port, hostPort, tt.hostPort)
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
	{IP.IsUnspecified, nil, false},
	{IP.IsLoopback, IPv4(127, 0, 0, 1), true},
	{IP.IsLoopback, IPv4(127, 255, 255, 254), true},
	{IP.IsLoopback, IPv4(128, 1, 2, 3), false},
	{IP.IsLoopback, IPv6loopback, true},
	{IP.IsLoopback, IPv6linklocalallrouters, false},
	{IP.IsLoopback, nil, false},
	{IP.IsMulticast, IPv4(224, 0, 0, 0), true},
	{IP.IsMulticast, IPv4(239, 0, 0, 0), true},
	{IP.IsMulticast, IPv4(240, 0, 0, 0), false},
	{IP.IsMulticast, IPv6linklocalallnodes, true},
	{IP.IsMulticast, IP{0xff, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true},
	{IP.IsMulticast, IP{0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsMulticast, nil, false},
	{IP.IsInterfaceLocalMulticast, IPv4(224, 0, 0, 0), false},
	{IP.IsInterfaceLocalMulticast, IPv4(0xff, 0x01, 0, 0), false},
	{IP.IsInterfaceLocalMulticast, IPv6interfacelocalallnodes, true},
	{IP.IsInterfaceLocalMulticast, nil, false},
	{IP.IsLinkLocalMulticast, IPv4(224, 0, 0, 0), true},
	{IP.IsLinkLocalMulticast, IPv4(239, 0, 0, 0), false},
	{IP.IsLinkLocalMulticast, IPv4(0xff, 0x02, 0, 0), false},
	{IP.IsLinkLocalMulticast, IPv6linklocalallrouters, true},
	{IP.IsLinkLocalMulticast, IP{0xff, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsLinkLocalMulticast, nil, false},
	{IP.IsLinkLocalUnicast, IPv4(169, 254, 0, 0), true},
	{IP.IsLinkLocalUnicast, IPv4(169, 255, 0, 0), false},
	{IP.IsLinkLocalUnicast, IPv4(0xfe, 0x80, 0, 0), false},
	{IP.IsLinkLocalUnicast, IP{0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, true},
	{IP.IsLinkLocalUnicast, IP{0xfe, 0xc0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsLinkLocalUnicast, nil, false},
	{IP.IsGlobalUnicast, IPv4(240, 0, 0, 0), true},
	{IP.IsGlobalUnicast, IPv4(232, 0, 0, 0), false},
	{IP.IsGlobalUnicast, IPv4(169, 254, 0, 0), false},
	{IP.IsGlobalUnicast, IPv4bcast, false},
	{IP.IsGlobalUnicast, IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0x1, 0x23, 0, 0x12, 0, 0x1}, true},
	{IP.IsGlobalUnicast, IP{0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsGlobalUnicast, IP{0xff, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false},
	{IP.IsGlobalUnicast, nil, false},
}

func name(f interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

func TestIPAddrScope(t *testing.T) {
	for _, tt := range ipAddrScopeTests {
		if ok := tt.scope(tt.in); ok != tt.ok {
			t.Errorf("%s(%q) = %v, want %v", name(tt.scope), tt.in, ok, tt.ok)
		}
		ip := tt.in.To4()
		if ip == nil {
			continue
		}
		if ok := tt.scope(ip); ok != tt.ok {
			t.Errorf("%s(%q) = %v, want %v", name(tt.scope), ip, ok, tt.ok)
		}
	}
}

func BenchmarkIPEqual(b *testing.B) {
	b.Run("IPv4", func(b *testing.B) {
		benchmarkIPEqual(b, IPv4len)
	})
	b.Run("IPv6", func(b *testing.B) {
		benchmarkIPEqual(b, IPv6len)
	})
}

func benchmarkIPEqual(b *testing.B, size int) {
	ips := make([]IP, 1000)
	for i := range ips {
		ips[i] = make(IP, size)
		rand.Read(ips[i])
	}
	// Half of the N are equal.
	for i := 0; i < b.N/2; i++ {
		x := ips[i%len(ips)]
		y := ips[i%len(ips)]
		x.Equal(y)
	}
	// The other half are not equal.
	for i := 0; i < b.N/2; i++ {
		x := ips[i%len(ips)]
		y := ips[(i+1)%len(ips)]
		x.Equal(y)
	}
}
