// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"reflect"
	"testing"
	"os"
)

func isEqual(a, b []byte) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return bytes.Equal(a, b)
}

var parseiptests = []struct {
	in  string
	out IP
}{
	{"127.0.1.2", IPv4(127, 0, 1, 2)},
	{"127.0.0.1", IPv4(127, 0, 0, 1)},
	{"127.0.0.256", nil},
	{"abc", nil},
	{"123:", nil},
	{"::ffff:127.0.0.1", IPv4(127, 0, 0, 1)},
	{"2001:4860:0:2001::68",
		IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01,
			0, 0, 0, 0, 0, 0, 0x00, 0x68,
		},
	},
	{"::ffff:4a7d:1363", IPv4(74, 125, 19, 99)},
}

func TestParseIP(t *testing.T) {
	for _, tt := range parseiptests {
		if out := ParseIP(tt.in); !isEqual(out, tt.out) {
			t.Errorf("ParseIP(%#q) = %v, want %v", tt.in, out, tt.out)
		}
	}
}

var ipstringtests = []struct {
	in  IP
	out string
}{
	// cf. RFC 5952 (A Recommendation for IPv6 Address Text Representation)
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0,
		0, 0, 0x1, 0x23, 0, 0x12, 0, 0x1},
		"2001:db8::123:12:1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0x1},
		"2001:db8::1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0x1,
		0, 0, 0, 0x1, 0, 0, 0, 0x1},
		"2001:db8:0:1:0:1:0:1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0x1, 0, 0,
		0, 0x1, 0, 0, 0, 0x1, 0, 0},
		"2001:db8:1:0:1:0:1:0"},
	{IP{0x20, 0x1, 0, 0, 0, 0, 0, 0,
		0, 0x1, 0, 0, 0, 0, 0, 0x1},
		"2001::1:0:0:1"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0,
		0, 0x1, 0, 0, 0, 0, 0, 0},
		"2001:db8:0:0:1::"},
	{IP{0x20, 0x1, 0xd, 0xb8, 0, 0, 0, 0,
		0, 0x1, 0, 0, 0, 0, 0, 0x1},
		"2001:db8::1:0:0:1"},
	{IP{0x20, 0x1, 0xD, 0xB8, 0, 0, 0, 0,
		0, 0xA, 0, 0xB, 0, 0xC, 0, 0xD},
		"2001:db8::a:b:c:d"},
}

func TestIPString(t *testing.T) {
	for _, tt := range ipstringtests {
		if out := tt.in.String(); out != tt.out {
			t.Errorf("IP.String(%v) = %#q, want %#q", tt.in, out, tt.out)
		}
	}
}

var parsecidrtests = []struct {
	in   string
	ip   IP
	mask IPMask
	err  os.Error
}{
	{"135.104.0.0/32", IPv4(135, 104, 0, 0), IPv4Mask(255, 255, 255, 255), nil},
	{"0.0.0.0/24", IPv4(0, 0, 0, 0), IPv4Mask(255, 255, 255, 0), nil},
	{"135.104.0.0/24", IPv4(135, 104, 0, 0), IPv4Mask(255, 255, 255, 0), nil},
	{"135.104.0.1/32", IPv4(135, 104, 0, 1), IPv4Mask(255, 255, 255, 255), nil},
	{"135.104.0.1/24", nil, nil, &ParseError{"CIDR address", "135.104.0.1/24"}},
	{"::1/128", ParseIP("::1"), IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")), nil},
	{"abcd:2345::/127", ParseIP("abcd:2345::"), IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe")), nil},
	{"abcd:2345::/65", ParseIP("abcd:2345::"), IPMask(ParseIP("ffff:ffff:ffff:ffff:8000::")), nil},
	{"abcd:2345::/64", ParseIP("abcd:2345::"), IPMask(ParseIP("ffff:ffff:ffff:ffff::")), nil},
	{"abcd:2345::/63", ParseIP("abcd:2345::"), IPMask(ParseIP("ffff:ffff:ffff:fffe::")), nil},
	{"abcd:2345::/33", ParseIP("abcd:2345::"), IPMask(ParseIP("ffff:ffff:8000::")), nil},
	{"abcd:2345::/32", ParseIP("abcd:2345::"), IPMask(ParseIP("ffff:ffff::")), nil},
	{"abcd:2344::/31", ParseIP("abcd:2344::"), IPMask(ParseIP("ffff:fffe::")), nil},
	{"abcd:2300::/24", ParseIP("abcd:2300::"), IPMask(ParseIP("ffff:ff00::")), nil},
	{"abcd:2345::/24", nil, nil, &ParseError{"CIDR address", "abcd:2345::/24"}},
	{"2001:DB8::/48", ParseIP("2001:DB8::"), IPMask(ParseIP("ffff:ffff:ffff::")), nil},
}

func TestParseCIDR(t *testing.T) {
	for _, tt := range parsecidrtests {
		if ip, mask, err := ParseCIDR(tt.in); !isEqual(ip, tt.ip) || !isEqual(mask, tt.mask) || !reflect.DeepEqual(err, tt.err) {
			t.Errorf("ParseCIDR(%q) = %v, %v, %v; want %v, %v, %v", tt.in, ip, mask, err, tt.ip, tt.mask, tt.err)
		}
	}
}

var splitjointests = []struct {
	Host string
	Port string
	Join string
}{
	{"www.google.com", "80", "www.google.com:80"},
	{"127.0.0.1", "1234", "127.0.0.1:1234"},
	{"::1", "80", "[::1]:80"},
}

func TestSplitHostPort(t *testing.T) {
	for _, tt := range splitjointests {
		if host, port, err := SplitHostPort(tt.Join); host != tt.Host || port != tt.Port || err != nil {
			t.Errorf("SplitHostPort(%q) = %q, %q, %v; want %q, %q, nil", tt.Join, host, port, err, tt.Host, tt.Port)
		}
	}
}

func TestJoinHostPort(t *testing.T) {
	for _, tt := range splitjointests {
		if join := JoinHostPort(tt.Host, tt.Port); join != tt.Join {
			t.Errorf("JoinHostPort(%q, %q) = %q; want %q", tt.Host, tt.Port, join, tt.Join)
		}
	}
}
