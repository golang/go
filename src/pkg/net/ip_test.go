// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"testing"
)

func isEqual(a, b IP) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil || len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type parseIPTest struct {
	in  string
	out IP
}

var parseiptests = []parseIPTest{
	{"127.0.1.2", IPv4(127, 0, 1, 2)},
	{"127.0.0.1", IPv4(127, 0, 0, 1)},
	{"127.0.0.256", nil},
	{"abc", nil},
	{"::ffff:127.0.0.1", IPv4(127, 0, 0, 1)},
	{"2001:4860:0:2001::68",
		IP{0x20, 0x01, 0x48, 0x60, 0, 0, 0x20, 0x01,
			0, 0, 0, 0, 0, 0, 0x00, 0x68,
		},
	},
	{"::ffff:4a7d:1363", IPv4(74, 125, 19, 99)},
}

func TestParseIP(t *testing.T) {
	for i := 0; i < len(parseiptests); i++ {
		tt := parseiptests[i]
		if out := ParseIP(tt.in); !isEqual(out, tt.out) {
			t.Errorf("ParseIP(%#q) = %v, want %v", tt.in, out, tt.out)
		}
	}
}

type ipStringTest struct {
	in  IP
	out string
}

var ipstringtests = []ipStringTest{
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
	for i := 0; i < len(ipstringtests); i++ {
		tt := ipstringtests[i]
		if out := tt.in.String(); out != tt.out {
			t.Errorf("IP.String(%v) = %#q, want %#q", tt.in, out, tt.out)
		}
	}
}
