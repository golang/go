// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"testing"
)

func IPv4(a, b, c, d byte) []byte {
	return []byte{ 0,0,0,0, 0,0,0,0, 0,0,255,255, a,b,c,d }
}

func Equal(a []byte, b []byte) bool {
	if a == b {
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

type ParseIPTest struct {
	in string;
	out []byte;
}
var parseiptests = []ParseIPTest {
	ParseIPTest{"127.0.1.2", IPv4(127, 0, 1, 2)},
	ParseIPTest{"127.0.0.1", IPv4(127, 0, 0, 1)},
	ParseIPTest{"127.0.0.256", nil},
	ParseIPTest{"abc", nil},
	ParseIPTest{"::ffff:127.0.0.1", IPv4(127, 0, 0, 1)},
	ParseIPTest{"2001:4860:0:2001::68",
		[]byte{0x20,0x01, 0x48,0x60, 0,0, 0x20,0x01, 0,0, 0,0, 0,0, 0x00,0x68}},
	ParseIPTest{"::ffff:4a7d:1363", IPv4(74, 125, 19, 99)},
}

export func TestParseIP(t *testing.T) {
	for i := 0; i < len(parseiptests); i++ {
		tt := parseiptests[i];
		if out := ParseIP(tt.in); !Equal(out, tt.out) {
			t.Errorf("ParseIP(%#q) = %v, want %v", tt.in, out, tt.out);
		}
	}
}
