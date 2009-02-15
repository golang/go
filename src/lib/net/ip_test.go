// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"testing"
)

func ipv4(a, b, c, d byte) []byte {
	return []byte( 0,0,0,0, 0,0,0,0, 0,0,255,255, a,b,c,d )
}

func isEqual(a []byte, b []byte) bool {
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
	in string;
	out []byte;
}
var parseiptests = []parseIPTest (
	parseIPTest("127.0.1.2", ipv4(127, 0, 1, 2)),
	parseIPTest("127.0.0.1", ipv4(127, 0, 0, 1)),
	parseIPTest("127.0.0.256", nil),
	parseIPTest("abc", nil),
	parseIPTest("::ffff:127.0.0.1", ipv4(127, 0, 0, 1)),
	parseIPTest("2001:4860:0:2001::68",
		[]byte(0x20,0x01, 0x48,0x60, 0,0, 0x20,0x01, 0,0, 0,0, 0,0, 0x00,0x68)),
	parseIPTest("::ffff:4a7d:1363", ipv4(74, 125, 19, 99)),
)

func TestParseIP(t *testing.T) {
	for i := 0; i < len(parseiptests); i++ {
		tt := parseiptests[i];
		if out := ParseIP(tt.in); !isEqual(out, tt.out) {
			t.Errorf("ParseIP(%#q) = %v, want %v", tt.in, out, tt.out);
		}
	}
}
