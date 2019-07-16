// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "testing"

var parsePortTests = []struct {
	service     string
	port        int
	needsLookup bool
}{
	{"", 0, false},

	// Decimal number literals
	{"-1073741825", -1 << 30, false},
	{"-1073741824", -1 << 30, false},
	{"-1073741823", -(1<<30 - 1), false},
	{"-123456789", -123456789, false},
	{"-1", -1, false},
	{"-0", 0, false},
	{"0", 0, false},
	{"+0", 0, false},
	{"+1", 1, false},
	{"65535", 65535, false},
	{"65536", 65536, false},
	{"123456789", 123456789, false},
	{"1073741822", 1<<30 - 2, false},
	{"1073741823", 1<<30 - 1, false},
	{"1073741824", 1<<30 - 1, false},
	{"1073741825", 1<<30 - 1, false},

	// Others
	{"abc", 0, true},
	{"9pfs", 0, true},
	{"123badport", 0, true},
	{"bad123port", 0, true},
	{"badport123", 0, true},
	{"123456789badport", 0, true},
	{"-2147483649badport", 0, true},
	{"2147483649badport", 0, true},
}

func TestParsePort(t *testing.T) {
	// The following test cases are cribbed from the strconv
	for _, tt := range parsePortTests {
		if port, needsLookup := parsePort(tt.service); port != tt.port || needsLookup != tt.needsLookup {
			t.Errorf("parsePort(%q) = %d, %t; want %d, %t", tt.service, port, needsLookup, tt.port, tt.needsLookup)
		}
	}
}
