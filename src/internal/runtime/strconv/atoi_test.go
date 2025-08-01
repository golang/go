// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"internal/runtime/strconv"
	"testing"
)

const intSize = 32 << (^uint(0) >> 63)

type atoi64Test struct {
	in  string
	out int64
	ok  bool
}

var atoi64tests = []atoi64Test{
	{"", 0, false},
	{"0", 0, true},
	{"-0", 0, true},
	{"1", 1, true},
	{"-1", -1, true},
	{"12345", 12345, true},
	{"-12345", -12345, true},
	{"012345", 12345, true},
	{"-012345", -12345, true},
	{"12345x", 0, false},
	{"-12345x", 0, false},
	{"98765432100", 98765432100, true},
	{"-98765432100", -98765432100, true},
	{"20496382327982653440", 0, false},
	{"-20496382327982653440", 0, false},
	{"9223372036854775807", 1<<63 - 1, true},
	{"-9223372036854775807", -(1<<63 - 1), true},
	{"9223372036854775808", 0, false},
	{"-9223372036854775808", -1 << 63, true},
	{"9223372036854775809", 0, false},
	{"-9223372036854775809", 0, false},
}

func TestAtoi(t *testing.T) {
	switch intSize {
	case 32:
		for i := range atoi32tests {
			test := &atoi32tests[i]
			out, ok := strconv.Atoi(test.in)
			if test.out != int32(out) || test.ok != ok {
				t.Errorf("Atoi(%q) = (%v, %v) want (%v, %v)",
					test.in, out, ok, test.out, test.ok)
			}
		}
	case 64:
		for i := range atoi64tests {
			test := &atoi64tests[i]
			out, ok := strconv.Atoi(test.in)
			if test.out != int64(out) || test.ok != ok {
				t.Errorf("Atoi(%q) = (%v, %v) want (%v, %v)",
					test.in, out, ok, test.out, test.ok)
			}
		}
	}
}

type atoi32Test struct {
	in  string
	out int32
	ok  bool
}

var atoi32tests = []atoi32Test{
	{"", 0, false},
	{"0", 0, true},
	{"-0", 0, true},
	{"1", 1, true},
	{"-1", -1, true},
	{"12345", 12345, true},
	{"-12345", -12345, true},
	{"012345", 12345, true},
	{"-012345", -12345, true},
	{"12345x", 0, false},
	{"-12345x", 0, false},
	{"987654321", 987654321, true},
	{"-987654321", -987654321, true},
	{"2147483647", 1<<31 - 1, true},
	{"-2147483647", -(1<<31 - 1), true},
	{"2147483648", 0, false},
	{"-2147483648", -1 << 31, true},
	{"2147483649", 0, false},
	{"-2147483649", 0, false},
}

func TestAtoi32(t *testing.T) {
	for i := range atoi32tests {
		test := &atoi32tests[i]
		out, ok := strconv.Atoi32(test.in)
		if test.out != out || test.ok != ok {
			t.Errorf("Atoi32(%q) = (%v, %v) want (%v, %v)",
				test.in, out, ok, test.out, test.ok)
		}
	}
}
