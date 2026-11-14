// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"strings"
	"testing"
)

func TestIntegerMarshalSFV(t *testing.T) {
	data := []struct {
		in       int64
		expected string
		valid    bool
	}{
		{10, "10", true},
		{-10, "-10", true},
		{0, "0", true},
		{-999999999999999, "-999999999999999", true},
		{999999999999999, "999999999999999", true},
		{-9999999999999999, "", false},
		{9999999999999999, "", false},
	}

	var b strings.Builder

	for _, d := range data {
		b.Reset()

		err := marshalInteger(&b, d.in)
		if d.valid && err != nil {
			t.Errorf("error not expected for %v, got %v", d.in, err)
		} else if !d.valid && err == nil {
			t.Errorf("error expected for %v, got %v", d.in, err)
		}

		if b.String() != d.expected {
			t.Errorf("got %v; want %v", b.String(), d.expected)
		}
	}
}

func TestParseIntegerOrDecimal(t *testing.T) {
	data := []struct {
		in       string
		expected interface{}
		valid    bool
	}{
		{"1871", int64(1871), false},
		{"-1871", int64(-1871), false},
		{"18.71", 18.71, false},
		{"-18.71", -18.71, false},
		{"1871next", int64(1871), false},
		{"-18.71next", -18.71, false},
		{"-18.710", -18.71, false},
		{"a", 0, true},
		{"10.", 0, true},
		{"10.1234", 0, true},
		{"-", 0, true},
		{"1234567890123456", 0, true},
		{"123456789012345.6", 0, true},
		{"1234567890123.", 0, true},
		{"-9999999999999991", 0, true},
		{"9999999999999991", 0, true},
	}

	for _, d := range data {
		s := &scanner{data: d.in}

		i, err := parseNumber(s)
		if d.valid && err == nil {
			t.Errorf("parseIntegerOrDecimal(%s): error expected", d.in)
		}

		if !d.valid && d.expected != i {
			t.Errorf("parseIntegerOrDecimal(%s) = %v, %v; %v, <nil> expected", d.in, i, err, d.expected)
		}
	}
}
