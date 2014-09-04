// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for some of the internal functions.

package main

import (
	"fmt"
	"testing"
)

// Helpers to save typing in the test cases.
type u []uint64
type uu [][]uint64

type SplitTest struct {
	input  u
	output uu
	signed bool
}

var (
	m2  = uint64(2)
	m1  = uint64(1)
	m0  = uint64(0)
	m_1 = ^uint64(0)     // -1 when signed.
	m_2 = ^uint64(0) - 1 // -2 when signed.
)

var splitTests = []SplitTest{
	// No need for a test for the empty case; that's picked off before splitIntoRuns.
	// Single value.
	{u{1}, uu{u{1}}, false},
	// Out of order.
	{u{3, 2, 1}, uu{u{1, 2, 3}}, true},
	// Out of order.
	{u{3, 2, 1}, uu{u{1, 2, 3}}, false},
	// A gap at the beginning.
	{u{1, 33, 32, 31}, uu{u{1}, u{31, 32, 33}}, true},
	// A gap in the middle, in mixed order.
	{u{33, 7, 32, 31, 9, 8}, uu{u{7, 8, 9}, u{31, 32, 33}}, true},
	// Gaps throughout
	{u{33, 44, 1, 32, 45, 31}, uu{u{1}, u{31, 32, 33}, u{44, 45}}, true},
	// Unsigned values spanning 0.
	{u{m1, m0, m_1, m2, m_2}, uu{u{m0, m1, m2}, u{m_2, m_1}}, false},
	// Signed values spanning 0
	{u{m1, m0, m_1, m2, m_2}, uu{u{m_2, m_1, m0, m1, m2}}, true},
}

func TestSplitIntoRuns(t *testing.T) {
Outer:
	for n, test := range splitTests {
		values := make([]Value, len(test.input))
		for i, v := range test.input {
			values[i] = Value{"", v, test.signed, fmt.Sprint(v)}
		}
		runs := splitIntoRuns(values)
		if len(runs) != len(test.output) {
			t.Errorf("#%d: %v: got %d runs; expected %d", n, test.input, len(runs), len(test.output))
			continue
		}
		for i, run := range runs {
			if len(run) != len(test.output[i]) {
				t.Errorf("#%d: got %v; expected %v", n, runs, test.output)
				continue Outer
			}
			for j, v := range run {
				if v.value != test.output[i][j] {
					t.Errorf("#%d: got %v; expected %v", n, runs, test.output)
					continue Outer
				}
			}
		}
	}
}
