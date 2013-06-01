// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"testing"
)

var roundDownTests = []struct {
	v, expected int
}{
	{1, 1},
	{9, 1},
	{10, 10},
	{11, 10},
	{100, 100},
	{101, 100},
	{999, 100},
	{1000, 1000},
	{1001, 1000},
}

func TestRoundDown10(t *testing.T) {
	for _, tt := range roundDownTests {
		actual := testing.RoundDown10(tt.v)
		if tt.expected != actual {
			t.Errorf("roundDown10(%d): expected %d, actual %d", tt.v, tt.expected, actual)
		}
	}
}

var roundUpTests = []struct {
	v, expected int
}{
	{0, 1},
	{1, 1},
	{2, 2},
	{5, 5},
	{9, 10},
	{999, 1000},
	{1000, 1000},
	{1400, 2000},
	{1700, 2000},
	{4999, 5000},
	{5000, 5000},
	{5001, 10000},
}

func TestRoundUp(t *testing.T) {
	for _, tt := range roundUpTests {
		actual := testing.RoundUp(tt.v)
		if tt.expected != actual {
			t.Errorf("roundUp(%d): expected %d, actual %d", tt.v, tt.expected, actual)
		}
	}
}
