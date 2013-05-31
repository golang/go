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
	{10, 1},
	{11, 10},
	{100, 10},
	//	{101, 100}, // issue 5599
	{1000, 100},
	//	{1001, 1000}, // issue 5599
}

func TestRoundDown10(t *testing.T) {
	for _, tt := range roundDownTests {
		actual := testing.RoundDown10(tt.v)
		if tt.expected != actual {
			t.Errorf("roundDown10: expected %v, actual %v", tt.expected, actual)
		}
	}
}
