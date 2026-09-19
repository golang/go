// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import "testing"

func TestMkWidth(t *testing.T) {
	tests := []struct {
		num, denom int
		want       ScalableWidth
	}{
		{4, 8, ScalableWidth{1, 2}},
		{3, 9, ScalableWidth{1, 3}},
		{12, 18, ScalableWidth{2, 3}},
		{0, 5, ScalableWidth{0, 1}},
		// We should never have a negative scalable width, but test GCD on them just in case.
		{-4, 8, ScalableWidth{-1, 2}},
		{4, -8, ScalableWidth{-1, 2}},
		{-4, -8, ScalableWidth{1, 2}},
	}
	for _, tc := range tests {
		got := mkWidth(tc.num, tc.denom)
		if got != tc.want {
			t.Errorf("mkWidth(%d, %d) = %v; want %v", tc.num, tc.denom, got, tc.want)
		}
	}
}
