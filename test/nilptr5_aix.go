// errorcheck -0 -d=nil

//go:build aix

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that nil checks are removed.
// Optimization is enabled.

package p

func f5(p *float32, q *float64, r *float32, s *float64) float64 {
	x := float64(*p) // ERROR "generated nil check"
	y := *q          // ERROR "generated nil check"
	*r = 7           // ERROR "removed nil check"
	*s = 9           // ERROR "removed nil check"
	return x + y
}

type T [29]byte

func f6(p, q *T) {
	x := *p // ERROR "generated nil check"
	*q = x  // ERROR "removed nil check"
}

// make sure to remove nil check for memory move (issue #18003)
func f8(t *[8]int) [8]int {
	return *t // ERROR "generated nil check"
}

// On AIX, a write nil check is removed, but a read nil check
// remains (for the write barrier).
func f9(x **int, y *int) {
	*x = y // ERROR "generated nil check" "removed nil check"
}
