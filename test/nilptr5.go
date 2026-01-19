// errorcheck -0 -d=nil

//go:build !wasm && !aix

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that nil checks are removed.
// Optimization is enabled.

package p

func f5(p *float32, q *float64, r *float32, s *float64) float64 {
	x := float64(*p) // ERROR "removed nil check"
	y := *q          // ERROR "removed nil check"
	*r = 7           // ERROR "removed nil check"
	*s = 9           // ERROR "removed nil check"
	return x + y
}

type T struct{ b [29]byte }

func f6(p, q *T) {
	x := *p // ERROR "removed nil check"
	*q = x  // ERROR "removed nil check"
}

// make sure to remove nil check for memory move (issue #18003)
func f8(t *struct{ b [8]int }) struct{ b [8]int } {
	return *t // ERROR "removed nil check"
}

// nil check is removed for pointer write (which involves a
// write barrier).
func f9(x **int, y *int) {
	*x = y // ERROR "removed nil check"
}
