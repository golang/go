// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regression coverage for the late spill/reload pair coalescer on arm64
// (cmd/compile/internal/arm64.pairSpills). When the coalescer fused two
// adjacent AMOVD reloads into a single LDP, paths that branched directly to
// the second reload would silently skip the fused load and observe a stale
// or uninitialized register. The fix records branch and jump-table targets
// before fusing and refuses to fuse when the second instruction is one.
// These functions exercise patterns that produce strictly-adjacent reloads
// and conditional branches over calls, including the shape from
// runtime.schedule.
//
// The check is end-to-end: each function returns a value that depends on
// every spilled variable, so a missing reload produces a wrong result and
// the program panics.

package main

import "fmt"

//go:noinline
func sink(int) {}

//go:noinline
func sink2(int, int) {}

func callTwoVars(p, q *int) (int, int) {
	a := *p
	b := *q
	sink(0)
	return a, b
}

func condCallTwoVars(c bool, p, q *int) (int, int) {
	a := *p
	b := *q
	if c {
		sink(0)
	}
	return a, b
}

// condCallThreeVars mimics the shape that produced the original
// miscompile: a conditional call surrounded by values that need to
// survive it, with the join landing on the second of two adjacent
// reloads.
func condCallThreeVars(c bool, p, q, r *int) (int, int, int) {
	a := *p
	b := *q
	d := *r
	if c {
		sink2(a, b)
	}
	return a, b, d
}

func loopReload(p []int) int {
	s := 0
	for _, v := range p {
		sink(v)
		s += v
	}
	return s
}

func nestedConds(c1, c2 bool, p, q, r *int) (int, int, int) {
	a := *p
	b := *q
	d := *r
	if c1 {
		sink(a)
		if c2 {
			sink(b)
		}
	}
	return a, b, d
}

func main() {
	x, y, z := 7, 11, 13
	if a, b := callTwoVars(&x, &y); a != 7 || b != 11 {
		panic(fmt.Sprintf("callTwoVars = %d, %d", a, b))
	}
	for _, c := range []bool{false, true} {
		if a, b := condCallTwoVars(c, &x, &y); a != 7 || b != 11 {
			panic(fmt.Sprintf("condCallTwoVars(%v) = %d, %d", c, a, b))
		}
		if a, b, d := condCallThreeVars(c, &x, &y, &z); a != 7 || b != 11 || d != 13 {
			panic(fmt.Sprintf("condCallThreeVars(%v) = %d, %d, %d", c, a, b, d))
		}
	}
	for _, c1 := range []bool{false, true} {
		for _, c2 := range []bool{false, true} {
			if a, b, d := nestedConds(c1, c2, &x, &y, &z); a != 7 || b != 11 || d != 13 {
				panic(fmt.Sprintf("nestedConds(%v,%v) = %d, %d, %d", c1, c2, a, b, d))
			}
		}
	}
	if s := loopReload([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}); s != 55 {
		panic(fmt.Sprintf("loopReload = %d", s))
	}
}
