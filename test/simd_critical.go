// errorcheck -0 -d=ssa/critical/debug=1

//go:build goexperiment.simd && amd64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that blocks created by the critical pass to split critical
// edges inherit the CPU features of the edge they sit on, so that
// later consumers (e.g. regalloc-inserted shuffle copies) can use
// feature-dependent instruction encodings there. There are three
// paths through the pass, each exercised below:
//
//  1. a fresh split block for an edge into a block with a single phi
//  2. a fresh split block for an edge into a block with several phis
//     (or none)
//  3. a split block reused for several predecessor edges carrying the
//     same phi argument, which keeps only the features all of its
//     predecessors guarantee

package foo

import "simd/archsimd"

var cond bool

// Case 1: the merge block has a single phi (x), so the split block for
// the critical edge is created on the single-phi path. It inherits
// avx from its predecessor and successor.
func singlePhi(a, b archsimd.Int64x4) archsimd.Int64x4 {
	x := a.Add(b)
	if cond { // ERROR "split critical edge" "split-edge block b[0-9]+ has features avx$"
		x = x.Add(a)
	}
	return x
}

// Case 2: the merge block has two phis (x and y), so the split block
// for the critical edge is created on the no-single-phi path.
func multiPhi(a, b archsimd.Int64x4) (archsimd.Int64x4, archsimd.Int64x4) {
	x := a.Add(b)
	y := b.Sub(a)
	if cond { // ERROR "split critical edge" "split-edge block b[0-9]+ has features avx$"
		x = x.Add(a)
		y = y.Sub(b)
	}
	return x, y
}

// Case 3: the short-circuit && evaluates each operand in its own
// block, and both false edges jump to the single-phi merge block with
// the same phi argument (the zero value of x). The split block is
// created for the edge from the second operand's block, whose features
// are avx+avx2+avx512 (it is dominated by the first operand's block
// and holds the 512-bit ops); when it is reused for the edge from the
// first operand's block, which only guarantees avx, its features must
// drop to the common subset rather than keep avx512.
func reuseIntersect(s []int64, s8 []int64) {
	var x archsimd.Int64x4
	if archsimd.LoadInt64x4(s).IsZero() && // ERROR "split critical edge" "reused split-edge block b[0-9]+ has features avx$" "split-edge block b[0-9]+ has features avx$"
		archsimd.LoadInt64x8(s8).Equal(archsimd.LoadInt64x8(s8)).ToBits() != 0 { // ERROR "split critical edge" "split-edge block b[0-9]+ has features avx[+]avx2[+]avx512$"
		x = archsimd.LoadInt64x4(s)
	}
	x.Store(s)
}
