// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && (amd64 || arm64)

package simd_test

// clmul64 returns the 128-bit carryless product of a and b.
func clmul64(a, b uint64) (lo, hi uint64) {
	for i := range 64 {
		if b>>i&1 != 0 {
			lo ^= a << i
			if i > 0 {
				hi ^= a >> (64 - i)
			}
		}
	}
	return lo, hi
}

// clmulSlice returns a slice simulation of the CarrylessMul method that,
// for each pair of elements, multiplies element i of the pair of x by element
// j of the pair of y.
func clmulSlice(i, j int) func(x, y []uint64) []uint64 {
	return func(x, y []uint64) []uint64 {
		z := make([]uint64, len(x))
		for k := 0; k < len(x); k += 2 {
			z[k], z[k+1] = clmul64(x[k+i], y[k+j])
		}
		return z
	}
}
