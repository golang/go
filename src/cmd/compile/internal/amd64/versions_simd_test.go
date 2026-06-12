// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && goexperiment.simd && !boringcrypto

package amd64_test

import "simd/archsimd"

//go:noinline
func testLoop2(n int, x float64) float64 {
	var r float64
	for range n {
		if archsimd.X86.AVX2() {
			v := archsimd.BroadcastFloat64x2(x)
			v = v.Mul(v)
			r += v.GetElem(0)
		}
	}
	return r
}

//go:noinline
func testLoop3(n int, x float64) float64 {
	var r float64
	if !archsimd.X86.AVX2() {
		n = 0
	}
	for range n {
		v := archsimd.BroadcastFloat64x2(x)
		v = v.Mul(v)
		r += v.GetElem(0)
	}
	return r
}
