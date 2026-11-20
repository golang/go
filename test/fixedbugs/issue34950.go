// run

//go:build goexperiment.simdstable && amd64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "simd/archsimd"

//go:noinline
func f(k bool, v archsimd.Int32x4) int32 {
	n := int32(0)
	if k {
		n = 1
	}
	if archsimd.X86.AVX() {
		n += v.Elem(0)
	}
	return n
}

func main() {
	_ = f(true, archsimd.Int32x4{})
}
