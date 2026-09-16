// run

//go:build goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"simd/archsimd"
)

//go:noinline
func twoRegions() int {
	n := 0
	if archsimd.X86.AVX512() {
		var v archsimd.Int32x16
		n += int(v.ReduceSum())
	}
	if archsimd.X86.AVX512() {
		var w archsimd.Int32x16
		n += int(w.ReduceSum())
	}
	return n
}

func main() {
	_ = twoRegions()
}
