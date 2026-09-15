// errorcheck -0 -d=ssa/decompose_addr/debug=1

//go:build goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for variables that can be decomposed by the decomposeAddr pass.

package main

import "simd/archsimd"

type SIMDPoint struct {
	x, y archsimd.Int16x8
}

//go:noinline
func simpleStructSIMD(c bool) archsimd.Int16x8 {
	p := SIMDPoint{} // ERROR "promoted p"
	ptr := &p
	if c {
		ptr.x = ptr.x.Add(ptr.y)
	} else {
		ptr.y = ptr.y.Sub(ptr.x)
	}
	return ptr.x.Add(ptr.y)
}
