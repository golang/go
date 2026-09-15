// errorcheck -0 -d=ssa/mem2reg/debug=4

//go:build goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for variables accessed through pointers that can
// be promoted to register by the mem2reg pass.

package main

import "simd/archsimd"

// Control flow: if/else
//
//go:noinline
func ifElseAddrSIMD(c bool) archsimd.Int8x16 {
	x := archsimd.Int8x16{} // ERROR "promoted x in cross block case"
	p := &x
	if c {
		*p = (*p).Add(*p)
	} else {
		*p = archsimd.LoadInt8x16Array(&[16]int8{1})
	}
	return *p
}
