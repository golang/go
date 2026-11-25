// errorcheck -0 -d=ssa/cpufeatures/debug=1,ssa/rewrite_tern/debug=1

//go:build goexperiment.simdstable && amd64 && !amd64.v4

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests rely on an AVX-512 check to credit code with AVX-512. With
// GOAMD64=v4, the compiler folds the check away, which changes which
// expressions it rewrites into ternary-logic instructions.

package foo

import "simd/archsimd"

func ternTricky1(x, y, z archsimd.Int32x8) archsimd.Int32x8 {
	// Int32x8 is a 256-bit vector and does not guarantee AVX-512
	// a is a 3-variable logical expression occurring outside AVX-512 feature check
	a := x.Xor(y).Xor(z)
	var w archsimd.Int32x8
	if !archsimd.X86.AVX512() { // ERROR "has features avx$"
		// do nothing
	} else {
		w = y.AndNot(a) // ERROR "has features avx[+]avx2[+]avx512" "Rewriting.*ternInt"
	}
	// a is a common subexpression
	return a.Or(w) // ERROR "has features avx$"
}

func ternTricky2(x, y, z archsimd.Int32x8) archsimd.Int32x8 {
	// Int32x8 is a 256-bit vector and does not guarantee AVX-512
	var a, w archsimd.Int32x8
	if !archsimd.X86.AVX512() { // ERROR "has features avx$"
		// do nothing
	} else {
		a = x.Xor(y).Xor(z)
		w = y.AndNot(a) // ERROR "has features avx[+]avx2[+]avx512" "Rewriting.*ternInt"
	}
	// a is a common subexpression
	return a.Or(w) // ERROR "has features avx$"
}
