// errorcheck -0 -d=ssa/rewrite_tern/debug=1

//go:build goexperiment.simdstable && amd64.v4

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// With GOAMD64=v4, the compiler folds away checks for AVX-512, so code after
// an "if" on AVX-512 only follows the AVX-512 code and can use AVX-512 too.
// This is the GOAMD64=v4 counterpart of the tests in simdnotv3.go.

package foo

import "simd/archsimd"

func ternAfterCheck(x, y, z archsimd.Int32x8) archsimd.Int32x8 {
	a := x.Xor(y).Xor(z)
	var w archsimd.Int32x8
	if archsimd.X86.AVX512() {
		w = y.AndNot(a)
	}
	return a.Or(w) // ERROR "Rewriting.*ternInt"
}
