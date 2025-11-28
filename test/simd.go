// errorcheck -0 -d=ssa/cpufeatures/debug=1,ssa/rewrite_tern/debug=1

//go:build goexperiment.simd && amd64

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "simd"

func f1(x simd.Int8x16) {
	return // ERROR "has features avx"
}

func g1() simd.Int8x16 {
	var x simd.Int8x16
	return x // ERROR "has features avx$"
}

type T1 simd.Int8x16

func (x T1) h() {
	return // ERROR "has features avx$"
}

func f2(x simd.Int8x64) {
	return // ERROR "has features avx[+]avx2[+]avx512$"
}

func g2() simd.Int8x64 {
	var x simd.Int8x64
	return x // ERROR "has features avx[+]avx2[+]avx512$"
}

type T2 simd.Int8x64

func (x T2) h() {
	return // ERROR "has features avx[+]avx2[+]avx512$"
}

var a int

func f() {
	if a == 0 {
		if !simd.X86.AVX512() {
			return
		}
		println("has avx512") // ERROR "has features avx[+]avx2[+]avx512$"
	} else {
		if !simd.X86.AVX2() {
			return
		}
		println("has avx2") // ERROR "has features avx[+]avx2$"
	}
	println("has something")
} // ERROR "has features avx[+]avx2$"

func g() {
	if simd.X86.AVX2() { // ERROR "has features avx[+]avx2$"
		for range 5 { // ERROR "has features avx[+]avx2$"
			if a < 0 { // ERROR "has features avx[+]avx2$"
				a++ // ERROR "has features avx[+]avx2$"
			}
		}
	}
	println("ahoy!") // ERROR "has features avx[+]avx2$" // this is an artifact of flaky block numbering and why isn't it fused?
	if a > 0 {
		a--
	}
}

//go:noinline
func p() bool {
	return true
}

func hasIrreducibleLoop() {
	if simd.X86.AVX2() {
		goto a // ERROR "has features avx[+]avx2$"
	} else {
		goto b
	}
a:
	println("a")
	if p() {
		goto c
	}
b:
	println("b")
	if p() {
		goto a
	}
c:
	println("c")
}

func ternRewrite(m, w, x, y, z simd.Int32x16) (t0, t1, t2 simd.Int32x16) {
	if !simd.X86.AVX512() { // ERROR "has features avx[+]avx2[+]avx512$"
		return // ERROR "has features avx[+]avx2[+]avx512$" // all blocks have it because of the vector size
	}
	t0 = w.Xor(y).Xor(z)                            // ERROR "Rewriting.*ternInt"
	t1 = m.And(w.Xor(y).Xor(z.Not()))               // ERROR "Rewriting.*ternInt"
	t2 = x.Xor(y).Xor(z).And(x.Xor(y).Xor(z.Not())) // ERROR "Rewriting.*ternInt"
	return                                          // ERROR "has features avx[+]avx2[+]avx512$"
}

func ternTricky1(x, y, z simd.Int32x8) simd.Int32x8 {
	// Int32x8 is a 256-bit vector and does not guarantee AVX-512
	// a is a 3-variable logical expression occurring outside AVX-512 feature check
	a := x.Xor(y).Xor(z)
	var w simd.Int32x8
	if !simd.X86.AVX512() { // ERROR "has features avx$"
		// do nothing
	} else {
		w = y.AndNot(a) // ERROR "has features avx[+]avx2[+]avx512" "Rewriting.*ternInt"
	}
	// a is a common subexpression
	return a.Or(w) // ERROR "has features avx$"
}

func ternTricky2(x, y, z simd.Int32x8) simd.Int32x8 {
	// Int32x8 is a 256-bit vector and does not guarantee AVX-512
	var a, w simd.Int32x8
	if !simd.X86.AVX512() { // ERROR "has features avx$"
		// do nothing
	} else {
		a = x.Xor(y).Xor(z)
		w = y.AndNot(a) // ERROR "has features avx[+]avx2[+]avx512" "Rewriting.*ternInt"
	}
	// a is a common subexpression
	return a.Or(w) // ERROR "has features avx$"
}

func ternTricky3(x, y, z simd.Int32x8) simd.Int32x8 {
	// Int32x8 is a 256-bit vector and does not guarantee AVX-512
	a := x.Xor(y).Xor(z)
	w := y.AndNot(a)
	if !simd.X86.AVX512() { // ERROR "has features avx$"
		return a // ERROR "has features avx$"
	}
	// a is a common subexpression
	return a.Or(w) // ERROR "has features avx[+]avx2[+]avx512"  // This does not rewrite, do we want it to?
}
