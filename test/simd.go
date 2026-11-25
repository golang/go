// errorcheck -0 -d=ssa/cpufeatures/debug=1,ssa/rewrite_tern/debug=1

//go:build goexperiment.simdstable && amd64

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import "simd/archsimd"

func f1(x archsimd.Int8x16) {
	return // ERROR "has features avx"
}

func f2(x archsimd.Int8x64) {
	return // ERROR "has features avx[+]avx2[+]avx512$"
}

func g2() archsimd.Int8x64 {
	var x archsimd.Int8x64
	return x // ERROR "has features avx[+]avx2[+]avx512$"
}

type T2 archsimd.Int8x64

func (x T2) h() {
	return // ERROR "has features avx[+]avx2[+]avx512$"
}

func ternRewrite(m, w, x, y, z archsimd.Int32x16) (t0, t1, t2 archsimd.Int32x16) {
	if !archsimd.X86.AVX512() { // ERROR "has features avx[+]avx2[+]avx512$"
		return // ERROR "has features avx[+]avx2[+]avx512$" // all blocks have it because of the vector size
	}
	t0 = w.Xor(y).Xor(z)                            // ERROR "Rewriting.*ternInt"
	t1 = m.And(w.Xor(y).Xor(z.Not()))               // ERROR "Rewriting.*ternInt"
	t2 = x.Xor(y).Xor(z).And(x.Xor(y).Xor(z.Not())) // ERROR "Rewriting.*ternInt"
	return                                          // ERROR "has features avx[+]avx2[+]avx512$"
}

func ternShared(a, b, c, d, e archsimd.Int32x16) (p1, p2 archsimd.Int32x16) {
	// x is rewritten on its own, because p1 has too many inputs to absorb
	// it. p2 also uses x, and must not be rewritten as if x were still
	// an Or of two of its inputs.
	x := a.Or(b.Or(c)) // ERROR "Rewriting.*ternInt"
	p1 = x.And(d)
	p2 = x.Xor(e)
	return p1, p2 // ERROR "has features avx[+]avx2[+]avx512$"
}

type notSIMD struct {
	v archsimd.Int8x16
	n int
}

var cond bool

// A SIMD-typed zero emits no code (it is just a reference to the fixed
// all-zeros register), so it must imply no CPU features: this function
// must stay compilable for machines without AVX and produce no feature
// diagnostics at all.
func zeroOnly(p *notSIMD) {
	p.v = archsimd.Int8x16{}
	p.n++
}
