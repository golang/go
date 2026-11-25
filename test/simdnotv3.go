// errorcheck -0 -d=ssa/cpufeatures/debug=1,ssa/rewrite_tern/debug=1

//go:build goexperiment.simdstable && amd64 && !amd64.v3

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check the CPU features that the analysis credits code with,
// based on the code's types and feature checks. With GOAMD64=v3, the compiler
// assumes AVX and AVX2 in all code that uses SIMD, and with GOAMD64=v4,
// AVX-512 too, which changes those features and which expressions it rewrites
// into ternary-logic instructions.

package foo

import (
	"fmt"
	"simd/archsimd"
)

func g1() archsimd.Int8x16 {
	var x archsimd.Int8x16
	return x // ERROR "has features avx$"
}

type T1 archsimd.Int8x16

func (x T1) h() {
	return // ERROR "has features avx$"
}

var a int

func f() {
	if a == 0 {
		if !archsimd.X86.AVX512() {
			return
		}
		println("has avx512") // ERROR "has features avx[+]avx2[+]avx512$"
	} else {
		if !archsimd.X86.AVX2() {
			return
		}
		println("has avx2") // ERROR "has features avx[+]avx2$"
	}
	println("has something")
} // ERROR "has features avx[+]avx2$"

func g() {
	if archsimd.X86.AVX2() { // ERROR "has features avx[+]avx2$"
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
	if archsimd.X86.AVX2() {
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

func ternTricky3(x, y, z archsimd.Int32x8) archsimd.Int32x8 {
	// Int32x8 is a 256-bit vector and does not guarantee AVX-512
	a := x.Xor(y).Xor(z)
	w := y.AndNot(a)
	if !archsimd.X86.AVX512() { // ERROR "has features avx$"
		return a // ERROR "has features avx$"
	}
	// a is a common subexpression
	return a.Or(w) // ERROR "has features avx[+]avx2[+]avx512"  // This does not rewrite, do we want it to?
}

func vpternlogdPanic() {
	// This uses vectors rather than masks because the ternary-logic rewrite
	// leaves 512-bit mask logic alone.
	results := archsimd.Int64x8{}

	// The loop header holds only rematerializable SIMD zeros, which emit
	// no code, so it claims no features.
	for {
		results = archsimd.Mask64x8FromBits(0).ToInt64x8().Or( // ERROR "has features avx[+]avx2[+]avx512"
			archsimd.Float64x8{}.Less(
				archsimd.BroadcastFloat64x8(0)).ToInt64x8()).Or(results) // ERROR "Rewriting.*ternInt"
		fmt.Print(results.And(results.And(archsimd.Int64x8{})))
	}
}

type notSIMD struct {
	v archsimd.Int8x16
	n int
}

var cond bool

// Only the block executing a real SIMD instruction claims features;
// they do not leak into the unconditionally executed parts of the
// function through the no-code zero or the merge.
func zeroMerge(p *notSIMD, s []int8) {
	x := archsimd.Int8x16{}
	if cond {
		x = archsimd.LoadInt8x16(s) // ERROR "has features avx$"
	}
	p.v = x
	p.n++
}

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
