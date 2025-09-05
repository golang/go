// errorcheck -0 -d=ssa/cpufeatures/debug=1

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
		if !simd.HasAVX512() {
			return
		}
		println("has avx512") // ERROR "has features avx[+]avx2[+]avx512$"
	} else {
		if !simd.HasAVX2() {
			return
		}
		println("has avx2") // ERROR "has features avx[+]avx2$"
	}
	println("has something")
} // ERROR "has features avx[+]avx2$"

func g() {
	if simd.HasAVX2() { // ERROR "has features avx[+]avx2$"
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
	if simd.HasAVX2() {
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
