// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

var globl int64
var globl32 int32

func BenchmarkLoadAdd(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s ^= x[i] + y[i]
		}
		globl = s
	}
}

// Added for ppc64 extswsli on power9
func BenchmarkExtShift(b *testing.B) {
	x := make([]int32, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s ^= int64(x[i]+32) * 8
		}
		globl = s
	}
}

func BenchmarkModify(b *testing.B) {
	a := make([]int64, 1024)
	v := globl
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += v
		}
	}
}

func BenchmarkMullImm(b *testing.B) {
	x := make([]int32, 1024)
	for i := 0; i < b.N; i++ {
		var s int32
		for i := range x {
			s += x[i] * 100
		}
		globl32 = s
	}
}

func BenchmarkConstModify(b *testing.B) {
	a := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += 3
		}
	}
}

func BenchmarkBitSet(b *testing.B) {
	const N = 64 * 8
	a := make([]uint64, N/64)
	for i := 0; i < b.N; i++ {
		for j := uint64(0); j < N; j++ {
			a[j/64] |= 1 << (j % 64)
		}
	}
}

func BenchmarkBitClear(b *testing.B) {
	const N = 64 * 8
	a := make([]uint64, N/64)
	for i := 0; i < b.N; i++ {
		for j := uint64(0); j < N; j++ {
			a[j/64] &^= 1 << (j % 64)
		}
	}
}

func BenchmarkBitToggle(b *testing.B) {
	const N = 64 * 8
	a := make([]uint64, N/64)
	for i := 0; i < b.N; i++ {
		for j := uint64(0); j < N; j++ {
			a[j/64] ^= 1 << (j % 64)
		}
	}
}

func BenchmarkBitSetConst(b *testing.B) {
	const N = 64
	a := make([]uint64, N)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] |= 1 << 37
		}
	}
}

func BenchmarkBitClearConst(b *testing.B) {
	const N = 64
	a := make([]uint64, N)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] &^= 1 << 37
		}
	}
}

func BenchmarkBitToggleConst(b *testing.B) {
	const N = 64
	a := make([]uint64, N)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] ^= 1 << 37
		}
	}
}

func BenchmarkMulNeg(b *testing.B) {
	x := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = (-x[i]) * 11
		}
		globl = s
	}
}

func BenchmarkMul2Neg(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = (-x[i]) * (-y[i])
		}
		globl = s
	}
}

func BenchmarkSimplifyNegMul(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = -(-x[i] * y[i])
		}
		globl = s
	}
}

func BenchmarkSimplifyNegDiv(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := range y {
		y[i] = 42
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = -(-x[i] / y[i])
		}
		globl = s
	}
}
