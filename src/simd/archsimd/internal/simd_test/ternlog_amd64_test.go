// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

//go:noinline
func f(a, b, c archsimd.Int32x16) archsimd.Int32x16 {
	return a.AndNot(b).Xor(c) // a &^ b ^ c, fused into one VPTERNLOGD
}

func TestTernLogAndnot(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("needs AVX-512")
		return
	}
	A := int32(0b1100)
	B := int32(0b1010)
	C := int32(0)
	a := archsimd.BroadcastInt32x16(A)
	b := archsimd.BroadcastInt32x16(B)
	c := archsimd.BroadcastInt32x16(C)
	var r [16]int32
	f(a, b, c).StoreArray(&r)
	want := A &^ B
	got := r[0]
	if want != got {
		t.Errorf("got %04b, want %04b", got, want)
	}
}

//go:noinline
func ternLoadD128(x, y, z *[4]int32) archsimd.Int32x4 {
	if !archsimd.X86.AVX512() {
		return archsimd.Int32x4{}
	}
	a, b, c := archsimd.LoadInt32x4Array(x), archsimd.LoadInt32x4Array(y), archsimd.LoadInt32x4Array(z)
	return a.AndNot(b).Xor(c)
}

//go:noinline
func ternLoadD256(x, y, z *[8]int32) archsimd.Int32x8 {
	if !archsimd.X86.AVX512() {
		return archsimd.Int32x8{}
	}
	a, b, c := archsimd.LoadInt32x8Array(x), archsimd.LoadInt32x8Array(y), archsimd.LoadInt32x8Array(z)
	return a.AndNot(b).Xor(c)
}

//go:noinline
func ternLoadD512(x, y, z *[16]int32) archsimd.Int32x16 {
	a, b, c := archsimd.LoadInt32x16Array(x), archsimd.LoadInt32x16Array(y), archsimd.LoadInt32x16Array(z)
	return a.AndNot(b).Xor(c)
}

//go:noinline
func ternLoadQ128(x, y, z *[2]int64) archsimd.Int64x2 {
	if !archsimd.X86.AVX512() {
		return archsimd.Int64x2{}
	}
	a, b, c := archsimd.LoadInt64x2Array(x), archsimd.LoadInt64x2Array(y), archsimd.LoadInt64x2Array(z)
	return a.AndNot(b).Xor(c)
}

//go:noinline
func ternLoadQ256(x, y, z *[4]int64) archsimd.Int64x4 {
	if !archsimd.X86.AVX512() {
		return archsimd.Int64x4{}
	}
	a, b, c := archsimd.LoadInt64x4Array(x), archsimd.LoadInt64x4Array(y), archsimd.LoadInt64x4Array(z)
	return a.AndNot(b).Xor(c)
}

//go:noinline
func ternLoadQ512(x, y, z *[8]int64) archsimd.Int64x8 {
	a, b, c := archsimd.LoadInt64x8Array(x), archsimd.LoadInt64x8Array(y), archsimd.LoadInt64x8Array(z)
	return a.AndNot(b).Xor(c)
}

var (
	ternLoadGlobX, ternLoadGlobY, ternLoadGlobZ [16]int32
)

//go:noinline
func ternLoadD512Global() {
	a := archsimd.LoadInt32x16Array(&ternLoadGlobX)
	b := archsimd.LoadInt32x16Array(&ternLoadGlobY)
	c := archsimd.LoadInt32x16Array(&ternLoadGlobZ)
	a.Xor(b).Xor(c).StoreArray(&ternLoadGlobX)
}

func TestTernLogMergedLoad(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("needs AVX-512")
		return
	}
	// Distinct 8-bit truth-table inputs covering all 8 (a, b, c) bit combinations.
	const (
		A32 int32 = 0b11110000
		B32 int32 = 0b11001100
		C32 int32 = 0b10101010
		A64 int64 = 0b11110000
		B64 int64 = 0b11001100
		C64 int64 = 0b10101010
	)
	want32 := (A32 &^ B32) ^ C32
	want64 := (A64 &^ B64) ^ C64

	var x4, y4, z4, r4 [4]int32
	for i := range x4 {
		x4[i], y4[i], z4[i] = A32, B32, C32
	}
	ternLoadD128(&x4, &y4, &z4).StoreArray(&r4)
	if r4[0] != want32 {
		t.Errorf("ternLoadD128: got %08b, want %08b", r4[0], want32)
	}

	var x8, y8, z8, r8 [8]int32
	for i := range x8 {
		x8[i], y8[i], z8[i] = A32, B32, C32
	}
	ternLoadD256(&x8, &y8, &z8).StoreArray(&r8)
	if r8[0] != want32 {
		t.Errorf("ternLoadD256: got %08b, want %08b", r8[0], want32)
	}

	var x16, y16, z16, r16 [16]int32
	for i := range x16 {
		x16[i], y16[i], z16[i] = A32, B32, C32
	}
	ternLoadD512(&x16, &y16, &z16).StoreArray(&r16)
	if r16[0] != want32 {
		t.Errorf("ternLoadD512: got %08b, want %08b", r16[0], want32)
	}

	var qx2, qy2, qz2, qr2 [2]int64
	for i := range qx2 {
		qx2[i], qy2[i], qz2[i] = A64, B64, C64
	}
	ternLoadQ128(&qx2, &qy2, &qz2).StoreArray(&qr2)
	if qr2[0] != want64 {
		t.Errorf("ternLoadQ128: got %08b, want %08b", qr2[0], want64)
	}

	var qx4, qy4, qz4, qr4 [4]int64
	for i := range qx4 {
		qx4[i], qy4[i], qz4[i] = A64, B64, C64
	}
	ternLoadQ256(&qx4, &qy4, &qz4).StoreArray(&qr4)
	if qr4[0] != want64 {
		t.Errorf("ternLoadQ256: got %08b, want %08b", qr4[0], want64)
	}

	var qx8, qy8, qz8, qr8 [8]int64
	for i := range qx8 {
		qx8[i], qy8[i], qz8[i] = A64, B64, C64
	}
	ternLoadQ512(&qx8, &qy8, &qz8).StoreArray(&qr8)
	if qr8[0] != want64 {
		t.Errorf("ternLoadQ512: got %08b, want %08b", qr8[0], want64)
	}

	for i := range ternLoadGlobX {
		ternLoadGlobX[i], ternLoadGlobY[i], ternLoadGlobZ[i] = A32, B32, C32
	}
	ternLoadD512Global()
	if wantGlob := A32 ^ B32 ^ C32; ternLoadGlobX[0] != wantGlob {
		t.Errorf("ternLoadD512Global: got %08b, want %08b", ternLoadGlobX[0], wantGlob)
	}
}
