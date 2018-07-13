// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to arithmetic
// simplifications and optimizations on float types.
// For codegen tests on integer types, see arithmetic.go.

// --------------------- //
//    Strength-reduce    //
// --------------------- //

func Mul2(f float64) float64 {
	// 386/sse2:"ADDSD",-"MULSD"
	// 386/387:"FADDDP",-"FMULDP"
	// amd64:"ADDSD",-"MULSD"
	// arm/7:"ADDD",-"MULD"
	// arm64:"FADDD",-"FMULD"
	return f * 2.0
}

func DivPow2(f1, f2, f3 float64) (float64, float64, float64) {
	// 386/sse2:"MULSD",-"DIVSD"
	// 386/387:"FMULDP",-"FDIVDP"
	// amd64:"MULSD",-"DIVSD"
	// arm/7:"MULD",-"DIVD"
	// arm64:"FMULD",-"FDIVD"
	x := f1 / 16.0

	// 386/sse2:"MULSD",-"DIVSD"
	// 386/387:"FMULDP",-"FDIVDP"
	// amd64:"MULSD",-"DIVSD"
	// arm/7:"MULD",-"DIVD"
	// arm64:"FMULD",-"FDIVD"
	y := f2 / 0.125

	// 386/sse2:"ADDSD",-"DIVSD",-"MULSD"
	// 386/387:"FADDDP",-"FDIVDP",-"FMULDP"
	// amd64:"ADDSD",-"DIVSD",-"MULSD"
	// arm/7:"ADDD",-"MULD",-"DIVD"
	// arm64:"FADDD",-"FMULD",-"FDIVD"
	z := f3 / 0.5

	return x, y, z
}

// ----------- //
//    Fused    //
// ----------- //

func FusedAdd32(x, y, z float32) float32 {
	// s390x:"FMADDS\t"
	// ppc64le:"FMADDS\t"
	return x*y + z
}

func FusedSub32(x, y, z float32) float32 {
	// s390x:"FMSUBS\t"
	// ppc64le:"FMSUBS\t"
	return x*y - z
}

func FusedAdd64(x, y, z float64) float64 {
	// s390x:"FMADD\t"
	// ppc64le:"FMADD\t"
	return x*y + z
}

func FusedSub64(x, y, z float64) float64 {
	// s390x:"FMSUB\t"
	// ppc64le:"FMSUB\t"
	return x*y - z
}

// ---------------- //
//    Non-floats    //
// ---------------- //

// We should make sure that the compiler doesn't generate floating point
// instructions for non-float operations on Plan 9, because floating point
// operations are not allowed in the note handler.

func ArrayZero() [16]byte {
	// amd64:"MOVUPS"
	// plan9/amd64/:-"MOVUPS"
	var a [16]byte
	return a
}

func ArrayCopy(a [16]byte) (b [16]byte) {
	// amd64:"MOVUPS"
	// plan9/amd64/:-"MOVUPS"
	b = a
	return
}
