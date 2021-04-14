// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math"

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
	// ppc64:"FADD",-"FMUL"
	// ppc64le:"FADD",-"FMUL"
	return f * 2.0
}

func DivPow2(f1, f2, f3 float64) (float64, float64, float64) {
	// 386/sse2:"MULSD",-"DIVSD"
	// 386/387:"FMULDP",-"FDIVDP"
	// amd64:"MULSD",-"DIVSD"
	// arm/7:"MULD",-"DIVD"
	// arm64:"FMULD",-"FDIVD"
	// ppc64:"FMUL",-"FDIV"
	// ppc64le:"FMUL",-"FDIV"
	x := f1 / 16.0

	// 386/sse2:"MULSD",-"DIVSD"
	// 386/387:"FMULDP",-"FDIVDP"
	// amd64:"MULSD",-"DIVSD"
	// arm/7:"MULD",-"DIVD"
	// arm64:"FMULD",-"FDIVD"
	// ppc64:"FMUL",-"FDIVD"
	// ppc64le:"FMUL",-"FDIVD"
	y := f2 / 0.125

	// 386/sse2:"ADDSD",-"DIVSD",-"MULSD"
	// 386/387:"FADDDP",-"FDIVDP",-"FMULDP"
	// amd64:"ADDSD",-"DIVSD",-"MULSD"
	// arm/7:"ADDD",-"MULD",-"DIVD"
	// arm64:"FADDD",-"FMULD",-"FDIVD"
	// ppc64:"FADD",-"FMUL",-"FDIV"
	// ppc64le:"FADD",-"FMUL",-"FDIV"
	z := f3 / 0.5

	return x, y, z
}

func getPi() float64 {
	// 386/387:"FLDPI"
	return math.Pi
}

func indexLoad(b0 []float32, b1 float32, idx int) float32 {
	// arm64:`FMOVS\s\(R[0-9]+\)\(R[0-9]+\),\sF[0-9]+`
	return b0[idx] * b1
}

func indexStore(b0 []float64, b1 float64, idx int) {
	// arm64:`FMOVD\sF[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`
	b0[idx] = b1
}

// ----------- //
//    Fused    //
// ----------- //

func FusedAdd32(x, y, z float32) float32 {
	// s390x:"FMADDS\t"
	// ppc64:"FMADDS\t"
	// ppc64le:"FMADDS\t"
	// arm64:"FMADDS"
	return x*y + z
}

func FusedSub32_a(x, y, z float32) float32 {
	// s390x:"FMSUBS\t"
	// ppc64:"FMSUBS\t"
	// ppc64le:"FMSUBS\t"
	return x*y - z
}

func FusedSub32_b(x, y, z float32) float32 {
	// arm64:"FMSUBS"
	return z - x*y
}

func FusedAdd64(x, y, z float64) float64 {
	// s390x:"FMADD\t"
	// ppc64:"FMADD\t"
	// ppc64le:"FMADD\t"
	// arm64:"FMADDD"
	return x*y + z
}

func FusedSub64_a(x, y, z float64) float64 {
	// s390x:"FMSUB\t"
	// ppc64:"FMSUB\t"
	// ppc64le:"FMSUB\t"
	return x*y - z
}

func FusedSub64_b(x, y, z float64) float64 {
	// arm64:"FMSUBD"
	return z - x*y
}

func Cmp(f float64) bool {
	// arm64:"FCMPD","(BGT|BLE|BMI|BPL)",-"CSET\tGT",-"CBZ"
	return f > 4 || f < -4
}

func CmpZero64(f float64) bool {
	// s390x:"LTDBR",-"FCMPU"
	return f <= 0
}

func CmpZero32(f float32) bool {
	// s390x:"LTEBR",-"CEBR"
	return f <= 0
}

func CmpWithSub(a float64, b float64) bool {
	f := a - b
	// s390x:-"LTDBR"
	return f <= 0
}

func CmpWithAdd(a float64, b float64) bool {
	f := a + b
	// s390x:-"LTDBR"
	return f <= 0
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
