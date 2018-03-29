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
	// 386:"ADDSD|FADDDP",-"MULSD",-"FMULDP"
	// amd64:"ADDSD",-"MULSD"
	// arm:"ADDD",-"MULD"
	// arm64:"FADDD",-"FMULD"
	return f * 2.0
}

func DivPow2(f1, f2, f3 float64) (float64, float64, float64) {
	// 386:"MULSD|FMULDP",-"DIVSD",-"FDIVDP"
	// amd64:"MULSD",-"DIVSD"
	// arm:"MULD",-"DIVD"
	// arm64:"FMULD",-"FDIVD"
	x := f1 / 16.0

	// 386:"MULSD|FMULDP",-"DIVSD",-"FDIVDP"
	// amd64:"MULSD",-"DIVSD"
	// arm:"MULD",-"DIVD"
	// arm64:"FMULD",-"FDIVD"
	y := f2 / 0.125

	// 386:"ADDSD|FADDDP",-"DIVSD",-"MULSD",-"FDIVDP",-"FMULDP"
	// amd64:"ADDSD",-"DIVSD",-"MULSD"
	// arm:"ADDD",-"MULD",-"DIVD"
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
