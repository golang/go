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
	// 386/sse2:"ADDSD" -"MULSD"
	// amd64:"ADDSD" -"MULSD"
	// arm/7:"ADDD" -"MULD"
	// arm64:"FADDD" -"FMULD"
	// loong64:"ADDD" -"MULD"
	// ppc64x:"FADD" -"FMUL"
	// riscv64:"FADDD" -"FMULD"
	return f * 2.0
}

func DivPow2(f1, f2, f3 float64) (float64, float64, float64) {
	// 386/sse2:"MULSD" -"DIVSD"
	// amd64:"MULSD" -"DIVSD"
	// arm/7:"MULD" -"DIVD"
	// arm64:"FMULD" -"FDIVD"
	// loong64:"MULD" -"DIVD"
	// ppc64x:"FMUL" -"FDIV"
	// riscv64:"FMULD" -"FDIVD"
	x := f1 / 16.0

	// 386/sse2:"MULSD" -"DIVSD"
	// amd64:"MULSD" -"DIVSD"
	// arm/7:"MULD" -"DIVD"
	// arm64:"FMULD" -"FDIVD"
	// loong64:"MULD" -"DIVD"
	// ppc64x:"FMUL" -"FDIVD"
	// riscv64:"FMULD" -"FDIVD"
	y := f2 / 0.125

	// 386/sse2:"ADDSD" -"DIVSD" -"MULSD"
	// amd64:"ADDSD" -"DIVSD" -"MULSD"
	// arm/7:"ADDD" -"MULD" -"DIVD"
	// arm64:"FADDD" -"FMULD" -"FDIVD"
	// loong64:"ADDD" -"MULD" -"DIVD"
	// ppc64x:"FADD" -"FMUL" -"FDIV"
	// riscv64:"FADDD" -"FMULD" -"FDIVD"
	z := f3 / 0.5

	return x, y, z
}

func indexLoad(b0 []float32, b1 float32, idx int) float32 {
	// arm64:`FMOVS\s\(R[0-9]+\)\(R[0-9]+<<2\),\sF[0-9]+`
	// loong64:`MOVF\s\(R[0-9]+\)\(R[0-9]+\),\sF[0-9]+`
	return b0[idx] * b1
}

func indexStore(b0 []float64, b1 float64, idx int) {
	// arm64:`FMOVD\sF[0-9]+,\s\(R[0-9]+\)\(R[0-9]+<<3\)`
	// loong64:`MOVD\sF[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`
	b0[idx] = b1
}

// ----------- //
//    Fused    //
// ----------- //

func FusedAdd32(x, y, z float32) float32 {
	// s390x:"FMADDS "
	// ppc64x:"FMADDS "
	// arm64:"FMADDS"
	// loong64:"FMADDF "
	// riscv64:"FMADDS "
	// amd64/v3:"VFMADD231SS "
	return x*y + z
}

func FusedSub32_a(x, y, z float32) float32 {
	// s390x:"FMSUBS "
	// ppc64x:"FMSUBS "
	// riscv64:"FMSUBS "
	// loong64:"FMSUBF "
	return x*y - z
}

func FusedSub32_b(x, y, z float32) float32 {
	// arm64:"FMSUBS"
	// loong64:"FNMSUBF "
	// riscv64:"FNMSUBS "
	return z - x*y
}

func FusedAdd64(x, y, z float64) float64 {
	// s390x:"FMADD "
	// ppc64x:"FMADD "
	// arm64:"FMADDD"
	// loong64:"FMADDD "
	// riscv64:"FMADDD "
	// amd64/v3:"VFMADD231SD "
	return x*y + z
}

func FusedSub64_a(x, y, z float64) float64 {
	// s390x:"FMSUB "
	// ppc64x:"FMSUB "
	// riscv64:"FMSUBD "
	// loong64:"FMSUBD "
	return x*y - z
}

func FusedSub64_b(x, y, z float64) float64 {
	// arm64:"FMSUBD"
	// loong64:"FNMSUBD "
	// riscv64:"FNMSUBD "
	return z - x*y
}

func Cmp(f float64) bool {
	// arm64:"FCMPD" "(BGT|BLE|BMI|BPL)" -"CSET GT" -"CBZ"
	return f > 4 || f < -4
}

func CmpZero64(f float64) bool {
	// s390x:"LTDBR" -"FCMPU"
	return f <= 0
}

func CmpZero32(f float32) bool {
	// s390x:"LTEBR" -"CEBR"
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

func ArrayZero() [16]byte {
	// amd64:"MOVUPS"
	var a [16]byte
	return a
}

func ArrayCopy(a [16]byte) (b [16]byte) {
	// amd64:"MOVUPS"
	b = a
	return
}

// ---------------- //
//  Float Min/Max   //
// ---------------- //

func Float64Min(a, b float64) float64 {
	// amd64:"MINSD"
	// arm64:"FMIND"
	// loong64:"FMIND"
	// riscv64:"FMIN"
	// ppc64/power9:"XSMINJDP"
	// ppc64/power10:"XSMINJDP"
	// s390x: "WFMINDB"
	return min(a, b)
}

func Float64Max(a, b float64) float64 {
	// amd64:"MINSD"
	// arm64:"FMAXD"
	// loong64:"FMAXD"
	// riscv64:"FMAX"
	// ppc64/power9:"XSMAXJDP"
	// ppc64/power10:"XSMAXJDP"
	// s390x: "WFMAXDB"
	return max(a, b)
}

func Float32Min(a, b float32) float32 {
	// amd64:"MINSS"
	// arm64:"FMINS"
	// loong64:"FMINF"
	// riscv64:"FMINS"
	// ppc64/power9:"XSMINJDP"
	// ppc64/power10:"XSMINJDP"
	// s390x: "WFMINSB"
	return min(a, b)
}

func Float32Max(a, b float32) float32 {
	// amd64:"MINSS"
	// arm64:"FMAXS"
	// loong64:"FMAXF"
	// riscv64:"FMAXS"
	// ppc64/power9:"XSMAXJDP"
	// ppc64/power10:"XSMAXJDP"
	// s390x: "WFMAXSB"
	return max(a, b)
}

// ------------------------ //
//  Constant Optimizations  //
// ------------------------ //

func Float32ConstantZero() float32 {
	// arm64:"FMOVS ZR,"
	return 0.0
}

func Float32ConstantChipFloat() float32 {
	// arm64:"FMOVS [$]\\(2\\.25\\),"
	return 2.25
}

func Float32Constant() float32 {
	// arm64:"FMOVS [$]f32\\.42440000\\(SB\\)"
	// ppc64x/power8:"FMOVS [$]f32\\.42440000\\(SB\\)"
	// ppc64x/power9:"FMOVS [$]f32\\.42440000\\(SB\\)"
	// ppc64x/power10:"XXSPLTIDP [$]1111752704,"
	return 49.0
}

func Float64ConstantZero() float64 {
	// arm64:"FMOVD ZR,"
	return 0.0
}

func Float64ConstantChipFloat() float64 {
	// arm64:"FMOVD [$]\\(2\\.25\\),"
	return 2.25
}

func Float64Constant() float64 {
	// arm64:"FMOVD [$]f64\\.4048800000000000\\(SB\\)"
	// ppc64x/power8:"FMOVD [$]f64\\.4048800000000000\\(SB\\)"
	// ppc64x/power9:"FMOVD [$]f64\\.4048800000000000\\(SB\\)"
	// ppc64x/power10:"XXSPLTIDP [$]1111752704,"
	return 49.0
}

func Float32DenormalConstant() float32 {
	// ppc64x:"FMOVS [$]f32\\.00400000\\(SB\\)"
	return 0x1p-127
}

// A float64 constant which can be exactly represented as a
// denormal float32 value. On ppc64x, denormal values cannot
// be used with XXSPLTIDP.
func Float64DenormalFloat32Constant() float64 {
	// ppc64x:"FMOVD [$]f64\\.3800000000000000\\(SB\\)"
	return 0x1p-127
}

func Float32ConstantStore(p *float32) {
	// amd64:"MOVL [$]1085133554"
	// riscv64: "MOVF [$]f32.40add2f2"
	*p = 5.432
}

func Float64ConstantStore(p *float64) {
	// amd64: "MOVQ [$]4617801906721357038"
	// riscv64: "MOVD [$]f64.4015ba5e353f7cee"
	*p = 5.432
}

// ------------------------ //
//  Subnormal tests         //
// ------------------------ //

func isSubnormal(x float64) bool {
	// riscv64:"FCLASSD" -"FABSD"
	return math.Abs(x) < 2.2250738585072014e-308
}

func isNormal(x float64) bool {
	// riscv64:"FCLASSD" -"FABSD"
	return math.Abs(x) >= 0x1p-1022
}

func isPosSubnormal(x float64) bool {
	// riscv64:"FCLASSD"
	return x > 0 && x < 2.2250738585072014e-308
}

func isNegSubnormal(x float64) bool {
	// riscv64:"FCLASSD"
	return x < 0 && x > -0x1p-1022
}

func isPosNormal(x float64) bool {
	// riscv64:"FCLASSD"
	return x >= 2.2250738585072014e-308
}

func isNegNormal(x float64) bool {
	// riscv64:"FCLASSD"
	return x <= -2.2250738585072014e-308
}
