// asmcheck -gcflags=-d=converthash=qy

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math"

var sink64 [8]float64

func approx(x float64) {
	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD [$]2"
	// s390x:"FIDBR [$]6"
	// arm64:"FRINTPD"
	// ppc64x:"FRIP"
	// wasm:"F64Ceil"
	sink64[0] = math.Ceil(x)

	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD [$]1"
	// s390x:"FIDBR [$]7"
	// arm64:"FRINTMD"
	// ppc64x:"FRIM"
	// wasm:"F64Floor"
	sink64[1] = math.Floor(x)

	// s390x:"FIDBR [$]1"
	// arm64:"FRINTAD"
	// ppc64x:"FRIN"
	sink64[2] = math.Round(x)

	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD [$]3"
	// s390x:"FIDBR [$]5"
	// arm64:"FRINTZD"
	// ppc64x:"FRIZ"
	// wasm:"F64Trunc"
	sink64[3] = math.Trunc(x)

	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD [$]0"
	// s390x:"FIDBR [$]4"
	// arm64:"FRINTND"
	// wasm:"F64Nearest"
	sink64[4] = math.RoundToEven(x)
}

func sqrt(x float64) float64 {
	// amd64:"SQRTSD"
	// 386/sse2:"SQRTSD" 386/softfloat:-"SQRTD"
	// arm64:"FSQRTD"
	// arm/7:"SQRTD"
	// loong64:"SQRTD"
	// mips/hardfloat:"SQRTD" mips/softfloat:-"SQRTD"
	// mips64/hardfloat:"SQRTD" mips64/softfloat:-"SQRTD"
	// wasm:"F64Sqrt"
	// ppc64x:"FSQRT"
	// riscv64: "FSQRTD"
	return math.Sqrt(x)
}

func sqrt32(x float32) float32 {
	// amd64:"SQRTSS"
	// 386/sse2:"SQRTSS" 386/softfloat:-"SQRTS"
	// arm64:"FSQRTS"
	// arm/7:"SQRTF"
	// loong64:"SQRTF"
	// mips/hardfloat:"SQRTF" mips/softfloat:-"SQRTF"
	// mips64/hardfloat:"SQRTF" mips64/softfloat:-"SQRTF"
	// wasm:"F32Sqrt"
	// ppc64x:"FSQRTS"
	// riscv64: "FSQRTS"
	return float32(math.Sqrt(float64(x)))
}

// Check that it's using integer registers
func abs(x, y float64) {
	// amd64:"BTRQ [$]63"
	// arm64:"FABSD "
	// loong64:"ABSD "
	// s390x:"LPDFR " -"MOVD "     (no integer load/store)
	// ppc64x:"FABS "
	// riscv64:"FABSD "
	// wasm:"F64Abs"
	// arm/6:"ABSD "
	// mips64/hardfloat:"ABSD "
	// mips/hardfloat:"ABSD "
	sink64[0] = math.Abs(x)

	// amd64:"BTRQ [$]63" "PXOR"    (TODO: this should be BTSQ)
	// s390x:"LNDFR " -"MOVD "     (no integer load/store)
	// ppc64x:"FNABS "
	sink64[1] = -math.Abs(y)
}

// Check that it's using integer registers
func abs32(x float32) float32 {
	// s390x:"LPDFR" -"LDEBR" -"LEDBR"     (no float64 conversion)
	return float32(math.Abs(float64(x)))
}

// Check that it's using integer registers
func copysign(a, b, c float64) {
	// amd64:"BTRQ [$]63" "ANDQ" "ORQ"
	// loong64:"FCOPYSGD"
	// s390x:"CPSDR" -"MOVD"         (no integer load/store)
	// ppc64x:"FCPSGN"
	// riscv64:"FSGNJD"
	// wasm:"F64Copysign"
	sink64[0] = math.Copysign(a, b)

	// amd64:"BTSQ [$]63"
	// loong64:"FCOPYSGD"
	// s390x:"LNDFR " -"MOVD "     (no integer load/store)
	// ppc64x:"FCPSGN"
	// riscv64:"FSGNJD"
	// arm64:"ORR", -"AND"
	sink64[1] = math.Copysign(c, -1)

	// Like math.Copysign(c, -1), but with integer operations. Useful
	// for platforms that have a copysign opcode to see if it's detected.
	// s390x:"LNDFR " -"MOVD "     (no integer load/store)
	sink64[2] = math.Float64frombits(math.Float64bits(a) | 1<<63)

	// amd64:"ANDQ" "ORQ"
	// loong64:"FCOPYSGD"
	// s390x:"CPSDR " -"MOVD "     (no integer load/store)
	// ppc64x:"FCPSGN"
	// riscv64:"FSGNJD"
	sink64[3] = math.Copysign(-1, c)
}

func fma(x, y, z float64) float64 {
	// amd64/v3:-".*x86HasFMA"
	// amd64:"VFMADD231SD"
	// arm/6:"FMULAD"
	// arm64:"FMADDD"
	// loong64:"FMADDD"
	// s390x:"FMADD"
	// ppc64x:"FMADD"
	// riscv64:"FMADDD"
	return math.FMA(x, y, z)
}

func fms(x, y, z float64) float64 {
	// riscv64:"FMSUBD"
	return math.FMA(x, y, -z)
}

func fnms(x, y, z float64) float64 {
	// riscv64:"FNMSUBD" -"FNMADDD"
	return math.FMA(-x, y, z)
}

func fnma(x, y, z float64) float64 {
	// riscv64:"FNMADDD" -"FNMSUBD"
	return math.FMA(x, -y, -z)
}

func isPosInf(x float64) bool {
	// riscv64:"FCLASSD"
	return math.IsInf(x, 1)
}

func isPosInfEq(x float64) bool {
	// riscv64:"FCLASSD"
	return x == math.Inf(1)
}

func isPosInfCmp(x float64) bool {
	// riscv64:"FCLASSD"
	return x > math.MaxFloat64
}

func isNotPosInf(x float64) bool {
	// riscv64:"FCLASSD"
	return !math.IsInf(x, 1)
}

func isNotPosInfEq(x float64) bool {
	// riscv64:"FCLASSD"
	return x != math.Inf(1)
}

func isNotPosInfCmp(x float64) bool {
	// riscv64:"FCLASSD"
	return x <= math.MaxFloat64
}

func isNegInf(x float64) bool {
	// riscv64:"FCLASSD"
	return math.IsInf(x, -1)
}

func isNegInfEq(x float64) bool {
	// riscv64:"FCLASSD"
	return x == math.Inf(-1)
}

func isNegInfCmp(x float64) bool {
	// riscv64:"FCLASSD"
	return x < -math.MaxFloat64
}

func isNotNegInf(x float64) bool {
	// riscv64:"FCLASSD"
	return !math.IsInf(x, -1)
}

func isNotNegInfEq(x float64) bool {
	// riscv64:"FCLASSD"
	return x != math.Inf(-1)
}

func isNotNegInfCmp(x float64) bool {
	// riscv64:"FCLASSD"
	return x >= -math.MaxFloat64
}

func fromFloat64(f64 float64) uint64 {
	// amd64:"MOVQ X.*, [^X].*"
	// arm64:"FMOVD F.*, R.*"
	// loong64:"MOVV F.*, R.*"
	// ppc64x:"MFVSRD"
	// mips64/hardfloat:"MOVV F.*, R.*"
	// riscv64:"FMVXD"
	return math.Float64bits(f64+1) + 1
}

func fromFloat32(f32 float32) uint32 {
	// amd64:"MOVL X.*, [^X].*"
	// arm64:"FMOVS F.*, R.*"
	// loong64:"MOVW F.*, R.*"
	// mips64/hardfloat:"MOVW F.*, R.*"
	// riscv64:"FMVXW"
	return math.Float32bits(f32+1) + 1
}

func toFloat64(u64 uint64) float64 {
	// amd64:"MOVQ [^X].*, X.*"
	// arm64:"FMOVD R.*, F.*"
	// loong64:"MOVV R.*, F.*"
	// ppc64x:"MTVSRD"
	// mips64/hardfloat:"MOVV R.*, F.*"
	// riscv64:"FMVDX"
	return math.Float64frombits(u64+1) + 1
}

func toFloat32(u32 uint32) float32 {
	// amd64:"MOVL [^X].*, X.*"
	// arm64:"FMOVS R.*, F.*"
	// loong64:"MOVW R.*, F.*"
	// mips64/hardfloat:"MOVW R.*, F.*"
	// riscv64:"FMVWX"
	return math.Float32frombits(u32+1) + 1
}

// Test that comparisons with constants converted to float
// are evaluated at compile-time

func constantCheck64() bool {
	// amd64:"(MOVB [$]0)|(XORL [A-Z][A-Z0-9]+, [A-Z][A-Z0-9]+)" -"FCMP" -"MOVB [$]1"
	// s390x:"MOV(B|BZ|D) [$]0," -"FCMPU" -"MOV(B|BZ|D) [$]1,"
	return 0.5 == float64(uint32(1)) || 1.5 > float64(uint64(1<<63))
}

func constantCheck32() bool {
	// amd64:"MOV(B|L) [$]1" -"FCMP" -"MOV(B|L) [$]0"
	// s390x:"MOV(B|BZ|D) [$]1," -"FCMPU" -"MOV(B|BZ|D) [$]0,"
	return float32(0.5) <= float32(int64(1)) && float32(1.5) >= float32(int32(-1<<31))
}

// Test that integer constants are converted to floating point constants
// at compile-time

func constantConvert32(x float32) float32 {
	// amd64:"MOVSS [$]f32.3f800000\\(SB\\)"
	// s390x:"FMOVS [$]f32.3f800000\\(SB\\)"
	// ppc64x/power8:"FMOVS [$]f32.3f800000\\(SB\\)"
	// ppc64x/power9:"FMOVS [$]f32.3f800000\\(SB\\)"
	// ppc64x/power10:"XXSPLTIDP [$]1065353216, VS0"
	// arm64:"FMOVS [$]\\(1.0\\)"
	if x > math.Float32frombits(0x3f800000) {
		return -x
	}
	return x
}

func constantConvertInt32(x uint32) uint32 {
	// amd64:-"MOVSS"
	// s390x:-"FMOVS"
	// ppc64x:-"FMOVS"
	// arm64:-"FMOVS"
	if x > math.Float32bits(1) {
		return -x
	}
	return x
}

func nanGenerate64() float64 {
	// Test to make sure we don't generate a NaN while constant propagating.
	// See issue 36400.
	zero := 0.0
	// amd64:-"DIVSD"
	inf := 1 / zero // +inf. We can constant propagate this one.
	negone := -1.0

	// amd64:"DIVSD"
	z0 := zero / zero
	// amd64/v1,amd64/v2:"MULSD"
	z1 := zero * inf
	// amd64:"SQRTSD"
	z2 := math.Sqrt(negone)
	// amd64/v3:"VFMADD231SD"
	return z0 + z1 + z2
}

func nanGenerate32() float32 {
	zero := float32(0.0)
	// amd64:-"DIVSS"
	inf := 1 / zero // +inf. We can constant propagate this one.

	// amd64:"DIVSS"
	z0 := zero / zero
	// amd64/v1,amd64/v2:"MULSS"
	z1 := zero * inf
	// amd64/v3:"VFMADD231SS"
	return z0 + z1
}

func outOfBoundsConv(i32 *[2]int32, u32 *[2]uint32, i64 *[2]int64, u64 *[2]uint64) {
	// arm64: "FCVTZSDW"
	// amd64: "CVTTSD2SL", "CVTSD2SS"
	i32[0] = int32(two40())
	// arm64: "FCVTZSDW"
	// amd64: "CVTTSD2SL", "CVTSD2SS"
	i32[1] = int32(-two40())
	// arm64: "FCVTZSDW"
	// amd64: "CVTTSD2SL", "CVTSD2SS"
	u32[0] = uint32(two41())
	// on arm64, this uses an explicit <0 comparison, so it constant folds.
	// on amd64, this uses an explicit <0 comparison, so it constant folds.
	// amd64: "MOVL [$]0,"
	u32[1] = uint32(minus1())
	// arm64: "FCVTZSD"
	// amd64: "CVTTSD2SQ"
	i64[0] = int64(two80())
	// arm64: "FCVTZSD"
	// amd64: "CVTTSD2SQ"
	i64[1] = int64(-two80())
	// arm64: "FCVTZUD"
	// amd64: "CVTTSD2SQ"
	u64[0] = uint64(two81())
	// arm64: "FCVTZUD"
	// on amd64, this uses an explicit <0 comparison, so it constant folds.
	// amd64: "MOVQ [$]0,"
	u64[1] = uint64(minus1())
}

func two40() float64 {
	return 1 << 40
}
func two41() float64 {
	return 1 << 41
}
func two80() float64 {
	return 1 << 80
}
func two81() float64 {
	return 1 << 81
}
func minus1() float64 {
	return -1
}
