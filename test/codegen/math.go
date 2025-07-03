// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math"

var sink64 [8]float64

func approx(x float64) {
	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD\t[$]2"
	// s390x:"FIDBR\t[$]6"
	// arm64:"FRINTPD"
	// ppc64x:"FRIP"
	// wasm:"F64Ceil"
	sink64[0] = math.Ceil(x)

	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD\t[$]1"
	// s390x:"FIDBR\t[$]7"
	// arm64:"FRINTMD"
	// ppc64x:"FRIM"
	// wasm:"F64Floor"
	sink64[1] = math.Floor(x)

	// s390x:"FIDBR\t[$]1"
	// arm64:"FRINTAD"
	// ppc64x:"FRIN"
	sink64[2] = math.Round(x)

	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD\t[$]3"
	// s390x:"FIDBR\t[$]5"
	// arm64:"FRINTZD"
	// ppc64x:"FRIZ"
	// wasm:"F64Trunc"
	sink64[3] = math.Trunc(x)

	// amd64/v2:-".*x86HasSSE41" amd64/v3:-".*x86HasSSE41"
	// amd64:"ROUNDSD\t[$]0"
	// s390x:"FIDBR\t[$]4"
	// arm64:"FRINTND"
	// wasm:"F64Nearest"
	sink64[4] = math.RoundToEven(x)
}

func sqrt(x float64) float64 {
	// amd64:"SQRTSD"
	// 386/sse2:"SQRTSD" 386/softfloat:-"SQRTD"
	// arm64:"FSQRTD"
	// arm/7:"SQRTD"
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
	// mips/hardfloat:"SQRTF" mips/softfloat:-"SQRTF"
	// mips64/hardfloat:"SQRTF" mips64/softfloat:-"SQRTF"
	// wasm:"F32Sqrt"
	// ppc64x:"FSQRTS"
	// riscv64: "FSQRTS"
	return float32(math.Sqrt(float64(x)))
}

// Check that it's using integer registers
func abs(x, y float64) {
	// amd64:"BTRQ\t[$]63"
	// arm64:"FABSD\t"
	// s390x:"LPDFR\t",-"MOVD\t"     (no integer load/store)
	// ppc64x:"FABS\t"
	// riscv64:"FABSD\t"
	// wasm:"F64Abs"
	// arm/6:"ABSD\t"
	// mips64/hardfloat:"ABSD\t"
	// mips/hardfloat:"ABSD\t"
	sink64[0] = math.Abs(x)

	// amd64:"BTRQ\t[$]63","PXOR"    (TODO: this should be BTSQ)
	// s390x:"LNDFR\t",-"MOVD\t"     (no integer load/store)
	// ppc64x:"FNABS\t"
	sink64[1] = -math.Abs(y)
}

// Check that it's using integer registers
func abs32(x float32) float32 {
	// s390x:"LPDFR",-"LDEBR",-"LEDBR"     (no float64 conversion)
	return float32(math.Abs(float64(x)))
}

// Check that it's using integer registers
func copysign(a, b, c float64) {
	// amd64:"BTRQ\t[$]63","ANDQ","ORQ"
	// s390x:"CPSDR",-"MOVD"         (no integer load/store)
	// ppc64x:"FCPSGN"
	// riscv64:"FSGNJD"
	// wasm:"F64Copysign"
	sink64[0] = math.Copysign(a, b)

	// amd64:"BTSQ\t[$]63"
	// s390x:"LNDFR\t",-"MOVD\t"     (no integer load/store)
	// ppc64x:"FCPSGN"
	// riscv64:"FSGNJD"
	// arm64:"ORR", -"AND"
	sink64[1] = math.Copysign(c, -1)

	// Like math.Copysign(c, -1), but with integer operations. Useful
	// for platforms that have a copysign opcode to see if it's detected.
	// s390x:"LNDFR\t",-"MOVD\t"     (no integer load/store)
	sink64[2] = math.Float64frombits(math.Float64bits(a) | 1<<63)

	// amd64:"ANDQ","ORQ"
	// s390x:"CPSDR\t",-"MOVD\t"     (no integer load/store)
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
	// riscv64:"FNMSUBD",-"FNMADDD"
	return math.FMA(-x, y, z)
}

func fnma(x, y, z float64) float64 {
	// riscv64:"FNMADDD",-"FNMSUBD"
	return math.FMA(x, -y, -z)
}

func fromFloat64(f64 float64) uint64 {
	// amd64:"MOVQ\tX.*, [^X].*"
	// arm64:"FMOVD\tF.*, R.*"
	// loong64:"MOVV\tF.*, R.*"
	// ppc64x:"MFVSRD"
	// mips64/hardfloat:"MOVV\tF.*, R.*"
	return math.Float64bits(f64+1) + 1
}

func fromFloat32(f32 float32) uint32 {
	// amd64:"MOVL\tX.*, [^X].*"
	// arm64:"FMOVS\tF.*, R.*"
	// loong64:"MOVW\tF.*, R.*"
	// mips64/hardfloat:"MOVW\tF.*, R.*"
	return math.Float32bits(f32+1) + 1
}

func toFloat64(u64 uint64) float64 {
	// amd64:"MOVQ\t[^X].*, X.*"
	// arm64:"FMOVD\tR.*, F.*"
	// loong64:"MOVV\tR.*, F.*"
	// ppc64x:"MTVSRD"
	// mips64/hardfloat:"MOVV\tR.*, F.*"
	return math.Float64frombits(u64+1) + 1
}

func toFloat32(u32 uint32) float32 {
	// amd64:"MOVL\t[^X].*, X.*"
	// arm64:"FMOVS\tR.*, F.*"
	// loong64:"MOVW\tR.*, F.*"
	// mips64/hardfloat:"MOVW\tR.*, F.*"
	return math.Float32frombits(u32+1) + 1
}

// Test that comparisons with constants converted to float
// are evaluated at compile-time

func constantCheck64() bool {
	// amd64:"(MOVB\t[$]0)|(XORL\t[A-Z][A-Z0-9]+, [A-Z][A-Z0-9]+)",-"FCMP",-"MOVB\t[$]1"
	// s390x:"MOV(B|BZ|D)\t[$]0,",-"FCMPU",-"MOV(B|BZ|D)\t[$]1,"
	return 0.5 == float64(uint32(1)) || 1.5 > float64(uint64(1<<63))
}

func constantCheck32() bool {
	// amd64:"MOV(B|L)\t[$]1",-"FCMP",-"MOV(B|L)\t[$]0"
	// s390x:"MOV(B|BZ|D)\t[$]1,",-"FCMPU",-"MOV(B|BZ|D)\t[$]0,"
	return float32(0.5) <= float32(int64(1)) && float32(1.5) >= float32(int32(-1<<31))
}

// Test that integer constants are converted to floating point constants
// at compile-time

func constantConvert32(x float32) float32 {
	// amd64:"MOVSS\t[$]f32.3f800000\\(SB\\)"
	// s390x:"FMOVS\t[$]f32.3f800000\\(SB\\)"
	// ppc64x/power8:"FMOVS\t[$]f32.3f800000\\(SB\\)"
	// ppc64x/power9:"FMOVS\t[$]f32.3f800000\\(SB\\)"
	// ppc64x/power10:"XXSPLTIDP\t[$]1065353216, VS0"
	// arm64:"FMOVS\t[$]\\(1.0\\)"
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
