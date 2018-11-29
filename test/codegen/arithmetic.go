// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to arithmetic
// simplifications and optimizations on integer types.
// For codegen tests on float types, see floats.go.

// ----------------- //
//    Subtraction    //
// ----------------- //

var ef int
func SubMem(arr []int, b, c, d int) int {
	// 386:`SUBL\s[A-Z]+,\s8\([A-Z]+\)`
	// amd64:`SUBQ\s[A-Z]+,\s16\([A-Z]+\)`
	arr[2] -= b
	// 386:`SUBL\s[A-Z]+,\s12\([A-Z]+\)`
	// amd64:`SUBQ\s[A-Z]+,\s24\([A-Z]+\)`
	arr[3] -= b
	// 386:`DECL\s16\([A-Z]+\)`
	arr[4]--
	// 386:`ADDL\s[$]-20,\s20\([A-Z]+\)`
	arr[5] -= 20
	// 386:`SUBL\s\([A-Z]+\)\([A-Z]+\*4\),\s[A-Z]+`
	ef -= arr[b]
	// 386:`SUBL\s[A-Z]+,\s\([A-Z]+\)\([A-Z]+\*4\)`
	arr[c] -= b
	// 386:`ADDL\s[$]-15,\s\([A-Z]+\)\([A-Z]+\*4\)`
	arr[d] -= 15
	// 386:`DECL\s\([A-Z]+\)\([A-Z]+\*4\)`
	arr[b]--
	// amd64:`DECQ\s64\([A-Z]+\)`
	arr[8]--
	// 386:"SUBL\t4"
	// amd64:"SUBQ\t8"
	return arr[0] - arr[1]
}

// -------------------- //
//    Multiplication    //
// -------------------- //

func Pow2Muls(n1, n2 int) (int, int) {
	// amd64:"SHLQ\t[$]5",-"IMULQ"
	// 386:"SHLL\t[$]5",-"IMULL"
	// arm:"SLL\t[$]5",-"MUL"
	// arm64:"LSL\t[$]5",-"MUL"
	// ppc64:"SLD\t[$]5",-"MUL"
	// ppc64le:"SLD\t[$]5",-"MUL"
	a := n1 * 32

	// amd64:"SHLQ\t[$]6",-"IMULQ"
	// 386:"SHLL\t[$]6",-"IMULL"
	// arm:"SLL\t[$]6",-"MUL"
	// arm64:`NEG\sR[0-9]+<<6,\sR[0-9]+`,-`LSL`,-`MUL`
	// ppc64:"SLD\t[$]6","NEG\\sR[0-9]+,\\sR[0-9]+",-"MUL"
	// ppc64le:"SLD\t[$]6","NEG\\sR[0-9]+,\\sR[0-9]+",-"MUL"
	b := -64 * n2

	return a, b
}

func Mul_96(n int) int {
	// amd64:`SHLQ\t[$]5`,`LEAQ\t\(.*\)\(.*\*2\),`
	return n * 96
}

func MulMemSrc(a []uint32, b []float32) {
	// 386:`IMULL\s4\([A-Z]+\),\s[A-Z]+`
	a[0] *= a[1]
	// 386/sse2:`MULSS\s4\([A-Z]+\),\sX[0-9]+`
	b[0] *= b[1]
}

// Multiplications merging tests

func MergeMuls1(n int) int {
	// amd64:"IMUL3Q\t[$]46"
	// 386:"IMUL3L\t[$]46"
	return 15*n + 31*n // 46n
}

func MergeMuls2(n int) int {
	// amd64:"IMUL3Q\t[$]23","ADDQ\t[$]29"
	// 386:"IMUL3L\t[$]23","ADDL\t[$]29"
	return 5*n + 7*(n+1) + 11*(n+2) // 23n + 29
}

func MergeMuls3(a, n int) int {
	// amd64:"ADDQ\t[$]19",-"IMULQ\t[$]19"
	// 386:"ADDL\t[$]19",-"IMULL\t[$]19"
	return a*n + 19*n // (a+19)n
}

func MergeMuls4(n int) int {
	// amd64:"IMUL3Q\t[$]14"
	// 386:"IMUL3L\t[$]14"
	return 23*n - 9*n // 14n
}

func MergeMuls5(a, n int) int {
	// amd64:"ADDQ\t[$]-19",-"IMULQ\t[$]19"
	// 386:"ADDL\t[$]-19",-"IMULL\t[$]19"
	return a*n - 19*n // (a-19)n
}

// -------------- //
//    Division    //
// -------------- //

func DivMemSrc(a []float64) {
	// 386/sse2:`DIVSD\s8\([A-Z]+\),\sX[0-9]+`
	a[0] /= a[1]
}

func Pow2Divs(n1 uint, n2 int) (uint, int) {
	// 386:"SHRL\t[$]5",-"DIVL"
	// amd64:"SHRQ\t[$]5",-"DIVQ"
	// arm:"SRL\t[$]5",-".*udiv"
	// arm64:"LSR\t[$]5",-"UDIV"
	// ppc64:"SRD"
	// ppc64le:"SRD"
	a := n1 / 32 // unsigned

	// amd64:"SARQ\t[$]6",-"IDIVQ"
	// 386:"SARL\t[$]6",-"IDIVL"
	// arm:"SRA\t[$]6",-".*udiv"
	// arm64:"ASR\t[$]6",-"SDIV"
	// ppc64:"SRAD"
	// ppc64le:"SRAD"
	b := n2 / 64 // signed

	return a, b
}

// Check that constant divisions get turned into MULs
func ConstDivs(n1 uint, n2 int) (uint, int) {
	// amd64:"MOVQ\t[$]-1085102592571150095","MULQ",-"DIVQ"
	a := n1 / 17 // unsigned

	// amd64:"MOVQ\t[$]-1085102592571150095","IMULQ",-"IDIVQ"
	b := n2 / 17 // signed

	return a, b
}

func FloatDivs(a []float32) float32 {
	// amd64:`DIVSS\s8\([A-Z]+\),\sX[0-9]+`
	return a[1] / a[2]
}

func Pow2Mods(n1 uint, n2 int) (uint, int) {
	// 386:"ANDL\t[$]31",-"DIVL"
	// amd64:"ANDQ\t[$]31",-"DIVQ"
	// arm:"AND\t[$]31",-".*udiv"
	// arm64:"AND\t[$]31",-"UDIV"
	// ppc64:"ANDCC\t[$]31"
	// ppc64le:"ANDCC\t[$]31"
	a := n1 % 32 // unsigned

	// 386:-"IDIVL"
	// amd64:-"IDIVQ"
	// arm:-".*udiv"
	// arm64:-"REM"
	b := n2 % 64 // signed

	return a, b
}

// Check that constant modulo divs get turned into MULs
func ConstMods(n1 uint, n2 int) (uint, int) {
	// amd64:"MOVQ\t[$]-1085102592571150095","MULQ",-"DIVQ"
	a := n1 % 17 // unsigned

	// amd64:"MOVQ\t[$]-1085102592571150095","IMULQ",-"IDIVQ"
	b := n2 % 17 // signed

	return a, b
}

// Check that fix-up code is not generated for divisions where it has been proven that
// that the divisor is not -1 or that the dividend is > MinIntNN.
func NoFix64A(divr int64) (int64, int64) {
	var d int64 = 42
	var e int64 = 84
	if divr > 5 {
		d /= divr // amd64:-"JMP"
		e %= divr // amd64:-"JMP"
	}
	return d, e
}

func NoFix64B(divd int64) (int64, int64) {
	var d int64
	var e int64
	var divr int64 = -1
	if divd > -9223372036854775808 {
		d = divd / divr // amd64:-"JMP"
		e = divd % divr // amd64:-"JMP"
	}
	return d, e
}

func NoFix32A(divr int32) (int32, int32) {
	var d int32 = 42
	var e int32 = 84
	if divr > 5 {
		// amd64:-"JMP"
		// 386:-"JMP"
		d /= divr
		// amd64:-"JMP"
		// 386:-"JMP"
		e %= divr
	}
	return d, e
}

func NoFix32B(divd int32) (int32, int32) {
	var d int32
	var e int32
	var divr int32 = -1
	if divd > -2147483648 {
		// amd64:-"JMP"
		// 386:-"JMP"
		d = divd / divr
		// amd64:-"JMP"
		// 386:-"JMP"
		e = divd % divr
	}
	return d, e
}

func NoFix16A(divr int16) (int16, int16) {
	var d int16 = 42
	var e int16 = 84
	if divr > 5 {
		// amd64:-"JMP"
		// 386:-"JMP"
		d /= divr
		// amd64:-"JMP"
		// 386:-"JMP"
		e %= divr
	}
	return d, e
}

func NoFix16B(divd int16) (int16, int16) {
	var d int16
	var e int16
	var divr int16 = -1
	if divd > -32768 {
		// amd64:-"JMP"
		// 386:-"JMP"
		d = divd / divr
		// amd64:-"JMP"
		// 386:-"JMP"
		e = divd % divr
	}
	return d, e
}

// Check that len() and cap() calls divided by powers of two are
// optimized into shifts and ands

func LenDiv1(a []int) int {
	// 386:"SHRL\t[$]10"
	// amd64:"SHRQ\t[$]10"
	// ppc64:"SRD"\t[$]10"
	// ppc64le:"SRD"\t[$]10"
	return len(a) / 1024
}

func LenDiv2(s string) int {
	// 386:"SHRL\t[$]11"
	// amd64:"SHRQ\t[$]11"
	// ppc64:"SRD\t[$]11"
	// ppc64le:"SRD\t[$]11"
	return len(s) / (4097 >> 1)
}

func LenMod1(a []int) int {
	// 386:"ANDL\t[$]1023"
	// amd64:"ANDQ\t[$]1023"
	// ppc64:"ANDCC\t[$]1023"
	// ppc64le:"ANDCC\t[$]1023"
	return len(a) % 1024
}

func LenMod2(s string) int {
	// 386:"ANDL\t[$]2047"
	// amd64:"ANDQ\t[$]2047"
	// ppc64:"ANDCC\t[$]2047"
	// ppc64le:"ANDCC\t[$]2047"
	return len(s) % (4097 >> 1)
}

func CapDiv(a []int) int {
	// 386:"SHRL\t[$]12"
	// amd64:"SHRQ\t[$]12"
	// ppc64:"SRD\t[$]12"
	// ppc64le:"SRD\t[$]12"
	return cap(a) / ((1 << 11) + 2048)
}

func CapMod(a []int) int {
	// 386:"ANDL\t[$]4095"
	// amd64:"ANDQ\t[$]4095"
	// ppc64:"ANDCC\t[$]4095"
	// ppc64le:"ANDCC\t[$]4095"
	return cap(a) % ((1 << 11) + 2048)
}

func AddMul(x int) int {
	// amd64:"LEAQ\t1"
	return 2*x + 1
}

func MULA(a, b, c uint32) (uint32, uint32, uint32) {
	// arm:`MULA`,-`MUL\s`
	// arm64:`MADDW`,-`MULW`
	r0 := a*b + c
	// arm:`MULA`,-`MUL\s`
	// arm64:`MADDW`,-`MULW`
	r1 := c*79 + a
	// arm:`ADD`,-`MULA`,-`MUL\s`
	// arm64:`ADD`,-`MADD`,-`MULW`
	r2 := b*64 + c
	return r0, r1, r2
}

func MULS(a, b, c uint32) (uint32, uint32, uint32) {
	// arm/7:`MULS`,-`MUL\s`
	// arm/6:`SUB`,`MUL\s`,-`MULS`
	// arm64:`MSUBW`,-`MULW`
	r0 := c - a*b
	// arm/7:`MULS`,-`MUL\s`
	// arm/6:`SUB`,`MUL\s`,-`MULS`
	// arm64:`MSUBW`,-`MULW`
	r1 := a - c*79
	// arm/7:`SUB`,-`MULS`,-`MUL\s`
	// arm64:`SUB`,-`MSUBW`,-`MULW`
	r2 := c - b*64
	return r0, r1, r2
}
