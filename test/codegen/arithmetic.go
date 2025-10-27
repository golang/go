// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to arithmetic
// simplifications and optimizations on integer types.
// For codegen tests on float types, see floats.go.

// ----------------- //
//    Addition       //
// ----------------- //

func AddLargeConst(a uint64, out []uint64) {
	// ppc64x/power10:"ADD [$]4294967296,"
	// ppc64x/power9:"MOVD [$]1", "SLD [$]32" "ADD R[0-9]*"
	// ppc64x/power8:"MOVD [$]1", "SLD [$]32" "ADD R[0-9]*"
	out[0] = a + 0x100000000
	// ppc64x/power10:"ADD [$]-8589934592,"
	// ppc64x/power9:"MOVD [$]-1", "SLD [$]33" "ADD R[0-9]*"
	// ppc64x/power8:"MOVD [$]-1", "SLD [$]33" "ADD R[0-9]*"
	out[1] = a + 0xFFFFFFFE00000000
	// ppc64x/power10:"ADD [$]1234567,"
	// ppc64x/power9:"ADDIS [$]19,", "ADD [$]-10617,"
	// ppc64x/power8:"ADDIS [$]19,", "ADD [$]-10617,"
	out[2] = a + 1234567
	// ppc64x/power10:"ADD [$]-1234567,"
	// ppc64x/power9:"ADDIS [$]-19,", "ADD [$]10617,"
	// ppc64x/power8:"ADDIS [$]-19,", "ADD [$]10617,"
	out[3] = a - 1234567
	// ppc64x/power10:"ADD [$]2147450879,"
	// ppc64x/power9:"ADDIS [$]32767,", "ADD [$]32767,"
	// ppc64x/power8:"ADDIS [$]32767,", "ADD [$]32767,"
	out[4] = a + 0x7FFF7FFF
	// ppc64x/power10:"ADD [$]-2147483647,"
	// ppc64x/power9:"ADDIS [$]-32768,", "ADD [$]1,"
	// ppc64x/power8:"ADDIS [$]-32768,", "ADD [$]1,"
	out[5] = a - 2147483647
	// ppc64x:"ADDIS [$]-32768,", ^"ADD "
	out[6] = a - 2147483648
	// ppc64x:"ADD [$]2147450880,", ^"ADDIS "
	out[7] = a + 0x7FFF8000
	// ppc64x:"ADD [$]-32768,", ^"ADDIS "
	out[8] = a - 32768
	// ppc64x/power10:"ADD [$]-32769,"
	// ppc64x/power9:"ADDIS [$]-1,", "ADD [$]32767,"
	// ppc64x/power8:"ADDIS [$]-1,", "ADD [$]32767,"
	out[9] = a - 32769
}

func AddLargeConst2(a int, out []int) {
	// loong64: -"ADDVU" "ADDV16"
	out[0] = a + 0x10000
}

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
	// 386:"SUBL 4"
	// amd64:"SUBQ 8"
	return arr[0] - arr[1]
}

func SubFromConst(a int) int {
	// ppc64x: `SUBC R[0-9]+,\s[$]40,\sR`
	// riscv64: "ADDI \\$-40" "NEG"
	b := 40 - a
	return b
}

func SubFromConstNeg(a int) int {
	// arm64: "ADD \\$40"
	// loong64: "ADDV[U] \\$40"
	// mips: "ADD[U] \\$40"
	// mips64: "ADDV[U] \\$40"
	// ppc64x: `ADD [$]40,\sR[0-9]+,\sR`
	// riscv64: "ADDI \\$40" -"NEG"
	c := 40 - (-a)
	return c
}

func SubSubFromConst(a int) int {
	// arm64: "ADD \\$20"
	// loong64: "ADDV[U] \\$20"
	// mips: "ADD[U] \\$20"
	// mips64: "ADDV[U] \\$20"
	// ppc64x: `ADD [$]20,\sR[0-9]+,\sR`
	// riscv64: "ADDI \\$20" -"NEG"
	c := 40 - (20 - a)
	return c
}

func AddSubFromConst(a int) int {
	// ppc64x: `SUBC R[0-9]+,\s[$]60,\sR`
	// riscv64: "ADDI \\$-60" "NEG"
	c := 40 + (20 - a)
	return c
}

func NegSubFromConst(a int) int {
	// arm64: "SUB \\$20"
	// loong64: "ADDV[U] \\$-20"
	// mips: "ADD[U] \\$-20"
	// mips64: "ADDV[U] \\$-20"
	// ppc64x: `ADD [$]-20,\sR[0-9]+,\sR`
	// riscv64: "ADDI \\$-20"
	c := -(20 - a)
	return c
}

func NegAddFromConstNeg(a int) int {
	// arm64: "SUB \\$40" "NEG"
	// loong64: "ADDV[U] \\$-40" "SUBV"
	// mips: "ADD[U] \\$-40" "SUB"
	// mips64: "ADDV[U] \\$-40" "SUBV"
	// ppc64x: `SUBC R[0-9]+,\s[$]40,\sR`
	// riscv64: "ADDI \\$-40" "NEG"
	c := -(-40 + a)
	return c
}

func SubSubNegSimplify(a, b int) int {
	// amd64:"NEGQ"
	// arm64:"NEG"
	// loong64:"SUBV"
	// mips:"SUB"
	// mips64:"SUBV"
	// ppc64x:"NEG"
	// riscv64:"NEG" -"SUB"
	r := (a - b) - a
	return r
}

func SubAddSimplify(a, b int) int {
	// amd64:-"SUBQ" -"ADDQ"
	// arm64:-"SUB" -"ADD"
	// loong64:-"SUBV" -"ADDV"
	// mips:-"SUB" -"ADD"
	// mips64:-"SUBV" -"ADDV"
	// ppc64x:-"SUB" -"ADD"
	// riscv64:-"SUB" -"ADD"
	r := a + (b - a)
	return r
}

func SubAddSimplify2(a, b, c int) (int, int, int, int, int, int) {
	// amd64:-"ADDQ"
	// arm64:-"ADD"
	// mips:"SUB" -"ADD"
	// mips64:"SUBV" -"ADDV"
	// loong64:"SUBV" -"ADDV"
	r := (a + b) - (a + c)
	// amd64:-"ADDQ"
	r1 := (a + b) - (c + a)
	// amd64:-"ADDQ"
	r2 := (b + a) - (a + c)
	// amd64:-"ADDQ"
	r3 := (b + a) - (c + a)
	// amd64:-"SUBQ"
	// arm64:-"SUB"
	// mips:"ADD" -"SUB"
	// mips64:"ADDV" -"SUBV"
	// loong64:"ADDV" -"SUBV"
	r4 := (a - c) + (c + b)
	// amd64:-"SUBQ"
	r5 := (a - c) + (b + c)
	return r, r1, r2, r3, r4, r5
}

func SubAddNegSimplify(a, b int) int {
	// amd64:"NEGQ" -"ADDQ" -"SUBQ"
	// arm64:"NEG" -"ADD" -"SUB"
	// loong64:"SUBV" -"ADDV"
	// mips:"SUB" -"ADD"
	// mips64:"SUBV" -"ADDV"
	// ppc64x:"NEG" -"ADD" -"SUB"
	// riscv64:"NEG" -"ADD" -"SUB"
	r := a - (b + a)
	return r
}

func AddAddSubSimplify(a, b, c int) int {
	// amd64:-"SUBQ"
	// arm64:"ADD" -"SUB"
	// loong64:"ADDV" -"SUBV"
	// mips:"ADD" -"SUB"
	// mips64:"ADDV" -"SUBV"
	// ppc64x:-"SUB"
	// riscv64:"ADD" "ADD" -"SUB"
	r := a + (b + (c - a))
	return r
}

func NegToInt32(a int) int {
	// riscv64: "NEGW" -"MOVW"
	r := int(int32(-a))
	return r
}

// -------------------- //
//    Multiplication    //
// -------------------- //

func Pow2Muls(n1, n2 int) (int, int) {
	// amd64:"SHLQ [$]5" -"IMULQ"
	// 386:"SHLL [$]5" -"IMULL"
	// arm:"SLL [$]5" -"MUL"
	// arm64:"LSL [$]5" -"MUL"
	// loong64:"SLLV [$]5" -"MULV"
	// ppc64x:"SLD [$]5" -"MUL"
	a := n1 * 32

	// amd64:"SHLQ [$]6" -"IMULQ"
	// 386:"SHLL [$]6" -"IMULL"
	// arm:"SLL [$]6" -"MUL"
	// arm64:`NEG\sR[0-9]+<<6,\sR[0-9]+`,-`LSL`,-`MUL`
	// loong64:"SLLV [$]6" -"MULV"
	// ppc64x:"SLD [$]6" "NEG\\sR[0-9]+,\\sR[0-9]+" -"MUL"
	b := -64 * n2

	return a, b
}

func Mul_2(n1 int32, n2 int64) (int32, int64) {
	// amd64:"ADDL", -"SHLL"
	a := n1 * 2
	// amd64:"ADDQ", -"SHLQ"
	b := n2 * 2

	return a, b
}

func Mul_96(n int) int {
	// amd64:`SHLQ [$]5`,`LEAQ \(.*\)\(.*\*2\),`,-`IMULQ`
	// 386:`SHLL [$]5`,`LEAL \(.*\)\(.*\*2\),`,-`IMULL`
	// arm64:`LSL [$]5`,`ADD\sR[0-9]+<<1,\sR[0-9]+`,-`MUL`
	// arm:`SLL [$]5`,`ADD\sR[0-9]+<<1,\sR[0-9]+`,-`MUL`
	// loong64:"SLLV [$]5" "ALSLV [$]1,"
	// s390x:`SLD [$]5`,`SLD [$]6`,-`MULLD`
	return n * 96
}

func Mul_n120(n int) int {
	// loong64:"SLLV [$]3" "SLLV [$]7" "SUBVU" -"MULV"
	// s390x:`SLD [$]3`,`SLD [$]7`,-`MULLD`
	return n * -120
}

func MulMemSrc(a []uint32, b []float32) {
	// 386:`IMULL\s4\([A-Z]+\),\s[A-Z]+`
	a[0] *= a[1]
	// 386/sse2:`MULSS\s4\([A-Z]+\),\sX[0-9]+`
	// amd64:`MULSS\s4\([A-Z]+\),\sX[0-9]+`
	b[0] *= b[1]
}

// Multiplications merging tests

func MergeMuls1(n int) int {
	// amd64:"IMUL3Q [$]46"
	// 386:"IMUL3L [$]46"
	// ppc64x:"MULLD [$]46"
	return 15*n + 31*n // 46n
}

func MergeMuls2(n int) int {
	// amd64:"IMUL3Q [$]23" "(ADDQ [$]29)|(LEAQ 29)"
	// 386:"IMUL3L [$]23" "ADDL [$]29"
	// ppc64x/power9:"MADDLD" -"MULLD [$]23" -"ADD [$]29"
	// ppc64x/power8:"MULLD [$]23" "ADD [$]29"
	return 5*n + 7*(n+1) + 11*(n+2) // 23n + 29
}

func MergeMuls3(a, n int) int {
	// amd64:"ADDQ [$]19" -"IMULQ [$]19"
	// 386:"ADDL [$]19" -"IMULL [$]19"
	// ppc64x:"ADD [$]19" -"MULLD [$]19"
	return a*n + 19*n // (a+19)n
}

func MergeMuls4(n int) int {
	// amd64:"IMUL3Q [$]14"
	// 386:"IMUL3L [$]14"
	// ppc64x:"MULLD [$]14"
	return 23*n - 9*n // 14n
}

func MergeMuls5(a, n int) int {
	// amd64:"ADDQ [$]-19" -"IMULQ [$]19"
	// 386:"ADDL [$]-19" -"IMULL [$]19"
	// ppc64x:"ADD [$]-19" -"MULLD [$]19"
	return a*n - 19*n // (a-19)n
}

// Multiplications folded negation

func FoldNegMul(a int) int {
	// loong64:"SUBVU" "ALSLV [$]2" "ALSLV [$]1"
	return (-a) * 11
}

func Fold2NegMul(a, b int) int {
	// loong64:"MULV" -"SUBVU R[0-9], R0,"
	return (-a) * (-b)
}

// -------------- //
//    Division    //
// -------------- //

func DivMemSrc(a []float64) {
	// 386/sse2:`DIVSD\s8\([A-Z]+\),\sX[0-9]+`
	// amd64:`DIVSD\s8\([A-Z]+\),\sX[0-9]+`
	a[0] /= a[1]
}

func Pow2Divs(n1 uint, n2 int) (uint, int) {
	// 386:"SHRL [$]5" -"DIVL"
	// amd64:"SHRQ [$]5" -"DIVQ"
	// arm:"SRL [$]5" -".*udiv"
	// arm64:"LSR [$]5" -"UDIV"
	// ppc64x:"SRD"
	a := n1 / 32 // unsigned

	// amd64:"SARQ [$]6" -"IDIVQ"
	// 386:"SARL [$]6" -"IDIVL"
	// arm:"SRA [$]6" -".*udiv"
	// arm64:"ASR [$]6" -"SDIV"
	// ppc64x:"SRAD"
	b := n2 / 64 // signed

	return a, b
}

// Check that constant divisions get turned into MULs
func ConstDivs(n1 uint, n2 int) (uint, int) {
	// amd64:"MOVQ [$]-1085102592571150095" "MULQ" -"DIVQ"
	// 386:"MOVL [$]-252645135" "MULL" -"DIVL"
	// arm64:`MOVD`,`UMULH`,-`DIV`
	// arm:`MOVW`,`MUL`,-`.*udiv`
	a := n1 / 17 // unsigned

	// amd64:"MOVQ [$]-1085102592571150095" "IMULQ" -"IDIVQ"
	// 386:"IMULL" -"IDIVL"
	// arm64:`SMULH`,-`DIV`
	// arm:`MOVW`,`MUL`,-`.*udiv`
	b := n2 / 17 // signed

	return a, b
}

func FloatDivs(a []float32) float32 {
	// amd64:`DIVSS\s8\([A-Z]+\),\sX[0-9]+`
	// 386/sse2:`DIVSS\s8\([A-Z]+\),\sX[0-9]+`
	return a[1] / a[2]
}

func Pow2Mods(n1 uint, n2 int) (uint, int) {
	// 386:"ANDL [$]31" -"DIVL"
	// amd64:"ANDL [$]31" -"DIVQ"
	// arm:"AND [$]31" -".*udiv"
	// arm64:"AND [$]31" -"UDIV"
	// ppc64x:"RLDICL"
	a := n1 % 32 // unsigned

	// 386:"SHRL" -"IDIVL"
	// amd64:"SHRQ" -"IDIVQ"
	// arm:"SRA" -".*udiv"
	// arm64:"ASR" -"REM"
	// ppc64x:"SRAD"
	b := n2 % 64 // signed

	return a, b
}

// Check that signed divisibility checks get converted to AND on low bits
func Pow2DivisibleSigned(n1, n2 int) (bool, bool) {
	// 386:"TESTL [$]63" -"DIVL" -"SHRL"
	// amd64:"TESTQ [$]63" -"DIVQ" -"SHRQ"
	// arm:"AND [$]63" -".*udiv" -"SRA"
	// arm64:"TST [$]63" -"UDIV" -"ASR" -"AND"
	// ppc64x:"ANDCC" -"RLDICL" -"SRAD" -"CMP"
	a := n1%64 == 0 // signed divisible

	// 386:"TESTL [$]63" -"DIVL" -"SHRL"
	// amd64:"TESTQ [$]63" -"DIVQ" -"SHRQ"
	// arm:"AND [$]63" -".*udiv" -"SRA"
	// arm64:"TST [$]63" -"UDIV" -"ASR" -"AND"
	// ppc64x:"ANDCC" -"RLDICL" -"SRAD" -"CMP"
	b := n2%64 != 0 // signed indivisible

	return a, b
}

// Check that constant modulo divs get turned into MULs
func ConstMods(n1 uint, n2 int) (uint, int) {
	// amd64:"MOVQ [$]-1085102592571150095" "MULQ" -"DIVQ"
	// 386:"MOVL [$]-252645135" "MULL" -"DIVL"
	// arm64:`MOVD`,`UMULH`,-`DIV`
	// arm:`MOVW`,`MUL`,-`.*udiv`
	a := n1 % 17 // unsigned

	// amd64:"MOVQ [$]-1085102592571150095" "IMULQ" -"IDIVQ"
	// 386: "IMULL" -"IDIVL"
	// arm64:`SMULH`,-`DIV`
	// arm:`MOVW`,`MUL`,-`.*udiv`
	b := n2 % 17 // signed

	return a, b
}

// Check that divisibility checks x%c==0 are converted to MULs and rotates
func DivisibleU(n uint) (bool, bool) {
	// amd64:"MOVQ [$]-6148914691236517205" "IMULQ" "ROLQ [$]63" -"DIVQ"
	// 386:"IMUL3L [$]-1431655765" "ROLL [$]31" -"DIVQ"
	// arm64:"MOVD [$]-6148914691236517205" "MOVD [$]3074457345618258602" "MUL" "ROR" -"DIV"
	// arm:"MUL" "CMP [$]715827882" -".*udiv"
	// ppc64x:"MULLD" "ROTL [$]63"
	even := n%6 == 0

	// amd64:"MOVQ [$]-8737931403336103397" "IMULQ" -"ROLQ" -"DIVQ"
	// 386:"IMUL3L [$]678152731" -"ROLL" -"DIVQ"
	// arm64:"MOVD [$]-8737931403336103397" "MUL" -"ROR" -"DIV"
	// arm:"MUL" "CMP [$]226050910" -".*udiv"
	// ppc64x:"MULLD" -"ROTL"
	odd := n%19 == 0

	return even, odd
}

func Divisible(n int) (bool, bool) {
	// amd64:"IMULQ" "ADD" "ROLQ [$]63" -"DIVQ"
	// 386:"IMUL3L [$]-1431655765" "ADDL [$]715827882" "ROLL [$]31" -"DIVQ"
	// arm64:"MOVD [$]-6148914691236517205" "MOVD [$]3074457345618258602" "MUL" "ADD R" "ROR" -"DIV"
	// arm:"MUL" "ADD [$]715827882" -".*udiv"
	// ppc64x/power8:"MULLD" "ADD" "ROTL [$]63"
	// ppc64x/power9:"MADDLD" "ROTL [$]63"
	even := n%6 == 0

	// amd64:"IMULQ" "ADD" -"ROLQ" -"DIVQ"
	// 386:"IMUL3L [$]678152731" "ADDL [$]113025455" -"ROLL" -"DIVQ"
	// arm64:"MUL" "MOVD [$]485440633518672410" "ADD" -"ROR" -"DIV"
	// arm:"MUL" "ADD [$]113025455" -".*udiv"
	// ppc64x/power8:"MULLD" "ADD" -"ROTL"
	// ppc64x/power9:"MADDLD" -"ROTL"
	odd := n%19 == 0

	return even, odd
}

// Check that fix-up code is not generated for divisions where it has been proven that
// that the divisor is not -1 or that the dividend is > MinIntNN.
func NoFix64A(divr int64) (int64, int64) {
	var d int64 = 42
	var e int64 = 84
	if divr > 5 {
		d /= divr // amd64:-"JMP"
		e %= divr // amd64:-"JMP"
		// The following statement is to avoid conflict between the above check
		// and the normal JMP generated at the end of the block.
		d += e
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
		d += e
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
		d += e
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
		d += e
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
		d += e
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
		d += e
	}
	return d, e
}

// Check that len() and cap() calls divided by powers of two are
// optimized into shifts and ands

func LenDiv1(a []int) int {
	// 386:"SHRL [$]10"
	// amd64:"SHRQ [$]10"
	// arm64:"LSR [$]10" -"SDIV"
	// arm:"SRL [$]10" -".*udiv"
	// ppc64x:"SRD" [$]10"
	return len(a) / 1024
}

func LenDiv2(s string) int {
	// 386:"SHRL [$]11"
	// amd64:"SHRQ [$]11"
	// arm64:"LSR [$]11" -"SDIV"
	// arm:"SRL [$]11" -".*udiv"
	// ppc64x:"SRD [$]11"
	return len(s) / (4097 >> 1)
}

func LenMod1(a []int) int {
	// 386:"ANDL [$]1023"
	// amd64:"ANDL [$]1023"
	// arm64:"AND [$]1023" -"SDIV"
	// arm/6:"AND" -".*udiv"
	// arm/7:"BFC" -".*udiv" -"AND"
	// ppc64x:"RLDICL"
	return len(a) % 1024
}

func LenMod2(s string) int {
	// 386:"ANDL [$]2047"
	// amd64:"ANDL [$]2047"
	// arm64:"AND [$]2047" -"SDIV"
	// arm/6:"AND" -".*udiv"
	// arm/7:"BFC" -".*udiv" -"AND"
	// ppc64x:"RLDICL"
	return len(s) % (4097 >> 1)
}

func CapDiv(a []int) int {
	// 386:"SHRL [$]12"
	// amd64:"SHRQ [$]12"
	// arm64:"LSR [$]12" -"SDIV"
	// arm:"SRL [$]12" -".*udiv"
	// ppc64x:"SRD [$]12"
	return cap(a) / ((1 << 11) + 2048)
}

func CapMod(a []int) int {
	// 386:"ANDL [$]4095"
	// amd64:"ANDL [$]4095"
	// arm64:"AND [$]4095" -"SDIV"
	// arm/6:"AND" -".*udiv"
	// arm/7:"BFC" -".*udiv" -"AND"
	// ppc64x:"RLDICL"
	return cap(a) % ((1 << 11) + 2048)
}

func AddMul(x int) int {
	// amd64:"LEAQ 1"
	return 2*x + 1
}

func AddShift(a, b int) int {
	// loong64: "ALSLV"
	return a + (b << 4)
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
	// ppc64x:`ADD`,-`MULLD`
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
	// ppc64x:`SUB`,-`MULLD`
	r2 := c - b*64
	return r0, r1, r2
}

func addSpecial(a, b, c uint32) (uint32, uint32, uint32) {
	// amd64:`INCL`
	a++
	// amd64:`DECL`
	b--
	// amd64:`SUBL.*-128`
	c += 128
	return a, b, c
}

// Divide -> shift rules usually require fixup for negative inputs.
// If the input is non-negative, make sure the fixup is eliminated.
func divInt(v int64) int64 {
	if v < 0 {
		return 0
	}
	// amd64:-`.*SARQ.*63,`, -".*SHRQ", ".*SARQ.*[$]9,"
	return v / 512
}

// The reassociate rules "x - (z + C) -> (x - z) - C" and
// "(z + C) -x -> C + (z - x)" can optimize the following cases.
func constantFold1(i0, j0, i1, j1, i2, j2, i3, j3 int) (int, int, int, int) {
	// arm64:"SUB" "ADD [$]2"
	// ppc64x:"SUB" "ADD [$]2"
	r0 := (i0 + 3) - (j0 + 1)
	// arm64:"SUB" "SUB [$]4"
	// ppc64x:"SUB" "ADD [$]-4"
	r1 := (i1 - 3) - (j1 + 1)
	// arm64:"SUB" "ADD [$]4"
	// ppc64x:"SUB" "ADD [$]4"
	r2 := (i2 + 3) - (j2 - 1)
	// arm64:"SUB" "SUB [$]2"
	// ppc64x:"SUB" "ADD [$]-2"
	r3 := (i3 - 3) - (j3 - 1)
	return r0, r1, r2, r3
}

// The reassociate rules "x - (z + C) -> (x - z) - C" and
// "(C - z) - x -> C - (z + x)" can optimize the following cases.
func constantFold2(i0, j0, i1, j1 int) (int, int) {
	// arm64:"ADD" "MOVD [$]2" "SUB"
	// ppc64x: `SUBC R[0-9]+,\s[$]2,\sR`
	r0 := (3 - i0) - (j0 + 1)
	// arm64:"ADD" "MOVD [$]4" "SUB"
	// ppc64x: `SUBC R[0-9]+,\s[$]4,\sR`
	r1 := (3 - i1) - (j1 - 1)
	return r0, r1
}

func constantFold3(i, j int) int {
	// arm64: "LSL [$]5," "SUB R[0-9]+<<1," -"ADD"
	// ppc64x:"MULLD [$]30" "MULLD"
	r := (5 * i) * (6 * j)
	return r
}

// ----------------- //
//  Integer Min/Max  //
// ----------------- //

func Int64Min(a, b int64) int64 {
	// amd64: "CMPQ" "CMOVQLT"
	// arm64: "CMP" "CSEL"
	// riscv64/rva20u64:"BLT "
	// riscv64/rva22u64,riscv64/rva23u64:"MIN "
	return min(a, b)
}

func Int64Max(a, b int64) int64 {
	// amd64: "CMPQ" "CMOVQGT"
	// arm64: "CMP" "CSEL"
	// riscv64/rva20u64:"BLT "
	// riscv64/rva22u64,riscv64/rva23u64:"MAX "
	return max(a, b)
}

func Uint64Min(a, b uint64) uint64 {
	// amd64: "CMPQ" "CMOVQCS"
	// arm64: "CMP" "CSEL"
	// riscv64/rva20u64:"BLTU"
	// riscv64/rva22u64,riscv64/rva23u64:"MINU"
	return min(a, b)
}

func Uint64Max(a, b uint64) uint64 {
	// amd64: "CMPQ" "CMOVQHI"
	// arm64: "CMP" "CSEL"
	// riscv64/rva20u64:"BLTU"
	// riscv64/rva22u64,riscv64/rva23u64:"MAXU"
	return max(a, b)
}
