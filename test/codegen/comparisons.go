// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "unsafe"

// This file contains code generation tests related to the comparison
// operators.

// -------------- //
//    Equality    //
// -------------- //

// Check that compare to constant string use 2/4/8 byte compares

func CompareString1(s string) bool {
	// amd64:`CMPW\t\(.*\), [$]`
	// arm64:`MOVHU\t\(.*\), [R]`,`CMPW\t[$]`
	// ppc64le:`MOVHZ\t\(.*\), [R]`,`CMPW\t.*, [$]`
	// s390x:`MOVHBR\t\(.*\), [R]`,`CMPW\t.*, [$]`
	return s == "xx"
}

func CompareString2(s string) bool {
	// amd64:`CMPL\t\(.*\), [$]`
	// arm64:`MOVWU\t\(.*\), [R]`,`CMPW\t.*, [R]`
	// ppc64le:`MOVWZ\t\(.*\), [R]`,`CMPW\t.*, [R]`
	// s390x:`MOVWBR\t\(.*\), [R]`,`CMPW\t.*, [$]`
	return s == "xxxx"
}

func CompareString3(s string) bool {
	// amd64:`CMPQ\t\(.*\), [A-Z]`
	// arm64:-`CMPW\t`
	// ppc64:-`CMPW\t`
	// ppc64le:-`CMPW\t`
	// s390x:-`CMPW\t`
	return s == "xxxxxxxx"
}

// Check that arrays compare use 2/4/8 byte compares

func CompareArray1(a, b [2]byte) bool {
	// amd64:`CMPW\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	// arm64:-`MOVBU\t`
	// ppc64le:-`MOVBZ\t`
	// s390x:-`MOVBZ\t`
	return a == b
}

func CompareArray2(a, b [3]uint16) bool {
	// amd64:`CMPL\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	// amd64:`CMPW\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	return a == b
}

func CompareArray3(a, b [3]int16) bool {
	// amd64:`CMPL\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	// amd64:`CMPW\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	return a == b
}

func CompareArray4(a, b [12]int8) bool {
	// amd64:`CMPQ\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	// amd64:`CMPL\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	return a == b
}

func CompareArray5(a, b [15]byte) bool {
	// amd64:`CMPQ\t""[.+_a-z0-9]+\(SP\), [A-Z]`
	return a == b
}

// This was a TODO in mapaccess1_faststr
func CompareArray6(a, b unsafe.Pointer) bool {
	// amd64:`CMPL\t\(.*\), [A-Z]`
	// arm64:`MOVWU\t\(.*\), [R]`,`CMPW\t.*, [R]`
	// ppc64le:`MOVWZ\t\(.*\), [R]`,`CMPW\t.*, [R]`
	// s390x:`MOVWBR\t\(.*\), [R]`,`CMPW\t.*, [R]`
	return *((*[4]byte)(a)) != *((*[4]byte)(b))
}

// -------------- //
//    Ordering    //
// -------------- //

// Test that LEAQ/ADDQconst are folded into SETx ops

var r bool

func CmpFold(x uint32) {
	// amd64:`SETHI\t.*\(SB\)`
	r = x > 4
}

// Test that direct comparisons with memory are generated when
// possible

func CmpMem1(p int, q *int) bool {
	// amd64:`CMPQ\t\(.*\), [A-Z]`
	return p < *q
}

func CmpMem2(p *int, q int) bool {
	// amd64:`CMPQ\t\(.*\), [A-Z]`
	return *p < q
}

func CmpMem3(p *int) bool {
	// amd64:`CMPQ\t\(.*\), [$]7`
	return *p < 7
}

func CmpMem4(p *int) bool {
	// amd64:`CMPQ\t\(.*\), [$]7`
	return 7 < *p
}

func CmpMem5(p **int) {
	// amd64:`CMPL\truntime.writeBarrier\(SB\), [$]0`
	*p = nil
}

func CmpMem6(a []int) int {
	// 386:`CMPL\s8\([A-Z]+\),`
	// amd64:`CMPQ\s16\([A-Z]+\),`
	if a[1] > a[2] {
		return 1
	} else {
		return 2
	}
}

// Check tbz/tbnz are generated when comparing against zero on arm64

func CmpZero1(a int32, ptr *int) {
	if a < 0 { // arm64:"TBZ"
		*ptr = 0
	}
}

func CmpZero2(a int64, ptr *int) {
	if a < 0 { // arm64:"TBZ"
		*ptr = 0
	}
}

func CmpZero3(a int32, ptr *int) {
	if a >= 0 { // arm64:"TBNZ"
		*ptr = 0
	}
}

func CmpZero4(a int64, ptr *int) {
	if a >= 0 { // arm64:"TBNZ"
		*ptr = 0
	}
}

func CmpToZero(a, b, d int32, e, f int64) int32 {
	// arm:`TST`,-`AND`
	// arm64:`TSTW`,-`AND`
	// 386:`TESTL`,-`ANDL`
	// amd64:`TESTL`,-`ANDL`
	c0 := a&b < 0
	// arm:`CMN`,-`ADD`
	// arm64:`CMNW`,-`ADD`
	c1 := a+b < 0
	// arm:`TEQ`,-`XOR`
	c2 := a^b < 0
	// arm64:`TST`,-`AND`
	// amd64:`TESTQ`,-`ANDQ`
	c3 := e&f < 0
	// arm64:`CMN`,-`ADD`
	c4 := e+f < 0
	// not optimized to single CMNW/CMN due to further use of b+d
	// arm64:`ADD`,-`CMNW`
	// arm:`ADD`,-`CMN`
	c5 := b+d == 0
	// not optimized to single TSTW/TST due to further use of a&d
	// arm64:`AND`,-`TSTW`
	// arm:`AND`,-`TST`
	// 386:`ANDL`
	c6 := a&d >= 0
	// arm64:`TST\sR[0-9]+<<3,\sR[0-9]+`
	c7 := e&(f<<3) < 0
	// arm64:`CMN\sR[0-9]+<<3,\sR[0-9]+`
	c8 := e+(f<<3) < 0
	if c0 {
		return 1
	} else if c1 {
		return 2
	} else if c2 {
		return 3
	} else if c3 {
		return 4
	} else if c4 {
		return 5
	} else if c5 {
		return b + d
	} else if c6 {
		return a & d
	} else if c7 {
		return 7
	} else if c8 {
		return 8
	} else {
		return 0
	}
}

func CmpLogicalToZero(a, b, c uint32, d, e uint64) uint64 {

	// ppc64:"ANDCC",-"CMPW"
	// ppc64le:"ANDCC",-"CMPW"
	// wasm:"I64Eqz",-"I32Eqz",-"I64ExtendI32U",-"I32WrapI64"
	if a&63 == 0 {
		return 1
	}

	// ppc64:"ANDCC",-"CMP"
	// ppc64le:"ANDCC",-"CMP"
	// wasm:"I64Eqz",-"I32Eqz",-"I64ExtendI32U",-"I32WrapI64"
	if d&255 == 0 {
		return 1
	}

	// ppc64:"ANDCC",-"CMP"
	// ppc64le:"ANDCC",-"CMP"
	// wasm:"I64Eqz",-"I32Eqz",-"I64ExtendI32U",-"I32WrapI64"
	if d&e == 0 {
		return 1
	}
	// ppc64:"ORCC",-"CMP"
	// ppc64le:"ORCC",-"CMP"
	// wasm:"I64Eqz",-"I32Eqz",-"I64ExtendI32U",-"I32WrapI64"
	if d|e == 0 {
		return 1
	}

	// ppc64:"XORCC",-"CMP"
	// ppc64le:"XORCC",-"CMP"
	// wasm:"I64Eqz","I32Eqz",-"I64ExtendI32U",-"I32WrapI64"
	if e^d == 0 {
		return 1
	}
	return 0
}

// The following CmpToZero_ex* check that cmp|cmn with bmi|bpl are generated for
// 'comparing to zero' expressions

// var + const
// 'x-const' might be canonicalized to 'x+(-const)', so we check both
// CMN and CMP for subtraction expressions to make the pattern robust.
func CmpToZero_ex1(a int64, e int32) int {
	// arm64:`CMN`,-`ADD`,`(BMI|BPL)`
	if a+3 < 0 {
		return 1
	}

	// arm64:`CMN`,-`ADD`,`BEQ`,`(BMI|BPL)`
	if a+5 <= 0 {
		return 1
	}

	// arm64:`CMN`,-`ADD`,`(BMI|BPL)`
	if a+13 >= 0 {
		return 2
	}

	// arm64:`CMP|CMN`,-`(ADD|SUB)`,`(BMI|BPL)`
	if a-7 < 0 {
		return 3
	}

	// arm64:`CMP|CMN`,-`(ADD|SUB)`,`(BMI|BPL)`
	if a-11 >= 0 {
		return 4
	}

	// arm64:`CMP|CMN`,-`(ADD|SUB)`,`BEQ`,`(BMI|BPL)`
	if a-19 > 0 {
		return 4
	}

	// arm64:`CMNW`,-`ADDW`,`(BMI|BPL)`
	// arm:`CMN`,-`ADD`,`(BMI|BPL)`
	if e+3 < 0 {
		return 5
	}

	// arm64:`CMNW`,-`ADDW`,`(BMI|BPL)`
	// arm:`CMN`,-`ADD`,`(BMI|BPL)`
	if e+13 >= 0 {
		return 6
	}

	// arm64:`CMPW|CMNW`,`(BMI|BPL)`
	// arm:`CMP|CMN`, -`(ADD|SUB)`, `(BMI|BPL)`
	if e-7 < 0 {
		return 7
	}

	// arm64:`CMPW|CMNW`,`(BMI|BPL)`
	// arm:`CMP|CMN`, -`(ADD|SUB)`, `(BMI|BPL)`
	if e-11 >= 0 {
		return 8
	}

	return 0
}

// var + var
// TODO: optimize 'var - var'
func CmpToZero_ex2(a, b, c int64, e, f, g int32) int {
	// arm64:`CMN`,-`ADD`,`(BMI|BPL)`
	if a+b < 0 {
		return 1
	}

	// arm64:`CMN`,-`ADD`,`BEQ`,`(BMI|BPL)`
	if a+c <= 0 {
		return 1
	}

	// arm64:`CMN`,-`ADD`,`(BMI|BPL)`
	if b+c >= 0 {
		return 2
	}

	// arm64:`CMNW`,-`ADDW`,`(BMI|BPL)`
	// arm:`CMN`,-`ADD`,`(BMI|BPL)`
	if e+f < 0 {
		return 5
	}

	// arm64:`CMNW`,-`ADDW`,`(BMI|BPL)`
	// arm:`CMN`,-`ADD`,`(BMI|BPL)`
	if f+g >= 0 {
		return 6
	}
	return 0
}

// var + var*var
func CmpToZero_ex3(a, b, c, d int64, e, f, g, h int32) int {
	// arm64:`CMN`,-`MADD`,`MUL`,`(BMI|BPL)`
	if a+b*c < 0 {
		return 1
	}

	// arm64:`CMN`,-`MADD`,`MUL`,`(BMI|BPL)`
	if b+c*d >= 0 {
		return 2
	}

	// arm64:`CMNW`,-`MADDW`,`MULW`,`BEQ`,`(BMI|BPL)`
	// arm:`CMN`,-`MULA`,`MUL`,`BEQ`,`(BMI|BPL)`
	if e+f*g > 0 {
		return 5
	}

	// arm64:`CMNW`,-`MADDW`,`MULW`,`BEQ`,`(BMI|BPL)`
	// arm:`CMN`,-`MULA`,`MUL`,`BEQ`,`(BMI|BPL)`
	if f+g*h <= 0 {
		return 6
	}
	return 0
}

// var - var*var
func CmpToZero_ex4(a, b, c, d int64, e, f, g, h int32) int {
	// arm64:`CMP`,-`MSUB`,`MUL`,`BEQ`,`(BMI|BPL)`
	if a-b*c > 0 {
		return 1
	}

	// arm64:`CMP`,-`MSUB`,`MUL`,`(BMI|BPL)`
	if b-c*d >= 0 {
		return 2
	}

	// arm64:`CMPW`,-`MSUBW`,`MULW`,`(BMI|BPL)`
	if e-f*g < 0 {
		return 5
	}

	// arm64:`CMPW`,-`MSUBW`,`MULW`,`(BMI|BPL)`
	if f-g*h >= 0 {
		return 6
	}
	return 0
}

func CmpToZero_ex5(e, f int32, u uint32) int {
	// arm:`CMN`,-`ADD`,`BEQ`,`(BMI|BPL)`
	if e+f<<1 > 0 {
		return 1
	}

	// arm:`CMP`,-`SUB`,`(BMI|BPL)`
	if f-int32(u>>2) >= 0 {
		return 2
	}
	return 0
}
func UintLtZero(a uint8, b uint16, c uint32, d uint64) int {
	// amd64: -`(TESTB|TESTW|TESTL|TESTQ|JCC|JCS)`
	// arm64: -`(CMPW|CMP|BHS|BLO)`
	if a < 0 || b < 0 || c < 0 || d < 0 {
		return 1
	}
	return 0
}

func UintGeqZero(a uint8, b uint16, c uint32, d uint64) int {
	// amd64: -`(TESTB|TESTW|TESTL|TESTQ|JCS|JCC)`
	// arm64: -`(CMPW|CMP|BLO|BHS)`
	if a >= 0 || b >= 0 || c >= 0 || d >= 0 {
		return 1
	}
	return 0
}

func UintGtZero(a uint8, b uint16, c uint32, d uint64) int {
	// arm64: `(CBN?ZW)`, `(CBN?Z[^W])`, -`(CMPW|CMP|BLS|BHI)`
	if a > 0 || b > 0 || c > 0 || d > 0 {
		return 1
	}
	return 0
}

func UintLeqZero(a uint8, b uint16, c uint32, d uint64) int {
	// arm64: `(CBN?ZW)`, `(CBN?Z[^W])`, -`(CMPW|CMP|BHI|BLS)`
	if a <= 0 || b <= 0 || c <= 0 || d <= 0 {
		return 1
	}
	return 0
}

func UintLtOne(a uint8, b uint16, c uint32, d uint64) int {
	// arm64: `(CBN?ZW)`, `(CBN?Z[^W])`, -`(CMPW|CMP|BHS|BLO)`
	if a < 1 || b < 1 || c < 1 || d < 1 {
		return 1
	}
	return 0
}

func UintGeqOne(a uint8, b uint16, c uint32, d uint64) int {
	// arm64: `(CBN?ZW)`, `(CBN?Z[^W])`, -`(CMPW|CMP|BLO|BHS)`
	if a >= 1 || b >= 1 || c >= 1 || d >= 1 {
		return 1
	}
	return 0
}

func CmpToZeroU_ex1(a uint8, b uint16, c uint32, d uint64) int {
	// wasm:"I64Eqz"-"I64LtU"
	if 0 < a {
		return 1
	}
	// wasm:"I64Eqz"-"I64LtU"
	if 0 < b {
		return 1
	}
	// wasm:"I64Eqz"-"I64LtU"
	if 0 < c {
		return 1
	}
	// wasm:"I64Eqz"-"I64LtU"
	if 0 < d {
		return 1
	}
	return 0
}

func CmpToZeroU_ex2(a uint8, b uint16, c uint32, d uint64) int {
	// wasm:"I64Eqz"-"I64LeU"
	if a <= 0 {
		return 1
	}
	// wasm:"I64Eqz"-"I64LeU"
	if b <= 0 {
		return 1
	}
	// wasm:"I64Eqz"-"I64LeU"
	if c <= 0 {
		return 1
	}
	// wasm:"I64Eqz"-"I64LeU"
	if d <= 0 {
		return 1
	}
	return 0
}

func CmpToOneU_ex1(a uint8, b uint16, c uint32, d uint64) int {
	// wasm:"I64Eqz"-"I64LtU"
	if a < 1 {
		return 1
	}
	// wasm:"I64Eqz"-"I64LtU"
	if b < 1 {
		return 1
	}
	// wasm:"I64Eqz"-"I64LtU"
	if c < 1 {
		return 1
	}
	// wasm:"I64Eqz"-"I64LtU"
	if d < 1 {
		return 1
	}
	return 0
}

func CmpToOneU_ex2(a uint8, b uint16, c uint32, d uint64) int {
	// wasm:"I64Eqz"-"I64LeU"
	if 1 <= a {
		return 1
	}
	// wasm:"I64Eqz"-"I64LeU"
	if 1 <= b {
		return 1
	}
	// wasm:"I64Eqz"-"I64LeU"
	if 1 <= c {
		return 1
	}
	// wasm:"I64Eqz"-"I64LeU"
	if 1 <= d {
		return 1
	}
	return 0
}

// Check that small memequals are replaced with eq instructions

func equalConstString1() bool {
	a := string("A")
	b := string("Z")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a == b
}

func equalVarString1(a string) bool {
	b := string("Z")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a[:1] == b
}

func equalConstString2() bool {
	a := string("AA")
	b := string("ZZ")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a == b
}

func equalVarString2(a string) bool {
	b := string("ZZ")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a[:2] == b
}

func equalConstString4() bool {
	a := string("AAAA")
	b := string("ZZZZ")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a == b
}

func equalVarString4(a string) bool {
	b := string("ZZZZ")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a[:4] == b
}

func equalConstString8() bool {
	a := string("AAAAAAAA")
	b := string("ZZZZZZZZ")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a == b
}

func equalVarString8(a string) bool {
	b := string("ZZZZZZZZ")
	// amd64:-".*memequal"
	// arm64:-".*memequal"
	// ppc64:-".*memequal"
	// ppc64le:-".*memequal"
	return a[:8] == b
}
