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

func CmpFold(x uint32) bool {
	// amd64:`SETHI\t.*\(SP\)`
	return x > 4
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
