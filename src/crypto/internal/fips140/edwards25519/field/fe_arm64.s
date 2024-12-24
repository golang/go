// Copyright (c) 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// carryPropagate works exactly like carryPropagateGeneric and uses the
// same AND, ADD, and LSR+MADD instructions emitted by the compiler, but
// avoids loading R0-R4 twice and uses LDP and STP.
//
// See https://golang.org/issues/43145 for the main compiler issue.
//
// func carryPropagate(v *Element)
TEXT ·carryPropagate(SB),NOFRAME|NOSPLIT,$0-8
	MOVD v+0(FP), R20

	LDP 0(R20), (R0, R1)
	LDP 16(R20), (R2, R3)
	MOVD 32(R20), R4

	AND $0x7ffffffffffff, R0, R10
	AND $0x7ffffffffffff, R1, R11
	AND $0x7ffffffffffff, R2, R12
	AND $0x7ffffffffffff, R3, R13
	AND $0x7ffffffffffff, R4, R14

	ADD R0>>51, R11, R11
	ADD R1>>51, R12, R12
	ADD R2>>51, R13, R13
	ADD R3>>51, R14, R14
	// R4>>51 * 19 + R10 -> R10
	LSR $51, R4, R21
	MOVD $19, R22
	MADD R22, R10, R21, R10

	STP (R10, R11), 0(R20)
	STP (R12, R13), 16(R20)
	MOVD R14, 32(R20)

	RET
