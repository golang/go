// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT|NOFRAME,$0-56
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R0
	MOVD	b_base+24(FP), R3
	MOVD	b_len+32(FP), R1
	ADD	$56, RSP, R7
	B	cmpbody<>(SB)

TEXT bytes·Compare(SB),NOSPLIT|NOFRAME,$0-56
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R0
	MOVD	b_base+24(FP), R3
	MOVD	b_len+32(FP), R1
	ADD	$56, RSP, R7
	B	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT|NOFRAME,$0-40
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R0
	MOVD	b_base+16(FP), R3
	MOVD	b_len+24(FP), R1
	ADD	$40, RSP, R7
	B	cmpbody<>(SB)

// On entry:
// R0 is the length of a
// R1 is the length of b
// R2 points to the start of a
// R3 points to the start of b
// R7 points to return value (-1/0/1 will be written here)
//
// On exit:
// R4, R5, and R6 are clobbered
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMP	R2, R3
	BEQ	samebytes // same starting pointers; compare lengths
	CMP	R0, R1
	CSEL    LT, R1, R0, R6 // R6 is min(R0, R1)

	ADD	R2, R6	// R2 is current byte in a, R6 is last byte in a to compare
loop:
	CMP	R2, R6
	BEQ	samebytes // all compared bytes were the same; compare lengths
	MOVBU.P	1(R2), R4
	MOVBU.P	1(R3), R5
	CMP	R4, R5
	BEQ	loop
	// bytes differed
	MOVD	$1, R4
	CSNEG	LT, R4, R4, R4
	MOVD	R4, (R7)
	RET
samebytes:
	MOVD	$1, R4
	CMP	R0, R1
	CSNEG	LT, R4, R4, R4
	CSEL	EQ, ZR, R4, R4
	MOVD	R4, (R7)
	RET
