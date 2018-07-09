// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	a_base+0(FP), R2
	MOVW	a_len+4(FP), R0
	MOVW	b_base+12(FP), R3
	MOVW	b_len+16(FP), R1
	ADD	$28, R13, R7
	B	cmpbody<>(SB)

TEXT bytes·Compare(SB),NOSPLIT|NOFRAME,$0-28
	FUNCDATA $0, ·Compare·args_stackmap(SB)
	MOVW	a_base+0(FP), R2
	MOVW	a_len+4(FP), R0
	MOVW	b_base+12(FP), R3
	MOVW	b_len+16(FP), R1
	ADD	$28, R13, R7
	B	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT|NOFRAME,$0-20
	MOVW	a_base+0(FP), R2
	MOVW	a_len+4(FP), R0
	MOVW	b_base+8(FP), R3
	MOVW	b_len+12(FP), R1
	ADD	$20, R13, R7
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
	BEQ	samebytes
	CMP 	R0, R1
	MOVW 	R0, R6
	MOVW.LT	R1, R6	// R6 is min(R0, R1)

	ADD	R2, R6	// R2 is current byte in a, R6 is last byte in a to compare
loop:
	CMP	R2, R6
	BEQ	samebytes // all compared bytes were the same; compare lengths
	MOVBU.P	1(R2), R4
	MOVBU.P	1(R3), R5
	CMP	R4, R5
	BEQ	loop
	// bytes differed
	MOVW.LT	$1, R0
	MOVW.GT	$-1, R0
	MOVW	R0, (R7)
	RET
samebytes:
	CMP	R0, R1
	MOVW.LT	$1, R0
	MOVW.GT	$-1, R0
	MOVW.EQ	$0, R0
	MOVW	R0, (R7)
	RET
