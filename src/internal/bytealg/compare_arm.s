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
// R4, R5, R6 and R8 are clobbered
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMP	R2, R3
	BEQ	samebytes
	CMP 	R0, R1
	MOVW 	R0, R6
	MOVW.LT	R1, R6		// R6 is min(R0, R1)

	CMP	$0, R6
	BEQ	samebytes
	CMP	$4, R6
	ADD	R2, R6		// R2 is current byte in a, R6 is the end of the range to compare
	BLT	byte_loop	// length < 4
	AND	$3, R2, R8
	CMP	$0, R8
	BNE	byte_loop	// unaligned a, use byte-wise compare (TODO: try to align a)
aligned_a:
	AND	$3, R3, R8
	CMP	$0, R8
	BNE	byte_loop	// unaligned b, use byte-wise compare
	AND	$0xfffffffc, R6, R8
	// length >= 4
chunk4_loop:
	MOVW.P	4(R2), R4
	MOVW.P	4(R3), R5
	CMP	R4, R5
	BNE	cmp
	CMP	R2, R8
	BNE	chunk4_loop
	CMP	R2, R6
	BEQ	samebytes	// all compared bytes were the same; compare lengths
byte_loop:
	MOVBU.P	1(R2), R4
	MOVBU.P	1(R3), R5
	CMP	R4, R5
	BNE	ret
	CMP	R2, R6
	BNE	byte_loop
samebytes:
	CMP	R0, R1
	MOVW.LT	$1, R0
	MOVW.GT	$-1, R0
	MOVW.EQ	$0, R0
	MOVW	R0, (R7)
	RET
ret:
	// bytes differed
	MOVW.LT	$1, R0
	MOVW.GT	$-1, R0
	MOVW	R0, (R7)
	RET
cmp:
	SUB	$4, R2, R2
	SUB	$4, R3, R3
	B	byte_loop
