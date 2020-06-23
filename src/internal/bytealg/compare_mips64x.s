// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT,$0-56
	MOVV	a_base+0(FP), R3
	MOVV	b_base+24(FP), R4
	MOVV	a_len+8(FP), R1
	MOVV	b_len+32(FP), R2
	MOVV	$ret+48(FP), R9
	JMP	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT,$0-40
	MOVV	a_base+0(FP), R3
	MOVV	b_base+16(FP), R4
	MOVV	a_len+8(FP), R1
	MOVV	b_len+24(FP), R2
	MOVV	$ret+32(FP), R9
	JMP	cmpbody<>(SB)

// On entry:
// R1 length of a
// R2 length of b
// R3 points to the start of a
// R4 points to the start of b
// R9 points to the return value (-1/0/1)
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	R3, R4, samebytes // same start of a and b

	SGTU	R1, R2, R7
	BNE	R0, R7, r2_lt_r1
	MOVV	R1, R10
	JMP	entry
r2_lt_r1:
	MOVV	R2, R10	// R10 is min(R1, R2)
entry:
	ADDV	R3, R10, R8	// R3 start of a, R8 end of a
	BEQ	R3, R8, samebytes // length is 0

	SRLV	$4, R10		// R10 is number of chunks
	BEQ	R0, R10, byte_loop

	// make sure both a and b are aligned.
	OR	R3, R4, R11
	AND	$7, R11
	BNE	R0, R11, byte_loop

chunk16_loop:
	BEQ	R0, R10, byte_loop
	MOVV	(R3), R6
	MOVV	(R4), R7
	BNE	R6, R7, byte_loop
	MOVV	8(R3), R13
	MOVV	8(R4), R14
	ADDV	$16, R3
	ADDV	$16, R4
	SUBVU	$1, R10
	BEQ	R13, R14, chunk16_loop
	SUBV	$8, R3
	SUBV	$8, R4

byte_loop:
	BEQ	R3, R8, samebytes
	MOVBU	(R3), R6
	ADDVU	$1, R3
	MOVBU	(R4), R7
	ADDVU	$1, R4
	BEQ	R6, R7, byte_loop

byte_cmp:
	SGTU	R6, R7, R8 // R8 = 1 if (R6 > R7)
	BNE	R0, R8, ret
	MOVV	$-1, R8
	JMP	ret

samebytes:
	SGTU	R1, R2, R6
	SGTU	R2, R1, R7
	SUBV	R7, R6, R8

ret:
	MOVV	R8, (R9)
	RET
