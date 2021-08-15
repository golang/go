// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT,$0-56
	MOVV	a_base+0(FP), R6
	MOVV	b_base+24(FP), R7
	MOVV	a_len+8(FP), R4
	MOVV	b_len+32(FP), R5
	MOVV	$ret+48(FP), R13
	JMP	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT,$0-40
	MOVV	a_base+0(FP), R6
	MOVV	b_base+16(FP), R7
	MOVV	a_len+8(FP), R4
	MOVV	b_len+24(FP), R5
	MOVV	$ret+32(FP), R13
	JMP	cmpbody<>(SB)

// On entry:
// R4 length of a
// R5 length of b
// R6 points to the start of a
// R7 points to the start of b
// R13 points to the return value (-1/0/1)
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	R6, R7, samebytes // same start of a and b

	SGTU	R4, R5, R9
	BNE	R0, R9, r2_lt_r1
	MOVV	R4, R14
	JMP	entry
r2_lt_r1:
	MOVV	R5, R14	// R14 is min(R4, R5)
entry:
	ADDV	R6, R14, R12	// R6 start of a, R14 end of a
	BEQ	R6, R12, samebytes // length is 0

	SRLV	$4, R14		// R14 is number of chunks
	BEQ	R0, R14, byte_loop

	// make sure both a and b are aligned.
	OR	R6, R7, R15
	AND	$7, R15
	BNE	R0, R15, byte_loop

chunk16_loop:
	BEQ	R0, R14, byte_loop
	MOVV	(R6), R8
	MOVV	(R7), R9
	BNE	R8, R9, byte_loop
	MOVV	8(R6), R16
	MOVV	8(R7), R17
	ADDV	$16, R6
	ADDV	$16, R7
	SUBVU	$1, R14
	BEQ	R16, R17, chunk16_loop
	SUBV	$8, R6
	SUBV	$8, R7

byte_loop:
	BEQ	R6, R12, samebytes
	MOVBU	(R6), R8
	ADDVU	$1, R6
	MOVBU	(R7), R9
	ADDVU	$1, R7
	BEQ	R8, R9, byte_loop

byte_cmp:
	SGTU	R8, R9, R12 // R12 = 1 if (R8 > R9)
	BNE	R0, R12, ret
	MOVV	$-1, R12
	JMP	ret

samebytes:
	SGTU	R4, R5, R8
	SGTU	R5, R4, R9
	SUBV	R9, R8, R12

ret:
	MOVV	R12, (R13)
	RET
