// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT,$0-56
	// R4 = a_base
	// R5 = a_len
	// R6 = a_cap (unused)
	// R7 = b_base (want in R6)
	// R8 = b_len (want in R7)
	// R9 = b_cap (unused)
	MOVV	R7, R6
	MOVV	R8, R7
	JMP	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT,$0-40
	// R4 = a_base
	// R5 = a_len
	// R6 = b_base
	// R7 = b_len
	JMP	cmpbody<>(SB)

// On entry:
// R5 length of a
// R7 length of b
// R4 points to the start of a
// R6 points to the start of b
// R13 points to the return value (-1/0/1)
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	R4, R6, samebytes // same start of a and b

	SGTU	R5, R7, R9
	BNE	R0, R9, r2_lt_r1
	MOVV	R5, R14
	JMP	entry
r2_lt_r1:
	MOVV	R7, R14	// R14 is min(R4, R5)
entry:
	ADDV	R4, R14, R12	// R6 start of a, R14 end of a
	BEQ	R4, R12, samebytes // length is 0

	SRLV	$4, R14		// R14 is number of chunks
	BEQ	R0, R14, byte_loop

	// make sure both a and b are aligned.
	OR	R4, R6, R15
	AND	$7, R15
	BNE	R0, R15, byte_loop

	PCALIGN	$16
chunk16_loop:
	BEQ	R0, R14, byte_loop
	MOVV	(R4), R8
	MOVV	(R6), R9
	BNE	R8, R9, byte_loop
	MOVV	8(R4), R16
	MOVV	8(R6), R17
	ADDV	$16, R4
	ADDV	$16, R6
	SUBVU	$1, R14
	BEQ	R16, R17, chunk16_loop
	SUBV	$8, R4
	SUBV	$8, R6

byte_loop:
	BEQ	R4, R12, samebytes
	MOVBU	(R4), R8
	ADDVU	$1, R4
	MOVBU	(R6), R9
	ADDVU	$1, R6
	BEQ	R8, R9, byte_loop

byte_cmp:
	SGTU	R8, R9, R4 // R12 = 1 if (R8 > R9)
	BNE	R0, R4, ret
	MOVV	$-1, R4
	JMP	ret

samebytes:
	SGTU	R5, R7, R8
	SGTU	R7, R5, R9
	SUBV	R9, R8, R4

ret:
	RET
