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
// for regabi the return value (-1/0/1) in R4
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0
	BEQ	R4, R6, cmp_len		// same start of a and b, then compare lengths

	SGTU	R5, R7, R9
	BNE	R9, b_lt_a
	MOVV	R5, R14
	JMP	entry
b_lt_a:
	MOVV	R7, R14			// R14 is min(R5, R7)
entry:
	ADDV	R4, R14, R12		// R4 start of a, R12 end of a
	BEQ	R4, R12, cmp_len	// minlength is 0

tail:
	MOVV	$2, R15
	BLT	R14, R15, cmp1		// min < 2
	SLLV	$1, R15
	BLT	R14, R15, cmp2		// min < 4
	SLLV	$1, R15
	BLT	R14, R15, cmp4		// min < 8
	SLLV	$1, R15
	BLT	R14, R15, cmp8		// min < 16
	SLLV	$1, R15
	BLT	R14, R15, cmp16		// min < 32

// When min >= 32 bytes, enter the cmp32_loop loop processing:
// take out 4 8-bytes from a and b in turn for comparison.
cmp32_loop:
	MOVV	(R4), R8
	MOVV	(R6), R9
	MOVV	8(R4), R10
	MOVV	8(R6), R11
	BNE	R8, R9, cmp8a
	BNE	R10, R11, cmp8b
	MOVV	16(R4), R8
	MOVV	16(R6), R9
	MOVV	24(R4), R10
	MOVV	24(R6), R11
	BNE	R8, R9, cmp8a
	BNE	R10, R11, cmp8b
	ADDV	$32, R4
	ADDV	$32, R6
	SUBV	$32, R14
	BGE	R14, R15, cmp32_loop
	BEQ	R14, cmp_len

check16:
	MOVV	$16, R15
	BLT	R14, R15, check8
cmp16:
	MOVV	(R4), R8
	MOVV	(R6), R9
	MOVV	8(R4), R10
	MOVV	8(R6), R11
	BNE	R8, R9, cmp8a
	BNE	R10, R11, cmp8b
	ADDV	$16, R4
	ADDV	$16, R6
	SUBV	$16, R14
	BEQ	R14, cmp_len

check8:
	MOVV	$8, R15
	BLT	R14, R15, check4
cmp8:
	MOVV	(R4), R8
	MOVV	(R6), R9
	BNE	R8, R9, cmp8a
	ADDV	$8, R4
	ADDV	$8, R6
	SUBV	$8, R14
	BEQ	R14, cmp_len

check4:
	MOVV	$4, R15
	BLT	R14, R15, check2
cmp4:
	MOVW	(R4), R8
	MOVW	(R6), R9
	BNE	R8, R9, cmp8a
	ADDV	$4, R4
	ADDV	$4, R6
	SUBV	$4, R14
	BEQ	R14, cmp_len

check2:
	MOVV	$2, R15
	BLT	R14, R15, cmp1
cmp2:
	MOVH	(R4), R8
	MOVH	(R6), R9
	BNE	R8, R9, cmp8a
	ADDV	$2, R4
	ADDV	$2, R6
	SUBV	$2, R14
	BEQ	R14, cmp_len

cmp1:
	BEQ	R14, cmp_len
	MOVBU	(R4), R8
	MOVBU	(R6), R9
	BNE	R8, R9, byte_cmp
	JMP	cmp_len

	// Compare 8/4/2 bytes taken from R8/R9 that are known to differ.
cmp8a:
	MOVV	R8, R10
	MOVV	R9, R11

	// Compare 8/4/2 bytes taken from R10/R11 that are known to differ.
cmp8b:
	MOVV	$0xff, R15

	// Take single bytes from R10/R11 in turn for cyclic comparison.
cmp8_loop:
	AND	R10, R15, R8
	AND	R11, R15, R9
	BNE	R8, R9, byte_cmp
	SLLV	$8, R15
	JMP	cmp8_loop

	// Compare 1 bytes taken from R8/R9 that are known to differ.
byte_cmp:
	SGTU	R8, R9, R4		// R4 = 1 if (R8 > R9)
	BNE	R0, R4, ret
	MOVV	$-1, R4
	JMP	ret

cmp_len:
	SGTU	R5, R7, R8
	SGTU	R7, R5, R9
	SUBV	R9, R8, R4

ret:
	RET
