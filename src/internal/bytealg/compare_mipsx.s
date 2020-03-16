// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT,$0-28
	MOVW	a_base+0(FP), R3
	MOVW	b_base+12(FP), R4
	MOVW	a_len+4(FP), R1
	MOVW	b_len+16(FP), R2
	BEQ	R3, R4, samebytes
	SGTU	R1, R2, R7
	MOVW	R1, R8
	CMOVN	R7, R2, R8	// R8 is min(R1, R2)

	ADDU	R3, R8	// R3 is current byte in a, R8 is last byte in a to compare
loop:
	BEQ	R3, R8, samebytes

	MOVBU	(R3), R6
	ADDU	$1, R3
	MOVBU	(R4), R7
	ADDU	$1, R4
	BEQ	R6, R7 , loop

	SGTU	R6, R7, R8
	MOVW	$-1, R6
	CMOVZ	R8, R6, R8
	JMP	cmp_ret
samebytes:
	SGTU	R1, R2, R6
	SGTU	R2, R1, R7
	SUBU	R7, R6, R8
cmp_ret:
	MOVW	R8, ret+24(FP)
	RET

TEXT runtime·cmpstring(SB),NOSPLIT,$0-20
	MOVW	a_base+0(FP), R3
	MOVW	a_len+4(FP), R1
	MOVW	b_base+8(FP), R4
	MOVW	b_len+12(FP), R2
	BEQ	R3, R4, samebytes
	SGTU	R1, R2, R7
	MOVW	R1, R8
	CMOVN	R7, R2, R8	// R8 is min(R1, R2)

	ADDU	R3, R8	// R3 is current byte in a, R8 is last byte in a to compare
loop:
	BEQ	R3, R8, samebytes	// all compared bytes were the same; compare lengths

	MOVBU	(R3), R6
	ADDU	$1, R3
	MOVBU	(R4), R7
	ADDU	$1, R4
	BEQ	R6, R7 , loop
	// bytes differed
	SGTU	R6, R7, R8
	MOVW	$-1, R6
	CMOVZ	R8, R6, R8
	JMP	cmp_ret
samebytes:
	SGTU	R1, R2, R6
	SGTU	R2, R1, R7
	SUBU	R7, R6, R8
cmp_ret:
	MOVW	R8, ret+16(FP)
	RET
