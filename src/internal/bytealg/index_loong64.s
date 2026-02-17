// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Index<ABIInternal>(SB),NOSPLIT,$0-56
	MOVV	R7, R6		// R6 = separator pointer
	MOVV	R8, R7		// R7 = separator length
	JMP	indexbody<>(SB)

TEXT ·IndexString<ABIInternal>(SB),NOSPLIT,$0-40
	JMP	indexbody<>(SB)

// input:
//   R4 = string
//   R5 = length
//   R6 = separator pointer
//   R7 = separator length (2 <= len <= 64)
TEXT indexbody<>(SB),NOSPLIT,$0
	// main idea is to load 'sep' into separate register(s)
	// to avoid repeatedly re-load it again and again
	// for sebsequent substring comparisons
	SUBV	R7, R5, R8
	ADDV	R4, R8		// R8 contains the start of last substring for comparison
	ADDV	$1, R4, R9	// store base for later

	MOVV	$8, R5
	BGE	R7, R5, len_gt_or_eq_8
len_2_7:
	AND	$0x4, R7, R5
	BNE	R5, len_4_7

len_2_3:
	AND	$0x1, R7, R5
	BNE	R5, len_3

len_2:
	MOVHU	(R6), R10
loop_2:
	BLT	R8, R4, not_found
	MOVHU	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, loop_2
	JMP	found

len_3:
	MOVHU	(R6), R10
	MOVBU	2(R6), R11
loop_3:
	BLT	R8, R4, not_found
	MOVHU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_3
	MOVBU	1(R4), R13
	BNE	R11, R13, loop_3
	JMP	found

len_4_7:
	AND	$0x2, R7, R5
	BNE	R5, len_6_7
	AND	$0x1, R7, R5
	BNE	R5, len_5
len_4:
	MOVWU	(R6), R10
loop_4:
	BLT	R8, R4, not_found
	MOVWU	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, loop_4
	JMP	found

len_5:
	MOVWU	(R6), R10
	MOVBU	4(R6), R11
loop_5:
	BLT	R8, R4, not_found
	MOVWU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_5
	MOVBU	3(R4), R13
	BNE	R11, R13, loop_5
	JMP	found

len_6_7:
	AND	$0x1, R7, R5
	BNE	R5, len_7
len_6:
	MOVWU	(R6), R10
	MOVHU	4(R6), R11
loop_6:
	BLT	R8, R4, not_found
	MOVWU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_6
	MOVHU	3(R4), R13
	BNE	R11, R13, loop_6
	JMP	found

len_7:
	MOVWU	(R6), R10
	MOVWU	3(R6), R11
loop_7:
	BLT	R8, R4, not_found
	MOVWU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_7
	MOVWU	2(R4), R13
	BNE	R11, R13, loop_7
	JMP	found

len_gt_or_eq_8:
	BEQ	R5, R7, len_8
	MOVV	$17, R5
	BGE	R7, R5, len_gt_or_eq_17
	JMP	len_9_16
len_8:
	MOVV	(R6), R10
loop_8:
	BLT	R8, R4, not_found
	MOVV	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, loop_8
	JMP	found

len_9_16:
	MOVV	(R6), R10
	SUBV	$8, R7
	MOVV	(R6)(R7), R11
	SUBV	$1, R7
loop_9_16:
	BLT	R8, R4, not_found
	MOVV	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_9_16
	MOVV	(R4)(R7), R13
	BNE	R11, R13, loop_9_16
	JMP	found

len_gt_or_eq_17:
	MOVV	$25, R5
	BGE	R7, R5, len_gt_or_eq_25
len_17_24:
	MOVV	0(R6), R10
	MOVV	8(R6), R11
	SUBV	$8, R7
	MOVV	(R6)(R7), R12
	SUBV	$1, R7
loop_17_24:
	BLT	R8, R4, not_found
	MOVV	(R4), R13
	ADDV	$1, R4
	BNE	R10, R13, loop_17_24
	MOVV	7(R4), R14
	BNE	R11, R14, loop_17_24
	MOVV	(R4)(R7), R15
	BNE	R12, R15, loop_17_24
	JMP	found

len_gt_or_eq_25:
	MOVV	$33, R5
	BGE	R7, R5, len_gt_or_eq_33
	MOVBU   internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R10
	BNE	R10, lsx_len_25_32
len_25_32:
	MOVV	0(R6), R10
	MOVV	8(R6), R11
	MOVV	16(R6), R12
	SUBV	$8, R7
	MOVV	(R6)(R7), R13
	SUBV	$1, R7
loop_25_32:
	BLT	R8, R4, not_found
	MOVV	(R4), R14
	ADDV	$1, R4
	BNE	R10, R14, loop_25_32
	MOVV	7(R4), R15
	BNE	R11, R15, loop_25_32
	MOVV	15(R4), R16
	BNE	R12, R16, loop_25_32
	MOVV	(R4)(R7), R17
	BNE	R13, R17, loop_25_32
	JMP	found

	// On loong64, LSX is included if LASX is supported.
lasx_len_25_32:
lsx_len_25_32:
	VMOVQ	0(R6), V0
	SUBV	$16, R7
	VMOVQ	(R6)(R7), V1
	SUBV	$1, R7
lsx_loop_25_32:
	BLT	R8, R4, not_found
	VMOVQ	(R4), V2
	ADDV	$1, R4
	VSEQV	V0, V2, V2
	VSETANYEQV	V2, FCC0
	BFPT	FCC0, lsx_loop_25_32

	VMOVQ	(R4)(R7), V3
	VSEQV	V1, V3, V3
	VSETANYEQV	V3, FCC1
	BFPT	FCC1, lsx_loop_25_32
	JMP	found

len_gt_or_eq_33:
	MOVBU   internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R10
	MOVV	$49, R5
	BGE	R7, R5, len_gt_or_eq_49
len_33_48:
	BNE	R10, lasx_len_33_48
	JMP	lsx_len_33_48

len_gt_or_eq_49:
len_49_64:
	BNE	R10, lasx_len_49_64
	JMP	lsx_len_49_64

lsx_len_33_48:
	VMOVQ	0(R6), V0
	VMOVQ	16(R6), V1
	SUBV	$16, R7
	VMOVQ	(R6)(R7), V2
	SUBV	$1, R7
lsx_loop_33_48:
	BLT	R8, R4, not_found
	VMOVQ	0(R4), V3
	ADDV	$1, R4
	VSEQV	V0, V3, V3
	VSETANYEQV	V3, FCC0
	BFPT	FCC0, lsx_loop_33_48

	VMOVQ	15(R4), V4
	VSEQV	V1, V4, V4
	VSETANYEQV	V4, FCC1
	BFPT	FCC1, lsx_loop_33_48

	VMOVQ	(R4)(R7), V5
	VSEQV	V2, V5, V5
	VSETANYEQV	V5, FCC2
	BFPT	FCC2, lsx_loop_33_48
	JMP	found

lsx_len_49_64:
	VMOVQ	0(R6), V0
	VMOVQ	16(R6), V1
	VMOVQ	32(R6), V2
	SUBV	$16, R7
	VMOVQ	(R6)(R7), V3
	SUBV	$1, R7
lsx_loop_49_64:
	BLT	R8, R4, not_found
	VMOVQ	0(R4), V4
	ADDV	$1, R4
	VSEQV	V0, V4, V4
	VSETANYEQV	V4, FCC0
	BFPT	FCC0, lsx_loop_49_64

	VMOVQ	15(R4), V5
	VSEQV	V1, V5, V5
	VSETANYEQV	V5, FCC1
	BFPT	FCC1, lsx_loop_49_64

	VMOVQ	31(R4), V6
	VSEQV	V2, V6, V6
	VSETANYEQV	V6, FCC2
	BFPT	FCC2, lsx_loop_49_64

	VMOVQ	(R4)(R7), V7
	VSEQV	V3, V7, V7
	VSETANYEQV	V7, FCC3
	BFPT	FCC3, lsx_loop_49_64
	JMP	found

lasx_len_33_48:
lasx_len_49_64:
lasx_len_33_64:
	XVMOVQ	(R6), X0
	SUBV	$32, R7
	XVMOVQ	(R6)(R7), X1
	SUBV	$1, R7
lasx_loop_33_64:
	BLT	R8, R4, not_found
	XVMOVQ	(R4), X2
	ADDV	$1, R4
	XVSEQV	X0, X2, X3
	XVSETANYEQV	X3, FCC0
	BFPT	FCC0, lasx_loop_33_64

	XVMOVQ	(R4)(R7), X4
	XVSEQV	X1, X4, X5
	XVSETANYEQV	X5, FCC1
	BFPT	FCC1, lasx_loop_33_64
	JMP	found

found:
	SUBV	R9, R4
	RET

not_found:
	MOVV	$-1, R4
	RET
