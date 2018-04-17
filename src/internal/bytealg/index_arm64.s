// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Index(SB),NOSPLIT,$0-56
	MOVD	a_base+0(FP), R0
	MOVD	a_len+8(FP), R1
	MOVD	b_base+24(FP), R2
	MOVD	b_len+32(FP), R3
	MOVD	$ret+48(FP), R9
	B	indexbody<>(SB)

TEXT ·IndexString(SB),NOSPLIT,$0-40
	MOVD	a_base+0(FP), R0
	MOVD	a_len+8(FP), R1
	MOVD	b_base+16(FP), R2
	MOVD	b_len+24(FP), R3
	MOVD	$ret+32(FP), R9
	B	indexbody<>(SB)

// input:
//   R0: haystack
//   R1: length of haystack
//   R2: needle
//   R3: length of needle (2 <= len <= 8)
//   R9: address to put result
TEXT indexbody<>(SB),NOSPLIT,$0-56
	// main idea is to load 'sep' into separate register(s)
	// to avoid repeatedly re-load it again and again
	// for sebsequent substring comparisons
	SUB	R3, R1, R4
	// R4 contains the start of last substring for comparsion
	ADD	R0, R4, R4
	ADD	$1, R0, R8
	TBZ	$3, R3, len_2_7
len_8:
	// R5 contains 8-byte sep
	MOVD	(R2), R5
loop_8:
	// R6 contains substring for comparison
	CMP	R4, R0
	BHI	not_found
	MOVD.P	1(R0), R6
	CMP	R5, R6
	BNE	loop_8
	B	found
len_2_7:
	TBZ	$2, R3, len_2_3
	TBZ	$1, R3, len_4_5
	TBZ	$0, R3, len_6
len_7:
	// R5 and R6 contain 7-byte sep
	MOVWU	(R2), R5
	// 1-byte overlap with R5
	MOVWU	3(R2), R6
loop_7:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	loop_7
	MOVWU	2(R0), R3
	CMP	R6, R3
	BNE	loop_7
	B	found
len_6:
	// R5 and R6 contain 6-byte sep
	MOVWU	(R2), R5
	MOVHU	4(R2), R6
loop_6:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	loop_6
	MOVHU	3(R0), R3
	CMP	R6, R3
	BNE	loop_6
	B	found
len_4_5:
	TBZ	$0, R3, len_4
len_5:
	// R5 and R7 contain 5-byte sep
	MOVWU	(R2), R5
	MOVBU	4(R2), R7
loop_5:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	loop_5
	MOVBU	3(R0), R3
	CMP	R7, R3
	BNE	loop_5
	B	found
len_4:
	// R5 contains 4-byte sep
	MOVWU	(R2), R5
loop_4:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R6
	CMP	R5, R6
	BNE	loop_4
	B	found
len_2_3:
	TBZ	$0, R3, len_2
len_3:
	// R6 and R7 contain 3-byte sep
	MOVHU	(R2), R6
	MOVBU	2(R2), R7
loop_3:
	CMP	R4, R0
	BHI	not_found
	MOVHU.P	1(R0), R3
	CMP	R6, R3
	BNE	loop_3
	MOVBU	1(R0), R3
	CMP	R7, R3
	BNE	loop_3
	B	found
len_2:
	// R5 contains 2-byte sep
	MOVHU	(R2), R5
loop_2:
	CMP	R4, R0
	BHI	not_found
	MOVHU.P	1(R0), R6
	CMP	R5, R6
	BNE	loop_2
found:
	SUB	R8, R0, R0
	MOVD	R0, (R9)
	RET
not_found:
	MOVD	$-1, R0
	MOVD	R0, (R9)
	RET
