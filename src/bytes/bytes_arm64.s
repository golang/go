// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// countByte(s []byte, c byte) int
TEXT bytes·countByte(SB),NOSPLIT,$0-40
	MOVD	s_base+0(FP), R0
	MOVD	s_len+8(FP), R2
	MOVBU	c+24(FP), R1
	// R11 = count of byte to search
	MOVD	$0, R11
	// short path to handle 0-byte case
	CBZ	R2, done
	CMP	$0x20, R2
	// jump directly to tail if length < 32
	BLO	tail
	ANDS	$0x1f, R0, R9
	BEQ	chunk
	// Work with not 32-byte aligned head
	BIC	$0x1f, R0, R3
	ADD	$0x20, R3
head_loop:
	MOVBU.P	1(R0), R5
	CMP	R5, R1
	CINC	EQ, R11, R11
	SUB	$1, R2, R2
	CMP	R0, R3
	BNE	head_loop
	// Work with 32-byte aligned chunks
chunk:
	BIC	$0x1f, R2, R9
	// The first chunk can also be the last
	CBZ	R9, tail
	// R3 = end of 32-byte chunks
	ADD	R0, R9, R3
	MOVD	$1, R5
	VMOV	R5, V5.B16
	// R2 = length of tail
	SUB	R9, R2, R2
	// Duplicate R1 (byte to search) to 16 1-byte elements of V0
	VMOV	R1, V0.B16
	// Clear the low 64-bit element of V7 and V8
	VEOR	V7.B8, V7.B8, V7.B8
	VEOR	V8.B8, V8.B8, V8.B8
	// Count the target byte in 32-byte chunk
chunk_loop:
	VLD1.P	(R0), [V1.B16, V2.B16]
	CMP	R0, R3
	VCMEQ	V0.B16, V1.B16, V3.B16
	VCMEQ	V0.B16, V2.B16, V4.B16
	// Clear the higher 7 bits
	VAND	V5.B16, V3.B16, V3.B16
	VAND	V5.B16, V4.B16, V4.B16
	// Count lanes match the requested byte
	VADDP	V4.B16, V3.B16, V6.B16 // 32B->16B
	VUADDLV	V6.B16, V7
	// Accumulate the count in low 64-bit element of V8 when inside the loop
	VADD	V7, V8
	BNE	chunk_loop
	VMOV	V8.D[0], R6
	ADD	R6, R11, R11
	CBZ	R2, done
tail:
	// Work with tail shorter than 32 bytes
	MOVBU.P	1(R0), R5
	SUB	$1, R2, R2
	CMP	R5, R1
	CINC	EQ, R11, R11
	CBNZ	R2, tail
done:
	MOVD	R11, ret+32(FP)
	RET

// indexShortStr(s, sep []byte) int
// precondition: 2 <= len(sep) <= 8
TEXT bytes·indexShortStr(SB),NOSPLIT,$0-56
	// main idea is to load 'sep' into separate register(s)
	// to avoid repeatedly re-load it again and again
	// for sebsequent substring comparisons
	MOVD	s+0(FP), R0
	MOVD	s_len+8(FP), R1
	MOVD	sep+24(FP), R2
	MOVD	sep_len+32(FP), R3
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
	MOVD.P	1(R0), R6
	CMP	R5, R6
	BEQ	found
	CMP	R4, R0
	BLS	loop_8
	JMP	not_found
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
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	not_equal_7
	MOVWU	2(R0), R3
	CMP	R6, R3
	BEQ	found
not_equal_7:
	CMP	R4, R0
	BLS	loop_7
	JMP	not_found
len_6:
	// R5 and R6 contain 6-byte sep
	MOVWU	(R2), R5
	MOVHU	4(R2), R6
loop_6:
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	not_equal_6
	MOVHU	3(R0), R3
	CMP	R6, R3
	BEQ	found
not_equal_6:
	CMP	R4, R0
	BLS	loop_6
	JMP	not_found
len_4_5:
	TBZ	$0, R3, len_4
len_5:
	// R5 and R7 contain 5-byte sep
	MOVWU	(R2), R5
	MOVBU	4(R2), R7
loop_5:
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	not_equal_5
	MOVBU	3(R0), R3
	CMP	R7, R3
	BEQ	found
not_equal_5:
	CMP	R4, R0
	BLS	loop_5
	JMP	not_found
len_4:
	// R5 contains 4-byte sep
	MOVWU	(R2), R5
loop_4:
	MOVWU.P	1(R0), R6
	CMP	R5, R6
	BEQ	found
	CMP	R4, R0
	BLS	loop_4
	JMP	not_found
len_2_3:
	TBZ	$0, R3, len_2
len_3:
	// R6 and R7 contain 3-byte sep
	MOVHU	(R2), R6
	MOVBU	2(R2), R7
loop_3:
	MOVHU.P	1(R0), R3
	CMP	R6, R3
	BNE	not_equal_3
	MOVBU	1(R0), R3
	CMP	R7, R3
	BEQ	found
not_equal_3:
	CMP	R4, R0
	BLS	loop_3
	JMP	not_found
len_2:
	// R5 contains 2-byte sep
	MOVHU	(R2), R5
loop_2:
	MOVHU.P	1(R0), R6
	CMP	R5, R6
	BEQ	found
	CMP	R4, R0
	BLS	loop_2
not_found:
	MOVD	$-1, R0
	MOVD	R0, ret+48(FP)
	RET
found:
	SUB	R8, R0, R0
	MOVD	R0, ret+48(FP)
	RET
