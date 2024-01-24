// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Count(SB),NOSPLIT,$0-40
	MOVD	b_base+0(FP), R0
	MOVD	b_len+8(FP), R2
	MOVBU	c+24(FP), R1
	MOVD	$ret+32(FP), R8
	B	countbytebody<>(SB)

TEXT ·CountString(SB),NOSPLIT,$0-32
	MOVD	s_base+0(FP), R0
	MOVD	s_len+8(FP), R2
	MOVBU	c+16(FP), R1
	MOVD	$ret+24(FP), R8
	B	countbytebody<>(SB)

// input:
//   R0: data
//   R2: data len
//   R1: byte to find
//   R8: address to put result
TEXT countbytebody<>(SB),NOSPLIT,$0
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
	PCALIGN $16
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
	PCALIGN $16
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
	MOVD	R11, (R8)
	RET
