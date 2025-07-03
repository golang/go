// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// func Count(b []byte, c byte) int
// input:
//   R0: b ptr
//   R1: b len
//   R2: b cap
//   R3: c byte to search
// return:
//   R0: result
TEXT ·Count<ABIInternal>(SB),NOSPLIT,$0-40
	MOVD	R3, R2
	B	·CountString<ABIInternal>(SB)

// func CountString(s string, c byte) int
// input:
//   R0: s ptr
//   R1: s len
//   R2: c byte to search (due to ABIInternal upper bits can contain junk)
// return:
//   R0: result
TEXT ·CountString<ABIInternal>(SB),NOSPLIT,$0-32
	// R11 = count of byte to search
	MOVD	$0, R11
	// short path to handle 0-byte case
	CBZ	R1, done
	CMP	$0x20, R1
	// jump directly to head if length >= 32
	BHS	head
tail:
	// Work with tail shorter than 32 bytes
	MOVBU.P	1(R0), R5
	SUB	$1, R1, R1
	CMP	R2.UXTB, R5
	CINC	EQ, R11, R11
	CBNZ	R1, tail
done:
	MOVD	R11, R0
	RET
	PCALIGN	$16
head:
	ANDS	$0x1f, R0, R9
	BEQ	chunk
	// Work with not 32-byte aligned head
	BIC	$0x1f, R0, R3
	ADD	$0x20, R3
	PCALIGN $16
head_loop:
	MOVBU.P	1(R0), R5
	CMP	R2.UXTB, R5
	CINC	EQ, R11, R11
	SUB	$1, R1, R1
	CMP	R0, R3
	BNE	head_loop
chunk:
	BIC	$0x1f, R1, R9
	// The first chunk can also be the last
	CBZ	R9, tail
	// R3 = end of 32-byte chunks
	ADD	R0, R9, R3
	MOVD	$1, R5
	VMOV	R5, V5.B16
	// R1 = length of tail
	SUB	R9, R1, R1
	// Duplicate R2 (byte to search) to 16 1-byte elements of V0
	VMOV	R2, V0.B16
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
	CBZ	R1, done
	B	tail
