// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	// short path to handle 0-byte case
	CBZ	R2, equal
	B	memeqbody<>(SB)
equal:
	MOVD	$1, R0
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT,$0-17
	CMP	R0, R1
	BEQ	eq
	MOVD	8(R26), R2    // compiler stores size at offset 8 in the closure
	CBZ	R2, eq
	B	memeqbody<>(SB)
eq:
	MOVD	$1, R0
	RET

// input:
// R0: pointer a
// R1: pointer b
// R2: data len
// at return: result in R0
TEXT memeqbody<>(SB),NOSPLIT,$0
	CMP	$1, R2
	// handle 1-byte special case for better performance
	BEQ	one
	CMP	$16, R2
	// handle specially if length < 16
	BLO	tail
	BIC	$0x3f, R2, R3
	CBZ	R3, chunk16
	// work with 64-byte chunks
	ADD	R3, R0, R6	// end of chunks
chunk64_loop:
	VLD1.P	(R0), [V0.D2, V1.D2, V2.D2, V3.D2]
	VLD1.P	(R1), [V4.D2, V5.D2, V6.D2, V7.D2]
	VCMEQ	V0.D2, V4.D2, V8.D2
	VCMEQ	V1.D2, V5.D2, V9.D2
	VCMEQ	V2.D2, V6.D2, V10.D2
	VCMEQ	V3.D2, V7.D2, V11.D2
	VAND	V8.B16, V9.B16, V8.B16
	VAND	V8.B16, V10.B16, V8.B16
	VAND	V8.B16, V11.B16, V8.B16
	CMP	R0, R6
	VMOV	V8.D[0], R4
	VMOV	V8.D[1], R5
	CBZ	R4, not_equal
	CBZ	R5, not_equal
	BNE	chunk64_loop
	AND	$0x3f, R2, R2
	CBZ	R2, equal
chunk16:
	// work with 16-byte chunks
	BIC	$0xf, R2, R3
	CBZ	R3, tail
	ADD	R3, R0, R6	// end of chunks
chunk16_loop:
	LDP.P	16(R0), (R4, R5)
	LDP.P	16(R1), (R7, R9)
	EOR	R4, R7
	CBNZ	R7, not_equal
	EOR	R5, R9
	CBNZ	R9, not_equal
	CMP	R0, R6
	BNE	chunk16_loop
	AND	$0xf, R2, R2
	CBZ	R2, equal
tail:
	// special compare of tail with length < 16
	TBZ	$3, R2, lt_8
	MOVD	(R0), R4
	MOVD	(R1), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	SUB	$8, R2, R6	// offset of the last 8 bytes
	MOVD	(R0)(R6), R4
	MOVD	(R1)(R6), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	B	equal
lt_8:
	TBZ	$2, R2, lt_4
	MOVWU	(R0), R4
	MOVWU	(R1), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	SUB	$4, R2, R6	// offset of the last 4 bytes
	MOVWU	(R0)(R6), R4
	MOVWU	(R1)(R6), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	B	equal
lt_4:
	TBZ	$1, R2, lt_2
	MOVHU.P	2(R0), R4
	MOVHU.P	2(R1), R5
	CMP	R4, R5
	BNE	not_equal
lt_2:
	TBZ	$0, R2, equal
one:
	MOVBU	(R0), R4
	MOVBU	(R1), R5
	CMP	R4, R5
	BNE	not_equal
equal:
	MOVD	$1, R0
	RET
not_equal:
	MOVB	ZR, R0
	RET
