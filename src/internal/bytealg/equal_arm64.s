// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Equal(SB),NOSPLIT,$0-49
	MOVD	a_len+8(FP), R1
	MOVD	b_len+32(FP), R3
	CMP	R1, R3
	// unequal lengths are not equal
	BNE	not_equal
	// short path to handle 0-byte case
	CBZ	R1, equal
	MOVD	a_base+0(FP), R0
	MOVD	b_base+24(FP), R2
	MOVD	$ret+48(FP), R8
	B	memeqbody<>(SB)
equal:
	MOVD	$1, R0
	MOVB	R0, ret+48(FP)
	RET
not_equal:
	MOVB	ZR, ret+48(FP)
	RET

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-25
	MOVD	size+16(FP), R1
	// short path to handle 0-byte case
	CBZ	R1, equal
	MOVD	a+0(FP), R0
	MOVD	b+8(FP), R2
	MOVD	$ret+24(FP), R8
	B	memeqbody<>(SB)
equal:
	MOVD	$1, R0
	MOVB	R0, ret+24(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$40-17
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R4
	CMP	R3, R4
	BEQ	eq
	MOVD	8(R26), R5    // compiler stores size at offset 8 in the closure
	CBZ	R5, eq
	MOVD	R3, 8(RSP)
	MOVD	R4, 16(RSP)
	MOVD	R5, 24(RSP)
	BL	runtime·memequal(SB)
	MOVBU	32(RSP), R3
	MOVB	R3, ret+16(FP)
	RET
eq:
	MOVD	$1, R3
	MOVB	R3, ret+16(FP)
	RET

// input:
// R0: pointer a
// R1: data len
// R2: pointer b
// R8: address to put result
TEXT memeqbody<>(SB),NOSPLIT,$0
	CMP	$1, R1
	// handle 1-byte special case for better performance
	BEQ	one
	CMP	$16, R1
	// handle specially if length < 16
	BLO	tail
	BIC	$0x3f, R1, R3
	CBZ	R3, chunk16
	// work with 64-byte chunks
	ADD	R3, R0, R6	// end of chunks
chunk64_loop:
	VLD1.P	(R0), [V0.D2, V1.D2, V2.D2, V3.D2]
	VLD1.P	(R2), [V4.D2, V5.D2, V6.D2, V7.D2]
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
	AND	$0x3f, R1, R1
	CBZ	R1, equal
chunk16:
	// work with 16-byte chunks
	BIC	$0xf, R1, R3
	CBZ	R3, tail
	ADD	R3, R0, R6	// end of chunks
chunk16_loop:
	LDP.P	16(R0), (R4, R5)
	LDP.P	16(R2), (R7, R9)
	EOR	R4, R7
	CBNZ	R7, not_equal
	EOR	R5, R9
	CBNZ	R9, not_equal
	CMP	R0, R6
	BNE	chunk16_loop
	AND	$0xf, R1, R1
	CBZ	R1, equal
tail:
	// special compare of tail with length < 16
	TBZ	$3, R1, lt_8
	MOVD	(R0), R4
	MOVD	(R2), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	SUB	$8, R1, R6	// offset of the last 8 bytes
	MOVD	(R0)(R6), R4
	MOVD	(R2)(R6), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	B	equal
lt_8:
	TBZ	$2, R1, lt_4
	MOVWU	(R0), R4
	MOVWU	(R2), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	SUB	$4, R1, R6	// offset of the last 4 bytes
	MOVWU	(R0)(R6), R4
	MOVWU	(R2)(R6), R5
	EOR	R4, R5
	CBNZ	R5, not_equal
	B	equal
lt_4:
	TBZ	$1, R1, lt_2
	MOVHU.P	2(R0), R4
	MOVHU.P	2(R2), R5
	CMP	R4, R5
	BNE	not_equal
lt_2:
	TBZ	$0, R1, equal
one:
	MOVBU	(R0), R4
	MOVBU	(R2), R5
	CMP	R4, R5
	BNE	not_equal
equal:
	MOVD	$1, R0
	MOVB	R0, (R8)
	RET
not_equal:
	MOVB	ZR, (R8)
	RET
