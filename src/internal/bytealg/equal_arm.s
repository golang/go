// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Equal(SB),NOSPLIT,$0-25
	MOVW	a_len+4(FP), R1
	MOVW	b_len+16(FP), R3

	CMP	R1, R3		// unequal lengths are not equal
	B.NE	notequal

	MOVW	a_base+0(FP), R0
	MOVW	b_base+12(FP), R2
	MOVW	$ret+24(FP), R7
	B	memeqbody<>(SB)
notequal:
	MOVW	$0, R0
	MOVBU	R0, ret+24(FP)
	RET

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-13
	MOVW	a+0(FP), R0
	MOVW	b+4(FP), R2
	CMP	R0, R2
	B.EQ	eq
	MOVW	size+8(FP), R1
	MOVW	$ret+12(FP), R7
	B	memeqbody<>(SB)
eq:
	MOVW	$1, R0
	MOVB	R0, ret+12(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT|NOFRAME,$0-9
	MOVW	a+0(FP), R0
	MOVW	b+4(FP), R2
	CMP	R0, R2
	B.EQ	eq
	MOVW	4(R7), R1    // compiler stores size at offset 4 in the closure
	MOVW	$ret+8(FP), R7
	B	memeqbody<>(SB)
eq:
	MOVW	$1, R0
	MOVB	R0, ret+8(FP)
	RET

// Input:
// R0: data of a
// R1: length
// R2: data of b
// R7: points to return value
TEXT memeqbody<>(SB),NOSPLIT|NOFRAME,$0-0
	ADD	R0, R1		// end
loop:
	CMP	R0, R1
	B.EQ	equal		// reached the end
	MOVBU.P	1(R0), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	B.EQ	loop
notequal:
	MOVW	$0, R0
	MOVB	R0, (R7)
	RET
equal:
	MOVW	$1, R0
	MOVB	R0, (R7)
	RET
