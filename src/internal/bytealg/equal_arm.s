// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-13
	MOVW	a+0(FP), R0
	MOVW	b+4(FP), R2
	CMP	R0, R2
	B.EQ	eq
	MOVW	size+8(FP), R1
	CMP	$0, R1
	B.EQ	eq		// short path to handle 0-byte case
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
	MOVW	4(R7), R1	// compiler stores size at offset 4 in the closure
	CMP	$0, R1
	B.EQ	eq		// short path to handle 0-byte case
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
//
// On exit:
// R4, R5 and R6 are clobbered
TEXT memeqbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMP	$1, R1
	B.EQ	one		// 1-byte special case for better performance

	CMP	$4, R1
	ADD	R0, R1		// R1 is the end of the range to compare
	B.LT	byte_loop	// length < 4
	AND	$3, R0, R6
	CMP	$0, R6
	B.NE	byte_loop	// unaligned a, use byte-wise compare (TODO: try to align a)
	AND	$3, R2, R6
	CMP	$0, R6
	B.NE	byte_loop	// unaligned b, use byte-wise compare
	AND	$0xfffffffc, R1, R6
	// length >= 4
chunk4_loop:
	MOVW.P	4(R0), R4
	MOVW.P	4(R2), R5
	CMP	R4, R5
	B.NE	notequal
	CMP	R0, R6
	B.NE	chunk4_loop
	CMP	R0, R1
	B.EQ	equal		// reached the end
byte_loop:
	MOVBU.P	1(R0), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	B.NE	notequal
	CMP	R0, R1
	B.NE	byte_loop
equal:
	MOVW	$1, R0
	MOVB	R0, (R7)
	RET
one:
	MOVBU	(R0), R4
	MOVBU	(R2), R5
	CMP	R4, R5
	B.EQ	equal
notequal:
	MOVW	$0, R0
	MOVB	R0, (R7)
	RET
