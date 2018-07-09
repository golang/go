// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// TODO: share code with memequal?
TEXT ·Equal(SB),NOSPLIT,$0-25
	MOVW	a_len+4(FP), R1
	MOVW	b_len+16(FP), R3
	
	CMP	R1, R3		// unequal lengths are not equal
	B.NE	notequal

	MOVW	a_base+0(FP), R0
	MOVW	b_base+12(FP), R2
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
	MOVBU	R0, ret+24(FP)
	RET

equal:
	MOVW	$1, R0
	MOVBU	R0, ret+24(FP)
	RET

TEXT bytes·Equal(SB),NOSPLIT,$0-25
	FUNCDATA $0, ·Equal·args_stackmap(SB)
	JMP	·Equal(SB)

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-13
	MOVW	a+0(FP), R1
	MOVW	b+4(FP), R2
	MOVW	size+8(FP), R3
	ADD	R1, R3, R6
	MOVW	$1, R0
	MOVB	R0, ret+12(FP)
	CMP	R1, R2
	RET.EQ
loop:
	CMP	R1, R6
	RET.EQ
	MOVBU.P	1(R1), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	BEQ	loop

	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$16-9
	MOVW	a+0(FP), R0
	MOVW	b+4(FP), R1
	CMP	R0, R1
	BEQ	eq
	MOVW	4(R7), R2    // compiler stores size at offset 4 in the closure
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·memequal(SB)
	MOVB	16(R13), R0
	MOVB	R0, ret+8(FP)
	RET
eq:
	MOVW	$1, R0
	MOVB	R0, ret+8(FP)
	RET
