// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "go_asm.h"
#include "textflag.h"

#define	REGCTXT	R22

TEXT ·Equal(SB),NOSPLIT,$0-25
	MOVW	a_len+4(FP), R3
	MOVW	b_len+16(FP), R4
	BNE	R3, R4, noteq	// unequal lengths are not equal

	MOVW	a_base+0(FP), R1
	MOVW	b_base+12(FP), R2
	ADDU	R1, R3	// end

loop:
	BEQ	R1, R3, equal	// reached the end
	MOVBU	(R1), R6
	ADDU	$1, R1
	MOVBU	(R2), R7
	ADDU	$1, R2
	BEQ	R6, R7, loop

noteq:
	MOVB	R0, ret+24(FP)
	RET

equal:
	MOVW	$1, R1
	MOVB	R1, ret+24(FP)
	RET

TEXT bytes·Equal(SB),NOSPLIT,$0-25
	FUNCDATA $0, ·Equal·args_stackmap(SB)
	JMP	·Equal(SB)

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-13
	MOVW	a+0(FP), R1
	MOVW	b+4(FP), R2
	BEQ	R1, R2, eq
	MOVW	size+8(FP), R3
	ADDU	R1, R3, R4
loop:
	BNE	R1, R4, test
	MOVW	$1, R1
	MOVB	R1, ret+12(FP)
	RET
test:
	MOVBU	(R1), R6
	ADDU	$1, R1
	MOVBU	(R2), R7
	ADDU	$1, R2
	BEQ	R6, R7, loop

	MOVB	R0, ret+12(FP)
	RET
eq:
	MOVW	$1, R1
	MOVB	R1, ret+12(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$0-9
	MOVW	a+0(FP), R1
	MOVW	b+4(FP), R2
	BEQ	R1, R2, eq
	MOVW	4(REGCTXT), R3	// compiler stores size at offset 4 in the closure
	ADDU	R1, R3, R4
loop:
	BNE	R1, R4, test
	MOVW	$1, R1
	MOVB	R1, ret+8(FP)
	RET
test:
	MOVBU	(R1), R6
	ADDU	$1, R1
	MOVBU	(R2), R7
	ADDU	$1, R2
	BEQ	R6, R7, loop

	MOVB	R0, ret+8(FP)
	RET
eq:
	MOVW	$1, R1
	MOVB	R1, ret+8(FP)
	RET
