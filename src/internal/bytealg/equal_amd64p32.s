// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Equal(SB),NOSPLIT,$0-25
	MOVL	a_len+4(FP), BX
	MOVL	b_len+16(FP), CX
	CMPL	BX, CX
	JNE	neq
	MOVL	a_base+0(FP), SI
	MOVL	b_base+12(FP), DI
	CMPL	SI, DI
	JEQ	eq
	CALL	memeqbody<>(SB)
	MOVB	AX, ret+24(FP)
	RET
neq:
	MOVB	$0, ret+24(FP)
	RET
eq:
	MOVB    $1, ret+24(FP)
	RET

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-17
	MOVL	a+0(FP), SI
	MOVL	b+4(FP), DI
	CMPL	SI, DI
	JEQ	eq
	MOVL	size+8(FP), BX
	CALL	memeqbody<>(SB)
	MOVB	AX, ret+16(FP)
	RET
eq:
	MOVB    $1, ret+16(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$0-9
	MOVL    a+0(FP), SI
	MOVL    b+4(FP), DI
	CMPL    SI, DI
	JEQ     eq
	MOVL    4(DX), BX    // compiler stores size at offset 4 in the closure
	CALL    memeqbody<>(SB)
	MOVB    AX, ret+8(FP)
	RET
eq:
	MOVB    $1, ret+8(FP)
	RET

// a in SI
// b in DI
// count in BX
TEXT memeqbody<>(SB),NOSPLIT,$0-0
	XORQ	AX, AX

	CMPQ	BX, $8
	JB	small

	// 64 bytes at a time using xmm registers
hugeloop:
	CMPQ	BX, $64
	JB	bigloop
	MOVOU	(SI), X0
	MOVOU	(DI), X1
	MOVOU	16(SI), X2
	MOVOU	16(DI), X3
	MOVOU	32(SI), X4
	MOVOU	32(DI), X5
	MOVOU	48(SI), X6
	MOVOU	48(DI), X7
	PCMPEQB	X1, X0
	PCMPEQB	X3, X2
	PCMPEQB	X5, X4
	PCMPEQB	X7, X6
	PAND	X2, X0
	PAND	X6, X4
	PAND	X4, X0
	PMOVMSKB X0, DX
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$64, BX
	CMPL	DX, $0xffff
	JEQ	hugeloop
	RET

	// 8 bytes at a time using 64-bit register
bigloop:
	CMPQ	BX, $8
	JBE	leftover
	MOVQ	(SI), CX
	MOVQ	(DI), DX
	ADDQ	$8, SI
	ADDQ	$8, DI
	SUBQ	$8, BX
	CMPQ	CX, DX
	JEQ	bigloop
	RET

	// remaining 0-8 bytes
leftover:
	ADDQ	BX, SI
	ADDQ	BX, DI
	MOVQ	-8(SI), CX
	MOVQ	-8(DI), DX
	CMPQ	CX, DX
	SETEQ	AX
	RET

small:
	CMPQ	BX, $0
	JEQ	equal

	LEAQ	0(BX*8), CX
	NEGQ	CX

	CMPB	SI, $0xf8
	JA	si_high

	// load at SI won't cross a page boundary.
	MOVQ	(SI), SI
	JMP	si_finish
si_high:
	// address ends in 11111xxx. Load up to bytes we want, move to correct position.
	MOVQ	BX, DX
	ADDQ	SI, DX
	MOVQ	-8(DX), SI
	SHRQ	CX, SI
si_finish:

	// same for DI.
	CMPB	DI, $0xf8
	JA	di_high
	MOVQ	(DI), DI
	JMP	di_finish
di_high:
	MOVQ	BX, DX
	ADDQ	DI, DX
	MOVQ	-8(DX), DI
	SHRQ	CX, DI
di_finish:

	SUBQ	SI, DI
	SHLQ	CX, DI
equal:
	SETEQ	AX
	RET
