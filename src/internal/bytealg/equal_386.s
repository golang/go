// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-13
	MOVL	a+0(FP), SI
	MOVL	b+4(FP), DI
	CMPL	SI, DI
	JEQ	eq
	MOVL	size+8(FP), BX
	LEAL	ret+12(FP), AX
	JMP	memeqbody<>(SB)
eq:
	MOVB    $1, ret+12(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$0-9
	MOVL    a+0(FP), SI
	MOVL    b+4(FP), DI
	CMPL    SI, DI
	JEQ     eq
	MOVL    4(DX), BX    // compiler stores size at offset 4 in the closure
	LEAL	ret+8(FP), AX
	JMP	memeqbody<>(SB)
eq:
	MOVB    $1, ret+8(FP)
	RET

// a in SI
// b in DI
// count in BX
// address of result byte in AX
TEXT memeqbody<>(SB),NOSPLIT,$0-0
	CMPL	BX, $4
	JB	small

	// 64 bytes at a time using xmm registers
hugeloop:
	CMPL	BX, $64
	JB	bigloop
	CMPB	internal∕cpu·X86+const_offsetX86HasSSE2(SB), $1
	JNE	bigloop
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
	ADDL	$64, SI
	ADDL	$64, DI
	SUBL	$64, BX
	CMPL	DX, $0xffff
	JEQ	hugeloop
	MOVB	$0, (AX)
	RET

	// 4 bytes at a time using 32-bit register
bigloop:
	CMPL	BX, $4
	JBE	leftover
	MOVL	(SI), CX
	MOVL	(DI), DX
	ADDL	$4, SI
	ADDL	$4, DI
	SUBL	$4, BX
	CMPL	CX, DX
	JEQ	bigloop
	MOVB	$0, (AX)
	RET

	// remaining 0-4 bytes
leftover:
	MOVL	-4(SI)(BX*1), CX
	MOVL	-4(DI)(BX*1), DX
	CMPL	CX, DX
	SETEQ	(AX)
	RET

small:
	CMPL	BX, $0
	JEQ	equal

	LEAL	0(BX*8), CX
	NEGL	CX

	MOVL	SI, DX
	CMPB	DX, $0xfc
	JA	si_high

	// load at SI won't cross a page boundary.
	MOVL	(SI), SI
	JMP	si_finish
si_high:
	// address ends in 111111xx. Load up to bytes we want, move to correct position.
	MOVL	-4(SI)(BX*1), SI
	SHRL	CX, SI
si_finish:

	// same for DI.
	MOVL	DI, DX
	CMPB	DX, $0xfc
	JA	di_high
	MOVL	(DI), DI
	JMP	di_finish
di_high:
	MOVL	-4(DI)(BX*1), DI
	SHRL	CX, DI
di_finish:

	SUBL	SI, DI
	SHLL	CX, DI
equal:
	SETEQ	(AX)
	RET
