// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Equal(SB),NOSPLIT,$0-49
	MOVQ	a_len+8(FP), BX
	MOVQ	b_len+32(FP), CX
	CMPQ	BX, CX
	JNE	neq
	MOVQ	a_base+0(FP), SI
	MOVQ	b_base+24(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	LEAQ	ret+48(FP), AX
	JMP	memeqbody<>(SB)
neq:
	MOVB	$0, ret+48(FP)
	RET
eq:
	MOVB	$1, ret+48(FP)
	RET

TEXT bytes·Equal(SB),NOSPLIT,$0-49
	FUNCDATA $0, ·Equal·args_stackmap(SB)
	MOVQ	a_len+8(FP), BX
	MOVQ	b_len+32(FP), CX
	CMPQ	BX, CX
	JNE	neq
	MOVQ	a_base+0(FP), SI
	MOVQ	b_base+24(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	LEAQ	ret+48(FP), AX
	JMP	memeqbody<>(SB)
neq:
	MOVB	$0, ret+48(FP)
	RET
eq:
	MOVB	$1, ret+48(FP)
	RET

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-25
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	MOVQ	size+16(FP), BX
	LEAQ	ret+24(FP), AX
	JMP	memeqbody<>(SB)
eq:
	MOVB	$1, ret+24(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$0-17
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	MOVQ	8(DX), BX    // compiler stores size at offset 8 in the closure
	LEAQ	ret+16(FP), AX
	JMP	memeqbody<>(SB)
eq:
	MOVB	$1, ret+16(FP)
	RET

// a in SI
// b in DI
// count in BX
// address of result byte in AX
TEXT memeqbody<>(SB),NOSPLIT,$0-0
	CMPQ	BX, $8
	JB	small
	CMPQ	BX, $64
	JB	bigloop
	CMPB	internal∕cpu·X86+const_x86_HasAVX2(SB), $1
	JE	hugeloop_avx2
	
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
	MOVB	$0, (AX)
	RET

	// 64 bytes at a time using ymm registers
hugeloop_avx2:
	CMPQ	BX, $64
	JB	bigloop_avx2
	VMOVDQU	(SI), Y0
	VMOVDQU	(DI), Y1
	VMOVDQU	32(SI), Y2
	VMOVDQU	32(DI), Y3
	VPCMPEQB	Y1, Y0, Y4
	VPCMPEQB	Y2, Y3, Y5
	VPAND	Y4, Y5, Y6
	VPMOVMSKB Y6, DX
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$64, BX
	CMPL	DX, $0xffffffff
	JEQ	hugeloop_avx2
	VZEROUPPER
	MOVB	$0, (AX)
	RET

bigloop_avx2:
	VZEROUPPER

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
	MOVB	$0, (AX)
	RET

	// remaining 0-8 bytes
leftover:
	MOVQ	-8(SI)(BX*1), CX
	MOVQ	-8(DI)(BX*1), DX
	CMPQ	CX, DX
	SETEQ	(AX)
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
	MOVQ	-8(SI)(BX*1), SI
	SHRQ	CX, SI
si_finish:

	// same for DI.
	CMPB	DI, $0xf8
	JA	di_high
	MOVQ	(DI), DI
	JMP	di_finish
di_high:
	MOVQ	-8(DI)(BX*1), DI
	SHRQ	CX, DI
di_finish:

	SUBQ	SI, DI
	SHLQ	CX, DI
equal:
	SETEQ	(AX)
	RET

