// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "asm_amd64.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT,$0-25
	// AX = a    (want in SI)
	// BX = b    (want in DI)
	// CX = size (want in BX)
	CMPQ	AX, BX
	JNE	neq
	MOVQ	$1, AX	// return 1
	RET
neq:
	MOVQ	AX, SI
	MOVQ	BX, DI
	MOVQ	CX, BX
	JMP	memeqbody<>(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT,$0-17
	// AX = a       (want in SI)
	// BX = b       (want in DI)
	// 8(DX) = size (want in BX)
	CMPQ	AX, BX
	JNE	neq
	MOVQ	$1, AX	// return 1
	RET
neq:
	MOVQ	AX, SI
	MOVQ	BX, DI
	MOVQ	8(DX), BX    // compiler stores size at offset 8 in the closure
	JMP	memeqbody<>(SB)

// Input:
//   a in SI
//   b in DI
//   count in BX
// Output:
//   result in AX
TEXT memeqbody<>(SB),NOSPLIT,$0-0
	CMPQ	BX, $8
	JB	small
	CMPQ	BX, $64
	JB	bigloop
#ifndef hasAVX2
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX2(SB), $1
	JE	hugeloop_avx2

	// 64 bytes at a time using xmm registers
	PCALIGN $16
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
	XORQ	AX, AX	// return 0
	RET
#endif

	// 64 bytes at a time using ymm registers
	PCALIGN $16
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
	XORQ	AX, AX	// return 0
	RET

bigloop_avx2:
	VZEROUPPER

	// 8 bytes at a time using 64-bit register
	PCALIGN $16
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
	XORQ	AX, AX	// return 0
	RET

	// remaining 0-8 bytes
leftover:
	MOVQ	-8(SI)(BX*1), CX
	MOVQ	-8(DI)(BX*1), DX
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
	SETEQ	AX
	RET
