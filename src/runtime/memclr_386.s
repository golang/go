// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

#include "textflag.h"

// NOTE: Windows externalthreadhandler expects memclr to preserve DX.

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), BX
	XORL	AX, AX

	// MOVOU seems always faster than REP STOSL.
clr_tail:
	TESTL	BX, BX
	JEQ	clr_0
	CMPL	BX, $2
	JBE	clr_1or2
	CMPL	BX, $4
	JBE	clr_3or4
	CMPL	BX, $8
	JBE	clr_5through8
	CMPL	BX, $16
	JBE	clr_9through16
	TESTL	$0x4000000, runtime·cpuid_edx(SB) // check for sse2
	JEQ	nosse2
	PXOR	X0, X0
	CMPL	BX, $32
	JBE	clr_17through32
	CMPL	BX, $64
	JBE	clr_33through64
	CMPL	BX, $128
	JBE	clr_65through128
	CMPL	BX, $256
	JBE	clr_129through256
	// TODO: use branch table and BSR to make this just a single dispatch

clr_loop:
	MOVOU	X0, 0(DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, 64(DI)
	MOVOU	X0, 80(DI)
	MOVOU	X0, 96(DI)
	MOVOU	X0, 112(DI)
	MOVOU	X0, 128(DI)
	MOVOU	X0, 144(DI)
	MOVOU	X0, 160(DI)
	MOVOU	X0, 176(DI)
	MOVOU	X0, 192(DI)
	MOVOU	X0, 208(DI)
	MOVOU	X0, 224(DI)
	MOVOU	X0, 240(DI)
	SUBL	$256, BX
	ADDL	$256, DI
	CMPL	BX, $256
	JAE	clr_loop
	JMP	clr_tail

clr_1or2:
	MOVB	AX, (DI)
	MOVB	AX, -1(DI)(BX*1)
	RET
clr_0:
	RET
clr_3or4:
	MOVW	AX, (DI)
	MOVW	AX, -2(DI)(BX*1)
	RET
clr_5through8:
	MOVL	AX, (DI)
	MOVL	AX, -4(DI)(BX*1)
	RET
clr_9through16:
	MOVL	AX, (DI)
	MOVL	AX, 4(DI)
	MOVL	AX, -8(DI)(BX*1)
	MOVL	AX, -4(DI)(BX*1)
	RET
clr_17through32:
	MOVOU	X0, (DI)
	MOVOU	X0, -16(DI)(BX*1)
	RET
clr_33through64:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
clr_65through128:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, -64(DI)(BX*1)
	MOVOU	X0, -48(DI)(BX*1)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
clr_129through256:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, 64(DI)
	MOVOU	X0, 80(DI)
	MOVOU	X0, 96(DI)
	MOVOU	X0, 112(DI)
	MOVOU	X0, -128(DI)(BX*1)
	MOVOU	X0, -112(DI)(BX*1)
	MOVOU	X0, -96(DI)(BX*1)
	MOVOU	X0, -80(DI)(BX*1)
	MOVOU	X0, -64(DI)(BX*1)
	MOVOU	X0, -48(DI)(BX*1)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
nosse2:
	MOVL	BX, CX
	SHRL	$2, CX
	REP
	STOSL
	ANDL	$3, BX
	JNE	clr_tail
	RET
