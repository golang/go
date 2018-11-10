// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

#include "textflag.h"

// NOTE: Windows externalthreadhandler expects memclr to preserve DX.

// void runtime·memclrNoHeapPointers(void*, uintptr)
TEXT runtime·memclrNoHeapPointers(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), BX
	XORL	AX, AX

	// MOVOU seems always faster than REP STOSL.
tail:
	TESTL	BX, BX
	JEQ	_0
	CMPL	BX, $2
	JBE	_1or2
	CMPL	BX, $4
	JB	_3
	JE	_4
	CMPL	BX, $8
	JBE	_5through8
	CMPL	BX, $16
	JBE	_9through16
	CMPB	runtime·support_sse2(SB), $1
	JNE	nosse2
	PXOR	X0, X0
	CMPL	BX, $32
	JBE	_17through32
	CMPL	BX, $64
	JBE	_33through64
	CMPL	BX, $128
	JBE	_65through128
	CMPL	BX, $256
	JBE	_129through256
	// TODO: use branch table and BSR to make this just a single dispatch

loop:
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
	JAE	loop
	JMP	tail

_1or2:
	MOVB	AX, (DI)
	MOVB	AX, -1(DI)(BX*1)
	RET
_0:
	RET
_3:
	MOVW	AX, (DI)
	MOVB	AX, 2(DI)
	RET
_4:
	// We need a separate case for 4 to make sure we clear pointers atomically.
	MOVL	AX, (DI)
	RET
_5through8:
	MOVL	AX, (DI)
	MOVL	AX, -4(DI)(BX*1)
	RET
_9through16:
	MOVL	AX, (DI)
	MOVL	AX, 4(DI)
	MOVL	AX, -8(DI)(BX*1)
	MOVL	AX, -4(DI)(BX*1)
	RET
_17through32:
	MOVOU	X0, (DI)
	MOVOU	X0, -16(DI)(BX*1)
	RET
_33through64:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
_65through128:
	MOVOU	X0, (DI)
	MOVOU	X0, 16(DI)
	MOVOU	X0, 32(DI)
	MOVOU	X0, 48(DI)
	MOVOU	X0, -64(DI)(BX*1)
	MOVOU	X0, -48(DI)(BX*1)
	MOVOU	X0, -32(DI)(BX*1)
	MOVOU	X0, -16(DI)(BX*1)
	RET
_129through256:
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
	JNE	tail
	RET
