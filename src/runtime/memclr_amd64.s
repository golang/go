// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

#include "textflag.h"

// NOTE: Windows externalthreadhandler expects memclr to preserve DX.

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), DI
	MOVQ	n+8(FP), BX
	XORQ	AX, AX

	// MOVOU seems always faster than REP STOSQ.
tail:
	TESTQ	BX, BX
	JEQ	_0
	CMPQ	BX, $2
	JBE	_1or2
	CMPQ	BX, $4
	JBE	_3or4
	CMPQ	BX, $8
	JBE	_5through8
	CMPQ	BX, $16
	JBE	_9through16
	PXOR	X0, X0
	CMPQ	BX, $32
	JBE	_17through32
	CMPQ	BX, $64
	JBE	_33through64
	CMPQ	BX, $128
	JBE	_65through128
	CMPQ	BX, $256
	JBE	_129through256
	// TODO: use branch table and BSR to make this just a single dispatch
	// TODO: for really big clears, use MOVNTDQ.

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
	SUBQ	$256, BX
	ADDQ	$256, DI
	CMPQ	BX, $256
	JAE	loop
	JMP	tail

_1or2:
	MOVB	AX, (DI)
	MOVB	AX, -1(DI)(BX*1)
	RET
_0:
	RET
_3or4:
	MOVW	AX, (DI)
	MOVW	AX, -2(DI)(BX*1)
	RET
_5through8:
	MOVL	AX, (DI)
	MOVL	AX, -4(DI)(BX*1)
	RET
_9through16:
	MOVQ	AX, (DI)
	MOVQ	AX, -8(DI)(BX*1)
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
