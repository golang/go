// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), BX
	XORL	AX, AX

tail:
	TESTL	BX, BX
	JEQ	_0
	CMPL	BX, $2
	JBE	_1or2
	CMPL	BX, $4
	JBE	_3or4
	CMPL	BX, $8
	JBE	_5through8
	CMPL	BX, $16
	JBE	_9through16
	MOVL	BX, CX
	SHRL	$2, CX
	REP
	STOSL
	ANDL	$3, BX
	JNE	tail
	RET

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
	MOVL	AX, (DI)
	MOVL	AX, 4(DI)
	MOVL	AX, -8(DI)(BX*1)
	MOVL	AX, -4(DI)(BX*1)
	RET
