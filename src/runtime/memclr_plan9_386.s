// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), BX
	XORL	AX, AX

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
