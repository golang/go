// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), BX
	XORL	AX, AX

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
	MOVL	BX, CX
	SHRL	$2, CX
	REP
	STOSL
	ANDL	$3, BX
	JNE	clr_tail
	RET

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
