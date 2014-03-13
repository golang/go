// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../cmd/ld/textflag.h"

// void runtime·memclr(void*, uintptr)
TEXT runtime·memclr(SB), NOSPLIT, $0-16
	MOVQ	ptr+0(FP), DI
	MOVQ	n+8(FP), BX
	XORQ	AX, AX

clr_tail:
	TESTQ	BX, BX
	JEQ	clr_0
	CMPQ	BX, $2
	JBE	clr_1or2
	CMPQ	BX, $4
	JBE	clr_3or4
	CMPQ	BX, $8
	JBE	clr_5through8
	CMPQ	BX, $16
	JBE	clr_9through16
	MOVQ	BX, CX
	SHRQ	$2, CX
	REP
	STOSQ
	ANDQ	$3, BX
	JNE	clr_tail
	RET

clr_1or2:
	MOVB	AX, (DI)
	MOVB	AX, -1(DI)(BX*1)
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
	MOVQ	AX, (DI)
	MOVQ	AX, -8(DI)(BX*1)
	RET
