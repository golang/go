// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT,$0-28
	MOVL	a_base+0(FP), SI
	MOVL	a_len+4(FP), BX
	MOVL	b_base+12(FP), DI
	MOVL	b_len+16(FP), DX
	LEAL	ret+24(FP), AX
	JMP	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT,$0-20
	MOVL	a_base+0(FP), SI
	MOVL	a_len+4(FP), BX
	MOVL	b_base+8(FP), DI
	MOVL	b_len+12(FP), DX
	LEAL	ret+16(FP), AX
	JMP	cmpbody<>(SB)

// input:
//   SI = a
//   DI = b
//   BX = alen
//   DX = blen
//   AX = address of return word (set to 1/0/-1)
TEXT cmpbody<>(SB),NOSPLIT,$0-0
	MOVL	DX, BP
	SUBL	BX, DX // DX = blen-alen
	JLE	2(PC)
	MOVL	BX, BP // BP = min(alen, blen)
	CMPL	SI, DI
	JEQ	allsame
	CMPL	BP, $4
	JB	small
#ifdef GO386_softfloat
	JMP	mediumloop
#endif
largeloop:
	CMPL	BP, $16
	JB	mediumloop
	MOVOU	(SI), X0
	MOVOU	(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, BX
	XORL	$0xffff, BX	// convert EQ to NE
	JNE	diff16	// branch if at least one byte is not equal
	ADDL	$16, SI
	ADDL	$16, DI
	SUBL	$16, BP
	JMP	largeloop

diff16:
	BSFL	BX, BX	// index of first byte that differs
	XORL	DX, DX
	MOVB	(SI)(BX*1), CX
	CMPB	CX, (DI)(BX*1)
	SETHI	DX
	LEAL	-1(DX*2), DX	// convert 1/0 to +1/-1
	MOVL	DX, (AX)
	RET

mediumloop:
	CMPL	BP, $4
	JBE	_0through4
	MOVL	(SI), BX
	MOVL	(DI), CX
	CMPL	BX, CX
	JNE	diff4
	ADDL	$4, SI
	ADDL	$4, DI
	SUBL	$4, BP
	JMP	mediumloop

_0through4:
	MOVL	-4(SI)(BP*1), BX
	MOVL	-4(DI)(BP*1), CX
	CMPL	BX, CX
	JEQ	allsame

diff4:
	BSWAPL	BX	// reverse order of bytes
	BSWAPL	CX
	XORL	BX, CX	// find bit differences
	BSRL	CX, CX	// index of highest bit difference
	SHRL	CX, BX	// move a's bit to bottom
	ANDL	$1, BX	// mask bit
	LEAL	-1(BX*2), BX // 1/0 => +1/-1
	MOVL	BX, (AX)
	RET

	// 0-3 bytes in common
small:
	LEAL	(BP*8), CX
	NEGL	CX
	JEQ	allsame

	// load si
	CMPB	SI, $0xfc
	JA	si_high
	MOVL	(SI), SI
	JMP	si_finish
si_high:
	MOVL	-4(SI)(BP*1), SI
	SHRL	CX, SI
si_finish:
	SHLL	CX, SI

	// same for di
	CMPB	DI, $0xfc
	JA	di_high
	MOVL	(DI), DI
	JMP	di_finish
di_high:
	MOVL	-4(DI)(BP*1), DI
	SHRL	CX, DI
di_finish:
	SHLL	CX, DI

	BSWAPL	SI	// reverse order of bytes
	BSWAPL	DI
	XORL	SI, DI	// find bit differences
	JEQ	allsame
	BSRL	DI, CX	// index of highest bit difference
	SHRL	CX, SI	// move a's bit to bottom
	ANDL	$1, SI	// mask bit
	LEAL	-1(SI*2), BX // 1/0 => +1/-1
	MOVL	BX, (AX)
	RET

	// all the bytes in common are the same, so we just need
	// to compare the lengths.
allsame:
	XORL	BX, BX
	XORL	CX, CX
	TESTL	DX, DX
	SETLT	BX	// 1 if alen > blen
	SETEQ	CX	// 1 if alen == blen
	LEAL	-1(CX)(BX*2), BX	// 1,0,-1 result
	MOVL	BX, (AX)
	RET
