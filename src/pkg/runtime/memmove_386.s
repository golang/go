// Inferno's libkern/memmove-386.s
// http://code.google.com/p/inferno-os/source/browse/libkern/memmove-386.s
//
//         Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//         Revisions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).  All rights reserved.
//         Portions Copyright 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// +build !plan9

#include "textflag.h"

TEXT runtime·memmove(SB), NOSPLIT, $0-12
	MOVL	to+0(FP), DI
	MOVL	from+4(FP), SI
	MOVL	n+8(FP), BX

	// REP instructions have a high startup cost, so we handle small sizes
	// with some straightline code.  The REP MOVSL instruction is really fast
	// for large sizes.  The cutover is approximately 1K.  We implement up to
	// 128 because that is the maximum SSE register load (loading all data
	// into registers lets us ignore copy direction).
tail:
	TESTL	BX, BX
	JEQ	move_0
	CMPL	BX, $2
	JBE	move_1or2
	CMPL	BX, $4
	JBE	move_3or4
	CMPL	BX, $8
	JBE	move_5through8
	CMPL	BX, $16
	JBE	move_9through16
	TESTL	$0x4000000, runtime·cpuid_edx(SB) // check for sse2
	JEQ	nosse2
	CMPL	BX, $32
	JBE	move_17through32
	CMPL	BX, $64
	JBE	move_33through64
	CMPL	BX, $128
	JBE	move_65through128
	// TODO: use branch table and BSR to make this just a single dispatch

nosse2:
/*
 * check and set for backwards
 */
	CMPL	SI, DI
	JLS	back

/*
 * forward copy loop
 */
forward:	
	MOVL	BX, CX
	SHRL	$2, CX
	ANDL	$3, BX

	REP;	MOVSL
	JMP	tail
/*
 * check overlap
 */
back:
	MOVL	SI, CX
	ADDL	BX, CX
	CMPL	CX, DI
	JLS	forward
/*
 * whole thing backwards has
 * adjusted addresses
 */

	ADDL	BX, DI
	ADDL	BX, SI
	STD

/*
 * copy
 */
	MOVL	BX, CX
	SHRL	$2, CX
	ANDL	$3, BX

	SUBL	$4, DI
	SUBL	$4, SI
	REP;	MOVSL

	CLD
	ADDL	$4, DI
	ADDL	$4, SI
	SUBL	BX, DI
	SUBL	BX, SI
	JMP	tail

move_1or2:
	MOVB	(SI), AX
	MOVB	-1(SI)(BX*1), CX
	MOVB	AX, (DI)
	MOVB	CX, -1(DI)(BX*1)
	RET
move_0:
	RET
move_3or4:
	MOVW	(SI), AX
	MOVW	-2(SI)(BX*1), CX
	MOVW	AX, (DI)
	MOVW	CX, -2(DI)(BX*1)
	RET
move_5through8:
	MOVL	(SI), AX
	MOVL	-4(SI)(BX*1), CX
	MOVL	AX, (DI)
	MOVL	CX, -4(DI)(BX*1)
	RET
move_9through16:
	MOVL	(SI), AX
	MOVL	4(SI), CX
	MOVL	-8(SI)(BX*1), DX
	MOVL	-4(SI)(BX*1), BP
	MOVL	AX, (DI)
	MOVL	CX, 4(DI)
	MOVL	DX, -8(DI)(BX*1)
	MOVL	BP, -4(DI)(BX*1)
	RET
move_17through32:
	MOVOU	(SI), X0
	MOVOU	-16(SI)(BX*1), X1
	MOVOU	X0, (DI)
	MOVOU	X1, -16(DI)(BX*1)
	RET
move_33through64:
	MOVOU	(SI), X0
	MOVOU	16(SI), X1
	MOVOU	-32(SI)(BX*1), X2
	MOVOU	-16(SI)(BX*1), X3
	MOVOU	X0, (DI)
	MOVOU	X1, 16(DI)
	MOVOU	X2, -32(DI)(BX*1)
	MOVOU	X3, -16(DI)(BX*1)
	RET
move_65through128:
	MOVOU	(SI), X0
	MOVOU	16(SI), X1
	MOVOU	32(SI), X2
	MOVOU	48(SI), X3
	MOVOU	-64(SI)(BX*1), X4
	MOVOU	-48(SI)(BX*1), X5
	MOVOU	-32(SI)(BX*1), X6
	MOVOU	-16(SI)(BX*1), X7
	MOVOU	X0, (DI)
	MOVOU	X1, 16(DI)
	MOVOU	X2, 32(DI)
	MOVOU	X3, 48(DI)
	MOVOU	X4, -64(DI)(BX*1)
	MOVOU	X5, -48(DI)(BX*1)
	MOVOU	X6, -32(DI)(BX*1)
	MOVOU	X7, -16(DI)(BX*1)
	RET
