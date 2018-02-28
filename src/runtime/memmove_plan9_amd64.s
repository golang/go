// Derived from Inferno's libkern/memmove-386.s (adapted for amd64)
// https://bitbucket.org/inferno-os/inferno-os/src/default/libkern/memmove-386.s
//
//         Copyright © 1994-1999 Lucent Technologies Inc. All rights reserved.
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

#include "textflag.h"

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $0-24

	MOVQ	to+0(FP), DI
	MOVQ	from+8(FP), SI
	MOVQ	n+16(FP), BX

	// REP instructions have a high startup cost, so we handle small sizes
	// with some straightline code. The REP MOVSQ instruction is really fast
	// for large sizes. The cutover is approximately 1K.
tail:
	TESTQ	BX, BX
	JEQ	move_0
	CMPQ	BX, $2
	JBE	move_1or2
	CMPQ	BX, $4
	JBE	move_3or4
	CMPQ	BX, $8
	JB	move_5through7
	JE	move_8
	CMPQ	BX, $16
	JBE	move_9through16

/*
 * check and set for backwards
 */
	CMPQ	SI, DI
	JLS	back

/*
 * forward copy loop
 */
forward:
	MOVQ	BX, CX
	SHRQ	$3, CX
	ANDQ	$7, BX

	REP;	MOVSQ
	JMP	tail

back:
/*
 * check overlap
 */
	MOVQ	SI, CX
	ADDQ	BX, CX
	CMPQ	CX, DI
	JLS	forward
	
/*
 * whole thing backwards has
 * adjusted addresses
 */
	ADDQ	BX, DI
	ADDQ	BX, SI
	STD

/*
 * copy
 */
	MOVQ	BX, CX
	SHRQ	$3, CX
	ANDQ	$7, BX

	SUBQ	$8, DI
	SUBQ	$8, SI
	REP;	MOVSQ

	CLD
	ADDQ	$8, DI
	ADDQ	$8, SI
	SUBQ	BX, DI
	SUBQ	BX, SI
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
move_5through7:
	MOVL	(SI), AX
	MOVL	-4(SI)(BX*1), CX
	MOVL	AX, (DI)
	MOVL	CX, -4(DI)(BX*1)
	RET
move_8:
	// We need a separate case for 8 to make sure we write pointers atomically.
	MOVQ	(SI), AX
	MOVQ	AX, (DI)
	RET
move_9through16:
	MOVQ	(SI), AX
	MOVQ	-8(SI)(BX*1), CX
	MOVQ	AX, (DI)
	MOVQ	CX, -8(DI)(BX*1)
	RET
