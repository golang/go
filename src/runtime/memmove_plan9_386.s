// Inferno's libkern/memmove-386.s
// https://bitbucket.org/inferno-os/inferno-os/src/master/libkern/memmove-386.s
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

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $0-12
	MOVL	to+0(FP), DI
	MOVL	from+4(FP), SI
	MOVL	n+8(FP), BX

	// REP instructions have a high startup cost, so we handle small sizes
	// with some straightline code. The REP MOVSL instruction is really fast
	// for large sizes. The cutover is approximately 1K.
tail:
	TESTL	BX, BX
	JEQ	move_0
	CMPL	BX, $2
	JBE	move_1or2
	CMPL	BX, $4
	JB	move_3
	JE	move_4
	CMPL	BX, $8
	JBE	move_5through8
	CMPL	BX, $16
	JBE	move_9through16

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
move_3:
	MOVW	(SI), AX
	MOVB	2(SI), CX
	MOVW	AX, (DI)
	MOVB	CX, 2(DI)
	RET
move_4:
	// We need a separate case for 4 to make sure we write pointers atomically.
	MOVL	(SI), AX
	MOVL	AX, (DI)
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
