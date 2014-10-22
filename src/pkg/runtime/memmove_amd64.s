// Derived from Inferno's libkern/memmove-386.s (adapted for amd64)
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

// void runtime·memmove(void*, void*, uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $0-24

	MOVQ	to+0(FP), DI
	MOVQ	from+8(FP), SI
	MOVQ	n+16(FP), BX

	// REP instructions have a high startup cost, so we handle small sizes
	// with some straightline code.  The REP MOVSQ instruction is really fast
	// for large sizes.  The cutover is approximately 2K.
tail:
	// move_129through256 or smaller work whether or not the source and the
	// destination memory regions overlap because they load all data into
	// registers before writing it back.  move_256through2048 on the other
	// hand can be used only when the memory regions don't overlap or the copy
	// direction is forward.
	TESTQ	BX, BX
	JEQ	move_0
	CMPQ	BX, $2
	JBE	move_1or2
	CMPQ	BX, $4
	JBE	move_3or4
	CMPQ	BX, $8
	JBE	move_5through8
	CMPQ	BX, $16
	JBE	move_9through16
	CMPQ	BX, $32
	JBE	move_17through32
	CMPQ	BX, $64
	JBE	move_33through64
	CMPQ	BX, $128
	JBE	move_65through128
	CMPQ	BX, $256
	JBE	move_129through256
	// TODO: use branch table and BSR to make this just a single dispatch

/*
 * check and set for backwards
 */
	CMPQ	SI, DI
	JLS	back

/*
 * forward copy loop
 */
forward:
	CMPQ	BX, $2048
	JLS	move_256through2048

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
move_5through8:
	MOVL	(SI), AX
	MOVL	-4(SI)(BX*1), CX
	MOVL	AX, (DI)
	MOVL	CX, -4(DI)(BX*1)
	RET
move_9through16:
	MOVQ	(SI), AX
	MOVQ	-8(SI)(BX*1), CX
	MOVQ	AX, (DI)
	MOVQ	CX, -8(DI)(BX*1)
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
move_129through256:
	MOVOU	(SI), X0
	MOVOU	16(SI), X1
	MOVOU	32(SI), X2
	MOVOU	48(SI), X3
	MOVOU	64(SI), X4
	MOVOU	80(SI), X5
	MOVOU	96(SI), X6
	MOVOU	112(SI), X7
	MOVOU	-128(SI)(BX*1), X8
	MOVOU	-112(SI)(BX*1), X9
	MOVOU	-96(SI)(BX*1), X10
	MOVOU	-80(SI)(BX*1), X11
	MOVOU	-64(SI)(BX*1), X12
	MOVOU	-48(SI)(BX*1), X13
	MOVOU	-32(SI)(BX*1), X14
	MOVOU	-16(SI)(BX*1), X15
	MOVOU	X0, (DI)
	MOVOU	X1, 16(DI)
	MOVOU	X2, 32(DI)
	MOVOU	X3, 48(DI)
	MOVOU	X4, 64(DI)
	MOVOU	X5, 80(DI)
	MOVOU	X6, 96(DI)
	MOVOU	X7, 112(DI)
	MOVOU	X8, -128(DI)(BX*1)
	MOVOU	X9, -112(DI)(BX*1)
	MOVOU	X10, -96(DI)(BX*1)
	MOVOU	X11, -80(DI)(BX*1)
	MOVOU	X12, -64(DI)(BX*1)
	MOVOU	X13, -48(DI)(BX*1)
	MOVOU	X14, -32(DI)(BX*1)
	MOVOU	X15, -16(DI)(BX*1)
	RET
move_256through2048:
	SUBQ	$256, BX
	MOVOU	(SI), X0
	MOVOU	16(SI), X1
	MOVOU	32(SI), X2
	MOVOU	48(SI), X3
	MOVOU	64(SI), X4
	MOVOU	80(SI), X5
	MOVOU	96(SI), X6
	MOVOU	112(SI), X7
	MOVOU	128(SI), X8
	MOVOU	144(SI), X9
	MOVOU	160(SI), X10
	MOVOU	176(SI), X11
	MOVOU	192(SI), X12
	MOVOU	208(SI), X13
	MOVOU	224(SI), X14
	MOVOU	240(SI), X15
	MOVOU	X0, (DI)
	MOVOU	X1, 16(DI)
	MOVOU	X2, 32(DI)
	MOVOU	X3, 48(DI)
	MOVOU	X4, 64(DI)
	MOVOU	X5, 80(DI)
	MOVOU	X6, 96(DI)
	MOVOU	X7, 112(DI)
	MOVOU	X8, 128(DI)
	MOVOU	X9, 144(DI)
	MOVOU	X10, 160(DI)
	MOVOU	X11, 176(DI)
	MOVOU	X12, 192(DI)
	MOVOU	X13, 208(DI)
	MOVOU	X14, 224(DI)
	MOVOU	X15, 240(DI)
	CMPQ	BX, $256
	LEAQ	256(SI), SI
	LEAQ	256(DI), DI
	JGE	move_256through2048
	JMP	tail
