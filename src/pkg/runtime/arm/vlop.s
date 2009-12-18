// Inferno's libkern/vlop-arm.s
// http://code.google.com/p/inferno-os/source/browse/libkern/vlop-arm.s
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

#define UMULL(Rs,Rm,Rhi,Rlo,S)  WORD	 $((14<<28)|(4<<21)|(S<<20)|(Rhi<<16)|(Rlo<<12)|(Rs<<8)|(9<<4)|Rm)
#define UMLAL(Rs,Rm,Rhi,Rlo,S)  WORD	 $((14<<28)|(5<<21)|(S<<20)|(Rhi<<16)|(Rlo<<12)|(Rs<<8)|(9<<4)|Rm)
#define MUL(Rs,Rm,Rd,S) WORD	 $((14<<28)|(0<<21)|(S<<20)|(Rd<<16)|(Rs<<8)|(9<<4)|Rm)
arg=0

/* replaced use of R10 by R11 because the former can be the data segment base register */

TEXT	_mulv(SB), $0
	MOVW	0(FP), R0
	MOVW	8(FP), R2		/* l0 */
	MOVW	4(FP), R3	  /* h0 */
	MOVW	16(FP), R4	  /* l1 */
	MOVW	12(FP), R5	  /* h1 */
	UMULL(4, 2, 7, 6, 0)
	MUL(3, 4, 8, 0)
	ADD	R8, R7
	MUL(2, 5, 8, 0)
	ADD	R8, R7
	MOVW	R6, 4(R(arg))
	MOVW	R7, 0(R(arg))
	RET


Q	= 0
N	= 1
D	= 2
CC	= 3
TMP	= 11

TEXT	save<>(SB), 7, $0
	MOVW	R(Q), 0(FP)
	MOVW	R(N), 4(FP)
	MOVW	R(D), 8(FP)
	MOVW	R(CC), 12(FP)

	MOVW	R(TMP), R(Q)		/* numerator */
	MOVW	20(FP), R(D)		/* denominator */
	CMP	$0, R(D)
	BNE	s1
	SWI		 0
/*	  MOVW	-1(R(D)), R(TMP)	/* divide by zero fault */
s1:	 RET

TEXT	rest<>(SB), 7, $0
	MOVW	0(FP), R(Q)
	MOVW	4(FP), R(N)
	MOVW	8(FP), R(D)
	MOVW	12(FP), R(CC)
/*
 * return to caller
 * of rest<>
 */
	MOVW	0(R13), R14
	ADD	$20, R13
	B	(R14)

TEXT	div<>(SB), 7, $0
	MOVW	$32, R(CC)
/*
 * skip zeros 8-at-a-time
 */
e1:
	AND.S	$(0xff<<24),R(Q), R(N)
	BNE	e2
	SLL	$8, R(Q)
	SUB.S	$8, R(CC)
	BNE	e1
	RET
e2:
	MOVW	$0, R(N)

loop:
/*
 * shift R(N||Q) left one
 */
	SLL	$1, R(N)
	CMP	$0, R(Q)
	ORR.LT  $1, R(N)
	SLL	$1, R(Q)

/*
 * compare numerator to denominator
 * if less, subtract and set quotent bit
 */
	CMP	R(D), R(N)
	ORR.HS  $1, R(Q)
	SUB.HS  R(D), R(N)
	SUB.S	$1, R(CC)
	BNE	loop
	RET

TEXT	_div(SB), 7, $16
	BL	save<>(SB)
	CMP	$0, R(Q)
	BGE	d1
	RSB	$0, R(Q), R(Q)
	CMP	$0, R(D)
	BGE	d2
	RSB	$0, R(D), R(D)
d0:
	BL	div<>(SB)			/* none/both neg */
	MOVW	R(Q), R(TMP)
	B	out
d1:
	CMP	$0, R(D)
	BGE	d0
	RSB	$0, R(D), R(D)
d2:
	BL	div<>(SB)			/* one neg */
	RSB	$0, R(Q), R(TMP)
	B	out

TEXT	_mod(SB), 7, $16
	BL	save<>(SB)
	CMP	$0, R(D)
	RSB.LT	$0, R(D), R(D)
	CMP	$0, R(Q)
	BGE	m1
	RSB	$0, R(Q), R(Q)
	BL	div<>(SB)			/* neg numerator */
	RSB	$0, R(N), R(TMP)
	B	out
m1:
	BL	div<>(SB)			/* pos numerator */
	MOVW	R(N), R(TMP)
	B	out

TEXT	_divu(SB), 7, $16
	BL	save<>(SB)
	BL	div<>(SB)
	MOVW	R(Q), R(TMP)
	B	out

TEXT	_modu(SB), 7, $16
	BL	save<>(SB)
	BL	div<>(SB)
	MOVW	R(N), R(TMP)
	B	out

out:
	BL	rest<>(SB)
	B	out

// trampoline for _sfloat2. passes LR as arg0 and
// saves registers R0-R11 on the stack for mutation
// by _sfloat2
TEXT	_sfloat(SB), 7, $52 // 4 arg + 12*4 saved regs
	MOVW	R14, 4(R13)
	MOVW	R0, 8(R13)
	MOVW	$12(R13), R0
	MOVM.IA.W	[R1-R11], (R0)
	BL	_sfloat2(SB)
	MOVW	R0, 0(R13)
	MOVW	$12(R13), R0
	MOVM.IA.W	(R0), [R1-R11]
	MOVW	8(R13), R0
	RET
			

