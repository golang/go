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

	TEXT	memmove(SB), 7, $0

	MOVQ	to+0(FP), DI
	MOVQ	fr+8(FP), SI
	MOVLQSX	n+16(FP), BX
	CMPQ	BX, $0
	JLT	fault

/*
 * check and set for backwards
 * should we look closer for overlap?
 */
	CMPQ	SI, DI
	JLS	back

/*
 * forward copy loop
 */
	MOVQ	BX, CX
	SHRQ	$3, CX
	ANDQ	$7, BX

	REP;	MOVSQ
	MOVQ	BX, CX
	REP;	MOVSB

	MOVQ	to+0(FP),AX
	RET
/*
 * whole thing backwards has
 * adjusted addresses
 */
back:
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

	ADDQ	$7, DI
	ADDQ	$7, SI
	MOVQ	BX, CX
	REP;	MOVSB

	CLD
	MOVQ	to+0(FP),AX
	RET

/*
 * if called with negative count,
 * treat as error rather than
 * rotating all of memory
 */
fault:
	MOVQ	$0,SI
	MOVQ	0(SI), AX
	RET
