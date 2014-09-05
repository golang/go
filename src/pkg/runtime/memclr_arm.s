// Inferno's libkern/memset-arm.s
// http://code.google.com/p/inferno-os/source/browse/libkern/memset-arm.s
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

#include "textflag.h"

TO = 8
TOE = 11
N = 12
TMP = 12				/* N and TMP don't overlap */

TEXT runtime·memclr(SB),NOSPLIT,$0-8
	MOVW	ptr+0(FP), R(TO)
	MOVW	n+4(FP), R(N)
	MOVW	$0, R(0)

	ADD	R(N), R(TO), R(TOE)	/* to end pointer */

	CMP	$4, R(N)		/* need at least 4 bytes to copy */
	BLT	_1tail

_4align:				/* align on 4 */
	AND.S	$3, R(TO), R(TMP)
	BEQ	_4aligned

	MOVBU.P	R(0), 1(R(TO))		/* implicit write back */
	B	_4align

_4aligned:
	SUB	$31, R(TOE), R(TMP)	/* do 32-byte chunks if possible */
	CMP	R(TMP), R(TO)
	BHS	_4tail

	MOVW	R0, R1			/* replicate */
	MOVW	R0, R2
	MOVW	R0, R3
	MOVW	R0, R4
	MOVW	R0, R5
	MOVW	R0, R6
	MOVW	R0, R7

_f32loop:
	CMP	R(TMP), R(TO)
	BHS	_4tail

	MOVM.IA.W [R0-R7], (R(TO))
	B	_f32loop

_4tail:
	SUB	$3, R(TOE), R(TMP)	/* do remaining words if possible */
_4loop:
	CMP	R(TMP), R(TO)
	BHS	_1tail

	MOVW.P	R(0), 4(R(TO))		/* implicit write back */
	B	_4loop

_1tail:
	CMP	R(TO), R(TOE)
	BEQ	_return

	MOVBU.P	R(0), 1(R(TO))		/* implicit write back */
	B	_1tail

_return:
	RET
