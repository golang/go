// Inferno's libkern/memmove-arm.s
// http://code.google.com/p/inferno-os/source/browse/libkern/memmove-arm.s
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

TS = 0
TE = 1
FROM = 2
N = 3
TMP = 3					/* N and TMP don't overlap */
TMP1 = 4

TEXT memmove(SB), 7, $-4
_memmove:
	MOVW	R(TS), to+0(FP)		/* need to save for return value */
	MOVW	from+4(FP), R(FROM)
	MOVW	n+8(FP), R(N)

	ADD	R(N), R(TS), R(TE)	/* to end pointer */

	CMP	R(FROM), R(TS)
	BLS	_forward

_back:
	ADD	R(N), R(FROM)		/* from end pointer */
	CMP	$4, R(N)		/* need at least 4 bytes to copy */
	BLT	_b1tail

_b4align:				/* align destination on 4 */
	AND.S	$3, R(TE), R(TMP)
	BEQ	_b4aligned

	MOVBU.W	-1(R(FROM)), R(TMP)	/* pre-indexed */
	MOVBU.W	R(TMP), -1(R(TE))	/* pre-indexed */
	B	_b4align

_b4aligned:				/* is source now aligned? */
	AND.S	$3, R(FROM), R(TMP)
	BNE	_bunaligned

	ADD	$31, R(TS), R(TMP)	/* do 32-byte chunks if possible */
_b32loop:
	CMP	R(TMP), R(TE)
	BLS	_b4tail

	MOVM.DB.W (R(FROM)), [R4-R11]
	MOVM.DB.W [R4-R11], (R(TE))
	B	_b32loop

_b4tail:				/* do remaining words if possible */
	ADD	$3, R(TS), R(TMP)
_b4loop:
	CMP	R(TMP), R(TE)
	BLS	_b1tail

	MOVW.W	-4(R(FROM)), R(TMP1)	/* pre-indexed */
	MOVW.W	R(TMP1), -4(R(TE))	/* pre-indexed */
	B	_b4loop

_b1tail:				/* remaining bytes */
	CMP	R(TE), R(TS)
	BEQ	_return

	MOVBU.W	-1(R(FROM)), R(TMP)	/* pre-indexed */
	MOVBU.W	R(TMP), -1(R(TE))	/* pre-indexed */
	B	_b1tail

_forward:
	CMP	$4, R(N)		/* need at least 4 bytes to copy */
	BLT	_f1tail

_f4align:				/* align destination on 4 */
	AND.S	$3, R(TS), R(TMP)
	BEQ	_f4aligned

	MOVBU.P	1(R(FROM)), R(TMP)	/* implicit write back */
	MOVBU.P	R(TMP), 1(R(TS))	/* implicit write back */
	B	_f4align

_f4aligned:				/* is source now aligned? */
	AND.S	$3, R(FROM), R(TMP)
	BNE	_funaligned

	SUB	$31, R(TE), R(TMP)	/* do 32-byte chunks if possible */
_f32loop:
	CMP	R(TMP), R(TS)
	BHS	_f4tail

	MOVM.IA.W (R(FROM)), [R4-R11] 
	MOVM.IA.W [R4-R11], (R(TS))
	B	_f32loop

_f4tail:
	SUB	$3, R(TE), R(TMP)	/* do remaining words if possible */
_f4loop:
	CMP	R(TMP), R(TS)
	BHS	_f1tail

	MOVW.P	4(R(FROM)), R(TMP1)	/* implicit write back */
	MOVW.P	R4, 4(R(TS))		/* implicit write back */
	B	_f4loop

_f1tail:
	CMP	R(TS), R(TE)
	BEQ	_return

	MOVBU.P	1(R(FROM)), R(TMP)	/* implicit write back */
	MOVBU.P	R(TMP), 1(R(TS))	/* implicit write back */
	B	_f1tail

_return:
	MOVW	to+0(FP), R0
	RET

RSHIFT = 4
LSHIFT = 5
OFFSET = 6

BR0 = 7
BW0 = 8
BR1 = 8
BW1 = 9
BR2 = 9
BW2 = 10
BR3 = 10
BW3 = 11

_bunaligned:
	CMP	$2, R(TMP)		/* is R(TMP) < 2 ? */

	MOVW.LT	$8, R(RSHIFT)		/* (R(n)<<24)|(R(n-1)>>8) */
	MOVW.LT	$24, R(LSHIFT)
	MOVW.LT	$1, R(OFFSET)

	MOVW.EQ	$16, R(RSHIFT)		/* (R(n)<<16)|(R(n-1)>>16) */
	MOVW.EQ	$16, R(LSHIFT)
	MOVW.EQ	$2, R(OFFSET)

	MOVW.GT	$24, R(RSHIFT)		/* (R(n)<<8)|(R(n-1)>>24) */
	MOVW.GT	$8, R(LSHIFT)
	MOVW.GT	$3, R(OFFSET)

	ADD	$16, R(TS), R(TMP)	/* do 16-byte chunks if possible */
	CMP	R(TMP), R(TE)
	BLS	_b1tail

	AND	$~0x03, R(FROM)		/* align source */
	MOVW	(R(FROM)), R(BR0)	/* prime first block register */

_bu16loop:
	CMP	R(TMP), R(TE)
	BLS	_bu1tail

	MOVW	R(BR0)<<R(LSHIFT), R(BW3)
	MOVM.DB.W (R(FROM)), [R(BR0)-R(BR3)]
	ORR	R(BR3)>>R(RSHIFT), R(BW3)

	MOVW	R(BR3)<<R(LSHIFT), R(BW2)
	ORR	R(BR2)>>R(RSHIFT), R(BW2)

	MOVW	R(BR2)<<R(LSHIFT), R(BW1)
	ORR	R(BR1)>>R(RSHIFT), R(BW1)

	MOVW	R(BR1)<<R(LSHIFT), R(BW0)
	ORR	R(BR0)>>R(RSHIFT), R(BW0)

	MOVM.DB.W [R(BW0)-R(BW3)], (R(TE))
	B	_bu16loop

_bu1tail:
	ADD	R(OFFSET), R(FROM)
	B	_b1tail

FW0 = 7
FR0 = 8
FW1 = 8
FR1 = 9
FW2 = 9
FR2 = 10
FW3 = 10
FR3 = 11

_funaligned:
	CMP	$2, R(TMP)

	MOVW.LT	$8, R(RSHIFT)		/* (R(n+1)<<24)|(R(n)>>8) */
	MOVW.LT	$24, R(LSHIFT)
	MOVW.LT	$3, R(OFFSET)

	MOVW.EQ	$16, R(RSHIFT)		/* (R(n+1)<<16)|(R(n)>>16) */
	MOVW.EQ	$16, R(LSHIFT)
	MOVW.EQ	$2, R(OFFSET)

	MOVW.GT	$24, R(RSHIFT)		/* (R(n+1)<<8)|(R(n)>>24) */
	MOVW.GT	$8, R(LSHIFT)
	MOVW.GT	$1, R(OFFSET)

	SUB	$16, R(TE), R(TMP)	/* do 16-byte chunks if possible */
	CMP	R(TMP), R(TS)
	BHS	_f1tail

	AND	$~0x03, R(FROM)		/* align source */
	MOVW.P	4(R(FROM)), R(FR3)	/* prime last block register, implicit write back */

_fu16loop:
	CMP	R(TMP), R(TS)
	BHS	_fu1tail

	MOVW	R(FR3)>>R(RSHIFT), R(FW0)
	MOVM.IA.W (R(FROM)), [R(FR0)-R(FR3)]
	ORR	R(FR0)<<R(LSHIFT), R(FW0)

	MOVW	R(FR0)>>R(RSHIFT), R(FW1)
	ORR	R(FR1)<<R(LSHIFT), R(FW1)

	MOVW	R(FR1)>>R(RSHIFT), R(FW2)
	ORR	R(FR2)<<R(LSHIFT), R(FW2)

	MOVW	R(FR2)>>R(RSHIFT), R(FW3)
	ORR	R(FR3)<<R(LSHIFT), R(FW3)

	MOVM.IA.W [R(FW0)-R(FW3)], (R(TS))
	B	_fu16loop

_fu1tail:
	SUB	R(OFFSET), R(FROM)
	B	_f1tail
