// Inferno's libkern/memmove-arm.s
// http://code.google.com/p/inferno-os/source/browse/libkern/memmove-arm.s
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

// TE or TS are spilled to the stack during bulk register moves.
#define TS	R0
#define TE	R8

// Warning: the linker will use R11 to synthesize certain instructions. Please
// take care and double check with objdump.
#define FROM	R11
#define N	R12
#define TMP	R12				/* N and TMP don't overlap */
#define TMP1	R5

#define RSHIFT	R5
#define LSHIFT	R6
#define OFFSET	R7

#define BR0	R0					/* shared with TS */
#define BW0	R1
#define BR1	R1
#define BW1	R2
#define BR2	R2
#define BW2	R3
#define BR3	R3
#define BW3	R4

#define FW0	R1
#define FR0	R2
#define FW1	R2
#define FR1	R3
#define FW2	R3
#define FR2	R4
#define FW3	R4
#define FR3	R8					/* shared with TE */

TEXT runtime·memmove(SB), NOSPLIT, $4-12
_memmove:
	MOVW	to+0(FP), TS
	MOVW	from+4(FP), FROM
	MOVW	n+8(FP), N

	ADD	N, TS, TE	/* to end pointer */

	CMP	FROM, TS
	BLS	_forward

_back:
	ADD	N, FROM		/* from end pointer */
	CMP	$4, N		/* need at least 4 bytes to copy */
	BLT	_b1tail

_b4align:				/* align destination on 4 */
	AND.S	$3, TE, TMP
	BEQ	_b4aligned

	MOVBU.W	-1(FROM), TMP	/* pre-indexed */
	MOVBU.W	TMP, -1(TE)	/* pre-indexed */
	B	_b4align

_b4aligned:				/* is source now aligned? */
	AND.S	$3, FROM, TMP
	BNE	_bunaligned

	ADD	$31, TS, TMP	/* do 32-byte chunks if possible */
	MOVW	TS, savedts-4(SP)
_b32loop:
	CMP	TMP, TE
	BLS	_b4tail

	MOVM.DB.W (FROM), [R0-R7]
	MOVM.DB.W [R0-R7], (TE)
	B	_b32loop

_b4tail:				/* do remaining words if possible */
	MOVW	savedts-4(SP), TS
	ADD	$3, TS, TMP
_b4loop:
	CMP	TMP, TE
	BLS	_b1tail

	MOVW.W	-4(FROM), TMP1	/* pre-indexed */
	MOVW.W	TMP1, -4(TE)	/* pre-indexed */
	B	_b4loop

_b1tail:				/* remaining bytes */
	CMP	TE, TS
	BEQ	_return

	MOVBU.W	-1(FROM), TMP	/* pre-indexed */
	MOVBU.W	TMP, -1(TE)	/* pre-indexed */
	B	_b1tail

_forward:
	CMP	$4, N		/* need at least 4 bytes to copy */
	BLT	_f1tail

_f4align:				/* align destination on 4 */
	AND.S	$3, TS, TMP
	BEQ	_f4aligned

	MOVBU.P	1(FROM), TMP	/* implicit write back */
	MOVBU.P	TMP, 1(TS)	/* implicit write back */
	B	_f4align

_f4aligned:				/* is source now aligned? */
	AND.S	$3, FROM, TMP
	BNE	_funaligned

	SUB	$31, TE, TMP	/* do 32-byte chunks if possible */
	MOVW	TE, savedte-4(SP)
_f32loop:
	CMP	TMP, TS
	BHS	_f4tail

	MOVM.IA.W (FROM), [R1-R8] 
	MOVM.IA.W [R1-R8], (TS)
	B	_f32loop

_f4tail:
	MOVW	savedte-4(SP), TE
	SUB	$3, TE, TMP	/* do remaining words if possible */
_f4loop:
	CMP	TMP, TS
	BHS	_f1tail

	MOVW.P	4(FROM), TMP1	/* implicit write back */
	MOVW.P	TMP1, 4(TS)	/* implicit write back */
	B	_f4loop

_f1tail:
	CMP	TS, TE
	BEQ	_return

	MOVBU.P	1(FROM), TMP	/* implicit write back */
	MOVBU.P	TMP, 1(TS)	/* implicit write back */
	B	_f1tail

_return:
	MOVW	to+0(FP), R0
	RET

_bunaligned:
	CMP	$2, TMP		/* is TMP < 2 ? */

	MOVW.LT	$8, RSHIFT		/* (R(n)<<24)|(R(n-1)>>8) */
	MOVW.LT	$24, LSHIFT
	MOVW.LT	$1, OFFSET

	MOVW.EQ	$16, RSHIFT		/* (R(n)<<16)|(R(n-1)>>16) */
	MOVW.EQ	$16, LSHIFT
	MOVW.EQ	$2, OFFSET

	MOVW.GT	$24, RSHIFT		/* (R(n)<<8)|(R(n-1)>>24) */
	MOVW.GT	$8, LSHIFT
	MOVW.GT	$3, OFFSET

	ADD	$16, TS, TMP	/* do 16-byte chunks if possible */
	CMP	TMP, TE
	BLS	_b1tail

	BIC	$3, FROM		/* align source */
	MOVW	TS, savedts-4(SP)
	MOVW	(FROM), BR0	/* prime first block register */

_bu16loop:
	CMP	TMP, TE
	BLS	_bu1tail

	MOVW	BR0<<LSHIFT, BW3
	MOVM.DB.W (FROM), [BR0-BR3]
	ORR	BR3>>RSHIFT, BW3

	MOVW	BR3<<LSHIFT, BW2
	ORR	BR2>>RSHIFT, BW2

	MOVW	BR2<<LSHIFT, BW1
	ORR	BR1>>RSHIFT, BW1

	MOVW	BR1<<LSHIFT, BW0
	ORR	BR0>>RSHIFT, BW0

	MOVM.DB.W [BW0-BW3], (TE)
	B	_bu16loop

_bu1tail:
	MOVW	savedts-4(SP), TS
	ADD	OFFSET, FROM
	B	_b1tail

_funaligned:
	CMP	$2, TMP

	MOVW.LT	$8, RSHIFT		/* (R(n+1)<<24)|(R(n)>>8) */
	MOVW.LT	$24, LSHIFT
	MOVW.LT	$3, OFFSET

	MOVW.EQ	$16, RSHIFT		/* (R(n+1)<<16)|(R(n)>>16) */
	MOVW.EQ	$16, LSHIFT
	MOVW.EQ	$2, OFFSET

	MOVW.GT	$24, RSHIFT		/* (R(n+1)<<8)|(R(n)>>24) */
	MOVW.GT	$8, LSHIFT
	MOVW.GT	$1, OFFSET

	SUB	$16, TE, TMP	/* do 16-byte chunks if possible */
	CMP	TMP, TS
	BHS	_f1tail

	BIC	$3, FROM		/* align source */
	MOVW	TE, savedte-4(SP)
	MOVW.P	4(FROM), FR3	/* prime last block register, implicit write back */

_fu16loop:
	CMP	TMP, TS
	BHS	_fu1tail

	MOVW	FR3>>RSHIFT, FW0
	MOVM.IA.W (FROM), [FR0,FR1,FR2,FR3]
	ORR	FR0<<LSHIFT, FW0

	MOVW	FR0>>RSHIFT, FW1
	ORR	FR1<<LSHIFT, FW1

	MOVW	FR1>>RSHIFT, FW2
	ORR	FR2<<LSHIFT, FW2

	MOVW	FR2>>RSHIFT, FW3
	ORR	FR3<<LSHIFT, FW3

	MOVM.IA.W [FW0,FW1,FW2,FW3], (TS)
	B	_fu16loop

_fu1tail:
	MOVW	savedte-4(SP), TE
	SUB	OFFSET, FROM
	B	_f1tail
