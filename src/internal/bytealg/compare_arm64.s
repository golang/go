// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT|NOFRAME,$0-56
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R0
	MOVD	b_base+24(FP), R3
	MOVD	b_len+32(FP), R1
	MOVD	$ret+48(FP), R7
	B	cmpbody<>(SB)

TEXT bytes·Compare(SB),NOSPLIT|NOFRAME,$0-56
	FUNCDATA $0, ·Compare·args_stackmap(SB)
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R0
	MOVD	b_base+24(FP), R3
	MOVD	b_len+32(FP), R1
	MOVD	$ret+48(FP), R7
	B	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT|NOFRAME,$0-40
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R0
	MOVD	b_base+16(FP), R3
	MOVD	b_len+24(FP), R1
	MOVD	$ret+32(FP), R7
	B	cmpbody<>(SB)

// On entry:
// R0 is the length of a
// R1 is the length of b
// R2 points to the start of a
// R3 points to the start of b
// R7 points to return value (-1/0/1 will be written here)
//
// On exit:
// R4, R5, R6, R8, R9 and R10 are clobbered
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMP	R2, R3
	BEQ	samebytes         // same starting pointers; compare lengths
	CMP	R0, R1
	CSEL	LT, R1, R0, R6    // R6 is min(R0, R1)

	CMP	$0, R6
	BEQ	samebytes
	BIC	$0xf, R6, R10
	CBZ	R10, small        // length < 16
	ADD	R2, R10           // end of chunk16
	// length >= 16
chunk16_loop:
	LDP.P	16(R2), (R4, R8)
	LDP.P	16(R3), (R5, R9)
	CMP	R4, R5
	BNE	cmp
	CMP	R8, R9
	BNE	cmpnext
	CMP	R10, R2
	BNE	chunk16_loop
	AND	$0xf, R6, R6
	CBZ	R6, samebytes
	SUBS	$8, R6
	BLT	tail
	// the length of tail > 8 bytes
	MOVD.P	8(R2), R4
	MOVD.P	8(R3), R5
	CMP	R4, R5
	BNE	cmp
	SUB	$8, R6
	// compare last 8 bytes
tail:
	MOVD	(R2)(R6), R4
	MOVD	(R3)(R6), R5
	CMP	R4, R5
	BEQ	samebytes
cmp:
	REV	R4, R4
	REV	R5, R5
	CMP	R4, R5
ret:
	MOVD	$1, R4
	CNEG	HI, R4, R4
	MOVD	R4, (R7)
	RET
small:
	TBZ	$3, R6, lt_8
	MOVD	(R2), R4
	MOVD	(R3), R5
	CMP	R4, R5
	BNE	cmp
	SUBS	$8, R6
	BEQ	samebytes
	ADD	$8, R2
	ADD	$8, R3
	SUB	$8, R6
	B	tail
lt_8:
	TBZ	$2, R6, lt_4
	MOVWU	(R2), R4
	MOVWU	(R3), R5
	CMPW	R4, R5
	BNE	cmp
	SUBS	$4, R6
	BEQ	samebytes
	ADD	$4, R2
	ADD	$4, R3
lt_4:
	TBZ	$1, R6, lt_2
	MOVHU	(R2), R4
	MOVHU	(R3), R5
	CMPW	R4, R5
	BNE	cmp
	ADD	$2, R2
	ADD	$2, R3
lt_2:
	TBZ	$0, R6, samebytes
one:
	MOVBU	(R2), R4
	MOVBU	(R3), R5
	CMPW	R4, R5
	BNE	ret
samebytes:
	CMP	R1, R0
	CSET	NE, R4
	CNEG	LO, R4, R4
	MOVD	R4, (R7)
	RET
cmpnext:
	REV	R8, R4
	REV	R9, R5
	CMP	R4, R5
	B	ret
