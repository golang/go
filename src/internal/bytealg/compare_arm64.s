// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
#ifdef GOEXPERIMENT_regabiargs
	// R0 = a_base (want in R0)
	// R1 = a_len  (want in R1)
	// R2 = a_cap  (unused)
	// R3 = b_base (want in R2)
	// R4 = b_len  (want in R3)
	// R5 = b_cap  (unused)
	MOVD	R3, R2
	MOVD	R4, R3
#else
	MOVD	a_base+0(FP), R0
	MOVD	a_len+8(FP), R1
	MOVD	b_base+24(FP), R2
	MOVD	b_len+32(FP), R3
	MOVD	$ret+48(FP), R7
#endif
	B	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
#ifdef GOEXPERIMENT_regabiargs
	// R0 = a_base
	// R1 = a_len
	// R2 = b_base
	// R3 = b_len
#else
	MOVD	a_base+0(FP), R0
	MOVD	a_len+8(FP), R1
	MOVD	b_base+16(FP), R2
	MOVD	b_len+24(FP), R3
	MOVD	$ret+32(FP), R7
#endif
	B	cmpbody<>(SB)

// On entry:
// R0 points to the start of a
// R1 is the length of a
// R2 points to the start of b
// R3 is the length of b
#ifndef GOEXPERIMENT_regabiargs
// R7 points to return value (-1/0/1 will be written here)
#endif
//
// On exit:
#ifdef GOEXPERIMENT_regabiargs
// R0 is the result
#endif
// R4, R5, R6, R8, R9 and R10 are clobbered
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMP	R0, R2
	BEQ	samebytes         // same starting pointers; compare lengths
	CMP	R1, R3
	CSEL	LT, R3, R1, R6    // R6 is min(R1, R3)

	CBZ	R6, samebytes
	BIC	$0xf, R6, R10
	CBZ	R10, small        // length < 16
	ADD	R0, R10           // end of chunk16
	// length >= 16
chunk16_loop:
	LDP.P	16(R0), (R4, R8)
	LDP.P	16(R2), (R5, R9)
	CMP	R4, R5
	BNE	cmp
	CMP	R8, R9
	BNE	cmpnext
	CMP	R10, R0
	BNE	chunk16_loop
	AND	$0xf, R6, R6
	CBZ	R6, samebytes
	SUBS	$8, R6
	BLT	tail
	// the length of tail > 8 bytes
	MOVD.P	8(R0), R4
	MOVD.P	8(R2), R5
	CMP	R4, R5
	BNE	cmp
	SUB	$8, R6
	// compare last 8 bytes
tail:
	MOVD	(R0)(R6), R4
	MOVD	(R2)(R6), R5
	CMP	R4, R5
	BEQ	samebytes
cmp:
	REV	R4, R4
	REV	R5, R5
	CMP	R4, R5
ret:
	MOVD	$1, R0
	CNEG	HI, R0, R0
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R0, (R7)
#endif
	RET
small:
	TBZ	$3, R6, lt_8
	MOVD	(R0), R4
	MOVD	(R2), R5
	CMP	R4, R5
	BNE	cmp
	SUBS	$8, R6
	BEQ	samebytes
	ADD	$8, R0
	ADD	$8, R2
	SUB	$8, R6
	B	tail
lt_8:
	TBZ	$2, R6, lt_4
	MOVWU	(R0), R4
	MOVWU	(R2), R5
	CMPW	R4, R5
	BNE	cmp
	SUBS	$4, R6
	BEQ	samebytes
	ADD	$4, R0
	ADD	$4, R2
lt_4:
	TBZ	$1, R6, lt_2
	MOVHU	(R0), R4
	MOVHU	(R2), R5
	CMPW	R4, R5
	BNE	cmp
	ADD	$2, R0
	ADD	$2, R2
lt_2:
	TBZ	$0, R6, samebytes
one:
	MOVBU	(R0), R4
	MOVBU	(R2), R5
	CMPW	R4, R5
	BNE	ret
samebytes:
	CMP	R3, R1
	CSET	NE, R0
	CNEG	LO, R0, R0
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R0, (R7)
#endif
	RET
cmpnext:
	REV	R8, R4
	REV	R9, R5
	CMP	R4, R5
	B	ret
