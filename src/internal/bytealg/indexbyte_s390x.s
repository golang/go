// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB),NOSPLIT|NOFRAME,$0-40
	MOVD	b_base+0(FP), R3// b_base => R3
	MOVD	b_len+8(FP), R4 // b_len => R4
	MOVBZ	c+24(FP), R5    // c => R5
	MOVD	$ret+32(FP), R2 // &ret => R9
	BR	indexbytebody<>(SB)

TEXT ·IndexByteString(SB),NOSPLIT|NOFRAME,$0-32
	MOVD	s_base+0(FP), R3// s_base => R3
	MOVD	s_len+8(FP), R4 // s_len => R4
	MOVBZ	c+16(FP), R5    // c => R5
	MOVD	$ret+24(FP), R2 // &ret => R9
	BR	indexbytebody<>(SB)

// input:
// R3: s
// R4: s_len
// R5: c -- byte sought
// R2: &ret -- address to put index into
TEXT indexbytebody<>(SB),NOSPLIT|NOFRAME,$0
	CMPBEQ	R4, $0, notfound
	MOVD	R3, R6          // store base for later
	ADD	R3, R4, R8      // the address after the end of the string
	//if the length is small, use loop; otherwise, use vector or srst search
	CMPBGE	R4, $16, large

residual:
	CMPBEQ	R3, R8, notfound
	MOVBZ	0(R3), R7
	LA	1(R3), R3
	CMPBNE	R7, R5, residual

found:
	SUB	R6, R3
	SUB	$1, R3
	MOVD	R3, 0(R2)
	RET

notfound:
	MOVD	$-1, 0(R2)
	RET

large:
	MOVBZ	internal∕cpu·S390X+const_offsetS390xHasVX(SB), R1
	CMPBNE	R1, $0, vectorimpl

srstimpl:                       // no vector facility
	MOVBZ	R5, R0          // c needs to be in R0, leave until last minute as currently R0 is expected to be 0
srstloop:
	WORD	$0xB25E0083     // srst %r8, %r3 (search the range [R3, R8))
	BVS	srstloop        // interrupted - continue
	BGT	notfoundr0
foundr0:
	XOR	R0, R0          // reset R0
	SUB	R6, R8          // remove base
	MOVD	R8, 0(R2)
	RET
notfoundr0:
	XOR	R0, R0          // reset R0
	MOVD	$-1, 0(R2)
	RET

vectorimpl:
	//if the address is not 16byte aligned, use loop for the header
	MOVD	R3, R8
	AND	$15, R8
	CMPBGT	R8, $0, notaligned

aligned:
	ADD	R6, R4, R8
	MOVD	R8, R7
	AND	$-16, R7
	// replicate c across V17
	VLVGB	$0, R5, V19
	VREPB	$0, V19, V17

vectorloop:
	CMPBGE	R3, R7, residual
	VL	0(R3), V16    // load string to be searched into V16
	ADD	$16, R3
	VFEEBS	V16, V17, V18 // search V17 in V16 and set conditional code accordingly
	BVS	vectorloop

	// when vector search found c in the string
	VLGVB	$7, V18, R7   // load 7th element of V18 containing index into R7
	SUB	$16, R3
	SUB	R6, R3
	ADD	R3, R7
	MOVD	R7, 0(R2)
	RET

notaligned:
	MOVD	R3, R8
	AND	$-16, R8
	ADD     $16, R8
notalignedloop:
	CMPBEQ	R3, R8, aligned
	MOVBZ	0(R3), R7
	LA	1(R3), R3
	CMPBNE	R7, R5, notalignedloop
	BR	found
