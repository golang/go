// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "go_asm.h"
#include "textflag.h"

TEXT ·Equal(SB),NOSPLIT,$0-49
	MOVD	a_len+8(FP), R4
	MOVD	b_len+32(FP), R5
	CMP	R5, R4		// unequal lengths are not equal
	BNE	noteq
	MOVD	a_base+0(FP), R3
	MOVD	b_base+24(FP), R4
	BL	memeqbody<>(SB)

	MOVBZ	R9,ret+48(FP)
	RET

noteq:
	MOVBZ	$0,ret+48(FP)
	RET

equal:
	MOVD	$1,R3
	MOVBZ	R3,ret+48(FP)
	RET

TEXT bytes·Equal(SB),NOSPLIT,$0-49
	FUNCDATA $0, ·Equal·args_stackmap(SB)
	MOVD	a_len+8(FP), R4
	MOVD	b_len+32(FP), R5
	CMP	R5, R4		// unequal lengths are not equal
	BNE	noteq
	MOVD	a_base+0(FP), R3
	MOVD	b_base+24(FP), R4
	BL	memeqbody<>(SB)

	MOVBZ	R9,ret+48(FP)
	RET

noteq:
	MOVBZ	$0,ret+48(FP)
	RET

equal:
	MOVD	$1,R3
	MOVBZ	R3,ret+48(FP)
	RET

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-25
	MOVD    a+0(FP), R3
	MOVD    b+8(FP), R4
	MOVD    size+16(FP), R5

	BL	memeqbody<>(SB)
	MOVB    R9, ret+24(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$40-17
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R4
	CMP	R3, R4
	BEQ	eq
	MOVD	8(R11), R5    // compiler stores size at offset 8 in the closure
	BL	memeqbody<>(SB)
	MOVB	R9, ret+16(FP)
	RET
eq:
	MOVD	$1, R3
	MOVB	R3, ret+16(FP)
	RET

// Do an efficient memequal for ppc64
// R3 = s1
// R4 = s2
// R5 = len
// R9 = return value
TEXT memeqbody<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVD    R5,CTR
	CMP     R5,$8		// only optimize >=8
	BLT     simplecheck
	DCBT	(R3)		// cache hint
	DCBT	(R4)
	CMP	R5,$32		// optimize >= 32
	MOVD	R5,R6		// needed if setup8a branch
	BLT	setup8a		// 8 byte moves only
setup32a:                       // 8 byte aligned, >= 32 bytes
	SRADCC  $5,R5,R6        // number of 32 byte chunks to compare
	MOVD	R6,CTR
loop32a:
	MOVD    0(R3),R6        // doublewords to compare
	MOVD    0(R4),R7
	MOVD	8(R3),R8	//
	MOVD	8(R4),R9
	CMP     R6,R7           // bytes batch?
	BNE     noteq
	MOVD	16(R3),R6
	MOVD	16(R4),R7
	CMP     R8,R9		// bytes match?
	MOVD	24(R3),R8
	MOVD	24(R4),R9
	BNE     noteq
	CMP     R6,R7           // bytes match?
	BNE	noteq
	ADD     $32,R3		// bump up to next 32
	ADD     $32,R4
	CMP     R8,R9           // bytes match?
	BC      8,2,loop32a	// br ctr and cr
	BNE	noteq
	ANDCC	$24,R5,R6       // Any 8 byte chunks?
	BEQ	leftover	// and result is 0
setup8a:
	SRADCC  $3,R6,R6        // get the 8 byte count
	BEQ	leftover	// shifted value is 0
	MOVD    R6,CTR
loop8:
	MOVD    0(R3),R6        // doublewords to compare
	ADD	$8,R3
	MOVD    0(R4),R7
	ADD     $8,R4
	CMP     R6,R7           // match?
	BC	8,2,loop8	// bt ctr <> 0 && cr
	BNE     noteq
leftover:
	ANDCC   $7,R5,R6        // check for leftover bytes
	BEQ     equal
	MOVD    R6,CTR
	BR	simple
simplecheck:
	CMP	R5,$0
	BEQ	equal
simple:
	MOVBZ   0(R3), R6
	ADD	$1,R3
	MOVBZ   0(R4), R7
	ADD     $1,R4
	CMP     R6, R7
	BNE     noteq
	BC      8,2,simple
	BNE	noteq
	BR	equal
noteq:
	MOVD    $0, R9
	RET
equal:
	MOVD    $1, R9
	RET

