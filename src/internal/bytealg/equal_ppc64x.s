// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
	// R3 = a
	// R4 = b
	// R5 = size
	BR	memeqbody<>(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-17
	// R3 = a
	// R4 = b
	CMP	R3, R4
	BEQ	eq
	MOVD	8(R11), R5    // compiler stores size at offset 8 in the closure
	BR	memeqbody<>(SB)
eq:
	MOVD	$1, R3
	RET

// Do an efficient memequal for ppc64
// R3 = s1
// R4 = s2
// R5 = len
// On exit:
// R3 = return value
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
	MOVD	$16,R14		// index for VSX loads and stores
loop32a:
	LXVD2X  (R3+R0), VS32	// VS32 = V0
	LXVD2X  (R4+R0), VS33	// VS33 = V1
	VCMPEQUBCC V0, V1, V2	// compare, setting CR6
	BGE     CR6, noteq
	LXVD2X  (R3+R14), VS32
	LXVD2X  (R4+R14), VS33
	VCMPEQUBCC V0, V1, V2
	BGE     CR6, noteq
	ADD     $32,R3		// bump up to next 32
	ADD     $32,R4
	BC      16, 0, loop32a  // br ctr and cr
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
	MOVD	$0, R3
	RET
equal:
	MOVD	$1, R3
	RET

