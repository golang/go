// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
#ifndef GOEXPERIMENT_regabiargs
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R3
	MOVD	b_base+24(FP), R4
	MOVD	b_len+32(FP), R5
	LA	ret+48(FP), R6
#else
	// R2 = a_base
	// R3 = a_len
	// R4 = a_cap (unused)
	// R5 = b_base (want in R4)
	// R6 = b_len (want in R5)
	// R7 = b_cap (unused)
	MOVD	R5, R4
	MOVD	R6, R5
#endif
	BR	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
#ifndef GOEXPERIMENT_regabiargs
	MOVD	a_base+0(FP), R2
	MOVD	a_len+8(FP), R3
	MOVD	b_base+16(FP), R4
	MOVD	b_len+24(FP), R5
	LA	ret+32(FP), R6
#endif
	// R2 = a_base
	// R3 = a_len
	// R4 = b_base
	// R5 = b_len

	BR	cmpbody<>(SB)

// input:
//   R2 = a
//   R3 = alen
//   R4 = b
//   R5 = blen
//   For regabiargs output value( -1/0/1 ) stored in R2
//   For !regabiargs address of output word( stores -1/0/1 ) stored in R6
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R2, R4, cmplengths
	MOVD	R3, R7
	CMPBLE	R3, R5, amin
	MOVD	R5, R7
amin:
	CMPBEQ	R7, $0, cmplengths
	CMP	R7, $256
	BLE	tail
loop:
	CLC	$256, 0(R2), 0(R4)
	BGT	gt
	BLT	lt
	SUB	$256, R7
	MOVD	$256(R2), R2
	MOVD	$256(R4), R4
	CMP	R7, $256
	BGT	loop
tail:
	SUB	$1, R7
	EXRL	$cmpbodyclc<>(SB), R7
	BGT	gt
	BLT	lt
cmplengths:
	CMP	R3, R5
	BEQ	eq
	BLT	lt
gt:
	MOVD	$1, R2
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R2, 0(R6)
#endif
	RET
lt:
	MOVD	$-1, R2
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R2, 0(R6)
#endif
	RET
eq:
	MOVD	$0, R2
#ifndef GOEXPERIMENT_regabiargs
	MOVD	R2, 0(R6)
#endif
	RET

TEXT cmpbodyclc<>(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R2), 0(R4)
	RET
