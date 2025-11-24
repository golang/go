// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-25
#ifndef GOEXPERIMENT_regabiargs
	MOVD	a+0(FP), R2
	MOVD	b+8(FP), R3
	MOVD	size+16(FP), R4
	LA	ret+24(FP), R5
#endif
	BR	memeqbody<>(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-17
#ifndef GOEXPERIMENT_regabiargs
	MOVD	a+0(FP), R2
	MOVD	b+8(FP), R3
	LA	ret+16(FP), R5
#endif

	MOVD	8(R12), R4    // compiler stores size at offset 8 in the closure
	BR	memeqbody<>(SB)

// input:
//   R2 = a
//   R3 = b
//   R4 = len
//   For regabiargs output value( 0/1 ) stored in R2
//   For !regabiargs address of output byte( stores 0/1 ) stored in R5
//   a and b have the same length
TEXT memeqbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R2, R3, equal
loop:
	CMPBEQ	R4, $0, equal
	CMPBLT	R4, $32, tiny
	CMP	R4, $256
	BLT	tail
	CLC	$256, 0(R2), 0(R3)
	BNE	notequal
	SUB	$256, R4
	LA	256(R2), R2
	LA	256(R3), R3
	BR	loop
tail:
	SUB	$1, R4, R8
	EXRL	$memeqbodyclc<>(SB), R8
	BEQ	equal
notequal:
	MOVD	$0, R2
#ifndef GOEXPERIMENT_regabiargs
	MOVB	R2, 0(R5)
#endif
	RET
equal:
	MOVD	$1, R2
#ifndef GOEXPERIMENT_regabiargs
	MOVB	R2, 0(R5)
#endif
	RET
tiny:
	MOVD	$0, R1
	CMPBLT	R4, $16, lt16
	MOVD	0(R2), R8
	MOVD	0(R3), R9
	CMPBNE	R8, R9, notequal
	MOVD	8(R2), R8
	MOVD	8(R3), R9
	CMPBNE	R8, R9, notequal
	LA	16(R1), R1
	SUB	$16, R4
lt16:
	CMPBLT	R4, $8, lt8
	MOVD	0(R2)(R1*1), R8
	MOVD	0(R3)(R1*1), R9
	CMPBNE	R8, R9, notequal
	LA	8(R1), R1
	SUB	$8, R4
lt8:
	CMPBLT	R4, $4, lt4
	MOVWZ	0(R2)(R1*1), R8
	MOVWZ	0(R3)(R1*1), R9
	CMPBNE	R8, R9, notequal
	LA	4(R1), R1
	SUB	$4, R4
lt4:
#define CHECK(n) \
	CMPBEQ	R4, $n, equal \
	MOVB	n(R2)(R1*1), R8 \
	MOVB	n(R3)(R1*1), R9 \
	CMPBNE	R8, R9, notequal
	CHECK(0)
	CHECK(1)
	CHECK(2)
	CHECK(3)
	BR	equal

TEXT memeqbodyclc<>(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R2), 0(R3)
	RET
