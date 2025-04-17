// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// memequal(a, b unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-25
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R5
	MOVD	size+16(FP), R6
	LA	ret+24(FP), R7
	BR	memeqbody<>(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT|NOFRAME,$0-17
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R5
	MOVD	8(R12), R6    // compiler stores size at offset 8 in the closure
	LA	ret+16(FP), R7
	BR	memeqbody<>(SB)

// input:
//   R3 = a
//   R5 = b
//   R6 = len
//   R7 = address of output byte (stores 0 or 1 here)
//   a and b have the same length
TEXT memeqbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R3, R5, equal
loop:
	CMPBEQ	R6, $0, equal
	CMPBLT	R6, $32, tiny
	CMP	R6, $256
	BLT	tail
	CLC	$256, 0(R3), 0(R5)
	BNE	notequal
	SUB	$256, R6
	LA	256(R3), R3
	LA	256(R5), R5
	BR	loop
tail:
	SUB	$1, R6, R8
	EXRL	$memeqbodyclc<>(SB), R8
	BEQ	equal
notequal:
	MOVB	$0, 0(R7)
	RET
equal:
	MOVB	$1, 0(R7)
	RET
tiny:
	MOVD	$0, R2
	CMPBLT	R6, $16, lt16
	MOVD	0(R3), R8
	MOVD	0(R5), R9
	CMPBNE	R8, R9, notequal
	MOVD	8(R3), R8
	MOVD	8(R5), R9
	CMPBNE	R8, R9, notequal
	LA	16(R2), R2
	SUB	$16, R6
lt16:
	CMPBLT	R6, $8, lt8
	MOVD	0(R3)(R2*1), R8
	MOVD	0(R5)(R2*1), R9
	CMPBNE	R8, R9, notequal
	LA	8(R2), R2
	SUB	$8, R6
lt8:
	CMPBLT	R6, $4, lt4
	MOVWZ	0(R3)(R2*1), R8
	MOVWZ	0(R5)(R2*1), R9
	CMPBNE	R8, R9, notequal
	LA	4(R2), R2
	SUB	$4, R6
lt4:
#define CHECK(n) \
	CMPBEQ	R6, $n, equal \
	MOVB	n(R3)(R2*1), R8 \
	MOVB	n(R5)(R2*1), R9 \
	CMPBNE	R8, R9, notequal
	CHECK(0)
	CHECK(1)
	CHECK(2)
	CHECK(3)
	BR	equal

TEXT memeqbodyclc<>(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R3), 0(R5)
	RET
