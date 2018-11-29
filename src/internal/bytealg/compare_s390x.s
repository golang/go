// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB),NOSPLIT|NOFRAME,$0-56
	MOVD	a_base+0(FP), R3
	MOVD	a_len+8(FP), R4
	MOVD	b_base+24(FP), R5
	MOVD	b_len+32(FP), R6
	LA	ret+48(FP), R7
	BR	cmpbody<>(SB)

TEXT runtime·cmpstring(SB),NOSPLIT|NOFRAME,$0-40
	MOVD	a_base+0(FP), R3
	MOVD	a_len+8(FP), R4
	MOVD	b_base+16(FP), R5
	MOVD	b_len+24(FP), R6
	LA	ret+32(FP), R7
	BR	cmpbody<>(SB)

// input:
//   R3 = a
//   R4 = alen
//   R5 = b
//   R6 = blen
//   R7 = address of output word (stores -1/0/1 here)
TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R3, R5, cmplengths
	MOVD	R4, R8
	CMPBLE	R4, R6, amin
	MOVD	R6, R8
amin:
	CMPBEQ	R8, $0, cmplengths
	CMP	R8, $256
	BLE	tail
loop:
	CLC	$256, 0(R3), 0(R5)
	BGT	gt
	BLT	lt
	SUB	$256, R8
	MOVD	$256(R3), R3
	MOVD	$256(R5), R5
	CMP	R8, $256
	BGT	loop
tail:
	SUB	$1, R8
	EXRL	$cmpbodyclc<>(SB), R8
	BGT	gt
	BLT	lt
cmplengths:
	CMP	R4, R6
	BEQ	eq
	BLT	lt
gt:
	MOVD	$1, 0(R7)
	RET
lt:
	MOVD	$-1, 0(R7)
	RET
eq:
	MOVD	$0, 0(R7)
	RET

TEXT cmpbodyclc<>(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R3), 0(R5)
	RET
