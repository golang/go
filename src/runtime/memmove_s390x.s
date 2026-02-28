// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB),NOSPLIT|NOFRAME,$0-24
	MOVD	to+0(FP), R6
	MOVD	from+8(FP), R4
	MOVD	n+16(FP), R5

	CMPBEQ	R6, R4, done

start:
	CMPBLE	R5, $3, move0to3
	CMPBLE	R5, $7, move4to7
	CMPBLE	R5, $11, move8to11
	CMPBLE	R5, $15, move12to15
	CMPBNE	R5, $16, movemt16
	MOVD	0(R4), R7
	MOVD	8(R4), R8
	MOVD	R7, 0(R6)
	MOVD	R8, 8(R6)
	RET

movemt16:
	CMPBGT	R4, R6, forwards
	ADD	R5, R4, R7
	CMPBLE	R7, R6, forwards
	ADD	R5, R6, R8
backwards:
	MOVD	-8(R7), R3
	MOVD	R3, -8(R8)
	MOVD	-16(R7), R3
	MOVD	R3, -16(R8)
	ADD	$-16, R5
	ADD	$-16, R7
	ADD	$-16, R8
	CMP	R5, $16
	BGE	backwards
	BR	start

forwards:
	CMPBGT	R5, $64, forwards_fast
	MOVD	0(R4), R3
	MOVD	R3, 0(R6)
	MOVD	8(R4), R3
	MOVD	R3, 8(R6)
	ADD	$16, R4
	ADD	$16, R6
	ADD	$-16, R5
	CMP	R5, $16
	BGE	forwards
	BR	start

forwards_fast:
	CMP	R5, $256
	BLE	forwards_small
	MVC	$256, 0(R4), 0(R6)
	ADD	$256, R4
	ADD	$256, R6
	ADD	$-256, R5
	BR	forwards_fast

forwards_small:
	CMPBEQ	R5, $0, done
	ADD	$-1, R5
	EXRL	$memmove_exrl_mvc<>(SB), R5
	RET

move0to3:
	CMPBEQ	R5, $0, done
move1:
	CMPBNE	R5, $1, move2
	MOVB	0(R4), R3
	MOVB	R3, 0(R6)
	RET
move2:
	CMPBNE	R5, $2, move3
	MOVH	0(R4), R3
	MOVH	R3, 0(R6)
	RET
move3:
	MOVH	0(R4), R3
	MOVB	2(R4), R7
	MOVH	R3, 0(R6)
	MOVB	R7, 2(R6)
	RET

move4to7:
	CMPBNE	R5, $4, move5
	MOVW	0(R4), R3
	MOVW	R3, 0(R6)
	RET
move5:
	CMPBNE	R5, $5, move6
	MOVW	0(R4), R3
	MOVB	4(R4), R7
	MOVW	R3, 0(R6)
	MOVB	R7, 4(R6)
	RET
move6:
	CMPBNE	R5, $6, move7
	MOVW	0(R4), R3
	MOVH	4(R4), R7
	MOVW	R3, 0(R6)
	MOVH	R7, 4(R6)
	RET
move7:
	MOVW	0(R4), R3
	MOVH	4(R4), R7
	MOVB	6(R4), R8
	MOVW	R3, 0(R6)
	MOVH	R7, 4(R6)
	MOVB	R8, 6(R6)
	RET

move8to11:
	CMPBNE	R5, $8, move9
	MOVD	0(R4), R3
	MOVD	R3, 0(R6)
	RET
move9:
	CMPBNE	R5, $9, move10
	MOVD	0(R4), R3
	MOVB	8(R4), R7
	MOVD	R3, 0(R6)
	MOVB	R7, 8(R6)
	RET
move10:
	CMPBNE	R5, $10, move11
	MOVD	0(R4), R3
	MOVH	8(R4), R7
	MOVD	R3, 0(R6)
	MOVH	R7, 8(R6)
	RET
move11:
	MOVD	0(R4), R3
	MOVH	8(R4), R7
	MOVB	10(R4), R8
	MOVD	R3, 0(R6)
	MOVH	R7, 8(R6)
	MOVB	R8, 10(R6)
	RET

move12to15:
	CMPBNE	R5, $12, move13
	MOVD	0(R4), R3
	MOVW	8(R4), R7
	MOVD	R3, 0(R6)
	MOVW	R7, 8(R6)
	RET
move13:
	CMPBNE	R5, $13, move14
	MOVD	0(R4), R3
	MOVW	8(R4), R7
	MOVB	12(R4), R8
	MOVD	R3, 0(R6)
	MOVW	R7, 8(R6)
	MOVB	R8, 12(R6)
	RET
move14:
	CMPBNE	R5, $14, move15
	MOVD	0(R4), R3
	MOVW	8(R4), R7
	MOVH	12(R4), R8
	MOVD	R3, 0(R6)
	MOVW	R7, 8(R6)
	MOVH	R8, 12(R6)
	RET
move15:
	MOVD	0(R4), R3
	MOVW	8(R4), R7
	MOVH	12(R4), R8
	MOVB	14(R4), R10
	MOVD	R3, 0(R6)
	MOVW	R7, 8(R6)
	MOVH	R8, 12(R6)
	MOVB	R10, 14(R6)
done:
	RET

// DO NOT CALL - target for exrl (execute relative long) instruction.
TEXT memmove_exrl_mvc<>(SB),NOSPLIT|NOFRAME,$0-0
	MVC	$1, 0(R4), 0(R6)
	MOVD	R0, 0(R0)
	RET

