// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),NOSPLIT,$0
	ADD.S	$0, R0		// clear carry flag
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R4
	MOVW	x+12(FP), R2
	MOVW	y+24(FP), R3
	ADD	R4<<2, R1, R4
	B E1
L1:
	MOVW.P	4(R2), R5
	MOVW.P	4(R3), R6
	ADC.S	R6, R5
	MOVW.P	R5, 4(R1)
E1:
	TEQ	R1, R4
	BNE L1

	MOVW	$0, R0
	MOVW.CS	$1, R0
	MOVW	R0, c+36(FP)
	RET


// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SBC instead of ADC and label names)
TEXT ·subVV(SB),NOSPLIT,$0
	SUB.S	$0, R0		// clear borrow flag
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R4
	MOVW	x+12(FP), R2
	MOVW	y+24(FP), R3
	ADD	R4<<2, R1, R4
	B E2
L2:
	MOVW.P	4(R2), R5
	MOVW.P	4(R3), R6
	SBC.S	R6, R5
	MOVW.P	R5, 4(R1)
E2:
	TEQ	R1, R4
	BNE L2

	MOVW	$0, R0
	MOVW.CC	$1, R0
	MOVW	R0, c+36(FP)
	RET


// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),NOSPLIT,$0
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R4
	MOVW	x+12(FP), R2
	MOVW	y+24(FP), R3
	ADD	R4<<2, R1, R4
	TEQ	R1, R4
	BNE L3a
	MOVW	R3, c+28(FP)
	RET
L3a:
	MOVW.P	4(R2), R5
	ADD.S	R3, R5
	MOVW.P	R5, 4(R1)
	B	E3
L3:
	MOVW.P	4(R2), R5
	ADC.S	$0, R5
	MOVW.P	R5, 4(R1)
E3:
	TEQ	R1, R4
	BNE	L3

	MOVW	$0, R0
	MOVW.CS	$1, R0
	MOVW	R0, c+28(FP)
	RET


// func subVW(z, x []Word, y Word) (c Word)
TEXT ·subVW(SB),NOSPLIT,$0
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R4
	MOVW	x+12(FP), R2
	MOVW	y+24(FP), R3
	ADD	R4<<2, R1, R4
	TEQ	R1, R4
	BNE L4a
	MOVW	R3, c+28(FP)
	RET
L4a:
	MOVW.P	4(R2), R5
	SUB.S	R3, R5
	MOVW.P	R5, 4(R1)
	B	E4
L4:
	MOVW.P	4(R2), R5
	SBC.S	$0, R5
	MOVW.P	R5, 4(R1)
E4:
	TEQ	R1, R4
	BNE	L4

	MOVW	$0, R0
	MOVW.CC	$1, R0
	MOVW	R0, c+28(FP)
	RET


// func shlVU(z, x []Word, s uint) (c Word)
TEXT ·shlVU(SB),NOSPLIT,$0
	MOVW	z_len+4(FP), R5
	TEQ	$0, R5
	BEQ	X7

	MOVW	z+0(FP), R1
	MOVW	x+12(FP), R2
	ADD	R5<<2, R2, R2
	ADD	R5<<2, R1, R5
	MOVW	s+24(FP), R3
	TEQ	$0, R3	// shift 0 is special
	BEQ	Y7
	ADD	$4, R1	// stop one word early
	MOVW	$32, R4
	SUB	R3, R4
	MOVW	$0, R7

	MOVW.W	-4(R2), R6
	MOVW	R6<<R3, R7
	MOVW	R6>>R4, R6
	MOVW	R6, c+28(FP)
	B E7

L7:
	MOVW.W	-4(R2), R6
	ORR	R6>>R4, R7
	MOVW.W	R7, -4(R5)
	MOVW	R6<<R3, R7
E7:
	TEQ	R1, R5
	BNE	L7

	MOVW	R7, -4(R5)
	RET

Y7:	// copy loop, because shift 0 == shift 32
	MOVW.W	-4(R2), R6
	MOVW.W	R6, -4(R5)
	TEQ	R1, R5
	BNE Y7

X7:
	MOVW	$0, R1
	MOVW	R1, c+28(FP)
	RET


// func shrVU(z, x []Word, s uint) (c Word)
TEXT ·shrVU(SB),NOSPLIT,$0
	MOVW	z_len+4(FP), R5
	TEQ	$0, R5
	BEQ	X6

	MOVW	z+0(FP), R1
	MOVW	x+12(FP), R2
	ADD	R5<<2, R1, R5
	MOVW	s+24(FP), R3
	TEQ	$0, R3	// shift 0 is special
	BEQ Y6
	SUB	$4, R5	// stop one word early
	MOVW	$32, R4
	SUB	R3, R4
	MOVW	$0, R7

	// first word
	MOVW.P	4(R2), R6
	MOVW	R6>>R3, R7
	MOVW	R6<<R4, R6
	MOVW	R6, c+28(FP)
	B E6

	// word loop
L6:
	MOVW.P	4(R2), R6
	ORR	R6<<R4, R7
	MOVW.P	R7, 4(R1)
	MOVW	R6>>R3, R7
E6:
	TEQ	R1, R5
	BNE	L6

	MOVW	R7, 0(R1)
	RET

Y6:	// copy loop, because shift 0 == shift 32
	MOVW.P	4(R2), R6
	MOVW.P	R6, 4(R1)
	TEQ R1, R5
	BNE Y6

X6:
	MOVW	$0, R1
	MOVW	R1, c+28(FP)
	RET


// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVW	$0, R0
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R5
	MOVW	x+12(FP), R2
	MOVW	y+24(FP), R3
	MOVW	r+28(FP), R4
	ADD	R5<<2, R1, R5
	B E8

	// word loop
L8:
	MOVW.P	4(R2), R6
	MULLU	R6, R3, (R7, R6)
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW.P	R6, 4(R1)
	MOVW	R7, R4
E8:
	TEQ	R1, R5
	BNE	L8

	MOVW	R4, c+32(FP)
	RET


// func addMulVVW(z, x []Word, y Word) (c Word)
TEXT ·addMulVVW(SB),NOSPLIT,$0
	MOVW	$0, R0
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R5
	MOVW	x+12(FP), R2
	MOVW	y+24(FP), R3
	ADD	R5<<2, R1, R5
	MOVW	$0, R4
	B E9

	// word loop
L9:
	MOVW.P	4(R2), R6
	MULLU	R6, R3, (R7, R6)
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW	0(R1), R4
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW.P	R6, 4(R1)
	MOVW	R7, R4
E9:
	TEQ	R1, R5
	BNE	L9

	MOVW	R4, c+28(FP)
	RET


// func divWVW(z* Word, xn Word, x []Word, y Word) (r Word)
TEXT ·divWVW(SB),NOSPLIT,$0
	// ARM has no multiword division, so use portable code.
	B ·divWVW_g(SB)


// func divWW(x1, x0, y Word) (q, r Word)
TEXT ·divWW(SB),NOSPLIT,$0
	// ARM has no multiword division, so use portable code.
	B ·divWW_g(SB)


// func mulWW(x, y Word) (z1, z0 Word)
TEXT ·mulWW(SB),NOSPLIT,$0
	MOVW	x+0(FP), R1
	MOVW	y+4(FP), R2
	MULLU	R1, R2, (R4, R3)
	MOVW	R4, z1+8(FP)
	MOVW	R3, z0+12(FP)
	RET
