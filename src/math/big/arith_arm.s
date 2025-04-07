// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

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


// func lshVU(z, x []Word, s uint) (c Word)
TEXT ·lshVU(SB),NOSPLIT,$0
	MOVW	z_len+4(FP), R5
	TEQ	$0, R5
	BEQ	X7

	MOVW	z+0(FP), R1
	MOVW	x+12(FP), R2
	ADD	R5<<2, R2, R2
	ADD	R5<<2, R1, R5
	MOVW	s+24(FP), R3
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

X7:
	MOVW	$0, R1
	MOVW	R1, c+28(FP)
	RET



// func rshVU(z, x []Word, s uint) (c Word)
TEXT ·rshVU(SB),NOSPLIT,$0
	MOVW	z_len+4(FP), R5
	TEQ	$0, R5
	BEQ	X6

	MOVW	z+0(FP), R1
	MOVW	x+12(FP), R2
	ADD	R5<<2, R1, R5
	MOVW	s+24(FP), R3
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

X6:
	MOVW	$0, R1
	MOVW	R1, c+28(FP)
	RET

// func mulAddVWW(z, x []Word, m, a Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVW	$0, R0
	MOVW	z+0(FP), R1
	MOVW	z_len+4(FP), R5
	MOVW	x+12(FP), R2
	MOVW	m+24(FP), R3
	MOVW	a+28(FP), R4
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


// func addMulVVWW(z, x, y []Word, m, a Word) (c Word)
TEXT ·addMulVVWW(SB),NOSPLIT,$0
	MOVW	$0, R0
	MOVW	z+0(FP), R9
	MOVW	x+12(FP), R1
	MOVW	z_len+4(FP), R5
	MOVW	y+24(FP), R2
	MOVW	m+36(FP), R3
	ADD	R5<<2, R1, R5
	MOVW	a+40(FP), R4
	B E9

	// word loop
L9:
	MOVW.P	4(R2), R6
	MULLU	R6, R3, (R7, R6)
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW.P	4(R1), R4
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW.P	R6, 4(R9)
	MOVW	R7, R4
E9:
	TEQ	R1, R5
	BNE	L9

	MOVW	R4, c+44(FP)
	RET
