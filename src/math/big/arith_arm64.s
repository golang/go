// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// TODO: Consider re-implementing using Advanced SIMD
// once the assembler supports those instructions.

// func mulWW(x, y Word) (z1, z0 Word)
TEXT ·mulWW(SB),NOSPLIT,$0
	MOVD	x+0(FP), R0
	MOVD	y+8(FP), R1
	MUL	R0, R1, R2
	UMULH	R0, R1, R3
	MOVD	R3, z1+16(FP)
	MOVD	R2, z0+24(FP)
	RET


// func divWW(x1, x0, y Word) (q, r Word)
TEXT ·divWW(SB),NOSPLIT,$0
	B	·divWW_g(SB) // ARM64 has no multiword division


// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R10
	ADDS	$0, R0		// clear carry flag
	TBZ	$0, R0, two
	MOVD.P	8(R8), R11
	MOVD.P	8(R9), R15
	ADCS	R15, R11
	MOVD.P	R11, 8(R10)
	SUB	$1, R0
two:
	TBZ	$1, R0, loop
	LDP.P	16(R8), (R11, R12)
	LDP.P	16(R9), (R15, R16)
	ADCS	R15, R11
	ADCS	R16, R12
	STP.P	(R11, R12), 16(R10)
	SUB	$2, R0
loop:
	CBZ	R0, done	// careful not to touch the carry flag
	LDP.P	32(R8), (R11, R12)
	LDP	-16(R8), (R13, R14)
	LDP.P	32(R9), (R15, R16)
	LDP	-16(R9), (R17, R19)
	ADCS	R15, R11
	ADCS	R16, R12
	ADCS	R17, R13
	ADCS	R19, R14
	STP.P	(R11, R12), 32(R10)
	STP	(R13, R14), -16(R10)
	SUB	$4, R0
	B	loop
done:
	CSET	HS, R0		// extract carry flag
	MOVD	R0, c+72(FP)
	RET


// func subVV(z, x, y []Word) (c Word)
TEXT ·subVV(SB),NOSPLIT,$0
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R8
	MOVD	y+48(FP), R9
	MOVD	z+0(FP), R10
	CMP	R0, R0		// set carry flag
	TBZ	$0, R0, two
	MOVD.P	8(R8), R11
	MOVD.P	8(R9), R15
	SBCS	R15, R11
	MOVD.P	R11, 8(R10)
	SUB	$1, R0
two:
	TBZ	$1, R0, loop
	LDP.P	16(R8), (R11, R12)
	LDP.P	16(R9), (R15, R16)
	SBCS	R15, R11
	SBCS	R16, R12
	STP.P	(R11, R12), 16(R10)
	SUB	$2, R0
loop:
	CBZ	R0, done	// careful not to touch the carry flag
	LDP.P	32(R8), (R11, R12)
	LDP	-16(R8), (R13, R14)
	LDP.P	32(R9), (R15, R16)
	LDP	-16(R9), (R17, R19)
	SBCS	R15, R11
	SBCS	R16, R12
	SBCS	R17, R13
	SBCS	R19, R14
	STP.P	(R11, R12), 32(R10)
	STP	(R13, R14), -16(R10)
	SUB	$4, R0
	B	loop
done:
	CSET	LO, R0		// extract carry flag
	MOVD	R0, c+72(FP)
	RET


// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R3
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R1
	MOVD	y+48(FP), R2
	CBZ	R0, return_y
	MOVD.P	8(R1), R4
	ADDS	R2, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
loop:
	CBZ	R0, done // careful not to touch the carry flag
	MOVD.P	8(R1), R4
	ADCS	$0, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
	B	loop
done:
	CSET	HS, R0 // extract carry flag
	MOVD	R0, c+56(FP)
	RET
return_y: // z is empty; copy y to c
	MOVD	R2, c+56(FP)
	RET


// func subVW(z, x []Word, y Word) (c Word)
TEXT ·subVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R3
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R1
	MOVD	y+48(FP), R2
	CBZ	R0, rety
	MOVD.P	8(R1), R4
	SUBS	R2, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
loop:
	CBZ	R0, done // careful not to touch the carry flag
	MOVD.P	8(R1), R4
	SBCS	$0, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
	B	loop
done:
	CSET	LO, R0 // extract carry flag
	MOVD	R0, c+56(FP)
	RET
rety: // z is empty; copy y to c
	MOVD	R2, c+56(FP)
	RET


// func shlVU(z, x []Word, s uint) (c Word)
TEXT ·shlVU(SB),NOSPLIT,$0
	B ·shlVU_g(SB)


// func shrVU(z, x []Word, s uint) (c Word)
TEXT ·shrVU(SB),NOSPLIT,$0
	B ·shrVU_g(SB)


// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R1
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R2
	MOVD	y+48(FP), R3
	MOVD	r+56(FP), R4
loop:
	CBZ	R0, done
	MOVD.P	8(R2), R5
	UMULH	R5, R3, R7
	MUL	R5, R3, R6
	ADDS	R4, R6
	ADC	$0, R7
	MOVD.P	R6, 8(R1)
	MOVD	R7, R4
	SUB	$1, R0
	B	loop
done:
	MOVD	R4, c+64(FP)
	RET


// func addMulVVW(z, x []Word, y Word) (c Word)
TEXT ·addMulVVW(SB),NOSPLIT,$0
	MOVD	z+0(FP), R1
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R2
	MOVD	y+48(FP), R3
	MOVD	$0, R4

	TBZ	$0, R0, two

	MOVD.P	8(R2), R5
	MOVD	(R1), R6

	MUL	R5, R3, R7
	UMULH	R5, R3, R8

	ADDS	R7, R6
	ADC	$0, R8, R4

	MOVD.P	R6, 8(R1)
	SUB	$1, R0

two:
	TBZ	$1, R0, loop

	LDP.P	16(R2), (R5, R10)
	LDP	(R1), (R6, R11)

	MUL	R10, R3, R13
	UMULH	R10, R3, R12

	MUL	R5, R3, R7
	UMULH	R5, R3, R8

	ADDS	R4, R6
	ADCS	R13, R11
	ADC	$0, R12

	ADDS	R7, R6
	ADCS	R8, R11
	ADC	$0, R12, R4

	STP.P	(R6, R11), 16(R1)
	SUB	$2, R0

// The main loop of this code operates on a block of 4 words every iteration
// performing [R4:R12:R11:R10:R9] = R4 + R3 * [R8:R7:R6:R5] + [R12:R11:R10:R9]
// where R4 is carried from the previous iteration, R8:R7:R6:R5 hold the next
// 4 words of x, R3 is y and R12:R11:R10:R9 are part of the result z.
loop:
	CBZ	R0, done

	LDP.P	16(R2), (R5, R6)
	LDP.P	16(R2), (R7, R8)

	LDP	(R1), (R9, R10)
	ADDS	R4, R9
	MUL	R6, R3, R14
	ADCS	R14, R10
	MUL	R7, R3, R15
	LDP	16(R1), (R11, R12)
	ADCS	R15, R11
	MUL	R8, R3, R16
	ADCS	R16, R12
	UMULH	R8, R3, R20
	ADC	$0, R20

	MUL	R5, R3, R13
	ADDS	R13, R9
	UMULH	R5, R3, R17
	ADCS	R17, R10
	UMULH	R6, R3, R21
	STP.P	(R9, R10), 16(R1)
	ADCS	R21, R11
	UMULH	R7, R3, R19
	ADCS	R19, R12
	STP.P	(R11, R12), 16(R1)
	ADC	$0, R20, R4

	SUB	$4, R0
	B	loop

done:
	MOVD	R4, c+56(FP)
	RET

// func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
TEXT ·divWVW(SB),NOSPLIT,$0
	B ·divWVW_g(SB)
