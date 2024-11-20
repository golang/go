// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB), $0-32
	MOVD	$16, R0
	JMP		addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB), $0-32
	MOVD	$24, R0
	JMP		addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB), $0-32
	MOVD	$32, R0
	JMP		addMulVVWx(SB)

TEXT addMulVVWx(SB), NOFRAME|NOSPLIT, $0
	MOVD	z+0(FP), R1
	MOVD	x+8(FP), R2
	MOVD	y+16(FP), R3
	MOVD	$0, R4

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
	MOVD	R4, c+24(FP)
	RET
