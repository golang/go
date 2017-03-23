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
	MOVD	z+0(FP), R3
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R1
	MOVD	y+48(FP), R2
	ADDS	$0, R0 // clear carry flag
loop:
	CBZ	R0, done // careful not to touch the carry flag
	MOVD.P	8(R1), R4
	MOVD.P	8(R2), R5
	ADCS	R4, R5
	MOVD.P	R5, 8(R3)
	SUB	$1, R0
	B	loop
done:
	CSET	HS, R0 // extract carry flag
	MOVD	R0, c+72(FP)
	RET


// func subVV(z, x, y []Word) (c Word)
TEXT ·subVV(SB),NOSPLIT,$0
	MOVD	z+0(FP), R3
	MOVD	z_len+8(FP), R0
	MOVD	x+24(FP), R1
	MOVD	y+48(FP), R2
	CMP	R0, R0 // set carry flag
loop:
	CBZ	R0, done // careful not to touch the carry flag
	MOVD.P	8(R1), R4
	MOVD.P	8(R2), R5
	SBCS	R5, R4
	MOVD.P	R4, 8(R3)
	SUB	$1, R0
	B	loop
done:
	CSET	LO, R0 // extract carry flag
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
	B ·addMulVVW_g(SB)


// func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
TEXT ·divWVW(SB),NOSPLIT,$0
	B ·divWVW_g(SB)
