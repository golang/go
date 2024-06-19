// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go && loong64

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT ·addVV(SB),NOSPLIT,$0
	JMP ·addVV_g(SB)

TEXT ·subVV(SB),NOSPLIT,$0
	JMP ·subVV_g(SB)

// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),NOSPLIT,$0
	// input:
	//   R4: z
	//   R5: z_len
	//   R7: x
	//   R10: y
	MOVV	z+0(FP), R4
	MOVV	z_len+8(FP), R5
	MOVV	x+24(FP), R7
	MOVV	y+48(FP), R10
	MOVV	$0, R6
	SLLV	$3, R5
loop:
	BEQ	R5, R6, done
	MOVV	(R6)(R7), R8
	ADDV	R8, R10, R9	// x1 + c = z1, if z1 < x1 then z1 overflow
	SGTU	R8, R9, R10
	MOVV	R9, (R6)(R4)
	ADDV	$8, R6
	JMP	loop
done:
	MOVV	R10, c+56(FP)
	RET

// func subVW(z, x []Word, y Word) (c Word)
TEXT ·subVW(SB),NOSPLIT,$0
	// input:
	//   R4: z
	//   R5: z_len
	//   R7: x
	//   R10: y
	MOVV	z+0(FP), R4
	MOVV	z_len+8(FP), R5
	MOVV	x+24(FP), R7
	MOVV	y+48(FP), R10
	MOVV	$0, R6
	SLLV	$3, R5
loop:
	BEQ	R5, R6, done
	MOVV	(R6)(R7), R8
	SUBV	R10, R8, R11	// x1 - c = z1, if z1 > x1 then overflow
	SGTU	R11, R8, R10
	MOVV	R11, (R6)(R4)
	ADDV	$8, R6
	JMP	loop
done:
	MOVV	R10, c+56(FP)
	RET

TEXT ·lshVU(SB),NOSPLIT,$0
	JMP ·lshVU_g(SB)

TEXT ·rshVU(SB),NOSPLIT,$0
	JMP ·rshVU_g(SB)

// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	// input:
	//   R4: z
	//   R5: z_len
	//   R7: x
	//   R10: y
	//   R11: r
	MOVV	z+0(FP), R4
	MOVV	z_len+8(FP), R5
	MOVV	x+24(FP), R7
	MOVV	y+48(FP), R10
	MOVV	r+56(FP), R11
	SLLV	$3, R5
	MOVV	$0, R6
loop:
	BEQ	R5, R6, done
	MOVV	(R6)(R7), R8
	MULV	R8, R10, R9
	MULHVU	R8, R10, R12
	ADDV	R9, R11, R8
	SGTU	R9, R8, R11	// if (c' = lo + c) < lo then overflow
	MOVV	R8, (R6)(R4)
	ADDV	R12, R11
	ADDV	$8, R6
	JMP	loop
done:
	MOVV	R11, c+64(FP)
	RET

TEXT ·addMulVVWW(SB),NOSPLIT,$0
	JMP ·addMulVVWW_g(SB)
