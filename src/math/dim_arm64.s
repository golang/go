// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000

// func Dim(x, y float64) float64
TEXT ·Dim(SB),NOSPLIT,$0
	// (+Inf, +Inf) special case
	MOVD	$PosInf, R0
	MOVD	x+0(FP), R1
	MOVD	y+8(FP), R2
	CMP	R0, R1
	BNE	dim2
	CMP	R0, R2
	BEQ	bothInf
dim2:	// (-Inf, -Inf) special case
	MOVD	$NegInf, R0
	CMP	R0, R1
	BNE	dim3
	CMP	R0, R2
	BEQ	bothInf
dim3:	// normal case
	FMOVD	R1, F0
	FMOVD	R2, F1
	FMOVD	$0.0, F2
	FSUBD	F1, F0
	FMAXD	F0, F2, F0
	FMOVD	F0, ret+16(FP)
	RET
bothInf:
	MOVD	$NaN, R0
	MOVD	R0, ret+16(FP)
	RET

// func ·Max(x, y float64) float64
TEXT ·Max(SB),NOSPLIT,$0
	// +Inf special cases
	MOVD	$PosInf, R0
	MOVD	x+0(FP), R1
	CMP	R0, R1
	BEQ	isPosInf
	MOVD	y+8(FP), R2
	CMP	R0, R2
	BEQ	isPosInf
	// normal case
	FMOVD	R1, F0
	FMOVD	R2, F1
	FMAXD	F0, F1, F0
	FMOVD	F0, ret+16(FP)
	RET
isPosInf: // return +Inf
	MOVD	R0, ret+16(FP)
	RET

// func Min(x, y float64) float64
TEXT ·Min(SB),NOSPLIT,$0
	// -Inf special cases
	MOVD	$NegInf, R0
	MOVD	x+0(FP), R1
	CMP	R0, R1
	BEQ	isNegInf
	MOVD	y+8(FP), R2
	CMP	R0, R2
	BEQ	isNegInf
	// normal case
	FMOVD	R1, F0
	FMOVD	R2, F1
	FMIND	F0, F1, F0
	FMOVD	F0, ret+16(FP)
	RET
isNegInf: // return -Inf
	MOVD	R0, ret+16(FP)
	RET
