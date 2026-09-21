// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf	0x7FF0000000000000
#define NaN	0x7FF8000000000001

// func archHypot(p, q float64) float64
TEXT ·archHypot(SB),NOSPLIT,$0
	MOVD	p+0(FP), F0
	MOVD	q+8(FP), F1

	MOVV	$1, R4
	MOVV	R4, F9
	FCLASSD	F0, F4
	FCLASSD	F1, F5

	ABSD	F0, F0
	ABSD	F1, F1
	MOVV	F4, R4
	MOVV	F5, R5
	DIVD	F9, F9, F7	// F7 = 1.0
	SUBD	F7, F7, F4	// F4 = 0.0
	OR	R4, R5, R6

	// +Inf special case
	AND	$68, R6, R7
	BNE	R7, isInf

	// NaN special case
	AND	$2, R6, R8
	BNE	R8, isNaN

	// hypot = max * sqrt(1 + (min/max)**2)
	FMAXD	F0, F1, F5
	FMIND	F0, F1, F2
	CMPEQD	F5, F4, FCC0
	BFPT	isZero

	DIVD	F5, F2, F4
	FMADDD	F7, F4, F4, F4

	SQRTD	F4, F4
	MULD	F5, F4, F4
	MOVD	F4, ret+16(FP)
	RET
isNaN:
	MOVV	$NaN, R7
	MOVV	R7, ret+16(FP)	// return NaN
	RET
isInf:
	MOVV	$PosInf, R6
	MOVV	R6, ret+16(FP)	// return +Inf
	RET
isZero:
	MOVV	R0, ret+16(FP)	// return 0
	RET
