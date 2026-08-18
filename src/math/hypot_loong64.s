// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001

DATA hypotrodata<>+0(SB)/8, $1.0
GLOBL hypotrodata<>+0(SB), NOPTR|RODATA, $8

// func archHypot(p, q float64) float64
TEXT ·archHypot(SB),NOSPLIT,$0
	MOVD	p+0(FP), F0
	MOVD	q+8(FP), F1
	ABSD	F0, F0	// p = |p|
	ABSD	F1, F1	// q = |q|

	FCLASSD	F0, F2
	FCLASSD	F1, F3
	MOVV	F2, R4
	MOVV	F3, R5
	OR	R5, R4

	// +Inf special case
	AND	$64, R4, R5
	BNE	R5, isInf

	// NaN special case
	AND	$2, R4, R5
	BNE	R5, isNaN

	// hypot = max * sqrt(1 + (min/max)**2)
	MOVD	F0, F4
	FMAXD	F0, F1, F0	// F0 = max(p, q)
	FMIND	F4, F1, F1	// F1 = min(p, q)

	MOVV	F0, R6
	MOVV	F1, R7
	OR	R7, R6
	BEQ	R6, R0, isZero

	DIVD	F0, F1, F1
	MULD	F1, F1, F1
	MOVV	$hypotrodata<>+0(SB), R8
	MOVD	0(R8), F2
	ADDD	F2, F1, F1
	SQRTD	F1, F1
	MULD	F1, F0, F0
	MOVD	F0, ret+16(FP)
	RET
isNaN:
	MOVV	$NaN, R6
	MOVV	R6, ret+16(FP) // return NaN
	RET
isInf:
	MOVV	$PosInf, R6
	MOVV	R6, ret+16(FP) // return +Inf
	RET
isZero:
	MOVV	R0, ret+16(FP) // return 0
	RET
