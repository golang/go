// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000

TEXT ·archMax(SB),NOSPLIT,$0
	MOVD	x+0(FP), F0
	MOVD	y+8(FP), F1
	FCLASSD	F0, F2
	FCLASSD	F1, F3

	// combine x and y categories together to judge
	MOVV	F2, R4
	MOVV	F3, R5
	OR	R5, R4

	// +Inf special cases
	AND	$64, R4, R5
	BNE	R5, isPosInf

	// NaN special cases
	AND	$2, R4, R5
	BNE	R5, isMaxNaN

	// normal case
	FMAXD	F0, F1, F0
	MOVD	F0, ret+16(FP)
	RET

isMaxNaN:
	MOVV	$NaN, R6
	MOVV	R6, ret+16(FP)
	RET

isPosInf:
	MOVV	$PosInf, R6
	MOVV	R6, ret+16(FP)
	RET

TEXT ·archMin(SB),NOSPLIT,$0
	MOVD	x+0(FP), F0
	MOVD	y+8(FP), F1
	FCLASSD	F0, F2
	FCLASSD	F1, F3

	// combine x and y categories together to judge
	MOVV	F2, R4
	MOVV	F3, R5
	OR	R5, R4

	// -Inf special cases
	AND	$4, R4, R5
	BNE	R5, isNegInf

	// NaN special cases
	AND	$2, R4, R5
	BNE	R5, isMinNaN

	// normal case
	FMIND	F0, F1, F0
	MOVD	F0, ret+16(FP)
	RET

isMinNaN:
	MOVV	$NaN, R6
	MOVV	R6, ret+16(FP)
	RET

isNegInf:
	MOVV	$NegInf, R6
	MOVV	R6, ret+16(FP)
	RET
