// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Values returned from an FCLASS instruction.
#define	NegInf	0x001
#define	PosInf	0x080
#define	NaN	0x200

// func Max(x, y float64) float64
TEXT ·Max(SB),NOSPLIT,$0
	MOVD	x+0(FP), F0
	MOVD	y+8(FP), F1
	FCLASSD	F0, X5
	FCLASSD	F1, X6

	// +Inf special cases
	MOV	$PosInf, X7
	BEQ	X7, X5, isMaxX
	BEQ	X7, X6, isMaxY

	// NaN special cases
	MOV	$NaN, X7
	BEQ	X7, X5, isMaxX
	BEQ	X7, X6, isMaxY

	// normal case
	FMAXD	F0, F1, F0
	MOVD	F0, ret+16(FP)
	RET

isMaxX: // return x
	MOVD	F0, ret+16(FP)
	RET

isMaxY: // return y
	MOVD	F1, ret+16(FP)
	RET

// func Min(x, y float64) float64
TEXT ·Min(SB),NOSPLIT,$0
	MOVD	x+0(FP), F0
	MOVD	y+8(FP), F1
	FCLASSD	F0, X5
	FCLASSD	F1, X6

	// -Inf special cases
	MOV	$NegInf, X7
	BEQ	X7, X5, isMinX
	BEQ	X7, X6, isMinY

	// NaN special cases
	MOV	$NaN, X7
	BEQ	X7, X5, isMinX
	BEQ	X7, X6, isMinY

	// normal case
	FMIND	F0, F1, F0
	MOVD	F0, ret+16(FP)
	RET

isMinX: // return x
	MOVD	F0, ret+16(FP)
	RET

isMinY: // return y
	MOVD	F1, ret+16(FP)
	RET
