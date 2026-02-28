// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Based on dim_amd64.s

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000

// func ·Max(x, y float64) float64
TEXT ·archMax(SB),NOSPLIT,$0
	// +Inf special cases
	MOVD    $PosInf, R4
	MOVD    x+0(FP), R8
	CMPUBEQ R4, R8, isPosInf
	MOVD    y+8(FP), R9
	CMPUBEQ R4, R9, isPosInf
	// NaN special cases
	MOVD    $~(1<<63), R5 // bit mask
	MOVD    $PosInf, R4
	MOVD    R8, R2
	AND     R5, R2 // x = |x|
	CMPUBLT R4, R2, isMaxNaN
	MOVD    R9, R3
	AND     R5, R3 // y = |y|
	CMPUBLT R4, R3, isMaxNaN
	// ±0 special cases
	OR      R3, R2
	BEQ     isMaxZero

	FMOVD   x+0(FP), F1
	FMOVD   y+8(FP), F2
	FCMPU   F2, F1
	BGT     +3(PC)
	FMOVD   F1, ret+16(FP)
	RET
	FMOVD   F2, ret+16(FP)
	RET
isMaxNaN: // return NaN
	MOVD	$NaN, R4
isPosInf: // return +Inf
	MOVD    R4, ret+16(FP)
	RET
isMaxZero:
	MOVD    $(1<<63), R4 // -0.0
	CMPUBEQ R4, R8, +3(PC)
	MOVD    R8, ret+16(FP) // return 0
	RET
	MOVD    R9, ret+16(FP) // return other 0
	RET

// func archMin(x, y float64) float64
TEXT ·archMin(SB),NOSPLIT,$0
	// -Inf special cases
	MOVD    $NegInf, R4
	MOVD    x+0(FP), R8
	CMPUBEQ R4, R8, isNegInf
	MOVD    y+8(FP), R9
	CMPUBEQ R4, R9, isNegInf
	// NaN special cases
	MOVD    $~(1<<63), R5
	MOVD    $PosInf, R4
	MOVD    R8, R2
	AND     R5, R2 // x = |x|
	CMPUBLT R4, R2, isMinNaN
	MOVD    R9, R3
	AND     R5, R3 // y = |y|
	CMPUBLT R4, R3, isMinNaN
	// ±0 special cases
	OR      R3, R2
	BEQ     isMinZero

	FMOVD   x+0(FP), F1
	FMOVD   y+8(FP), F2
	FCMPU   F2, F1
	BLT     +3(PC)
	FMOVD   F1, ret+16(FP)
	RET
	FMOVD   F2, ret+16(FP)
	RET
isMinNaN: // return NaN
	MOVD	$NaN, R4
isNegInf: // return -Inf
	MOVD    R4, ret+16(FP)
	RET
isMinZero:
	MOVD    $(1<<63), R4 // -0.0
	CMPUBEQ R4, R8, +3(PC)
	MOVD    R9, ret+16(FP) // return other 0
	RET
	MOVD    R8, ret+16(FP) // return -0
	RET

