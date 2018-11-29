// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000

// func ·Max(x, y float64) float64
TEXT ·Max(SB),NOSPLIT,$0
	// +Inf special cases
	MOVQ    $PosInf, AX
	MOVQ    x+0(FP), R8
	CMPQ    AX, R8
	JEQ     isPosInf
	MOVQ    y+8(FP), R9
	CMPQ    AX, R9
	JEQ     isPosInf
	// NaN special cases
	MOVQ    $~(1<<63), DX // bit mask
	MOVQ    $PosInf, AX
	MOVQ    R8, BX
	ANDQ    DX, BX // x = |x|
	CMPQ    AX, BX
	JLT     isMaxNaN
	MOVQ    R9, CX
	ANDQ    DX, CX // y = |y|
	CMPQ    AX, CX
	JLT     isMaxNaN
	// ±0 special cases
	ORQ     CX, BX
	JEQ     isMaxZero

	MOVQ    R8, X0
	MOVQ    R9, X1
	MAXSD   X1, X0
	MOVSD   X0, ret+16(FP)
	RET
isMaxNaN: // return NaN
	MOVQ	$NaN, AX
isPosInf: // return +Inf
	MOVQ    AX, ret+16(FP)
	RET
isMaxZero:
	MOVQ    $(1<<63), AX // -0.0
	CMPQ    AX, R8
	JEQ     +3(PC)
	MOVQ    R8, ret+16(FP) // return 0
	RET
	MOVQ    R9, ret+16(FP) // return other 0
	RET

// func Min(x, y float64) float64
TEXT ·Min(SB),NOSPLIT,$0
	// -Inf special cases
	MOVQ    $NegInf, AX
	MOVQ    x+0(FP), R8
	CMPQ    AX, R8
	JEQ     isNegInf
	MOVQ    y+8(FP), R9
	CMPQ    AX, R9
	JEQ     isNegInf
	// NaN special cases
	MOVQ    $~(1<<63), DX
	MOVQ    $PosInf, AX
	MOVQ    R8, BX
	ANDQ    DX, BX // x = |x|
	CMPQ    AX, BX
	JLT     isMinNaN
	MOVQ    R9, CX
	ANDQ    DX, CX // y = |y|
	CMPQ    AX, CX
	JLT     isMinNaN
	// ±0 special cases
	ORQ     CX, BX
	JEQ     isMinZero

	MOVQ    R8, X0
	MOVQ    R9, X1
	MINSD   X1, X0
	MOVSD X0, ret+16(FP)
	RET
isMinNaN: // return NaN
	MOVQ	$NaN, AX
isNegInf: // return -Inf
	MOVQ    AX, ret+16(FP)
	RET
isMinZero:
	MOVQ    $(1<<63), AX // -0.0
	CMPQ    AX, R8
	JEQ     +3(PC)
	MOVQ    R9, ret+16(FP) // return other 0
	RET
	MOVQ    R8, ret+16(FP) // return -0
	RET

