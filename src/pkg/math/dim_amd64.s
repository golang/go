// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000

// func Dim(x, y float64) float64
TEXT ·Dim(SB),7,$0
	// (+Inf, +Inf) special case
	MOVQ    x+0(FP), BX
	MOVQ    y+8(FP), CX
	MOVQ    $PosInf, AX
	CMPQ    AX, BX
	JNE     dim2
	CMPQ    AX, CX
	JEQ     bothInf
dim2:	// (-Inf, -Inf) special case
	MOVQ    $NegInf, AX
	CMPQ    AX, BX
	JNE     dim3
	CMPQ    AX, CX
	JEQ     bothInf
dim3:	// (NaN, x) or (x, NaN)
	MOVQ    $~(1<<63), DX
	MOVQ    $NaN, AX
	ANDQ    DX, BX // x = |x|
	CMPQ    AX, BX
	JLE     isDimNaN
	ANDQ    DX, CX // y = |y|
	CMPQ    AX, CX
	JLE     isDimNaN

	MOVSD x+0(FP), X0
	SUBSD y+8(FP), X0
	MOVSD $(0.0), X1
	MAXSD X1, X0
	MOVSD X0, ret+16(FP)
	RET
bothInf: // Dim(-Inf, -Inf) or Dim(+Inf, +Inf)
	MOVQ    $NaN, AX
isDimNaN:
	MOVQ    AX, ret+16(FP)
	RET

// func ·Max(x, y float64) float64
TEXT ·Max(SB),7,$0
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
	MOVQ    $NaN, AX
	MOVQ    R8, BX
	ANDQ    DX, BX // x = |x|
	CMPQ    AX, BX
	JLE     isMaxNaN
	MOVQ    R9, CX
	ANDQ    DX, CX // y = |y|
	CMPQ    AX, CX
	JLE     isMaxNaN
	// ±0 special cases
	ORQ     CX, BX
	JEQ     isMaxZero

	MOVQ    R8, X0
	MOVQ    R9, X1
	MAXSD   X1, X0
	MOVSD   X0, ret+16(FP)
	RET
isMaxNaN: // return NaN
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

/*
	MOVQ    $0, AX
	CMPQ    AX, R8
	JNE     +3(PC)
	MOVQ    R8, ret+16(FP) // return 0
	RET
	MOVQ    R9, ret+16(FP) // return other 0
	RET
*/

// func Min(x, y float64) float64
TEXT ·Min(SB),7,$0
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
	MOVQ    $NaN, AX
	MOVQ    R8, BX
	ANDQ    DX, BX // x = |x|
	CMPQ    AX, BX
	JLE     isMinNaN
	MOVQ    R9, CX
	ANDQ    DX, CX // y = |y|
	CMPQ    AX, CX
	JLE     isMinNaN
	// ±0 special cases
	ORQ     CX, BX
	JEQ     isMinZero

	MOVQ    R8, X0
	MOVQ    R9, X1
	MINSD   X1, X0
	MOVSD X0, ret+16(FP)
	RET
isMinNaN: // return NaN
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

