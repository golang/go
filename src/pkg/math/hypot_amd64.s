// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define PosInf 0x7FF0000000000000
#define NaN 0x7FF8000000000001

// func Hypot(p, q float64) float64
TEXT ·Hypot(SB),7,$0
	// test bits for special cases
	MOVQ    p+0(FP), BX
	MOVQ    $~(1<<63), AX
	ANDQ    AX, BX // p = |p|
	MOVQ    q+8(FP), CX
	ANDQ    AX, CX // q = |q|
	MOVQ    $PosInf, AX
	CMPQ    AX, BX
	JLE     isInfOrNaN
	CMPQ    AX, CX
	JLE     isInfOrNaN
	// hypot = max * sqrt(1 + (min/max)**2)
	MOVQ    BX, X0
	MOVQ    CX, X1
	ORQ     CX, BX
	JEQ     isZero
	MOVAPD  X0, X2
	MAXSD   X1, X0
	MINSD   X2, X1
	DIVSD   X0, X1
	MULSD   X1, X1
	ADDSD   $1.0, X1
	SQRTSD  X1, X1
	MULSD   X1, X0
	MOVSD   X0, ret+16(FP)
	RET
isInfOrNaN:
	CMPQ    AX, BX
	JEQ     isInf
	CMPQ    AX, CX
	JEQ     isInf
	MOVQ    $NaN, AX
	MOVQ    AX, ret+16(FP) // return NaN
	RET
isInf:
	MOVQ    AX, ret+16(FP) // return +Inf
	RET
isZero:
	MOVQ    $0, AX
	MOVQ    AX, ret+16(FP) // return 0
	RET
