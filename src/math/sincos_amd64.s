// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// The method is based on a paper by Naoki Shibata: "Efficient evaluation
// methods of elementary functions suitable for SIMD computation", Proc.
// of International Supercomputing Conference 2010 (ISC'10), pp. 25 -- 32
// (May 2010). The paper is available at
// http://www.springerlink.com/content/340228x165742104/
//
// The original code and the constants below are from the author's
// implementation available at http://freshmeat.net/projects/sleef.
// The README file says, "The software is in public domain.
// You can use the software without any obligation."
//
// This code is a simplified version of the original.

#define PosOne 0x3FF0000000000000
#define PosInf 0x7FF0000000000000
#define NaN    0x7FF8000000000001
#define PI4A 0.7853981554508209228515625 // pi/4 split into three parts
#define PI4B 0.794662735614792836713604629039764404296875e-8
#define PI4C 0.306161699786838294306516483068750264552437361480769e-16
#define M4PI 1.273239544735162542821171882678754627704620361328125 // 4/pi
#define T0 1.0
#define T1 -8.33333333333333333333333e-02 // (-1.0/12)
#define T2 2.77777777777777777777778e-03 // (+1.0/360)
#define T3 -4.96031746031746031746032e-05 // (-1.0/20160)
#define T4 5.51146384479717813051146e-07 // (+1.0/1814400)

// func Sincos(d float64) (sin, cos float64)
TEXT ·Sincos(SB),NOSPLIT,$0
	// test for special cases
	MOVQ    $~(1<<63), DX // sign bit mask
	MOVQ    x+0(FP), BX
	ANDQ    BX, DX
	JEQ     isZero
	MOVQ    $PosInf, AX
	CMPQ    AX, DX
	JLE     isInfOrNaN
	// Reduce argument
	MOVQ    BX, X7 // x7= d
	MOVQ    DX, X0 // x0= |d|
	MOVSD   $M4PI, X2
	MULSD   X0, X2
	CVTTSD2SQ X2, BX // bx= q
	MOVQ    $1, AX
	ANDQ    BX, AX
	ADDQ    BX, AX
	CVTSQ2SD AX, X2
	MOVSD   $PI4A, X3
	MULSD   X2, X3
	SUBSD   X3, X0
	MOVSD   $PI4B, X3
	MULSD   X2, X3
	SUBSD   X3, X0
	MOVSD   $PI4C, X3
	MULSD   X2, X3
	SUBSD   X3, X0
	MULSD   $0.125, X0 // x0= x, x7= d, bx= q
	// Evaluate Taylor series
	MULSD   X0, X0
	MOVSD   $T4, X2
	MULSD   X0, X2
	ADDSD   $T3, X2
	MULSD   X0, X2
	ADDSD   $T2, X2
	MULSD   X0, X2
	ADDSD   $T1, X2
	MULSD   X0, X2
	ADDSD   $T0, X2
	MULSD   X2, X0 // x0= x, x7= d, bx= q
	// Apply double angle formula
	MOVSD   $4.0, X2
	SUBSD   X0, X2
	MULSD   X2, X0
	MOVSD   $4.0, X2
	SUBSD   X0, X2
	MULSD   X2, X0
	MOVSD   $4.0, X2
	SUBSD   X0, X2
	MULSD   X2, X0
	MULSD   $0.5, X0 // x0= x, x7= d, bx= q
	// sin = sqrt((2 - x) * x)
	MOVSD   $2.0, X2
	SUBSD   X0, X2
	MULSD   X0, X2
	SQRTSD  X2, X2 // x0= x, x2= z, x7= d, bx= q
	// cos = 1 - x
	MOVSD   $1.0, X1
	SUBSD   X0, X1 // x1= x, x2= z, x7= d, bx= q
	// if ((q + 1) & 2) != 0 { sin, cos = cos, sin }
	MOVQ    $1, DX
	ADDQ    BX, DX
	ANDQ    $2, DX
	SHRQ    $1, DX
	SUBQ	$1, DX
	MOVQ    DX, X3
	// sin = (y & z) | (^y & x)
	MOVAPD  X2, X0
	ANDPD   X3, X0 // x0= sin
	MOVAPD  X3, X4
	ANDNPD  X1, X4
	ORPD    X4, X0 // x0= sin, x1= x, x2= z, x3= y, x7= d, bx= q
	// cos = (y & x) | (^y & z)
	ANDPD   X3, X1 // x1= cos
	ANDNPD  X2, X3
	ORPD    X3, X1 // x0= sin, x1= cos, x7= d, bx= q
	// if ((q & 4) != 0) != (d < 0) { sin = -sin }
	MOVQ    BX, AX
	MOVQ    $61, CX
	SHLQ    CX, AX
	MOVQ    AX, X3
	XORPD   X7, X3
	MOVQ    $(1<<63), AX
	MOVQ    AX, X2 // x2= -0.0
	ANDPD   X2, X3
	ORPD    X3, X0 // x0= sin, x1= cos, x2= -0.0, bx= q
	// if ((q + 2) & 4) != 0 { cos = -cos }
	MOVQ    $2, AX
	ADDQ    AX, BX
	MOVQ    $61, CX
	SHLQ    CX, BX
	MOVQ    BX, X3
	ANDPD   X2, X3
	ORPD    X3, X1 // x0= sin, x1= cos
	// return (sin, cos)
	MOVSD   X0, sin+8(FP)
	MOVSD   X1, cos+16(FP)
	RET
isZero: // return (±0.0, 1.0)
	MOVQ    BX, sin+8(FP)
	MOVQ    $PosOne, AX
	MOVQ    AX, cos+16(FP)
	RET
isInfOrNaN: // return (NaN, NaN)
	MOVQ    $NaN, AX
	MOVQ    AX, sin+8(FP)
	MOVQ    AX, cos+16(FP)
	RET
