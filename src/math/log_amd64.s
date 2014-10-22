// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define HSqrt2 7.07106781186547524401e-01 // sqrt(2)/2
#define Ln2Hi  6.93147180369123816490e-01 // 0x3fe62e42fee00000
#define Ln2Lo  1.90821492927058770002e-10 // 0x3dea39ef35793c76
#define L1     6.666666666666735130e-01   // 0x3FE5555555555593
#define L2     3.999999999940941908e-01   // 0x3FD999999997FA04
#define L3     2.857142874366239149e-01   // 0x3FD2492494229359
#define L4     2.222219843214978396e-01   // 0x3FCC71C51D8E78AF
#define L5     1.818357216161805012e-01   // 0x3FC7466496CB03DE
#define L6     1.531383769920937332e-01   // 0x3FC39A09D078C69F
#define L7     1.479819860511658591e-01   // 0x3FC2F112DF3E5244
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000
#define PosInf 0x7FF0000000000000

// func Log(x float64) float64
TEXT ·Log(SB),NOSPLIT,$0
	// test bits for special cases
	MOVQ    x+0(FP), BX
	MOVQ    $~(1<<63), AX // sign bit mask
	ANDQ    BX, AX
	JEQ     isZero
	MOVQ    $0, AX
	CMPQ    AX, BX
	JGT     isNegative
	MOVQ    $PosInf, AX
	CMPQ    AX, BX
	JLE     isInfOrNaN
	// f1, ki := math.Frexp(x); k := float64(ki)
	MOVQ    BX, X0
	MOVQ    $0x000FFFFFFFFFFFFF, AX
	MOVQ    AX, X2
	ANDPD   X0, X2
	MOVSD   $0.5, X0 // 0x3FE0000000000000
	ORPD    X0, X2 // X2= f1
	SHRQ    $52, BX
	ANDL    $0x7FF, BX
	SUBL    $0x3FE, BX
	CVTSL2SD BX, X1 // x1= k, x2= f1
	// if f1 < math.Sqrt2/2 { k -= 1; f1 *= 2 }
	MOVSD   $HSqrt2, X0 // x0= 0.7071, x1= k, x2= f1
	CMPSD   X2, X0, 5 // cmpnlt; x0= 0 or ^0, x1= k, x2 = f1
	MOVSD   $1.0, X3 // x0= 0 or ^0, x1= k, x2 = f1, x3= 1
	ANDPD   X0, X3 // x0= 0 or ^0, x1= k, x2 = f1, x3= 0 or 1
	SUBSD   X3, X1 // x0= 0 or ^0, x1= k, x2 = f1, x3= 0 or 1
	MOVSD   $1.0, X0 // x0= 1, x1= k, x2= f1, x3= 0 or 1
	ADDSD   X0, X3 // x0= 1, x1= k, x2= f1, x3= 1 or 2
	MULSD   X3, X2 // x0= 1, x1= k, x2= f1
	// f := f1 - 1
	SUBSD   X0, X2 // x1= k, x2= f
	// s := f / (2 + f)
	MOVSD   $2.0, X0
	ADDSD   X2, X0
	MOVAPD  X2, X3
	DIVSD   X0, X3 // x1=k, x2= f, x3= s
	// s2 := s * s
	MOVAPD  X3, X4 // x1= k, x2= f, x3= s
	MULSD   X4, X4 // x1= k, x2= f, x3= s, x4= s2
	// s4 := s2 * s2
	MOVAPD  X4, X5 // x1= k, x2= f, x3= s, x4= s2
	MULSD   X5, X5 // x1= k, x2= f, x3= s, x4= s2, x5= s4
	// t1 := s2 * (L1 + s4*(L3+s4*(L5+s4*L7)))
	MOVSD   $L7, X6
	MULSD   X5, X6
	ADDSD   $L5, X6
	MULSD   X5, X6
	ADDSD   $L3, X6
	MULSD   X5, X6
	ADDSD   $L1, X6
	MULSD   X6, X4 // x1= k, x2= f, x3= s, x4= t1, x5= s4
	// t2 := s4 * (L2 + s4*(L4+s4*L6))
	MOVSD   $L6, X6
	MULSD   X5, X6
	ADDSD   $L4, X6
	MULSD   X5, X6
	ADDSD   $L2, X6
	MULSD   X6, X5 // x1= k, x2= f, x3= s, x4= t1, x5= t2
	// R := t1 + t2
	ADDSD   X5, X4 // x1= k, x2= f, x3= s, x4= R
	// hfsq := 0.5 * f * f
	MOVSD   $0.5, X0
	MULSD   X2, X0
	MULSD   X2, X0 // x0= hfsq, x1= k, x2= f, x3= s, x4= R
	// return k*Ln2Hi - ((hfsq - (s*(hfsq+R) + k*Ln2Lo)) - f)
	ADDSD   X0, X4 // x0= hfsq, x1= k, x2= f, x3= s, x4= hfsq+R
	MULSD   X4, X3 // x0= hfsq, x1= k, x2= f, x3= s*(hfsq+R)
	MOVSD   $Ln2Lo, X4
	MULSD   X1, X4 // x4= k*Ln2Lo
	ADDSD   X4, X3 // x0= hfsq, x1= k, x2= f, x3= s*(hfsq+R)+k*Ln2Lo
	SUBSD   X3, X0 // x0= hfsq-(s*(hfsq+R)+k*Ln2Lo), x1= k, x2= f
	SUBSD   X2, X0 // x0= (hfsq-(s*(hfsq+R)+k*Ln2Lo))-f, x1= k
	MULSD   $Ln2Hi, X1 // x0= (hfsq-(s*(hfsq+R)+k*Ln2Lo))-f, x1= k*Ln2Hi
	SUBSD   X0, X1 // x1= k*Ln2Hi-((hfsq-(s*(hfsq+R)+k*Ln2Lo))-f)
  	MOVSD   X1, ret+8(FP)
	RET
isInfOrNaN:
	MOVQ    BX, ret+8(FP) // +Inf or NaN, return x
	RET
isNegative:
	MOVQ    $NaN, AX
	MOVQ    AX, ret+8(FP) // return NaN
	RET
isZero:
	MOVQ    $NegInf, AX
	MOVQ    AX, ret+8(FP) // return -Inf
	RET
