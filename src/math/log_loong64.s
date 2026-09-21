// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define HSqrt2 7.07106781186547524401e-01 // sqrt(2)/2
#define Ln2Hi  6.93147180369123816490e-01
#define Ln2Lo  1.90821492927058770002e-10
#define L1     6.666666666666735130e-01
#define L2     3.999999999940941908e-01
#define L3     2.857142874366239149e-01
#define L4     2.222219843214978396e-01
#define L5     1.818357216161805012e-01
#define L6     1.531383769920937332e-01
#define L7     1.479819860511658591e-01
#define NaN     0x7FF8000000000001
#define NegInf  0xFFF0000000000000
#define PosInf  0x7FF0000000000000
#define FracMask 0x000FFFFFFFFFFFFF
#define HalfExp  0x3FE0000000000000 // bit pattern of 0.5, used to force f1 into [0.5,1)

DATA logrodata<>+0(SB)/8, $0.5
DATA logrodata<>+8(SB)/8, $1.0
DATA logrodata<>+16(SB)/8, $2.0
DATA logrodata<>+24(SB)/8, $HSqrt2
DATA logrodata<>+32(SB)/8, $Ln2Hi
DATA logrodata<>+40(SB)/8, $Ln2Lo
DATA logrodata<>+48(SB)/8, $L1
DATA logrodata<>+56(SB)/8, $L2
DATA logrodata<>+64(SB)/8, $L3
DATA logrodata<>+72(SB)/8, $L4
DATA logrodata<>+80(SB)/8, $L5
DATA logrodata<>+88(SB)/8, $L6
DATA logrodata<>+96(SB)/8, $L7
GLOBL logrodata<>+0(SB), NOPTR|RODATA, $104

// func archLog(x float64) float64
TEXT ·archLog(SB),NOSPLIT,$0
	MOVD	x+0(FP), F0
	MOVV	F0, R4			// R4 = raw bits of x

	MOVV	$~(1<<63), R5
	AND	R4, R5, R6		// R6 = |bits(x)|
	BEQ	R6, R0, isZero

	BLT	R4, R0, isNegative	// sign bit set -> x < 0

	MOVV	$PosInf, R8
	BGE	R4, R8, isInfOrNaN	// bits(x) >= PosInf pattern -> +Inf or NaN

	MOVV	$logrodata<>+0(SB), R13

	// f1, ki := Frexp(x)
	MOVV	$FracMask, R9
	AND	R4, R9, R10		// fraction bits
	MOVV	$HalfExp, R11
	OR	R11, R10, R10		// f1 bits, f1 in [0.5, 1)
	MOVV	R10, F1			// F1 = f1

	SRLV	$52, R4, R12
	AND	$0x7FF, R12, R12	// biased exponent
	ADDV	$-0x3FE, R12, R12	// R12 = ki
	MOVV	R12, F2
	FFINTDV	F2, F2			// F2 = k (float64)

	// if f1 < Sqrt2/2 { k -= 1; f1 *= 2 }
	MOVD	24(R13), F3		// HSqrt2
	CMPGTD	F3, F1, FCC0		// FCC0 = (HSqrt2 > f1)

	MOVD	8(R13), F4		// 1.0
	SUBD	F4, F2, F17
	MOVD	16(R13), F18		// 2.0
	MULD	F18, F1, F19
	FSEL	FCC0, F17, F2, F2
	FSEL	FCC0, F19, F1, F1

	SUBD	F4, F1, F3		// f := f1 - 1

	// s := f / (2 + f)
	MOVD	16(R13), F5
	ADDD	F3, F5, F5		// 2 + f
	MOVD	F3, F6
	DIVD	F5, F6, F6		// s

	MULD	F6, F6, F7		// s2
	MULD	F7, F7, F8		// s4

	// t1 := s2 * (L1 + s4*(L3 + s4*(L5 + s4*L7)))
	MOVD	96(R13), F9		// L7
	MOVD	80(R13), F10		// L5
	FMADDD	F10, F8, F9, F9		// F9 = L7*s4 + L5
	MOVD	64(R13), F10		// L3
	FMADDD	F10, F8, F9, F9		// F9 = F9*s4 + L3
	MOVD	48(R13), F10		// L1
	FMADDD	F10, F8, F9, F9		// F9 = F9*s4 + L1
	MULD	F7, F9, F9		// t1 = F9 * s2

	// t2 := s4 * (L2 + s4*(L4 + s4*L6))
	MOVD	88(R13), F11		// L6
	MOVD	72(R13), F12		// L4
	FMADDD	F12, F8, F11, F11	// F11 = L6*s4 + L4
	MOVD	56(R13), F12		// L2
	FMADDD	F12, F8, F11, F11	// F11 = F11*s4 + L2
	MULD	F8, F11, F11		// t2 = F11 * s4

	ADDD	F11, F9, F9		// R = t1 + t2

	// hfsq := 0.5 * f * f
	MOVD	0(R13), F14		// 0.5
	MULD	F3, F14, F14
	MULD	F3, F14, F14		// hfsq

	// return k*Ln2Hi - ((hfsq - (s*(hfsq+R) + k*Ln2Lo)) - f)
	ADDD	F14, F9, F9		// hfsq + R
	MOVD	40(R13), F15		// Ln2Lo
	MULD	F2, F15, F15		// k*Ln2Lo
	FMADDD	F15, F6, F9, F9		// F9 = s*(hfsq+R) + k*Ln2Lo
	SUBD	F9, F14, F14
	SUBD	F3, F14, F14
	MOVD	32(R13), F16		// Ln2Hi
	FMSUBD	F14, F2, F16, F16

	MOVD	F16, ret+8(FP)
	RET
isInfOrNaN:
	MOVD	F0, ret+8(FP)		// +Inf or NaN, return x
	RET
isNegative:
	MOVV	$NaN, R6
	MOVV	R6, ret+8(FP)
	RET
isZero:
	MOVV	$NegInf, R6
	MOVV	R6, ret+8(FP)
	RET
