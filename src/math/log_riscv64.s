// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define Ln2Hi  6.93147180369123816490e-01
#define Ln2Lo  1.90821492927058770002e-10
#define L1     6.666666666666735130e-01
#define L2     3.999999999940941908e-01
#define L3     2.857142874366239149e-01
#define L4     2.222219843214978396e-01
#define L5     1.818357216161805012e-01
#define L6     1.531383769920937332e-01
#define L7     1.479819860511658591e-01
#define Two52  4.503599627370496e+15	// 2**52
#define NaN    0x7FF8000000000001
#define NegInf 0xFFF0000000000000

// Mantissa of Sqrt2/2, shifted up by 12. Frexp yields f1 = 0x3FE<<52 | frac,
// so f1 and Sqrt2/2 share an exponent and f1 < Sqrt2/2 is exactly
// frac < mantissa(Sqrt2/2), which is compared with the sign and exponent
// fields already shifted out.
#define HSqrt2Frac12 0x6A09E667F3BCD000

DATA logrodata<>+0(SB)/8, $0.5
DATA logrodata<>+8(SB)/8, $1.0
DATA logrodata<>+16(SB)/8, $Ln2Hi
DATA logrodata<>+24(SB)/8, $Ln2Lo
DATA logrodata<>+32(SB)/8, $L1
DATA logrodata<>+40(SB)/8, $L2
DATA logrodata<>+48(SB)/8, $L3
DATA logrodata<>+56(SB)/8, $L4
DATA logrodata<>+64(SB)/8, $L5
DATA logrodata<>+72(SB)/8, $L6
DATA logrodata<>+80(SB)/8, $L7
DATA logrodata<>+88(SB)/8, $Two52
DATA logrodata<>+96(SB)/8, $HSqrt2Frac12
GLOBL logrodata<>+0(SB), NOPTR|RODATA, $104

// func archLog(x float64) float64
TEXT ·archLog(SB),NOSPLIT,$0-16
	// The fast path only ever looks at the bits of x, so load them
	// straight into an integer register instead of paying for a
	// floating-point load plus an FMV.X.D.
	MOV	x+0(FP), X5		// X5 = raw bits of x

	BLT	X5, X0, isNegative	// sign bit set -> x < 0 or x == -0

	SRL	$52, X5, X12		// X12 = biased exponent (sign bit is clear)
	MOV	$0x7FF, X6
	BEQ	X6, X12, isInfOrNaN	// +Inf or NaN, return x

	// Start the constant loads early: they are independent of the
	// bit twiddling below and hide its latency.
	MOV	$logrodata<>+0(SB), X13
	MOV	0(X13), X11		// bit pattern of 0.5
	MOV	96(X13), X14		// HSqrt2Frac12
	MOVD	8(X13), F4		// 1.0

	BEQ	X0, X12, isSmall	// +0 or subnormal

normalized:
	// f1, ki := Frexp(x); if f1 < Sqrt2/2 { f1 *= 2; ki-- }
	//
	SLL	$12, X5, X9		// fraction bits, shifted up by 12
	SLTU	X14, X9, X7		// X7 = 1 if f1 < Sqrt2/2
	SRL	$12, X9, X10		// fraction bits
	OR	X11, X10, X10		// exponent 0x3FE: f1 in [0.5, 1)
	SLL	$52, X7, X8
	ADD	X8, X10, X10		// exponent 0x3FF when f1 is doubled
	FMVDX	X10, F1			// F1 = f1, in [Sqrt2/2, Sqrt2)

	ADD	$-1022, X12, X12
	SUB	X7, X12, X12		// X12 = ki
	FCVTDL	X12, F2			// F2 = k (float64)

	// f1 is in [Sqrt2/2, Sqrt2), so f1-1 is exact and 2+f is the
	// correctly rounded sum of the same exact value as 1+f1.
	FSUBD	F4, F1, F5		// f := f1 - 1
	FADDD	F4, F1, F6		// 2 + f
	FDIVD	F6, F5, F7		// s := f / (2 + f)

	MOVD	32(X13), F17		// L1
	MOVD	40(X13), F18		// L2
	MOVD	48(X13), F19		// L3
	MOVD	56(X13), F20		// L4
	MOVD	64(X13), F21		// L5
	MOVD	72(X13), F22		// L6
	MOVD	80(X13), F23		// L7
	MOVD	16(X13), F15		// Ln2Hi
	MOVD	24(X13), F16		// Ln2Lo
	MOVD	0(X13), F14		// 0.5
	FMULD	F5, F14, F12
	FMULD	F5, F12, F12		// hfsq := 0.5 * f * f
	FMULD	F2, F16, F13		// k*Ln2Lo

	FMULD	F7, F7, F8		// s2
	FMULD	F8, F8, F9		// s4

	// t1 := s2 * (L1 + s4*(L3 + s4*(L5 + s4*L7)))
	// t2 := s4 * (L2 + s4*(L4 + s4*L6))
	FMADDD	F9, F23, F21, F10	// s4*L7 + L5
	FMADDD	F9, F22, F20, F11	// s4*L6 + L4
	FMADDD	F9, F10, F19, F10	// s4*F10 + L3
	FMADDD	F9, F11, F18, F11	// s4*F11 + L2
	FMADDD	F9, F10, F17, F10	// s4*F10 + L1
	FMULD	F9, F11, F11		// t2
	FMULD	F8, F10, F10		// t1

	FADDD	F11, F10, F10		// R := t1 + t2

	// return k*Ln2Hi - ((hfsq - (s*(hfsq+R) + k*Ln2Lo)) - f)
	FADDD	F12, F10, F10		// hfsq + R
	FMADDD	F7, F10, F13, F10	// s*(hfsq+R) + k*Ln2Lo
	FSUBD	F10, F12, F12
	FSUBD	F5, F12, F12
	FMSUBD	F2, F15, F12, F24

	MOVD	F24, ret+8(FP)
	RET
isSmall:
	BEQ	X0, X5, isZero

	FMVDX	X5, F0
	MOVD	88(X13), F1
	FMULD	F1, F0, F0
	FMVXD	F0, X5
	SRL	$52, X5, X12
	ADD	$-52, X12, X12
	JMP	normalized
isInfOrNaN:
	MOV	X5, ret+8(FP)		// +Inf or NaN, return x
	RET
isNegative:
	SLL	$1, X5, X6
	BEQ	X0, X6, isZero		// x == -0, return -Inf
	MOV	$NaN, X6
	MOV	X6, ret+8(FP)
	RET
isZero:
	MOV	$NegInf, X6
	MOV	X6, ret+8(FP)
	RET
