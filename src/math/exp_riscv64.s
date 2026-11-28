// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define NearZero	0x3e30000000000000	// 2**-28
#define PosInf		0x7ff0000000000000
#define FracMask	0x000fffffffffffff
#define C1		0x3cb0000000000000	// 2**-52
#define Exp2DenormalScale	0x3ca0000000000000	// 2**-53

DATA exprodata<>+0(SB)/8, $0.5
DATA exprodata<>+8(SB)/8, $1.0
DATA exprodata<>+16(SB)/8, $2.0
DATA exprodata<>+24(SB)/8, $6.93147180369123816490e-01	// Ln2Hi
DATA exprodata<>+32(SB)/8, $1.90821492927058770002e-10	// Ln2Lo
DATA exprodata<>+40(SB)/8, $-1.90821492927058770002e-10	// -Ln2Lo
DATA exprodata<>+48(SB)/8, $1.44269504088896338700e+00	// Log2e
GLOBL exprodata<>+0(SB), NOPTR|RODATA, $56

DATA expspecialdata<>+0(SB)/8, $7.09782712893383973096e+02	// Overflow
DATA expspecialdata<>+8(SB)/8, $-7.45133219101941108420e+02	// Underflow
DATA expspecialdata<>+16(SB)/8, $1.0239999999999999e+03	// Overflow2
DATA expspecialdata<>+24(SB)/8, $-1075.0			// Underflow2
DATA expspecialdata<>+32(SB)/8, $3.7252902984619141e-09	// NearZero
GLOBL expspecialdata<>+0(SB), NOPTR|RODATA, $40

DATA expmultidata<>+0(SB)/8, $1.66666666666666657415e-01	// P1
DATA expmultidata<>+8(SB)/8, $-2.77777777770155933842e-03	// P2
DATA expmultidata<>+16(SB)/8, $6.61375632143793436117e-05	// P3
DATA expmultidata<>+24(SB)/8, $-1.65339022054652515390e-06	// P4
DATA expmultidata<>+32(SB)/8, $4.13813679705723846039e-08	// P5
GLOBL expmultidata<>+0(SB), NOPTR|RODATA, $40

// Exp returns e**x, the base-e exponential of x.
// This is an assembly implementation of the method used for function Exp in file exp.go.
//
// func Exp(x float64) float64
TEXT ·archExp(SB),$0-16
	MOVD	x+0(FP), F0	// F0 = x

	MOV	$exprodata<>+0(SB), X5
	MOV	$expspecialdata<>+0(SB), X6
	FEQD	F0, F0, X7
	BEQ	X0, X7, isNaN		// x = NaN, return NaN

	MOVD	0(X6), F1	// Overflow
	FLTD	F0, F1, X7
	BNE	X0, X7, overflow	// x > Overflow, return PosInf

	MOVD	8(X6), F2	// Underflow
	FLTD	F2, F0, X7
	BNE	X0, X7, underflow	// x < Underflow, return 0

	FABSD	F0, F5
	MOVD	32(X6), F3	// NearZero
	FLTD	F3, F5, X7
	BNE	X0, X7, nearzero	// fabs(x) < NearZero, return 1 + x

	// argument reduction, x = k*ln2 + r,  |r| <= 0.5*ln2
	// computed as r = hi - lo for extra precision.
	MOVD	0(X5), F3	// 0.5
	MOVD	48(X5), F2	// Log2e
	FSGNJD	F0, F3, F3	// F3 = copysign(0.5, x)
	FMADDD	F0, F2, F3, F3	// Log2e*x + 0.5 or -0.5

	FCVTLD.RTZ	F3, X16	// float64 -> int64
	FCVTDL	X16, F3		// int64 -> float64

	MOVD	24(X5), F4	// Ln2Hi
	MOVD	32(X5), F5	// Ln2Lo
	FNMSUBD	F3, F4, F0, F4
	FMULD	F3, F5, F5
	FSUBD	F5, F4, F6
	FMULD	F6, F6, F7

	// compute c
	// FMADDD x, y, z, r computes r = x*y + z.
	MOV	$expmultidata<>+0(SB), X8
	MOVD	0(X8), F12	// P1
	MOVD	8(X8), F11	// P2
	MOVD	16(X8), F10	// P3
	MOVD	24(X8), F9	// P4
	MOVD	32(X8), F8	// P5
	// Estrin's scheme: A, B, r4 are mutually independent and
	// can be computed in parallel; shortens the dependency
	// chain from 4 serial FMAs to 3.
	FMULD	F7, F7, F18		// F18 = r4 = r2*r2
	FMADDD	F7, F11, F12, F19	// F19 = A = P1 + P2*r2
	FMADDD	F7, F9, F10, F20	// F20 = B = P3 + P4*r2
	FMADDD	F18, F8, F20, F20	// F20 = C = B + P5*r4
	FMADDD	F18, F20, F19, F13	// F13 = A + C*r4
	FNMSUBD	F7, F13, F6, F13

	// compute y
	MOVD	16(X5), F14	// 2.0
	FSUBD	F13, F14, F14
	FMULD	F6, F13, F15
	FDIVD	F14, F15, F15

	MOVD	8(X5), F17	// 1.0
	FSUBD	F15, F5, F15
	FSUBD	F4, F15, F15
	FSUBD	F15, F17, F16

	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	MOVD	F16, X15
	MOV	$FracMask, X20
	AND	X20, X15, X17	// fraction
	SRL	$52, X15, X18	// exponent
	ADD	X16, X18
	MOV	$1, X21
	BGE	X18, X21, normal
	ADD	$52, X18		// denormal
	MOV	$C1, X19
	MOVD	X19, F17
normal:
	SLL	$52, X18
	OR	X18, X17, X15
	MOVD	X15, F0
	FMULD	F17, F0, F0		// return m * x
	MOVD	F0, ret+8(FP)
	RET
nearzero:
	MOVD	8(X5), F17	// 1.0
	FADDD	F17, F0, F0
isNaN:
	MOVD	F0, ret+8(FP)
	RET
underflow:
	MOV	X0, ret+8(FP)
	RET
overflow:
	MOV	$PosInf, X15
	MOV	X15, ret+8(FP)
	RET


// Exp2 returns 2**x, the base-2 exponential of x.
// This is an assembly implementation of the method used for function Exp2 in file exp.go.
//
// func Exp2(x float64) float64
TEXT ·archExp2(SB),$0-16
	MOVD	x+0(FP), F0	// F0 = x

	MOV	$exprodata<>+0(SB), X5
	MOV	$expspecialdata<>+0(SB), X6
	FEQD	F0, F0, X7
	BEQ	X0, X7, isNaN		// x = NaN, return NaN

	MOVD	16(X6), F1	// Overflow2
	FLTD	F0, F1, X7
	BNE	X0, X7, overflow	// x > Overflow2, return PosInf

	MOVD	24(X6), F2	// Underflow2
	FLTD	F2, F0, X7
	BNE	X0, X7, underflow	// x < Underflow2, return 0

	// argument reduction; x = r*lg(e) + k with |r| <= ln(2)/2
	// computed as r = hi - lo for extra precision.
	MOVD	0(X5), F2	// 0.5
	FSGNJD	F0, F2, F3	// F3 = copysign(0.5, x)
	FADDD	F3, F0, F3	// F3 = x + 0.5 or -0.5

	FCVTLD.RTZ	F3, X16
	FCVTDL	X16, F3

	MOVD	24(X5), F4	// Ln2Hi
	MOVD	40(X5), F5	// -Ln2Lo
	FSUBD	F3, F0, F3
	FMULD	F3, F4, F4
	FMULD	F3, F5, F5
	FSUBD	F5, F4, F6
	FMULD	F6, F6, F7

	// compute c
	MOV	$expmultidata<>+0(SB), X8
	MOVD	0(X8), F12	// P1
	MOVD	8(X8), F11	// P2
	MOVD	16(X8), F10	// P3
	MOVD	24(X8), F9	// P4
	MOVD	32(X8), F8	// P5
	// Estrin's scheme: A, B, r4 are mutually independent and
	// can be computed in parallel; shortens the dependency
	// chain from 4 serial FMAs to 3.
	FMULD	F7, F7, F18		// F18 = r4 = r2*r2
	FMADDD	F7, F11, F12, F19	// F19 = A = P1 + P2*r2
	FMADDD	F7, F9, F10, F20	// F20 = B = P3 + P4*r2
	FMADDD	F18, F8, F20, F20	// F20 = C = B + P5*r4
	FMADDD	F18, F20, F19, F13	// F13 = A + C*r4
	FNMSUBD	F7, F13, F6, F13

	// compute y
	MOVD	16(X5), F14	// 2.0
	FSUBD	F13, F14, F14
	FMULD	F6, F13, F15
	FDIVD	F14, F15, F15

	MOVD	8(X5), F17	// 1.0
	FSUBD	F15, F5, F15
	FSUBD	F4, F15, F15
	FSUBD	F15, F17, F16

	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	MOVD	F16, X15
	MOV	$FracMask, X20
	SRL	$52, X15, X18	// exponent
	AND	X20, X15, X17	// fraction
	ADD	X16, X18
	MOV	$1, X21
	BGE	X18, X21, normal

	ADD	$53, X18		// shift into the normal range
	MOV	$Exp2DenormalScale, X19
	MOVD	X19, F17
normal:
	SLL	$52, X18
	OR	X18, X17, X15
	MOVD	X15, F0
	FMULD	F17, F0, F0
isNaN:
	MOVD	F0, ret+8(FP)
	RET
underflow:
	MOV	X0, ret+8(FP)
	RET
overflow:
	MOV	$PosInf, X15
	MOV	X15, ret+8(FP)
	RET
