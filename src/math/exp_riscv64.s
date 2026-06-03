// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define NearZero	0x3e30000000000000	// 2**-28
#define PosInf		0x7ff0000000000000
#define FracMask	0x000fffffffffffff
#define C1		0x3cb0000000000000	// 2**-52

DATA exprodata<>+0(SB)/8, $0.0
DATA exprodata<>+8(SB)/8, $0.5
DATA exprodata<>+16(SB)/8, $1.0
DATA exprodata<>+24(SB)/8, $2.0
DATA exprodata<>+32(SB)/8, $6.93147180369123816490e-01	// Ln2Hi
DATA exprodata<>+40(SB)/8, $1.90821492927058770002e-10	// Ln2Lo
DATA exprodata<>+48(SB)/8, $1.44269504088896338700e+00	// Log2e
DATA exprodata<>+56(SB)/8, $7.09782712893383973096e+02	// Overflow
DATA exprodata<>+64(SB)/8, $-7.45133219101941108420e+02	// Underflow
DATA exprodata<>+72(SB)/8, $1.0239999999999999e+03	// Overflow2
DATA exprodata<>+80(SB)/8, $-1.0740e+03			// Underflow2
DATA exprodata<>+88(SB)/8, $3.7252902984619141e-09	// NearZero
GLOBL exprodata<>+0(SB), NOPTR|RODATA, $96

DATA expmultirodata<>+0(SB)/8, $1.66666666666666657415e-01	// P1
DATA expmultirodata<>+8(SB)/8, $-2.77777777770155933842e-03	// P2
DATA expmultirodata<>+16(SB)/8, $6.61375632143793436117e-05	// P3
DATA expmultirodata<>+24(SB)/8, $-1.65339022054652515390e-06	// P4
DATA expmultirodata<>+32(SB)/8, $4.13813679705723846039e-08	// P5
GLOBL expmultirodata<>+0(SB), NOPTR|RODATA, $40

// Exp returns e**x, the base-e exponential of x.
// This is an assembly implementation of the method used for function Exp in file exp.go.
//
// func Exp(x float64) float64
TEXT ·archExp(SB),$0-16
	MOVD	x+0(FP), F0	// F0 = x

	MOV	$exprodata<>+0(SB), X5
	MOVD	56(X5), F1	// Overflow
	MOVD	64(X5), F2	// Underflow
	MOVD	88(X5), F3	// NearZero
	MOVD	16(X5), F17	// 1.0

	FEQD	F0, F0, X7
	BEQ	X0, X7, isNaN		// x = NaN, return NaN

	FLTD	F0, F1, X7
	BNE	X0, X7, overflow	// x > Overflow, return PosInf

	FLTD	F2, F0, X7
	BNE	X0, X7, underflow	// x < Underflow, return 0

	FABSD	F0, F5
	FLTD	F3, F5, X7
	BNE	X0, X7, nearzero	// fabs(x) < NearZero, return 1 + x

	// argument reduction, x = k*ln2 + r,  |r| <= 0.5*ln2
	// computed as r = hi - lo for extra precision.
	MOVD	0(X5), F5
	MOVD	8(X5), F3
	MOVD	48(X5), F2
	FLTD	F0, F5, X7
	BNE	X0, X7, add		// x > 0
sub:
	FMSUBD	F0, F2, F3, F3	// Log2e*x - 0.5
	JMP	2(PC)
add:
	FMADDD	F0, F2, F3, F3	// Log2e*x + 0.5

	FCVTLD.RTZ	F3, X16	// float64 -> int64
	FCVTDL	X16, F3		// int64 -> float64

	MOVD	32(X5), F4
	MOVD	40(X5), F5
	FNMSUBD	F3, F4, F0, F4
	FMULD	F3, F5, F5
	FSUBD	F5, F4, F6
	FMULD	F6, F6, F7

	// compute c
	// r=(FMA x y z) -> FMADDD z, y, x, r
	// r=(FMA x y z) -> FMADDD x, y, z, r
	MOV	$expmultirodata<>+0(SB), X6
	MOVD	32(X6), F8
	MOVD	24(X6), F9
	FMADDD	F7, F8, F9, F13
	MOVD	16(X6), F10
	FMADDD	F7, F13, F10, F13
	MOVD	8(X6), F11
	FMADDD	F7, F13, F11, F13
	MOVD	0(X6), F12
	FMADDD	F7, F13, F12, F13
	FNMSUBD	F7, F13, F6, F13

	// compute y
	MOVD	24(X5), F14
	FSUBD	F13, F14, F14
	FMULD	F6, F13, F15
	FDIVD	F14, F15, F15
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
	MOVD	72(X5), F1	// Overflow2
	MOVD	80(X5), F2	// Underflow2
	MOVD	88(X5), F3	// NearZero

	FEQD	F0, F0, X7
	BEQ	X0, X7, isNaN		// x = NaN, return NaN

	FLTD	F0, F1, X7
	BNE	X0, X7, overflow	// x > Overflow, return PosInf

	FLTD	F2, F0, X7
	BNE	X0, X7, underflow	// x < Underflow, return 0

	// argument reduction; x = r*lg(e) + k with |r| <= ln(2)/2
	// computed as r = hi - lo for extra precision.
	MOVD	0(X5), F10
	MOVD	8(X5), F2
	FLTD	F0, F10, X7
	BNE	X0, X7, add
sub:
	FSUBD	F2, F0, F3	// x - 0.5
	JMP	2(PC)
add:
	FADDD	F2, F0, F3	// x + 0.5

	FCVTLD.RTZ	F3, X16
	FCVTDL	X16, F3

	MOVD	32(X5), F4
	MOVD	40(X5), F5
	FSUBD	F3, F0, F3
	FMULD	F3, F4, F4
	FNMSUBD	F5, F3, F10, F5
	FSUBD	F5, F4, F6
	FMULD	F6, F6, F7

	// compute c
	MOV	$expmultirodata<>+0(SB), X6
	MOVD	32(X6), F8
	MOVD	24(X6), F9
	FMADDD	F7, F8, F9, F13
	MOVD	16(X6), F10
	FMADDD	F7, F13, F10, F13
	MOVD	8(X6), F11
	FMADDD	F7, F13, F11, F13
	MOVD	0(X6), F12
	FMADDD	F7, F13, F12, F13
	FNMSUBD	F7, F13, F6, F13

	// compute y
	MOVD	24(X5), F14
	FSUBD	F13, F14, F14
	FMULD	F6, F13, F15
	FDIVD	F14, F15, F15

	MOVD	16(X5), F17
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

	ADD	$52, X18		// denormal
	MOV	$C1, X19
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
