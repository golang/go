// Copyright 2025 The Go Authors. All rights reserved.
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

	MOVV	$exprodata<>+0(SB), R10
	MOVD	56(R10), F1	// Overflow
	MOVD	64(R10), F2	// Underflow
	MOVD	88(R10), F3	// NearZero
	MOVD	16(R10), F17	// 1.0

	CMPEQD	F0, F0, FCC0
	BFPF	isNaN		// x = NaN, return NaN

	CMPGTD	F0, F1, FCC0
	BFPT	overflow	// x > Overflow, return PosInf

	CMPGTD	F2, F0, FCC0
	BFPT	underflow	// x < Underflow, return 0

	ABSD	F0, F5
	CMPGTD	F3, F5, FCC0
	BFPT	nearzero	// fabs(x) < NearZero, return 1 + x

	// argument reduction, x = k*ln2 + r,  |r| <= 0.5*ln2
	// computed as r = hi - lo for extra precision.
	MOVD	0(R10), F5
	MOVD	8(R10), F3
	MOVD	48(R10), F2
	CMPGTD	F0, F5, FCC0
	BFPT	add		// x > 0
sub:
	FMSUBD	F3, F2, F0, F3	// Log2e*x - 0.5
	JMP	2(PC)
add:
	FMADDD	F3, F2, F0, F3	// Log2e*x + 0.5

	FTINTRZVD F3, F4	// float64 -> int64
	MOVV	F4, R5		// R5 = int(k)
	FFINTDV	F4, F3		// int64 -> float64

	MOVD	32(R10), F4
	MOVD	40(R10), F5
	FNMSUBD	F0, F3, F4, F4
	MULD	F3, F5, F5
	SUBD	F5, F4, F6
	MULD	F6, F6, F7

	// compute c
	MOVV	$expmultirodata<>+0(SB), R11
	MOVD	32(R11), F8
	MOVD	24(R11), F9
	FMADDD	F9, F8, F7, F13
	MOVD	16(R11), F10
	FMADDD	F10, F13, F7, F13
	MOVD	8(R11), F11
	FMADDD	F11, F13, F7, F13
	MOVD	0(R11), F12
	FMADDD	F12, F13, F7, F13
	FNMSUBD	F6, F13, F7, F13

	// compute y
	MOVD	24(R10), F14
	SUBD	F13, F14, F14
	MULD	F6, F13, F15
	DIVD	F14, F15, F15
	SUBD	F15, F5, F15
	SUBD	F4, F15, F15
	SUBD	F15, F17, F16

	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	MOVV	F16, R4
	MOVV	$FracMask, R9
	AND	R9, R4, R6	// fraction
	SRLV	$52, R4, R7	// exponent
	ADDV	R5, R7
	MOVV	$1, R12
	BGE	R7, R12, normal
	ADDV	$52, R7		// denormal
	MOVV	$C1, R8
	MOVV	R8, F17
normal:
	SLLV	$52, R7
	OR	R7, R6, R4
	MOVV	R4, F0
	MULD	F17, F0		// return m * x
	MOVD	F0, ret+8(FP)
	RET
nearzero:
	ADDD	F17, F0, F0
isNaN:
	MOVD	F0, ret+8(FP)
	RET
underflow:
	MOVV	R0, ret+8(FP)
	RET
overflow:
	MOVV	$PosInf, R4
	MOVV	R4, ret+8(FP)
	RET


// Exp2 returns 2**x, the base-2 exponential of x.
// This is an assembly implementation of the method used for function Exp2 in file exp.go.
//
// func Exp2(x float64) float64
TEXT ·archExp2(SB),$0-16
	MOVD	x+0(FP), F0	// F0 = x

	MOVV	$exprodata<>+0(SB), R10
	MOVD	72(R10), F1	// Overflow2
	MOVD	80(R10), F2	// Underflow2
	MOVD	88(R10), F3	// NearZero

	CMPEQD	F0, F0, FCC0
	BFPF	isNaN		// x = NaN, return NaN

	CMPGTD	F0, F1, FCC0
	BFPT	overflow	// x > Overflow, return PosInf

	CMPGTD	F2, F0, FCC0
	BFPT	underflow	// x < Underflow, return 0

	// argument reduction; x = r*lg(e) + k with |r| <= ln(2)/2
	// computed as r = hi - lo for extra precision.
	MOVD	0(R10), F10
	MOVD	8(R10), F2
	CMPGTD	F0, F10, FCC0
	BFPT	add
sub:
	SUBD	F2, F0, F3	// x - 0.5
	JMP	2(PC)
add:
	ADDD	F2, F0, F3	// x + 0.5

	FTINTRZVD F3, F4
	MOVV	F4, R5
	FFINTDV	F4, F3

	MOVD	32(R10), F4
	MOVD	40(R10), F5
	SUBD	F3, F0, F3
	MULD	F3, F4
	FNMSUBD	F10, F3, F5, F5
	SUBD	F5, F4, F6
	MULD	F6, F6, F7

	// compute c
	MOVV	$expmultirodata<>+0(SB), R11
	MOVD	32(R11), F8
	MOVD	24(R11), F9
	FMADDD	F9, F8, F7, F13
	MOVD	16(R11), F10
	FMADDD	F10, F13, F7, F13
	MOVD	8(R11), F11
	FMADDD	F11, F13, F7, F13
	MOVD	0(R11), F12
	FMADDD	F12, F13, F7, F13
	FNMSUBD	F6, F13, F7, F13

	// compute y
	MOVD	24(R10), F14
	SUBD	F13, F14, F14
	MULD	F6, F13, F15
	DIVD	F14, F15

	MOVD	16(R10), F17
	SUBD	F15, F5, F15
	SUBD	F4, F15, F15
	SUBD	F15, F17, F16

	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	MOVV	F16, R4
	MOVV	$FracMask, R9
	SRLV	$52, R4, R7	// exponent
	AND	R9, R4, R6	// fraction
	ADDV	R5, R7
	MOVV	$1, R12
	BGE	R7, R12, normal

	ADDV	$52, R7		// denormal
	MOVV	$C1, R8
	MOVV	R8, F17
normal:
	SLLV	$52, R7
	OR	R7, R6, R4
	MOVV	R4, F0
	MULD	F17, F0
isNaN:
	MOVD	F0, ret+8(FP)
	RET
underflow:
	MOVV	R0, ret+8(FP)
	RET
overflow:
	MOVV	$PosInf, R4
	MOVV	R4, ret+8(FP)
	RET
