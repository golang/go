// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define	Ln2Hi	6.93147180369123816490e-01
#define	Ln2Lo	1.90821492927058770002e-10
#define	Log2e	1.44269504088896338700e+00
#define	Overflow	7.09782712893383973096e+02
#define	Underflow	-7.45133219101941108420e+02
#define	Overflow2	1.0239999999999999e+03
#define	Underflow2	-1.0740e+03
#define	NearZero	0x3e30000000000000	// 2**-28
#define	PosInf	0x7ff0000000000000
#define	FracMask	0x000fffffffffffff
#define	C1	0x3cb0000000000000	// 2**-52
#define	P1	1.66666666666666657415e-01	// 0x3FC55555; 0x55555555
#define	P2	-2.77777777770155933842e-03	// 0xBF66C16C; 0x16BEBD93
#define	P3	6.61375632143793436117e-05	// 0x3F11566A; 0xAF25DE2C
#define	P4	-1.65339022054652515390e-06	// 0xBEBBBD41; 0xC5D26BF1
#define	P5	4.13813679705723846039e-08	// 0x3E663769; 0x72BEA4D0

// Exp returns e**x, the base-e exponential of x.
// This is an assembly implementation of the method used for function Exp in file exp.go.
//
// func Exp(x float64) float64
TEXT ·Exp(SB),$0-16
	FMOVD	x+0(FP), F0	// F0 = x
	FCMPD	F0, F0
	BNE	isNaN		// x = NaN, return NaN
	FMOVD	$Overflow, F1
	FCMPD	F1, F0
	BGT	overflow	// x > Overflow, return PosInf
	FMOVD	$Underflow, F1
	FCMPD	F1, F0
	BLT	underflow	// x < Underflow, return 0
	MOVD	$NearZero, R0
	FMOVD	R0, F2
	FABSD	F0, F3
	FMOVD	$1.0, F1	// F1 = 1.0
	FCMPD	F2, F3
	BLT	nearzero	// fabs(x) < NearZero, return 1 + x
	// argument reduction, x = k*ln2 + r,  |r| <= 0.5*ln2
	// computed as r = hi - lo for extra precision.
	FMOVD	$Log2e, F2
	FMOVD	$0.5, F3
	FNMSUBD	F0, F3, F2, F4	// Log2e*x - 0.5
	FMADDD	F0, F3, F2, F3	// Log2e*x + 0.5
	FCMPD	$0.0, F0
	FCSELD	LT, F4, F3, F3	// F3 = k
	FCVTZSD	F3, R1		// R1 = int(k)
	SCVTFD	R1, F3		// F3 = float64(int(k))
	FMOVD	$Ln2Hi, F4	// F4 = Ln2Hi
	FMOVD	$Ln2Lo, F5	// F5 = Ln2Lo
	FMSUBD	F3, F0, F4, F4	// F4 = hi = x - float64(int(k))*Ln2Hi
	FMULD	F3, F5		// F5 = lo = float64(int(k)) * Ln2Lo
	FSUBD	F5, F4, F6	// F6 = r = hi - lo
	FMULD	F6, F6, F7	// F7 = t = r * r
	// compute y
	FMOVD	$P5, F8		// F8 = P5
	FMOVD	$P4, F9		// F9 = P4
	FMADDD	F7, F9, F8, F13	// P4+t*P5
	FMOVD	$P3, F10	// F10 = P3
	FMADDD	F7, F10, F13, F13	// P3+t*(P4+t*P5)
	FMOVD	$P2, F11	// F11 = P2
	FMADDD	F7, F11, F13, F13	// P2+t*(P3+t*(P4+t*P5))
	FMOVD	$P1, F12	// F12 = P1
	FMADDD	F7, F12, F13, F13	// P1+t*(P2+t*(P3+t*(P4+t*P5)))
	FMSUBD	F7, F6, F13, F13	// F13 = c = r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
	FMOVD	$2.0, F14
	FSUBD	F13, F14
	FMULD	F6, F13, F15
	FDIVD	F14, F15	// F15 = (r*c)/(2-c)
	FSUBD	F15, F5, F15	// lo-(r*c)/(2-c)
	FSUBD	F4, F15, F15	// (lo-(r*c)/(2-c))-hi
	FSUBD	F15, F1, F16	// F16 = y = 1-((lo-(r*c)/(2-c))-hi)
	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	FMOVD	F16, R0
	AND	$FracMask, R0, R2	// fraction
	LSR	$52, R0, R5	// exponent
	ADD	R1, R5		// R1 = int(k)
	CMP	$1, R5
	BGE	normal
	ADD	$52, R5		// denormal
	MOVD	$C1, R8
	FMOVD	R8, F1		// m = 2**-52
normal:
	ORR	R5<<52, R2, R0
	FMOVD	R0, F0
	FMULD	F1, F0		// return m * x
	FMOVD	F0, ret+8(FP)
	RET
nearzero:
	FADDD	F1, F0
isNaN:
	FMOVD	F0, ret+8(FP)
	RET
underflow:
	MOVD	ZR, ret+8(FP)
	RET
overflow:
	MOVD	$PosInf, R0
	MOVD	R0, ret+8(FP)
	RET


// Exp2 returns 2**x, the base-2 exponential of x.
// This is an assembly implementation of the method used for function Exp2 in file exp.go.
//
// func Exp2(x float64) float64
TEXT ·Exp2(SB),$0-16
	FMOVD	x+0(FP), F0	// F0 = x
	FCMPD	F0, F0
	BNE	isNaN		// x = NaN, return NaN
	FMOVD	$Overflow2, F1
	FCMPD	F1, F0
	BGT	overflow	// x > Overflow, return PosInf
	FMOVD	$Underflow2, F1
	FCMPD	F1, F0
	BLT	underflow	// x < Underflow, return 0
	// argument reduction; x = r*lg(e) + k with |r| <= ln(2)/2
	// computed as r = hi - lo for extra precision.
	FMOVD	$0.5, F2
	FSUBD	F2, F0, F3	// x + 0.5
	FADDD	F2, F0, F4	// x - 0.5
	FCMPD	$0.0, F0
	FCSELD	LT, F3, F4, F3	// F3 = k
	FCVTZSD	F3, R1		// R1 = int(k)
	SCVTFD	R1, F3		// F3 = float64(int(k))
	FSUBD	F3, F0, F3	// t = x - float64(int(k))
	FMOVD	$Ln2Hi, F4	// F4 = Ln2Hi
	FMOVD	$Ln2Lo, F5	// F5 = Ln2Lo
	FMULD	F3, F4		// F4 = hi = t * Ln2Hi
	FNMULD	F3, F5		// F5 = lo = -t * Ln2Lo
	FSUBD	F5, F4, F6	// F6 = r = hi - lo
	FMULD	F6, F6, F7	// F7 = t = r * r
	// compute y
	FMOVD	$P5, F8		// F8 = P5
	FMOVD	$P4, F9		// F9 = P4
	FMADDD	F7, F9, F8, F13	// P4+t*P5
	FMOVD	$P3, F10	// F10 = P3
	FMADDD	F7, F10, F13, F13	// P3+t*(P4+t*P5)
	FMOVD	$P2, F11	// F11 = P2
	FMADDD	F7, F11, F13, F13	// P2+t*(P3+t*(P4+t*P5))
	FMOVD	$P1, F12	// F12 = P1
	FMADDD	F7, F12, F13, F13	// P1+t*(P2+t*(P3+t*(P4+t*P5)))
	FMSUBD	F7, F6, F13, F13	// F13 = c = r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
	FMOVD	$2.0, F14
	FSUBD	F13, F14
	FMULD	F6, F13, F15
	FDIVD	F14, F15	// F15 = (r*c)/(2-c)
	FMOVD	$1.0, F1	// F1 = 1.0
	FSUBD	F15, F5, F15	// lo-(r*c)/(2-c)
	FSUBD	F4, F15, F15	// (lo-(r*c)/(2-c))-hi
	FSUBD	F15, F1, F16	// F16 = y = 1-((lo-(r*c)/(2-c))-hi)
	// inline Ldexp(y, k), benefit:
	// 1, no parameter pass overhead.
	// 2, skip unnecessary checks for Inf/NaN/Zero
	FMOVD	F16, R0
	AND	$FracMask, R0, R2	// fraction
	LSR	$52, R0, R5	// exponent
	ADD	R1, R5		// R1 = int(k)
	CMP	$1, R5
	BGE	normal
	ADD	$52, R5		// denormal
	MOVD	$C1, R8
	FMOVD	R8, F1		// m = 2**-52
normal:
	ORR	R5<<52, R2, R0
	FMOVD	R0, F0
	FMULD	F1, F0		// return m * x
isNaN:
	FMOVD	F0, ret+8(FP)
	RET
underflow:
	MOVD	ZR, ret+8(FP)
	RET
overflow:
	MOVD	$PosInf, R0
	MOVD	R0, ret+8(FP)
	RET
