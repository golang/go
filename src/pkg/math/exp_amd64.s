// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

#define LN2 0.6931471805599453094172321214581766 // log_e(2)
#define LOG2E 1.4426950408889634073599246810018920 // 1/LN2
#define LN2U 0.69314718055966295651160180568695068359375 // upper half LN2
#define LN2L 0.28235290563031577122588448175013436025525412068e-12 // lower half LN2
#define T0 1.0
#define T1 0.5
#define T2 1.6666666666666666667e-1
#define T3 4.1666666666666666667e-2
#define T4 8.3333333333333333333e-3
#define T5 1.3888888888888888889e-3
#define T6 1.9841269841269841270e-4
#define T7 2.4801587301587301587e-5
#define PosInf 0x7FF0000000000000
#define NegInf 0xFFF0000000000000

// func Exp(x float64) float64
TEXT ·Exp(SB),7,$0
// test bits for not-finite
	MOVQ    x+0(FP), BX
	MOVQ    $~(1<<63), AX // sign bit mask
	MOVQ    BX, DX
	ANDQ    AX, DX
	MOVQ    $PosInf, AX
	CMPQ    AX, DX
	JLE     notFinite
	MOVQ    BX, X0
	MOVSD   $LOG2E, X1
	MULSD   X0, X1
	CVTSD2SL X1, BX // BX = exponent
	CVTSL2SD BX, X1
	MOVSD   $LN2U, X2
	MULSD   X1, X2
	SUBSD   X2, X0
	MOVSD   $LN2L, X2
	MULSD   X1, X2
	SUBSD   X2, X0
	// reduce argument
	MULSD   $0.0625, X0
	// Taylor series evaluation
	MOVSD   $T7, X1
	MULSD   X0, X1
	ADDSD   $T6, X1
	MULSD   X0, X1
	ADDSD   $T5, X1
	MULSD   X0, X1
	ADDSD   $T4, X1
	MULSD   X0, X1
	ADDSD   $T3, X1
	MULSD   X0, X1
	ADDSD   $T2, X1
	MULSD   X0, X1
	ADDSD   $T1, X1
	MULSD   X0, X1
	ADDSD   $T0, X1
	MULSD   X1, X0
	MOVSD   $2.0, X1
	ADDSD   X0, X1
	MULSD   X1, X0
	MOVSD   $2.0, X1
	ADDSD   X0, X1
	MULSD   X1, X0
	MOVSD   $2.0, X1
	ADDSD   X0, X1
	MULSD   X1, X0
	MOVSD   $2.0, X1
	ADDSD   X0, X1
	MULSD   X1, X0
	ADDSD   $1.0, X0
	// return fr * 2**exponent
	MOVL    $0x3FF, AX // bias
	ADDL    AX, BX
	JLE     underflow
	CMPL    BX, $0x7FF
	JGE     overflow
	MOVL    $52, CX
	SHLQ    CX, BX
	MOVQ    BX, X1
	MULSD   X1, X0
	MOVSD   X0, ret+8(FP)
	RET
notFinite:
	// test bits for -Inf
	MOVQ    $NegInf, AX
	CMPQ    AX, BX
	JNE     notNegInf
	// -Inf, return 0
underflow: // return 0
	MOVQ    $0, AX
	MOVQ    AX, ret+8(FP)
	RET
overflow: // return +Inf
	MOVQ    $PosInf, BX
notNegInf: // NaN or +Inf, return x
	MOVQ    BX, ret+8(FP)
	RET
