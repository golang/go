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

// func Exp(x float64) float64
TEXT ·Exp(SB),7,$0
// test bits for not-finite
	MOVQ    x+0(FP), AX
	MOVQ    $0x7ff0000000000000, BX
	ANDQ    BX, AX
	CMPQ    BX, AX
	JEQ     not_finite
	MOVSD   x+0(FP), X0
	MOVSD   $LOG2E, X1
	MULSD   X0, X1
	CVTTSD2SQ X1, BX // BX = exponent
	CVTSQ2SD BX, X1
	MOVSD   $LN2U, X2
	MULSD   X1, X2
	SUBSD   X2, X0
	MOVSD   $LN2L, X2
	MULSD   X1, X2
	SUBSD   X2, X0
	// reduce argument
	MOVSD   $0.0625, X1
	MULSD   X1, X0
	// Taylor series evaluation
	MOVSD   $2.4801587301587301587e-5, X1
	MULSD   X0, X1
	MOVSD   $1.9841269841269841270e-4, X2
	ADDSD   X2, X1
	MULSD   X0, X1
	MOVSD   $1.3888888888888888889e-3, X2
	ADDSD   X2, X1
	MULSD   X0, X1
	MOVSD   $8.3333333333333333333e-3, X2
	ADDSD   X2, X1
	MULSD   X0, X1
	MOVSD   $4.1666666666666666667e-2, X2
	ADDSD   X2, X1
	MULSD   X0, X1
	MOVSD   $1.6666666666666666667e-1, X2
	ADDSD   X2, X1
	MULSD   X0, X1
	MOVSD   $0.5, X2
	ADDSD   X2, X1
	MULSD   X0, X1
	MOVSD   $1.0, X2
	ADDSD   X2, X1
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
	MOVSD   $1.0, X1
	ADDSD   X1, X0
	// return ldexp(fr, exp)
	MOVQ    $0x3ff, AX // bias + 1
	ADDQ    AX, BX
	MOVQ    BX, X1
	MOVQ    $52, AX // shift
	MOVQ    AX, X2
	PSLLQ   X2, X1
	MULSD   X1, X0
	MOVSD   X0, r+8(FP)
	RET
not_finite:
// test bits for -Inf
	MOVQ    x+0(FP), AX
	MOVQ    $0xfff0000000000000, BX
	CMPQ    BX, AX
	JNE     not_neginf
	XORQ    AX, AX
	MOVQ    AX, r+8(FP)
	RET
not_neginf:
	MOVQ    AX, r+8(FP)
	RET
