// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// The method is based on a paper by Naoki Shibata: "Efficient evaluation
// methods of elementary functions suitable for SIMD computation", Proc.
// of International Supercomputing Conference 2010 (ISC'10), pp. 25 -- 32
// (May 2010). The paper is available at
// https://link.springer.com/article/10.1007/s00450-010-0108-2
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
#define PosInf 0x7FF0000000000000
#define NegInf 0xFFF0000000000000
#define Overflow 7.09782712893384e+02

DATA exprodata<>+0(SB)/8, $0.5
DATA exprodata<>+8(SB)/8, $1.0
DATA exprodata<>+16(SB)/8, $2.0
DATA exprodata<>+24(SB)/8, $1.6666666666666666667e-1
DATA exprodata<>+32(SB)/8, $4.1666666666666666667e-2
DATA exprodata<>+40(SB)/8, $8.3333333333333333333e-3
DATA exprodata<>+48(SB)/8, $1.3888888888888888889e-3
DATA exprodata<>+56(SB)/8, $1.9841269841269841270e-4
DATA exprodata<>+64(SB)/8, $2.4801587301587301587e-5
GLOBL exprodata<>+0(SB), RODATA, $72

// func Exp(x float64) float64
TEXT ·archExp(SB),NOSPLIT,$0
	// test bits for not-finite
	MOVQ    x+0(FP), BX
	MOVQ    $~(1<<63), AX // sign bit mask
	MOVQ    BX, DX
	ANDQ    AX, DX
	MOVQ    $PosInf, AX
	CMPQ    AX, DX
	JLE     notFinite
	// check if argument will overflow
	MOVQ    BX, X0
	MOVSD   $Overflow, X1
	COMISD  X1, X0
	JA      overflow
	MOVSD   $LOG2E, X1
	MULSD   X0, X1
	CVTSD2SL X1, BX // BX = exponent
	CVTSL2SD BX, X1
	CMPB ·useFMA(SB), $1
	JE   avxfma
	MOVSD   $LN2U, X2
	MULSD   X1, X2
	SUBSD   X2, X0
	MOVSD   $LN2L, X2
	MULSD   X1, X2
	SUBSD   X2, X0
	// reduce argument
	MULSD   $0.0625, X0
	// Taylor series evaluation
	MOVSD   exprodata<>+64(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+56(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+48(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+40(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+32(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+24(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+0(SB), X1
	MULSD   X0, X1
	ADDSD   exprodata<>+8(SB), X1
	MULSD   X1, X0
	MOVSD   exprodata<>+16(SB), X1
	ADDSD   X0, X1
	MULSD   X1, X0
	MOVSD   exprodata<>+16(SB), X1
	ADDSD   X0, X1
	MULSD   X1, X0
	MOVSD   exprodata<>+16(SB), X1
	ADDSD   X0, X1
	MULSD   X1, X0
	MOVSD   exprodata<>+16(SB), X1
	ADDSD   X0, X1
	MULSD   X1, X0
	ADDSD exprodata<>+8(SB), X0
	// return fr * 2**exponent
ldexp:
	ADDL    $0x3FF, BX // add bias
	JLE     denormal
	CMPL    BX, $0x7FF
	JGE     overflow
lastStep:
	SHLQ    $52, BX
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
	MOVQ    $0, ret+8(FP)
	RET
overflow: // return +Inf
	MOVQ    $PosInf, BX
notNegInf: // NaN or +Inf, return x
	MOVQ    BX, ret+8(FP)
	RET
denormal:
	CMPL    BX, $-52
	JL      underflow
	ADDL    $0x3FE, BX // add bias - 1
	SHLQ    $52, BX
	MOVQ    BX, X1
	MULSD   X1, X0
	MOVQ    $1, BX
	JMP     lastStep

avxfma:
	MOVSD   $LN2U, X2
	VFNMADD231SD X2, X1, X0
	MOVSD   $LN2L, X2
	VFNMADD231SD X2, X1, X0
	// reduce argument
	MULSD   $0.0625, X0
	// Taylor series evaluation
	MOVSD   exprodata<>+64(SB), X1
	VFMADD213SD exprodata<>+56(SB), X0, X1
	VFMADD213SD exprodata<>+48(SB), X0, X1
	VFMADD213SD exprodata<>+40(SB), X0, X1
	VFMADD213SD exprodata<>+32(SB), X0, X1
	VFMADD213SD exprodata<>+24(SB), X0, X1
	VFMADD213SD exprodata<>+0(SB), X0, X1
	VFMADD213SD exprodata<>+8(SB), X0, X1
	MULSD   X1, X0
	VADDSD exprodata<>+16(SB), X0, X1
	MULSD   X1, X0
	VADDSD exprodata<>+16(SB), X0, X1
	MULSD   X1, X0
	VADDSD exprodata<>+16(SB), X0, X1
	MULSD   X1, X0
	VADDSD exprodata<>+16(SB), X0, X1
	VFMADD213SD   exprodata<>+8(SB), X1, X0
	JMP ldexp
