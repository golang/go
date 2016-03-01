// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Hypot(p, q float64) float64
TEXT ·Hypot(SB),NOSPLIT,$0
// test bits for not-finite
	MOVL    p_hi+4(FP), AX   // high word p
	ANDL    $0x7ff00000, AX
	CMPL    AX, $0x7ff00000
	JEQ     not_finite
	MOVL    q_hi+12(FP), AX   // high word q
	ANDL    $0x7ff00000, AX
	CMPL    AX, $0x7ff00000
	JEQ     not_finite
	FMOVD   p+0(FP), F0  // F0=p
	FABS                 // F0=|p|
	FMOVD   q+8(FP), F0  // F0=q, F1=|p|
	FABS                 // F0=|q|, F1=|p|
	FUCOMI  F0, F1       // compare F0 to F1
	JCC     2(PC)        // jump if F0 >= F1
	FXCHD   F0, F1       // F0=|p| (larger), F1=|q| (smaller)
	FTST                 // compare F0 to 0
	FSTSW	AX
	ANDW    $0x4000, AX
	JNE     10(PC)       // jump if F0 = 0
	FXCHD   F0, F1       // F0=q (smaller), F1=p (larger)
	FDIVD   F1, F0       // F0=q(=q/p), F1=p
	FMULD   F0, F0       // F0=q*q, F1=p
	FLD1                 // F0=1, F1=q*q, F2=p
	FADDDP  F0, F1       // F0=1+q*q, F1=p
	FSQRT                // F0=sqrt(1+q*q), F1=p
	FMULDP  F0, F1       // F0=p*sqrt(1+q*q)
	FMOVDP  F0, ret+16(FP)
	RET
	FMOVDP  F0, F1       // F0=0
	FMOVDP  F0, ret+16(FP)
	RET
not_finite:
// test bits for -Inf or +Inf
	MOVL    p_hi+4(FP), AX  // high word p
	ORL     p_lo+0(FP), AX  // low word p
	ANDL    $0x7fffffff, AX
	CMPL    AX, $0x7ff00000
	JEQ     is_inf
	MOVL    q_hi+12(FP), AX  // high word q
	ORL     q_lo+8(FP), AX   // low word q
	ANDL    $0x7fffffff, AX
	CMPL    AX, $0x7ff00000
	JEQ     is_inf
	MOVL    $0x7ff80000, ret_hi+20(FP)  // return NaN = 0x7FF8000000000001
	MOVL    $0x00000001, ret_lo+16(FP)
	RET
is_inf:
	MOVL    AX, ret_hi+20(FP)  // return +Inf = 0x7FF0000000000000
	MOVL    $0x00000000, ret_lo+16(FP)
	RET
