// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Exp2(x float64) float64
TEXT ·Exp2(SB),NOSPLIT,$0
// test bits for not-finite
	MOVL    x_hi+4(FP), AX
	ANDL    $0x7ff00000, AX
	CMPL    AX, $0x7ff00000
	JEQ     not_finite
	FMOVD   x+0(FP), F0   // F0=x
	FMOVD   F0, F1        // F0=x, F1=x
	FRNDINT               // F0=int(x), F1=x
	FSUBD   F0, F1        // F0=int(x), F1=x-int(x)
	FXCHD   F0, F1        // F0=x-int(x), F1=int(x)
	F2XM1                 // F0=2**(x-int(x))-1, F1=int(x)
	FLD1                  // F0=1, F1=2**(x-int(x))-1, F2=int(x)
	FADDDP  F0, F1        // F0=2**(x-int(x)), F1=int(x)
	FSCALE                // F0=2**x, F1=int(x)
	FMOVDP  F0, F1        // F0=2**x
	FMOVDP  F0, ret+8(FP)
	RET
not_finite:
// test bits for -Inf
	MOVL    x_hi+4(FP), BX
	MOVL    x_lo+0(FP), CX
	CMPL    BX, $0xfff00000
	JNE     not_neginf
	CMPL    CX, $0
	JNE     not_neginf
	MOVL    $0, ret_lo+8(FP)
	MOVL    $0, ret_hi+12(FP)
	RET
not_neginf:
	MOVL    CX, ret_lo+8(FP)
	MOVL    BX, ret_hi+12(FP)
	RET
