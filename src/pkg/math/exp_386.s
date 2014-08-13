// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Exp(x float64) float64
TEXT ·Exp(SB),NOSPLIT,$0
// test bits for not-finite
	MOVL    x_hi+4(FP), AX
	ANDL    $0x7ff00000, AX
	CMPL    AX, $0x7ff00000
	JEQ     not_finite
	FLDL2E                // F0=log2(e)
	FMULD   x+0(FP), F0   // F0=x*log2(e)
	FMOVD   F0, F1        // F0=x*log2(e), F1=x*log2(e)
	FRNDINT               // F0=int(x*log2(e)), F1=x*log2(e)
	FSUBD   F0, F1        // F0=int(x*log2(e)), F1=x*log2(e)-int(x*log2(e))
	FXCHD   F0, F1        // F0=x*log2(e)-int(x*log2(e)), F1=int(x*log2(e))
	F2XM1                 // F0=2**(x*log2(e)-int(x*log2(e)))-1, F1=int(x*log2(e))
	FLD1                  // F0=1, F1=2**(x*log2(e)-int(x*log2(e)))-1, F2=int(x*log2(e))
	FADDDP  F0, F1        // F0=2**(x*log2(e)-int(x*log2(e))), F1=int(x*log2(e))
	FSCALE                // F0=e**x, F1=int(x*log2(e))
	FMOVDP  F0, F1        // F0=e**x
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
	FLDZ                  // F0=0
	FMOVDP  F0, ret+8(FP)
	RET
not_neginf:
	MOVL    CX, ret_lo+8(FP)
	MOVL    BX, ret_hi+12(FP)
	RET
