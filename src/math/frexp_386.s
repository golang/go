// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Frexp(f float64) (frac float64, exp int)
TEXT ·Frexp(SB),NOSPLIT,$0
	FMOVD   f+0(FP), F0   // F0=f
	FXAM
	FSTSW   AX
	SAHF
	JNP     nan_zero_inf
	JCS     nan_zero_inf
	FXTRACT               // F0=f (0<=f<1), F1=e
	FMULD  $(0.5), F0     // F0=f (0.5<=f<1), F1=e
	FMOVDP  F0, frac+8(FP)   // F0=e
	FLD1                  // F0=1, F1=e
	FADDDP  F0, F1        // F0=e+1
	FMOVLP  F0, exp+16(FP)  // (int=int32)
	RET
nan_zero_inf:
	FMOVDP  F0, frac+8(FP)   // F0=e
	MOVL    $0, exp+16(FP)  // exp=0
	RET
