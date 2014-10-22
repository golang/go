// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Log1p(x float64) float64
TEXT ·Log1p(SB),NOSPLIT,$0
	FMOVD   $(2.928932188134524e-01), F0
	FMOVD   x+0(FP), F0  // F0=x, F1=1-sqrt(2)/2 = 0.29289321881345247559915564
	FABS                 // F0=|x|, F1=1-sqrt(2)/2
	FUCOMPP F0, F1       // compare F0 to F1
	FSTSW   AX
	FLDLN2               // F0=log(2)
	ANDW    $0x0100, AX
	JEQ     use_fyl2x    // jump if F0 >= F1
	FMOVD   x+0(FP), F0  // F0=x, F1=log(2)
	FYL2XP1              // F0=log(1+x)=log2(1+x)*log(2)
	FMOVDP  F0, ret+8(FP)
	RET
use_fyl2x:
	FLD1                 // F0=1, F2=log(2)
	FADDD   x+0(FP), F0  // F0=1+x, F1=log(2)
	FYL2X                // F0=log(1+x)=log2(1+x)*log(2)
	FMOVDP  F0, ret+8(FP)
	RET

