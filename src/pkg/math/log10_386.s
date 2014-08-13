// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Log10(x float64) float64
TEXT ·Log10(SB),NOSPLIT,$0
	FLDLG2               // F0=log10(2)
	FMOVD   x+0(FP), F0  // F0=x, F1=log10(2)
	FYL2X                // F0=log10(x)=log2(x)*log10(2)
	FMOVDP  F0, ret+8(FP)
	RET

// func Log2(x float64) float64
TEXT ·Log2(SB),NOSPLIT,$0
	FLD1                 // F0=1
	FMOVD   x+0(FP), F0  // F0=x, F1=1
	FYL2X                // F0=log2(x)
	FMOVDP  F0, ret+8(FP)
	RET
