// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Modf(f float64) (int float64, frac float64)
TEXT ·Modf(SB),NOSPLIT,$0
	// special case for f == -0.0
	MOVL f+4(FP), DX	// high word
	MOVL f+0(FP), AX	// low word
	CMPL DX, $(1<<31)	// beginning of -0.0
	JNE notNegativeZero
	CMPL AX, $0			// could be denormalized
	JNE notNegativeZero
	MOVL AX, int+8(FP)
	MOVL DX, int+12(FP)
	MOVL AX, frac+16(FP)
	MOVL DX, frac+20(FP)
	RET
notNegativeZero:
	FMOVD   f+0(FP), F0  // F0=f
	FMOVD   F0, F1       // F0=f, F1=f
	FSTCW   -2(SP)       // save old Control Word
	MOVW    -2(SP), AX
	ORW     $0x0c00, AX  // Rounding Control set to truncate
	MOVW    AX, -4(SP)   // store new Control Word
	FLDCW   -4(SP)       // load new Control Word
	FRNDINT              // F0=trunc(f), F1=f
	FLDCW   -2(SP)       // load old Control Word
	FSUBD   F0, F1       // F0=trunc(f), F1=f-trunc(f)
	FMOVDP  F0, int+8(FP)  // F0=f-trunc(f)
	FMOVDP  F0, frac+16(FP)
	RET
