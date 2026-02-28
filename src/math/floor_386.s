// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func archCeil(x float64) float64
TEXT ·archCeil(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FSTCW   -2(SP)       // save old Control Word
	MOVW    -2(SP), AX
	ANDW    $0xf3ff, AX
	ORW     $0x0800, AX  // Rounding Control set to +Inf
	MOVW    AX, -4(SP)   // store new Control Word
	FLDCW   -4(SP)       // load new Control Word
	FRNDINT              // F0=Ceil(x)
	FLDCW   -2(SP)       // load old Control Word
	FMOVDP  F0, ret+8(FP)
	RET

// func archFloor(x float64) float64
TEXT ·archFloor(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FSTCW   -2(SP)       // save old Control Word
	MOVW    -2(SP), AX
	ANDW    $0xf3ff, AX
	ORW     $0x0400, AX  // Rounding Control set to -Inf
	MOVW    AX, -4(SP)   // store new Control Word
	FLDCW   -4(SP)       // load new Control Word
	FRNDINT              // F0=Floor(x)
	FLDCW   -2(SP)       // load old Control Word
	FMOVDP  F0, ret+8(FP)
	RET

// func archTrunc(x float64) float64
TEXT ·archTrunc(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FSTCW   -2(SP)       // save old Control Word
	MOVW    -2(SP), AX
	ORW     $0x0c00, AX  // Rounding Control set to truncate
	MOVW    AX, -4(SP)   // store new Control Word
	FLDCW   -4(SP)       // load new Control Word
	FRNDINT              // F0=Trunc(x)
	FLDCW   -2(SP)       // load old Control Word
	FMOVDP  F0, ret+8(FP)
	RET
