// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// func Modf(f float64) (int float64, frac float64)
TEXT ·Modf(SB),7,$0
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
