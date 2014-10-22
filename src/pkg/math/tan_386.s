// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Tan(x float64) float64
TEXT ·Tan(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FPTAN                // F0=1, F1=tan(x) if -2**63 < x < 2**63
	FSTSW   AX           // AX=status word
	ANDW    $0x0400, AX
	JNE     4(PC)        // jump if x outside range
	FMOVDP  F0, F0       // F0=tan(x)
	FMOVDP  F0, ret+8(FP)
	RET
	FLDPI                // F0=Pi, F1=x
	FADDD   F0, F0       // F0=2*Pi, F1=x
	FXCHD   F0, F1       // F0=x, F1=2*Pi
	FPREM1               // F0=reduced_x, F1=2*Pi
	FSTSW   AX           // AX=status word
	ANDW    $0x0400, AX
	JNE     -3(PC)       // jump if reduction incomplete
	FMOVDP  F0, F1       // F0=reduced_x
	FPTAN                // F0=1, F1=tan(reduced_x)
	FMOVDP  F0, F0       // F0=tan(reduced_x)
	FMOVDP  F0, ret+8(FP)
	RET
