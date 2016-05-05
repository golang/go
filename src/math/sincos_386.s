// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Sincos(x float64) (sin, cos float64)
TEXT ·Sincos(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FSINCOS              // F0=cos(x), F1=sin(x) if -2**63 < x < 2**63
	FSTSW   AX           // AX=status word
	ANDW    $0x0400, AX
	JNE     4(PC)        // jump if x outside range
	FMOVDP  F0, cos+16(FP) // F0=sin(x)
	FMOVDP  F0, sin+8(FP)
	RET
	FLDPI                // F0=Pi, F1=x
	FADDD   F0, F0       // F0=2*Pi, F1=x
	FXCHD   F0, F1       // F0=x, F1=2*Pi
	FPREM1               // F0=reduced_x, F1=2*Pi
	FSTSW   AX           // AX=status word
	ANDW    $0x0400, AX
	JNE     -3(PC)       // jump if reduction incomplete
	FMOVDP  F0, F1       // F0=reduced_x
	FSINCOS              // F0=cos(reduced_x), F1=sin(reduced_x)
	FMOVDP  F0, cos+16(FP) // F0=sin(reduced_x)
	FMOVDP  F0, sin+8(FP)
	RET
