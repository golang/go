// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Remainder(x, y float64) float64
TEXT ·Remainder(SB),NOSPLIT,$0
	FMOVD   y+8(FP), F0  // F0=y
	FMOVD   x+0(FP), F0  // F0=x, F1=y
	FPREM1               // F0=reduced_x, F1=y
	FSTSW   AX           // AX=status word
	ANDW    $0x0400, AX
	JNE     -3(PC)       // jump if reduction incomplete
	FMOVDP  F0, F1       // F0=x-q*y
	FMOVDP  F0, ret+16(FP)
	RET
