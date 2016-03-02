// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Atan(x float64) float64
TEXT ·Atan(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FLD1                 // F0=1, F1=x
	FPATAN               // F0=atan(F1/F0)
	FMOVDP  F0, ret+8(FP)
	RET
