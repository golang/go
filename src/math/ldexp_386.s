// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Ldexp(frac float64, exp int) float64
TEXT ·Ldexp(SB),NOSPLIT,$0
	FMOVL   exp+8(FP), F0   // F0=exp
	FMOVD   frac+0(FP), F0   // F0=frac, F1=e
	FSCALE                // F0=x*2**e, F1=e
	FMOVDP  F0, F1        // F0=x*2**e
	FMOVDP  F0, ret+12(FP)
	RET
