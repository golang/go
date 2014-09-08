// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Abs(x float64) float64
TEXT ·Abs(SB),NOSPLIT,$0
	FMOVD   x+0(FP), F0  // F0=x
	FABS                 // F0=|x|
	FMOVDP  F0, ret+8(FP)
	RET
