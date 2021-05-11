// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func archSqrt(x float64) float64
TEXT ·archSqrt(SB),NOSPLIT,$0
	FMOVD x+0(FP), F1
	FSQRT F1, F1
	FMOVD F1, ret+8(FP)
	RET
