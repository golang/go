// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func archFloor(x float64) float64
TEXT ·archFloor(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FIDBR	$7, F0, F0
	FMOVD	F0, ret+8(FP)
	RET

// func archCeil(x float64) float64
TEXT ·archCeil(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FIDBR	$6, F0, F0
	FMOVD	F0, ret+8(FP)
	RET

// func archTrunc(x float64) float64
TEXT ·archTrunc(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FIDBR	$5, F0, F0
	FMOVD	F0, ret+8(FP)
	RET
