// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Floor(x float64) float64
TEXT ·Floor(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FRINTMD	F0, F0
	FMOVD	F0, ret+8(FP)
	RET

// func Ceil(x float64) float64
TEXT ·Ceil(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FRINTPD	F0, F0
	FMOVD	F0, ret+8(FP)
	RET

// func Trunc(x float64) float64
TEXT ·Trunc(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FRINTZD	F0, F0
	FMOVD	F0, ret+8(FP)
	RET
