// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

// func Sqrt(x float64) float64
TEXT ·Sqrt(SB),NOSPLIT,$0
	FMOVD	x+0(FP), F0
	FSQRT	F0, F0
	FMOVD	F0, ret+8(FP)
	RET
