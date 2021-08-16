// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build mips || mipsle
// +build mips mipsle

#include "textflag.h"

// func archSqrt(x float64) float64
TEXT ·archSqrt(SB),NOSPLIT,$0
#ifdef GOMIPS_softfloat
	JMP ·sqrt(SB)
#else
	MOVD	x+0(FP), F0
	SQRTD	F0, F0
	MOVD	F0, ret+8(FP)
#endif
	RET
