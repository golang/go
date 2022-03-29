// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func archSqrt(x float64) float64
TEXT ·archSqrt(SB),NOSPLIT,$0
	MOVB	runtime·goarm(SB), R11
	CMP	$5, R11
	BEQ	arm5
	MOVD	x+0(FP),F0
	SQRTD	F0,F0
	MOVD	F0,ret+8(FP)
	RET
arm5:
	// Tail call to Go implementation.
	// Can't use JMP, as in softfloat mode SQRTD is rewritten
	// to a CALL, which makes this function have a frame.
	RET	·sqrt(SB)
