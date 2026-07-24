// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define NaN 0x7FF8000000000001

// func archHypot(p, q float64) float64
TEXT ·archHypot(SB), NOSPLIT, $0-24
	FMOVD	p+0(FP), F0
	FMOVD	q+8(FP), F1
	MOVD	$PosInf, R0
	FMOVD	R0, F30 // F30 is PosInf

	FABSD	F0, F0
	FABSD	F1, F1
	FCMPD	F30, F0
	BGE	isInf
	FCMPD	F30, F1
	BGE	isInf

	FCMPED	F0, F0
	BNE	isNaN
	FCMPED	F1, F1
	BNE	isNaN

	FMAXD	F0, F1, F2 // p is greater
	FMIND	F0, F1, F3 // q is less
	FCMPD	F2, 0.0
	BEQ	IsZero // if p == 0, return 0

	//      p   q
	FDIVD	F2, F3, F3
	FMULD 	F3, F3, F3
	FMOVD	$1.0, F4
	FADDD	F4, F3, F3
	FSQRTD	F3, F3
	FMULD	F3, F2, F3
	FMOVD	F3, ret+16(FP)
	RET

isNaN:
	MOVD	$NaN, R0
	FMOVD	R0, F29 // F29 is NaN
	FMOVD    F29, ret+16(FP) // return NaN
	RET
isInf:
	FMOVD    F30, ret+16(FP) // return +Inf
	RET
isZero:
	// R0 has been set to zero
	MOVD    R0, ret+16(FP) // return 0
	RET
