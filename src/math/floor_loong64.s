// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// derived from math/floor_riscv64.s

#include "textflag.h"

#define ROUNDFN(NAME, FUNC)	\
TEXT NAME(SB),NOSPLIT,$0;	\
	MOVD	x+0(FP), F0;	\
	MOVV	F0, R11;	\
	/* 1023: bias of exponent, [-2^53, 2^53]: exactly integer represent range */;	\
	MOVV	$1023+53, R12;	\
	/* Drop all fraction bits */;	\
	SRLV	$52, R11, R11;	\
	/* Remove sign bit */;	\
	AND	$0x7FF, R11, R11;	\
	BLTU	R12, R11, isExtremum;	\
normal:;	\
	FUNC	F0, F2;	\
	MOVV	F2, R10;	\
	BEQ	R10, R0, is0;	\
	FFINTDV	F2, F0;	\
/* Return either input is +-Inf, NaN(0x7FF) or out of precision limitation */;	\
isExtremum:;	\
	MOVD	F0, ret+8(FP);	\
	RET;	\
is0:;	\
	FCOPYSGD	F0, F2, F2;	\
	MOVD	F2, ret+8(FP);	\
	RET

// func archFloor(x float64) float64
ROUNDFN(·archFloor, FTINTRMVD)

// func archCeil(x float64) float64
ROUNDFN(·archCeil, FTINTRPVD)

// func archTrunc(x float64) float64
ROUNDFN(·archTrunc, FTINTRZVD)
