// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// RISC-V offered floating-point (FP) rounding by FP conversion instructions (FCVT)
// with rounding mode field.
// As Go spec expects FP rounding result in FP, we have to use FCVT integer
// back to FP (fp -> int -> fp).
// RISC-V only set Inexact flag during invalid FP-integer conversion without changing any data,
// on the other hand, RISC-V sets out of integer represent range yet valid FP into NaN.
// When it comes to integer-FP conversion, invalid FP like NaN, +-Inf will be
// converted into the closest valid FP, for example:
//
// `Floor(-Inf) -> int64(0x7fffffffffffffff) -> float64(9.22e+18)`
// `Floor(18446744073709549568.0) -> int64(0x7fffffffffffffff) -> float64(9.22e+18)`
//
// This ISA conversion limitation requires we skip all invalid or out of range FP
// before any normal rounding operations.

#define ROUNDFN(NAME, MODE) 	\
TEXT NAME(SB),NOSPLIT,$0; 	\
	MOVD	x+0(FP), F10; 	\
	FMVXD	F10, X10;	\
	/* Drop all fraction bits */;\
	SRL	$52, X10, X12;	\
	/* Remove sign bit */;	\
	AND	$0x7FF, X12, X12;\
	/* Return either input is +-Inf, NaN(0x7FF) or out of precision limitation */;\
	/* 1023: bias of exponent, [-2^53, 2^53]: exactly integer represent range */;\
	MOV	$1023+53, X11;	\
	BLTU	X11, X12, 4(PC);\
	FCVTLD.MODE F10, X11;	\
	FCVTDL	X11, F11;	\
	/* RISC-V rounds negative values to +0, restore original sign */;\
	FSGNJD	F10, F11, F10;	\
	MOVD	F10, ret+8(FP); \
	RET

// func archFloor(x float64) float64
ROUNDFN(·archFloor, RDN)

// func archCeil(x float64) float64
ROUNDFN(·archCeil, RUP)

// func archTrunc(x float64) float64
ROUNDFN(·archTrunc, RTZ)
