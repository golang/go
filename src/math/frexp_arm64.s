// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf 0x7FF0000000000000
#define Float64SmallestNormal 2.2250738585072014e-308

// func archFrexp(x float64) (frac float64, exp int)
TEXT ·archFrexp(SB),NOSPLIT,$0-24
	FMOVD	x+0(FP), F0
	MOVD	ZR, R1 // R1 is the ret exp
	FABSD	F0, F3 // F3 is the absolute of 'x'
	FCMPD	$(0.0), F3 // avoid the input floating number is -0.0
	BEQ	isInfOrNaNOrZero

	MOVD	$PosInf, R3
	FMOVD	R3, F30 // F30 is PosInf
	FCMPD	F30, F3
	BGE	isInfOrNaNOrZero // isInf
	FCMPED	F0, F0
	BNE	isInfOrNaNOrZero // isNaN

	FMOVD	F3, R0
	UBFX $52, R0, $11, R0

	FMOVD	$Float64SmallestNormal, F29
	FCMPD	F29, F3
	BGE	pass

	// if abs(x) < SmallestNormal ,then run the following special case
	FMOVD	$(4503599627370496.0), F2 // 4503599627370496.0 is (1<<52)
	FMULD	F2, F0, F0
	SUB	$(52), R1, R1 // set R1 to -52
	FMOVD	F0, R0
	// tmp = int((x>>shift)&mask)
	UBFX $52, R0, $11, R0

pass:
	// tmp = tmp - bias + 1
	SUB	$0x3FE, R0, R3
	// exp = exp + tmp
	ADD R3, R1, R1

	FMOVD	F0, R0
	AND	$0x800FFFFFFFFFFFFF, R0, R0
	ORR	$0x3FE0000000000000, R0, R0
	FMOVD	R0, F0
  	FMOVD	F0, frac+8(FP)
	MOVD	R1, exp+16(FP)
	RET

isInfOrNaNOrZero:
	FMOVD	F0, frac+8(FP)
	MOVD	R1, exp+16(FP)
	RET
