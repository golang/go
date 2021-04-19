// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go,ppc64le

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// func divWW(x1, x0, y Word) (q, r Word)
TEXT ·divWW(SB), NOSPLIT, $0
	MOVD x1+0(FP), R4
	MOVD x0+8(FP), R5
	MOVD y+16(FP), R6

	CMPU R4, R6
	BGE  divbigger

	// from the programmer's note in ch. 3 of the ISA manual, p.74
	DIVDEU R6, R4, R3
	DIVDU  R6, R5, R7
	MULLD  R6, R3, R8
	MULLD  R6, R7, R20
	SUB    R20, R5, R10
	ADD    R7, R3, R3
	SUB    R8, R10, R4
	CMPU   R4, R10
	BLT    adjust
	CMPU   R4, R6
	BLT    end

adjust:
	MOVD $1, R21
	ADD  R21, R3, R3
	SUB  R6, R4, R4

end:
	MOVD R3, q+24(FP)
	MOVD R4, r+32(FP)

	RET

divbigger:
	MOVD $-1, R7
	MOVD R7, q+24(FP)
	MOVD R7, r+32(FP)
	RET

