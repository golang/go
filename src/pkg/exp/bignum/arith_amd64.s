// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides fast assembly versions
// of the routines in arith.go.

// func Mul128(x, y uint64) (z1, z0 uint64)
// z1<<64 + z0 = x*y
//
TEXT ·Mul128(SB),7,$0
	MOVQ a+0(FP), AX
	MULQ a+8(FP)
	MOVQ DX, a+16(FP)
	MOVQ AX, a+24(FP)
	RET


// func MulAdd128(x, y, c uint64) (z1, z0 uint64)
// z1<<64 + z0 = x*y + c
//
TEXT ·MulAdd128(SB),7,$0
	MOVQ a+0(FP), AX
	MULQ a+8(FP)
	ADDQ a+16(FP), AX
	ADCQ $0, DX
	MOVQ DX, a+24(FP)
	MOVQ AX, a+32(FP)
	RET


// func Div128(x1, x0, y uint64) (q, r uint64)
// q = (x1<<64 + x0)/y + r
//
TEXT ·Div128(SB),7,$0
	MOVQ a+0(FP), DX
	MOVQ a+8(FP), AX
	DIVQ a+16(FP)
	MOVQ AX, a+24(FP)
	MOVQ DX, a+32(FP)
	RET
