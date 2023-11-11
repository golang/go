// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB), $0-16
	MOVW	$32, R5
	JMP		addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB), $0-16
	MOVW	$48, R5
	JMP		addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB), $0-16
	MOVW	$64, R5
	JMP		addMulVVWx(SB)

TEXT addMulVVWx(SB), NOFRAME|NOSPLIT, $0
	MOVW	$0, R0
	MOVW	z+0(FP), R1
	MOVW	x+4(FP), R2
	MOVW	y+8(FP), R3
	ADD	R5<<2, R1, R5
	MOVW	$0, R4
	B E9

L9:	MOVW.P	4(R2), R6
	MULLU	R6, R3, (R7, R6)
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW	0(R1), R4
	ADD.S	R4, R6
	ADC	R0, R7
	MOVW.P	R6, 4(R1)
	MOVW	R7, R4

E9:	TEQ	R1, R5
	BNE	L9

	MOVW	R4, c+12(FP)
	RET
