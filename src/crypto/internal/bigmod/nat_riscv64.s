// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB),$0-32
	MOV	$16, X30
	JMP	addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB),$0-32
	MOV	$24, X30
	JMP	addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB),$0-32
	MOV	$32, X30
	JMP	addMulVVWx(SB)

TEXT addMulVVWx(SB),NOFRAME|NOSPLIT,$0
	MOV	z+0(FP), X5
	MOV	x+8(FP), X7
	MOV	y+16(FP), X6
	MOV	$0, X29

	BEQZ	X30, done
loop:
	MOV	0*8(X5), X10	// z[0]
	MOV	1*8(X5), X13	// z[1]
	MOV	2*8(X5), X16	// z[2]
	MOV	3*8(X5), X19	// z[3]

	MOV	0*8(X7), X8	// x[0]
	MOV	1*8(X7), X11	// x[1]
	MOV	2*8(X7), X14	// x[2]
	MOV	3*8(X7), X17	// x[3]

	MULHU	X8, X6, X9	// z_hi[0] = x[0] * y
	MUL	X8, X6, X8	// z_lo[0] = x[0] * y
	ADD	X8, X10, X21	// z_lo[0] = x[0] * y + z[0]
	SLTU	X8, X21, X22
	ADD	X9, X22, X9	// z_hi[0] = x[0] * y + z[0]
	ADD	X21, X29, X10	// z_lo[0] = x[0] * y + z[0] + c
	SLTU	X21, X10, X22
	ADD	X9, X22, X29	// next c

	MULHU	X11, X6, X12	// z_hi[1] = x[1] * y
	MUL	X11, X6, X11	// z_lo[1] = x[1] * y
	ADD	X11, X13, X21	// z_lo[1] = x[1] * y + z[1]
	SLTU	X11, X21, X22
	ADD	X12, X22, X12	// z_hi[1] = x[1] * y + z[1]
	ADD	X21, X29, X13	// z_lo[1] = x[1] * y + z[1] + c
	SLTU	X21, X13, X22
	ADD	X12, X22, X29	// next c

	MULHU	X14, X6, X15	// z_hi[2] = x[2] * y
	MUL	X14, X6, X14	// z_lo[2] = x[2] * y
	ADD	X14, X16, X21	// z_lo[2] = x[2] * y + z[2]
	SLTU	X14, X21, X22
	ADD	X15, X22, X15	// z_hi[2] = x[2] * y + z[2]
	ADD	X21, X29, X16	// z_lo[2] = x[2] * y + z[2] + c
	SLTU	X21, X16, X22
	ADD	X15, X22, X29	// next c

	MULHU	X17, X6, X18	// z_hi[3] = x[3] * y
	MUL	X17, X6, X17	// z_lo[3] = x[3] * y
	ADD	X17, X19, X21	// z_lo[3] = x[3] * y + z[3]
	SLTU	X17, X21, X22
	ADD	X18, X22, X18	// z_hi[3] = x[3] * y + z[3]
	ADD	X21, X29, X19	// z_lo[3] = x[3] * y + z[3] + c
	SLTU	X21, X19, X22
	ADD	X18, X22, X29	// next c

	MOV	X10, 0*8(X5)	// z[0]
	MOV	X13, 1*8(X5)	// z[1]
	MOV	X16, 2*8(X5)	// z[2]
	MOV	X19, 3*8(X5)	// z[3]

	ADD	$32, X5
	ADD	$32, X7

	SUB	$4, X30
	BNEZ	X30, loop

done:
	MOV	X29, c+24(FP)
	RET
