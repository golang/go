// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// derived from crypto/internal/fips/bigmod/nat_riscv64.s

//go:build !purego

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB),$0-32
	MOVV	$16, R8
	JMP	addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB),$0-32
	MOVV	$24, R8
	JMP	addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB),$0-32
	MOVV	$32, R8
	JMP	addMulVVWx(SB)

TEXT addMulVVWx(SB),NOFRAME|NOSPLIT,$0
	MOVV	z+0(FP), R4
	MOVV	x+8(FP), R6
	MOVV	y+16(FP), R5
	MOVV	$0, R7

	BEQ	R8, R0, done
loop:
	MOVV	0*8(R4), R9	// z[0]
	MOVV	1*8(R4), R10	// z[1]
	MOVV	2*8(R4), R11	// z[2]
	MOVV	3*8(R4), R12	// z[3]

	MOVV	0*8(R6), R13	// x[0]
	MOVV	1*8(R6), R14	// x[1]
	MOVV	2*8(R6), R15	// x[2]
	MOVV	3*8(R6), R16	// x[3]

	MULHVU	R13, R5, R17	// z_hi[0] = x[0] * y
	MULV	R13, R5, R13	// z_lo[0] = x[0] * y
	ADDV	R13, R9, R18	// z_lo[0] = x[0] * y + z[0]
	SGTU	R13, R18, R19
	ADDV	R17, R19, R17	// z_hi[0] = x[0] * y + z[0]
	ADDV	R18, R7, R9	// z_lo[0] = x[0] * y + z[0] + c
	SGTU	R18, R9, R19
	ADDV	R17, R19, R7	// next c

	MULHVU	R14, R5, R24	// z_hi[1] = x[1] * y
	MULV	R14, R5, R14	// z_lo[1] = x[1] * y
	ADDV	R14, R10, R18	// z_lo[1] = x[1] * y + z[1]
	SGTU	R14, R18, R19
	ADDV	R24, R19, R24	// z_hi[1] = x[1] * y + z[1]
	ADDV	R18, R7, R10	// z_lo[1] = x[1] * y + z[1] + c
	SGTU	R18, R10, R19
	ADDV	R24, R19, R7	// next c

	MULHVU	R15, R5, R25	// z_hi[2] = x[2] * y
	MULV	R15, R5, R15	// z_lo[2] = x[2] * y
	ADDV	R15, R11, R18	// z_lo[2] = x[2] * y + z[2]
	SGTU	R15, R18, R19
	ADDV	R25, R19, R25	// z_hi[2] = x[2] * y + z[2]
	ADDV	R18, R7, R11	// z_lo[2] = x[2] * y + z[2] + c
	SGTU	R18, R11, R19
	ADDV	R25, R19, R7	// next c

	MULHVU	R16, R5, R26	// z_hi[3] = x[3] * y
	MULV	R16, R5, R16	// z_lo[3] = x[3] * y
	ADDV	R16, R12, R18	// z_lo[3] = x[3] * y + z[3]
	SGTU	R16, R18, R19
	ADDV	R26, R19, R26	// z_hi[3] = x[3] * y + z[3]
	ADDV	R18, R7, R12	// z_lo[3] = x[3] * y + z[3] + c
	SGTU	R18, R12, R19
	ADDV	R26, R19, R7	// next c

	MOVV	R9, 0*8(R4)	// z[0]
	MOVV	R10, 1*8(R4)	// z[1]
	MOVV	R11, 2*8(R4)	// z[2]
	MOVV	R12, 3*8(R4)	// z[3]

	ADDV	$32, R4
	ADDV	$32, R6

	SUBV	$4, R8
	BNE	R8, R0, loop

done:
	MOVV	R7, c+24(FP)
	RET
