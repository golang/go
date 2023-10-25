// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB), $0-32
	MOVD	$16, R5
	JMP		addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB), $0-32
	MOVD	$24, R5
	JMP		addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB), $0-32
	MOVD	$32, R5
	JMP		addMulVVWx(SB)

TEXT addMulVVWx(SB), NOFRAME|NOSPLIT, $0
	MOVD z+0(FP), R2
	MOVD x+8(FP), R8
	MOVD y+16(FP), R9

	MOVD $0, R1 // i*8 = 0
	MOVD $0, R7 // i = 0
	MOVD $0, R0 // make sure it's zero
	MOVD $0, R4 // c = 0

	MOVD   R5, R12
	AND    $-2, R12
	CMPBGE R5, $2, A6
	BR     E6

A6:
	MOVD   (R8)(R1*1), R6
	MULHDU R9, R6
	MOVD   (R2)(R1*1), R10
	ADDC   R10, R11        // add to low order bits
	ADDE   R0, R6
	ADDC   R4, R11
	ADDE   R0, R6
	MOVD   R6, R4
	MOVD   R11, (R2)(R1*1)

	MOVD   (8)(R8)(R1*1), R6
	MULHDU R9, R6
	MOVD   (8)(R2)(R1*1), R10
	ADDC   R10, R11           // add to low order bits
	ADDE   R0, R6
	ADDC   R4, R11
	ADDE   R0, R6
	MOVD   R6, R4
	MOVD   R11, (8)(R2)(R1*1)

	ADD $16, R1 // i*8 + 8
	ADD $2, R7  // i++

	CMPBLT R7, R12, A6
	BR     E6

L6:
	// TODO: drop unused single-step loop.
	MOVD   (R8)(R1*1), R6
	MULHDU R9, R6
	MOVD   (R2)(R1*1), R10
	ADDC   R10, R11        // add to low order bits
	ADDE   R0, R6
	ADDC   R4, R11
	ADDE   R0, R6
	MOVD   R6, R4
	MOVD   R11, (R2)(R1*1)

	ADD $8, R1 // i*8 + 8
	ADD $1, R7 // i++

E6:
	CMPBLT R7, R5, L6 // i < n

	MOVD R4, c+24(FP)
	RET
