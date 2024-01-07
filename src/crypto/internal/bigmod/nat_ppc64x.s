// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego && (ppc64 || ppc64le)

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB), $0-32
	MOVD	$16, R22 // R22 = z_len
	JMP		addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB), $0-32
	MOVD	$24, R22 // R22 = z_len
	JMP		addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB), $0-32
	MOVD	$32, R22 // R22 = z_len
	JMP		addMulVVWx(SB)

TEXT addMulVVWx(SB), NOFRAME|NOSPLIT, $0
	MOVD z+0(FP), R10	// R10 = z[]
	MOVD x+8(FP), R8	// R8 = x[]
	MOVD y+16(FP), R9	// R9 = y

	MOVD R0, R3		// R3 will be the index register
	CMP  R0, R22
	MOVD R0, R4		// R4 = c = 0
	MOVD R22, CTR		// Initialize loop counter
	BEQ  done
	PCALIGN $16

loop:
	MOVD  (R8)(R3), R20	// Load x[i]
	MOVD  (R10)(R3), R21	// Load z[i]
	MULLD  R9, R20, R6	// R6 = Low-order(x[i]*y)
	MULHDU R9, R20, R7	// R7 = High-order(x[i]*y)
	ADDC   R21, R6		// R6 = z0
	ADDZE  R7		// R7 = z1
	ADDC   R4, R6		// R6 = z0 + c + 0
	ADDZE  R7, R4           // c += z1
	MOVD   R6, (R10)(R3)	// Store z[i]
	ADD    $8, R3
	BC  16, 0, loop		// bdnz

done:
	MOVD R4, c+24(FP)
	RET
