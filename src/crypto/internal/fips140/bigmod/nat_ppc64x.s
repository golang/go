// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego && (ppc64 || ppc64le)

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB), $0-32
	MOVD	$4, R6 // R6 = z_len/4
	JMP		addMulVVWx<>(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB), $0-32
	MOVD	$6, R6 // R6 = z_len/4
	JMP		addMulVVWx<>(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB), $0-32
	MOVD	$8, R6 // R6 = z_len/4
	JMP		addMulVVWx<>(SB)

// This local function expects to be called only by
// callers above. R6 contains the z length/4
// since 4 values are processed for each
// loop iteration, and is guaranteed to be > 0.
// If other callers are added this function might
// need to change.
TEXT addMulVVWx<>(SB), NOSPLIT, $0
	MOVD	z+0(FP), R3
	MOVD	x+8(FP), R4
	MOVD	y+16(FP), R5

	MOVD	$0, R9		// R9 = c = 0
	MOVD	R6, CTR		// Initialize loop counter
	PCALIGN	$16

loop:
	MOVD	0(R4), R14	// x[i]
	MOVD	8(R4), R16	// x[i+1]
	MOVD	16(R4), R18	// x[i+2]
	MOVD	24(R4), R20	// x[i+3]
	MOVD	0(R3), R15	// z[i]
	MOVD	8(R3), R17	// z[i+1]
	MOVD	16(R3), R19	// z[i+2]
	MOVD	24(R3), R21	// z[i+3]
	MULLD	R5, R14, R10	// low x[i]*y
	MULHDU	R5, R14, R11	// high x[i]*y
	ADDC	R15, R10
	ADDZE	R11
	ADDC	R9, R10
	ADDZE	R11, R9
	MULLD	R5, R16, R14	// low x[i+1]*y
	MULHDU	R5, R16, R15	// high x[i+1]*y
	ADDC	R17, R14
	ADDZE	R15
	ADDC	R9, R14
	ADDZE	R15, R9
	MULLD	R5, R18, R16	// low x[i+2]*y
	MULHDU	R5, R18, R17	// high x[i+2]*y
	ADDC	R19, R16
	ADDZE	R17
	ADDC	R9, R16
	ADDZE	R17, R9
	MULLD	R5, R20, R18	// low x[i+3]*y
	MULHDU	R5, R20, R19	// high x[i+3]*y
	ADDC	R21, R18
	ADDZE	R19
	ADDC	R9, R18
	ADDZE	R19, R9
	MOVD	R10, 0(R3)	// z[i]
	MOVD	R14, 8(R3)	// z[i+1]
	MOVD	R16, 16(R3)	// z[i+2]
	MOVD	R18, 24(R3)	// z[i+3]
	ADD	$32, R3
	ADD	$32, R4
	BDNZ	loop

done:
	MOVD	R9, c+24(FP)
	RET
