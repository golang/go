// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func xorBytes(dst, a, b *byte, n int)
TEXT ·xorBytes(SB), NOSPLIT, $0
	MOVV	dst+0(FP), R4
	MOVV	a+8(FP), R5
	MOVV	b+16(FP), R6
	MOVV	n+24(FP), R7

	MOVV	$64, R9
	BGEU	R7, R9, loop64	// n >= 64
tail:
	SRLV	$1, R9
	BGEU	R7, R9, xor_32	// n >= 32 && n < 64
	SRLV	$1, R9
	BGEU	R7, R9, xor_16	// n >= 16 && n < 32
	SRLV	$1, R9
	BGEU	R7, R9, xor_8	// n >= 8 && n < 16
	SRLV	$1, R9
	BGEU	R7, R9, xor_4	// n >= 4 && n < 8
	SRLV	$1, R9
	BGEU	R7, R9, xor_2	// n >= 2 && n < 4
	SRLV	$1, R9
	BGEU	R7, R9, xor_1	// n = 1

loop64:
	MOVV	(R5), R10
	MOVV	8(R5), R11
	MOVV	16(R5), R12
	MOVV	24(R5), R13
	MOVV	(R6), R14
	MOVV	8(R6), R15
	MOVV	16(R6), R16
	MOVV	24(R6), R17
	XOR	R10, R14
	XOR	R11, R15
	XOR	R12, R16
	XOR	R13, R17
	MOVV	R14, (R4)
	MOVV	R15, 8(R4)
	MOVV	R16, 16(R4)
	MOVV	R17, 24(R4)
	MOVV	32(R5), R10
	MOVV	40(R5), R11
	MOVV	48(R5), R12
	MOVV	56(R5), R13
	MOVV	32(R6), R14
	MOVV	40(R6), R15
	MOVV	48(R6), R16
	MOVV	56(R6), R17
	XOR	R10, R14
	XOR	R11, R15
	XOR	R12, R16
	XOR	R13, R17
	MOVV	R14, 32(R4)
	MOVV	R15, 40(R4)
	MOVV	R16, 48(R4)
	MOVV	R17, 56(R4)
	ADDV	$64, R5
	ADDV	$64, R6
	ADDV	$64, R4
	SUBV	$64, R7
	// 64 in R9
	BGEU	R7, R9, loop64
	BEQ	R7, R0, end

xor_32_check:
	SRLV	$1, R9
	BLT	R7, R9, xor_16_check
xor_32:
	MOVV	(R5), R10
	MOVV	8(R5), R11
	MOVV	16(R5), R12
	MOVV	24(R5), R13
	MOVV	(R6), R14
	MOVV	8(R6), R15
	MOVV	16(R6), R16
	MOVV	24(R6), R17
	XOR	R10, R14
	XOR	R11, R15
	XOR	R12, R16
	XOR	R13, R17
	MOVV	R14, (R4)
	MOVV	R15, 8(R4)
	MOVV	R16, 16(R4)
	MOVV	R17, 24(R4)
	ADDV	$32, R5
	ADDV	$32, R6
	ADDV	$32, R4
	SUBV	$32, R7
	BEQ	R7, R0, end

xor_16_check:
	SRLV	$1, R9
	BLT	R7, R9, xor_8_check
xor_16:
	MOVV	(R5), R10
	MOVV	8(R5), R11
	MOVV	(R6), R12
	MOVV	8(R6), R13
	XOR	R10, R12
	XOR	R11, R13
	MOVV	R12, (R4)
	MOVV	R13, 8(R4)
	ADDV	$16, R5
	ADDV	$16, R6
	ADDV	$16, R4
	SUBV	$16, R7
	BEQ	R7, R0, end

xor_8_check:
	SRLV	$1, R9
	BLT	R7, R9, xor_4_check
xor_8:
	MOVV	(R5), R10
	MOVV	(R6), R11
	XOR	R10, R11
	MOVV	R11, (R4)
	ADDV	$8, R5
	ADDV	$8, R6
	ADDV	$8, R4
	SUBV	$8, R7
	BEQ	R7, R0, end

xor_4_check:
	SRLV	$1, R9
	BLT	R7, R9, xor_2_check
xor_4:
	MOVW	(R5), R10
	MOVW	(R6), R11
	XOR	R10, R11
	MOVW	R11, (R4)
	ADDV	$4, R5
	ADDV	$4, R6
	ADDV	$4, R4
	SUBV	$4, R7
	BEQ	R7, R0, end

xor_2_check:
	SRLV	$1, R9
	BLT	R7, R9, xor_1
xor_2:
	MOVH	(R5), R10
	MOVH	(R6), R11
	XOR	R10, R11
	MOVH	R11, (R4)
	ADDV	$2, R5
	ADDV	$2, R6
	ADDV	$2, R4
	SUBV	$2, R7
	BEQ	R7, R0, end

xor_1:
	MOVB	(R5), R10
	MOVB	(R6), R11
	XOR	R10, R11
	MOVB	R11, (R4)

end:
	RET
