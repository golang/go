// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (mips64 || mips64le) && !purego

#include "textflag.h"

// func xorBytes(dst, a, b *byte, n int)
TEXT ·xorBytes(SB), NOSPLIT|NOFRAME, $0
	MOVV	dst+0(FP), R1
	MOVV	a+8(FP), R2
	MOVV	b+16(FP), R3
	MOVV	n+24(FP), R4

xor_64_check:
	SGTU	$64, R4, R5 // R5 = 1 if (64 > R4)
	BNE	R5, xor_32_check
xor_64:
	MOVV	(R2), R6
	MOVV	8(R2), R7
	MOVV	16(R2), R8
	MOVV	24(R2), R9
	MOVV	(R3), R10
	MOVV	8(R3), R11
	MOVV	16(R3), R12
	MOVV	24(R3), R13
	XOR	R6, R10
	XOR	R7, R11
	XOR	R8, R12
	XOR	R9, R13
	MOVV	R10, (R1)
	MOVV	R11, 8(R1)
	MOVV	R12, 16(R1)
	MOVV	R13, 24(R1)
	MOVV	32(R2), R6
	MOVV	40(R2), R7
	MOVV	48(R2), R8
	MOVV	56(R2), R9
	MOVV	32(R3), R10
	MOVV	40(R3), R11
	MOVV	48(R3), R12
	MOVV	56(R3), R13
	XOR	R6, R10
	XOR	R7, R11
	XOR	R8, R12
	XOR	R9, R13
	MOVV	R10, 32(R1)
	MOVV	R11, 40(R1)
	MOVV	R12, 48(R1)
	MOVV	R13, 56(R1)
	ADDV	$64, R2
	ADDV	$64, R3
	ADDV	$64, R1
	SUBV	$64, R4
	SGTU	$64, R4, R5
	BEQ	R0, R5, xor_64
	BEQ	R0, R4, end

xor_32_check:
	SGTU	$32, R4, R5
	BNE	R5, xor_16_check
xor_32:
	MOVV	(R2), R6
	MOVV	8(R2), R7
	MOVV	16(R2), R8
	MOVV	24(R2), R9
	MOVV	(R3), R10
	MOVV	8(R3), R11
	MOVV	16(R3), R12
	MOVV	24(R3), R13
	XOR	R6, R10
	XOR	R7, R11
	XOR	R8, R12
	XOR	R9, R13
	MOVV	R10, (R1)
	MOVV	R11, 8(R1)
	MOVV	R12, 16(R1)
	MOVV	R13, 24(R1)
	ADDV	$32, R2
	ADDV	$32, R3
	ADDV	$32, R1
	SUBV	$32, R4
	BEQ	R0, R4, end

xor_16_check:
	SGTU	$16, R4, R5
	BNE	R5, xor_8_check
xor_16:
	MOVV	(R2), R6
	MOVV	8(R2), R7
	MOVV	(R3), R8
	MOVV	8(R3), R9
	XOR	R6, R8
	XOR	R7, R9
	MOVV	R8, (R1)
	MOVV	R9, 8(R1)
	ADDV	$16, R2
	ADDV	$16, R3
	ADDV	$16, R1
	SUBV	$16, R4
	BEQ	R0, R4, end

xor_8_check:
	SGTU	$8, R4, R5
	BNE	R5, xor_4_check
xor_8:
	MOVV	(R2), R6
	MOVV	(R3), R7
	XOR	R6, R7
	MOVV	R7, (R1)
	ADDV	$8, R1
	ADDV	$8, R2
	ADDV	$8, R3
	SUBV	$8, R4
	BEQ	R0, R4, end

xor_4_check:
	SGTU	$4, R4, R5
	BNE	R5, xor_2_check
xor_4:
	MOVW	(R2), R6
	MOVW	(R3), R7
	XOR	R6, R7
	MOVW	R7, (R1)
	ADDV	$4, R2
	ADDV	$4, R3
	ADDV	$4, R1
	SUBV	$4, R4
	BEQ	R0, R4, end

xor_2_check:
	SGTU	$2, R4, R5
	BNE	R5, xor_1
xor_2:
	MOVH	(R2), R6
	MOVH	(R3), R7
	XOR	R6, R7
	MOVH	R7, (R1)
	ADDV	$2, R2
	ADDV	$2, R3
	ADDV	$2, R1
	SUBV	$2, R4
	BEQ	R0, R4, end

xor_1:
	MOVB	(R2), R6
	MOVB	(R3), R7
	XOR	R6, R7
	MOVB	R7, (R1)

end:
	RET
