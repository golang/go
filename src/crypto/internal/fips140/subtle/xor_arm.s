// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func xorBytes(dst, a, b *byte, n int)
TEXT ·xorBytes(SB), NOSPLIT|NOFRAME, $0
	MOVW	dst+0(FP), R0
	MOVW	a+4(FP), R1
	MOVW	b+8(FP), R2
	MOVW	n+12(FP), R3

xor_32_check:
	CMP	$32, R3
	BLT	xor_16_check
xor_32_loop:
	MOVW	(R1), R4
	MOVW	4(R1), R5
	MOVW	8(R1), R6
	MOVW	(R2), R7
	MOVW	4(R2), R8
	MOVW	8(R2), R9
	EOR	R4, R7
	EOR	R5, R8
	EOR	R6, R9
	MOVW	R7, (R0)
	MOVW	R8, 4(R0)
	MOVW	R9, 8(R0)

	MOVW	12(R1), R4
	MOVW	16(R1), R5
	MOVW	20(R1), R6
	MOVW	12(R2), R7
	MOVW	16(R2), R8
	MOVW	20(R2), R9
	EOR	R4, R7
	EOR	R5, R8
	EOR	R6, R9
	MOVW	R7, 12(R0)
	MOVW	R8, 16(R0)
	MOVW	R9, 20(R0)

	MOVW	24(R1), R4
	MOVW	28(R1), R5
	MOVW	24(R2), R6
	MOVW	28(R2), R7
	EOR	 R4, R6
	EOR	 R5, R7
	MOVW	R6, 24(R0)
	MOVW	R7, 28(R0)

	ADD	$32, R1
	ADD	$32, R2
	ADD	$32, R0
	SUB	$32, R3
	CMP	$32, R3
	BGE	xor_32_loop
	CMP	$0, R3
	BEQ	end

xor_16_check:
	CMP	$16, R3
	BLT	xor_8_check
xor_16:
	MOVW	(R1), R4
	MOVW	4(R1), R5
	MOVW	(R2), R6
	MOVW	4(R2), R7
	EOR	R4, R6
	EOR	R5, R7
	MOVW	R6, (R0)
	MOVW	R7, 4(R0)

	MOVW	8(R1), R4
	MOVW	12(R1), R5
	MOVW	8(R2), R6
	MOVW	12(R2), R7
	EOR	R4, R6
	EOR	R5, R7
	MOVW	R6, 8(R0)
	MOVW	R7, 12(R0)
	ADD	$16, R1
	ADD	$16, R2
	ADD	$16, R0
	SUB	$16, R3
	CMP	$0, R3
	BEQ	end

xor_8_check:
	CMP	$8, R3
	BLT	xor_4_check
xor_8:
	MOVW	(R1), R4
	MOVW	4(R1), R5
	MOVW	(R2), R6
	MOVW	4(R2), R7
	EOR	R4, R6
	EOR	R5, R7
	MOVW	R6, (R0)
	MOVW	R7, 4(R0)

	ADD	$8, R0
	ADD	$8, R1
	ADD	$8, R2
	SUB	$8, R3
	CMP	$0, R3
	BEQ	end

xor_4_check:
	CMP	$4, R3
	BLT	xor_2_check
xor_4:
	MOVW	(R1), R4
	MOVW	(R2), R5
	EOR	R4, R5
	MOVW	R5, (R0)
	ADD	$4, R1
	ADD	$4, R2
	ADD	$4, R0
	SUB	$4, R3
	CMP	$0, R3
	BEQ	end

xor_2_check:
	CMP	$2, R3
	BLT	xor_1
xor_2:
	MOVH	(R1), R4
	MOVH	(R2), R5
	EOR	R4, R5
	MOVH	R5, (R0)
	ADD	$2, R1
	ADD	$2, R2
	ADD	$2, R0
	SUB	$2, R3
	CMP	$0, R3
	BEQ	end

xor_1:
	MOVB	(R1), R4
	MOVB	(R2), R5
	EOR	R4, R5
	MOVB	R5, (R0)

end:
	RET
