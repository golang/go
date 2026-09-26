// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "asm_riscv64.h"
#include "go_asm.h"
#include "textflag.h"

// func encodeChunk(encode *[64]byte, dst, src *byte, n int)
//
// On entry (ABI0 calling convention), stack layout:
//   encode+0(FP)    - *[64]byte (8 bytes)
//   dst+8(FP)       - *byte     (8 bytes)
//   src+16(FP)      - *byte     (8 bytes)
//   n+24(FP)        - int       (8 bytes)
//
// Register usage:
//   X5  = encode table base (preserved across loop)
//   X6  = src pointer
//   X7  = dst pointer
//   X28 = remaining count (n)
//   X30 = $12 (loop constant)
//   X31 = assembler temp (do not use)

TEXT ·encodeChunk(SB),NOSPLIT,$0-32
	MOV	n+24(FP), X28	// X28 = n
	BEQZ	X28, ret

	MOV	encode+0(FP), X5	// X5  = encode table base
	MOV	dst+8(FP), X7		// X7  = dst pointer
	MOV	src+16(FP), X6		// X6  = src pointer

	// Scalar 4x loop: per-byte MOVBU loads, no wide loads, no REV8.
	PCALIGN	$16
scalar_4x:
	MOV	$12, X30
	BLT	X28, X30, tail

	PCALIGN	$16
scalar_loop4:
	// Group 1: src[0..2] -> dst[0..3]
	MOVBU	(X6), X10
	MOVBU	1(X6), X11
	MOVBU	2(X6), X12
	SLLI	$16, X10, X10
	SLLI	$8, X11, X11
	OR	X11, X10, X10
	OR	X12, X10, X10

	SRLI	$18, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, (X7)

	SRLI	$12, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 1(X7)

	SRLI	$6, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 2(X7)

	ANDI	$0x3F, X10, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 3(X7)

	// Group 2: src[3..5] -> dst[4..7]
	MOVBU	3(X6), X10
	MOVBU	4(X6), X11
	MOVBU	5(X6), X12
	SLLI	$16, X10, X10
	SLLI	$8, X11, X11
	OR	X11, X10, X10
	OR	X12, X10, X10

	SRLI	$18, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 4(X7)

	SRLI	$12, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 5(X7)

	SRLI	$6, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 6(X7)

	ANDI	$0x3F, X10, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 7(X7)

	// Group 3: src[6..8] -> dst[8..11]
	MOVBU	6(X6), X10
	MOVBU	7(X6), X11
	MOVBU	8(X6), X12
	SLLI	$16, X10, X10
	SLLI	$8, X11, X11
	OR	X11, X10, X10
	OR	X12, X10, X10

	SRLI	$18, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 8(X7)

	SRLI	$12, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 9(X7)

	SRLI	$6, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 10(X7)

	ANDI	$0x3F, X10, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 11(X7)

	// Group 4: src[9..11] -> dst[12..15]
	MOVBU	9(X6), X10
	MOVBU	10(X6), X11
	MOVBU	11(X6), X12
	SLLI	$16, X10, X10
	SLLI	$8, X11, X11
	OR	X11, X10, X10
	OR	X12, X10, X10

	SRLI	$18, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 12(X7)

	SRLI	$12, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 13(X7)

	SRLI	$6, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 14(X7)

	ANDI	$0x3F, X10, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 15(X7)

	ADD	$12, X6
	ADD	$16, X7
	ADD	$-12, X28
	BGE	X28, X30, scalar_loop4

	BEQZ	X28, ret

	// Tail: remaining < 12 bytes, one group per iteration
	PCALIGN	$16
tail:
	MOVBU	(X6), X10
	MOVBU	1(X6), X11
	MOVBU	2(X6), X12

	SLLI	$16, X10, X10
	SLLI	$8, X11, X11
	OR	X11, X10, X10
	OR	X12, X10, X10

	SRLI	$18, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, (X7)

	SRLI	$12, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 1(X7)

	SRLI	$6, X10, X11
	ANDI	$0x3F, X11, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 2(X7)

	ANDI	$0x3F, X10, X11
	ADD	X5, X11, X11
	MOVBU	(X11), X11
	MOVB	X11, 3(X7)

	ADD	$3, X6
	ADD	$4, X7
	ADD	$-3, X28
	BNEZ	X28, tail

ret:
	RET
