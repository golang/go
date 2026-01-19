// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// func xorBytes(dst, a, b *byte, n int)
TEXT ·xorBytes(SB), NOSPLIT|NOFRAME, $0
	MOV	dst+0(FP), X10
	MOV	a+8(FP), X11
	MOV	b+16(FP), X12
	MOV	n+24(FP), X13

	MOV	$32, X15
	BLT	X13, X15, loop4_check

	// Check alignment - if alignment differs we have to do one byte at a time.
	AND	$7, X10, X5
	AND	$7, X11, X6
	AND	$7, X12, X7
	BNE	X5, X6, loop4_check
	BNE	X5, X7, loop4_check
	BEQZ	X5, loop64_check

	// Check one byte at a time until we reach 8 byte alignment.
	MOV	$8, X8
	SUB	X5, X8
	SUB	X8, X13
align:
	MOVBU	0(X11), X16
	MOVBU	0(X12), X17
	XOR	X16, X17
	MOVB	X17, 0(X10)
	ADD	$1, X10
	ADD	$1, X11
	ADD	$1, X12
	SUB	$1, X8
	BNEZ	X8, align

loop64_check:
	MOV	$64, X15
	BLT	X13, X15, tail32_check
	PCALIGN	$16
loop64:
	MOV	0(X11), X16
	MOV	0(X12), X17
	MOV	8(X11), X18
	MOV	8(X12), X19
	XOR	X16, X17
	XOR	X18, X19
	MOV	X17, 0(X10)
	MOV	X19, 8(X10)
	MOV	16(X11), X20
	MOV	16(X12), X21
	MOV	24(X11), X22
	MOV	24(X12), X23
	XOR	X20, X21
	XOR	X22, X23
	MOV	X21, 16(X10)
	MOV	X23, 24(X10)
	MOV	32(X11), X16
	MOV	32(X12), X17
	MOV	40(X11), X18
	MOV	40(X12), X19
	XOR	X16, X17
	XOR	X18, X19
	MOV	X17, 32(X10)
	MOV	X19, 40(X10)
	MOV	48(X11), X20
	MOV	48(X12), X21
	MOV	56(X11), X22
	MOV	56(X12), X23
	XOR	X20, X21
	XOR	X22, X23
	MOV	X21, 48(X10)
	MOV	X23, 56(X10)
	ADD	$64, X10
	ADD	$64, X11
	ADD	$64, X12
	SUB	$64, X13
	BGE	X13, X15, loop64
	BEQZ	X13, done

tail32_check:
	MOV	$32, X15
	BLT	X13, X15, tail16_check
	MOV	0(X11), X16
	MOV	0(X12), X17
	MOV	8(X11), X18
	MOV	8(X12), X19
	XOR	X16, X17
	XOR	X18, X19
	MOV	X17, 0(X10)
	MOV	X19, 8(X10)
	MOV	16(X11), X20
	MOV	16(X12), X21
	MOV	24(X11), X22
	MOV	24(X12), X23
	XOR	X20, X21
	XOR	X22, X23
	MOV	X21, 16(X10)
	MOV	X23, 24(X10)
	ADD	$32, X10
	ADD	$32, X11
	ADD	$32, X12
	SUB	$32, X13
	BEQZ	X13, done

tail16_check:
	MOV	$16, X15
	BLT	X13, X15, loop4_check
	MOV	0(X11), X16
	MOV	0(X12), X17
	MOV	8(X11), X18
	MOV	8(X12), X19
	XOR	X16, X17
	XOR	X18, X19
	MOV	X17, 0(X10)
	MOV	X19, 8(X10)
	ADD	$16, X10
	ADD	$16, X11
	ADD	$16, X12
	SUB	$16, X13
	BEQZ	X13, done

loop4_check:
	MOV	$4, X15
	BLT	X13, X15, loop1
	PCALIGN	$16
loop4:
	MOVBU	0(X11), X16
	MOVBU	0(X12), X17
	MOVBU	1(X11), X18
	MOVBU	1(X12), X19
	XOR	X16, X17
	XOR	X18, X19
	MOVB	X17, 0(X10)
	MOVB	X19, 1(X10)
	MOVBU	2(X11), X20
	MOVBU	2(X12), X21
	MOVBU	3(X11), X22
	MOVBU	3(X12), X23
	XOR	X20, X21
	XOR	X22, X23
	MOVB	X21, 2(X10)
	MOVB	X23, 3(X10)
	ADD	$4, X10
	ADD	$4, X11
	ADD	$4, X12
	SUB	$4, X13
	BGE	X13, X15, loop4

	PCALIGN	$16
loop1:
	BEQZ	X13, done
	MOVBU	0(X11), X16
	MOVBU	0(X12), X17
	XOR	X16, X17
	MOVB	X17, 0(X10)
	ADD	$1, X10
	ADD	$1, X11
	ADD	$1, X12
	SUB	$1, X13
	JMP	loop1

done:
	RET
