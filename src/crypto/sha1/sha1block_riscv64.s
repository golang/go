// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

#define LOAD(index) \
	MOVBU	((index*4)+0)(X29), X5; \
	MOVBU	((index*4)+1)(X29), X6; \
	MOVBU	((index*4)+2)(X29), X7; \
	MOVBU	((index*4)+3)(X29), X8; \
	SLL	$24, X5; \
	SLL	$16, X6; \
	OR	X5, X6, X5; \
	SLL	$8, X7; \
	OR	X5, X7, X5; \
	OR	X5, X8, X5; \
	MOVW	X5, (index*4)(X19)

#define SHUFFLE(index) \
	MOVWU	(((index)&0xf)*4)(X19), X5; \
	MOVWU	(((index-3)&0xf)*4)(X19), X6; \
	MOVWU	(((index-8)&0xf)*4)(X19), X7; \
	MOVWU	(((index-14)&0xf)*4)(X19), X8; \
	XOR	X6, X5; \
	XOR	X7, X5; \
	XOR	X8, X5; \
	RORW	$31, X5; \
	MOVW	X5, (((index)&0xf)*4)(X19)

// f = d ^ (b & (c ^ d))
#define FUNC1(a, b, c, d, e) \
	XOR	c, d, X7; \
	AND	b, X7; \
	XOR	d, X7

// f = b ^ c ^ d
#define FUNC2(a, b, c, d, e) \
	XOR	b, c, X7; \
	XOR	d, X7

// f = (b & c) | ((b | c) & d)
#define FUNC3(a, b, c, d, e) \
	OR	b, c, X8; \
	AND	b, c, X6; \
	AND	d, X8; \
	OR	X6, X8, X7

#define FUNC4 FUNC2

#define MIX(a, b, c, d, e, key) \
	RORW	$2, b; \
	ADD	X7, e; \
	RORW	$27, a, X8; \
	ADD	X5, e; \
	ADD	key, e; \
	ADD	X8, e

#define ROUND1(a, b, c, d, e, index) \
	LOAD(index); \
	FUNC1(a, b, c, d, e); \
	MIX(a, b, c, d, e, X15)

#define ROUND1x(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC1(a, b, c, d, e); \
	MIX(a, b, c, d, e, X15)

#define ROUND2(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC2(a, b, c, d, e); \
	MIX(a, b, c, d, e, X16)

#define ROUND3(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC3(a, b, c, d, e); \
	MIX(a, b, c, d, e, X17)

#define ROUND4(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC4(a, b, c, d, e); \
	MIX(a, b, c, d, e, X18)

// func block(dig *Digest, p []byte)
TEXT ·block(SB),NOSPLIT,$64-32
	MOV	p_base+8(FP), X29
	MOV	p_len+16(FP), X30
	SRL	$6, X30
	SLL	$6, X30

	ADD	X29, X30, X28
	BEQ	X28, X29, end

	ADD	$8, X2, X19	// message schedule buffer on stack

	MOV	dig+0(FP), X20
	MOVWU	(0*4)(X20), X10	// a = H0
	MOVWU	(1*4)(X20), X11	// b = H1
	MOVWU	(2*4)(X20), X12	// c = H2
	MOVWU	(3*4)(X20), X13	// d = H3
	MOVWU	(4*4)(X20), X14	// e = H4

	MOV	$·_K(SB), X21
	MOVW	(0*4)(X21), X15
	MOVW	(1*4)(X21), X16
	MOVW	(2*4)(X21), X17
	MOVW	(3*4)(X21), X18

loop:
	MOVW	X10, X22
	MOVW	X11, X23
	MOVW	X12, X24
	MOVW	X13, X25
	MOVW	X14, X26

	ROUND1(X10, X11, X12, X13, X14, 0)
	ROUND1(X14, X10, X11, X12, X13, 1)
	ROUND1(X13, X14, X10, X11, X12, 2)
	ROUND1(X12, X13, X14, X10, X11, 3)
	ROUND1(X11, X12, X13, X14, X10, 4)
	ROUND1(X10, X11, X12, X13, X14, 5)
	ROUND1(X14, X10, X11, X12, X13, 6)
	ROUND1(X13, X14, X10, X11, X12, 7)
	ROUND1(X12, X13, X14, X10, X11, 8)
	ROUND1(X11, X12, X13, X14, X10, 9)
	ROUND1(X10, X11, X12, X13, X14, 10)
	ROUND1(X14, X10, X11, X12, X13, 11)
	ROUND1(X13, X14, X10, X11, X12, 12)
	ROUND1(X12, X13, X14, X10, X11, 13)
	ROUND1(X11, X12, X13, X14, X10, 14)
	ROUND1(X10, X11, X12, X13, X14, 15)

	ROUND1x(X14, X10, X11, X12, X13, 16)
	ROUND1x(X13, X14, X10, X11, X12, 17)
	ROUND1x(X12, X13, X14, X10, X11, 18)
	ROUND1x(X11, X12, X13, X14, X10, 19)

	ROUND2(X10, X11, X12, X13, X14, 20)
	ROUND2(X14, X10, X11, X12, X13, 21)
	ROUND2(X13, X14, X10, X11, X12, 22)
	ROUND2(X12, X13, X14, X10, X11, 23)
	ROUND2(X11, X12, X13, X14, X10, 24)
	ROUND2(X10, X11, X12, X13, X14, 25)
	ROUND2(X14, X10, X11, X12, X13, 26)
	ROUND2(X13, X14, X10, X11, X12, 27)
	ROUND2(X12, X13, X14, X10, X11, 28)
	ROUND2(X11, X12, X13, X14, X10, 29)
	ROUND2(X10, X11, X12, X13, X14, 30)
	ROUND2(X14, X10, X11, X12, X13, 31)
	ROUND2(X13, X14, X10, X11, X12, 32)
	ROUND2(X12, X13, X14, X10, X11, 33)
	ROUND2(X11, X12, X13, X14, X10, 34)
	ROUND2(X10, X11, X12, X13, X14, 35)
	ROUND2(X14, X10, X11, X12, X13, 36)
	ROUND2(X13, X14, X10, X11, X12, 37)
	ROUND2(X12, X13, X14, X10, X11, 38)
	ROUND2(X11, X12, X13, X14, X10, 39)

	ROUND3(X10, X11, X12, X13, X14, 40)
	ROUND3(X14, X10, X11, X12, X13, 41)
	ROUND3(X13, X14, X10, X11, X12, 42)
	ROUND3(X12, X13, X14, X10, X11, 43)
	ROUND3(X11, X12, X13, X14, X10, 44)
	ROUND3(X10, X11, X12, X13, X14, 45)
	ROUND3(X14, X10, X11, X12, X13, 46)
	ROUND3(X13, X14, X10, X11, X12, 47)
	ROUND3(X12, X13, X14, X10, X11, 48)
	ROUND3(X11, X12, X13, X14, X10, 49)
	ROUND3(X10, X11, X12, X13, X14, 50)
	ROUND3(X14, X10, X11, X12, X13, 51)
	ROUND3(X13, X14, X10, X11, X12, 52)
	ROUND3(X12, X13, X14, X10, X11, 53)
	ROUND3(X11, X12, X13, X14, X10, 54)
	ROUND3(X10, X11, X12, X13, X14, 55)
	ROUND3(X14, X10, X11, X12, X13, 56)
	ROUND3(X13, X14, X10, X11, X12, 57)
	ROUND3(X12, X13, X14, X10, X11, 58)
	ROUND3(X11, X12, X13, X14, X10, 59)

	ROUND4(X10, X11, X12, X13, X14, 60)
	ROUND4(X14, X10, X11, X12, X13, 61)
	ROUND4(X13, X14, X10, X11, X12, 62)
	ROUND4(X12, X13, X14, X10, X11, 63)
	ROUND4(X11, X12, X13, X14, X10, 64)
	ROUND4(X10, X11, X12, X13, X14, 65)
	ROUND4(X14, X10, X11, X12, X13, 66)
	ROUND4(X13, X14, X10, X11, X12, 67)
	ROUND4(X12, X13, X14, X10, X11, 68)
	ROUND4(X11, X12, X13, X14, X10, 69)
	ROUND4(X10, X11, X12, X13, X14, 70)
	ROUND4(X14, X10, X11, X12, X13, 71)
	ROUND4(X13, X14, X10, X11, X12, 72)
	ROUND4(X12, X13, X14, X10, X11, 73)
	ROUND4(X11, X12, X13, X14, X10, 74)
	ROUND4(X10, X11, X12, X13, X14, 75)
	ROUND4(X14, X10, X11, X12, X13, 76)
	ROUND4(X13, X14, X10, X11, X12, 77)
	ROUND4(X12, X13, X14, X10, X11, 78)
	ROUND4(X11, X12, X13, X14, X10, 79)

	ADD	X22, X10
	ADD	X23, X11
	ADD	X24, X12
	ADD	X25, X13
	ADD	X26, X14

	ADD	$64, X29
	BNE	X28, X29, loop

end:
	MOVW	X10, (0*4)(X20)
	MOVW	X11, (1*4)(X20)
	MOVW	X12, (2*4)(X20)
	MOVW	X13, (3*4)(X20)
	MOVW	X14, (4*4)(X20)

	RET

GLOBL	·_K(SB),RODATA,$16
DATA	·_K+0(SB)/4, $0x5A827999
DATA	·_K+4(SB)/4, $0x6ED9EBA1
DATA	·_K+8(SB)/4, $0x8F1BBCDC
DATA	·_K+12(SB)/4, $0xCA62C1D6
