// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Loong64 version of md5block.go
// derived from crypto/md5/md5block_amd64.s

//go:build !purego

#define REGTMP	R30
#define REGTMP1 R12
#define REGTMP2 R18

#include "textflag.h"

// func block(dig *digest, p []byte)
TEXT	·block(SB),NOSPLIT,$0-32
	MOVV	dig+0(FP), R4
	MOVV	p+8(FP), R5
	MOVV	p_len+16(FP), R6
	AND	$~63, R6
	BEQ	R6, zero

	// p_len >= 64
	ADDV	R5, R6, R24
	MOVW	(0*4)(R4), R7
	MOVW	(1*4)(R4), R8
	MOVW	(2*4)(R4), R9
	MOVW	(3*4)(R4), R10

loop:
	MOVW	R7, R14
	MOVW	R8, R15
	MOVW	R9, R16
	MOVW	R10, R17

	MOVW	(0*4)(R5), R11
	MOVW	R10, REGTMP1

// F = ((c ^ d) & b) ^ d
#define ROUND1(a, b, c, d, index, const, shift) \
	ADDV	$const, a; \
	ADD	R11, a; \
	MOVW	(index*4)(R5), R11; \
	XOR	c, REGTMP1; \
	AND	b, REGTMP1; \
	XOR	d, REGTMP1; \
	ADD	REGTMP1, a; \
	ROTR	$(32-shift), a; \
	MOVW	c, REGTMP1; \
	ADD	b, a

	ROUND1(R7,  R8,  R9,  R10,  1, 0xd76aa478,  7);
	ROUND1(R10, R7,  R8,  R9,   2, 0xe8c7b756, 12);
	ROUND1(R9,  R10, R7,  R8,   3, 0x242070db, 17);
	ROUND1(R8,  R9,  R10, R7,   4, 0xc1bdceee, 22);
	ROUND1(R7,  R8,  R9,  R10,  5, 0xf57c0faf,  7);
	ROUND1(R10, R7,  R8,  R9,   6, 0x4787c62a, 12);
	ROUND1(R9,  R10, R7,  R8,   7, 0xa8304613, 17);
	ROUND1(R8,  R9,  R10, R7,   8, 0xfd469501, 22);
	ROUND1(R7,  R8,  R9,  R10,  9, 0x698098d8,  7);
	ROUND1(R10, R7,  R8,  R9,  10, 0x8b44f7af, 12);
	ROUND1(R9,  R10, R7,  R8,  11, 0xffff5bb1, 17);
	ROUND1(R8,  R9,  R10, R7,  12, 0x895cd7be, 22);
	ROUND1(R7,  R8,  R9,  R10, 13, 0x6b901122,  7);
	ROUND1(R10, R7,  R8,  R9,  14, 0xfd987193, 12);
	ROUND1(R9,  R10, R7,  R8,  15, 0xa679438e, 17);
	ROUND1(R8,  R9,  R10, R7,   1, 0x49b40821, 22);

	MOVW	(1*4)(R5), R11

// F = ((b ^ c) & d) ^ c
#define ROUND2(a, b, c, d, index, const, shift) \
	ADDV	$const, a; \
	ADD	R11, a; \
	MOVW	(index*4)(R5), R11; \
	XOR	b, c, REGTMP; \
	AND	REGTMP, d, REGTMP; \
	XOR	REGTMP, c, REGTMP; \
	ADD	REGTMP, a; \
	ROTR	$(32-shift), a; \
	ADD	b, a

	ROUND2(R7,  R8,  R9,  R10,  6, 0xf61e2562,  5);
	ROUND2(R10, R7,  R8,  R9,  11, 0xc040b340,  9);
	ROUND2(R9,  R10, R7,  R8,   0, 0x265e5a51, 14);
	ROUND2(R8,  R9,  R10, R7,   5, 0xe9b6c7aa, 20);
	ROUND2(R7,  R8,  R9,  R10, 10, 0xd62f105d,  5);
	ROUND2(R10, R7,  R8,  R9,  15,  0x2441453,  9);
	ROUND2(R9,  R10, R7,  R8,   4, 0xd8a1e681, 14);
	ROUND2(R8,  R9,  R10, R7,   9, 0xe7d3fbc8, 20);
	ROUND2(R7,  R8,  R9,  R10, 14, 0x21e1cde6,  5);
	ROUND2(R10, R7,  R8,  R9,   3, 0xc33707d6,  9);
	ROUND2(R9,  R10, R7,  R8,   8, 0xf4d50d87, 14);
	ROUND2(R8,  R9,  R10, R7,  13, 0x455a14ed, 20);
	ROUND2(R7,  R8,  R9,  R10,  2, 0xa9e3e905,  5);
	ROUND2(R10, R7,  R8,  R9,   7, 0xfcefa3f8,  9);
	ROUND2(R9,  R10, R7,  R8,  12, 0x676f02d9, 14);
	ROUND2(R8,  R9,  R10, R7,   5, 0x8d2a4c8a, 20);

	MOVW	(5*4)(R5), R11
	MOVW	R9, REGTMP1

// F = b ^ c ^ d
#define ROUND3(a, b, c, d, index, const, shift) \
	ADDV	$const, a; \
	ADD	R11, a; \
	MOVW	(index*4)(R5), R11; \
	XOR	d, REGTMP1; \
	XOR	b, REGTMP1; \
	ADD	REGTMP1, a; \
	ROTR	$(32-shift), a; \
	MOVW	b, REGTMP1; \
	ADD	b, a

	ROUND3(R7,  R8,  R9,  R10,  8, 0xfffa3942,  4);
	ROUND3(R10, R7,  R8,  R9,  11, 0x8771f681, 11);
	ROUND3(R9,  R10, R7,  R8,  14, 0x6d9d6122, 16);
	ROUND3(R8,  R9,  R10, R7,   1, 0xfde5380c, 23);
	ROUND3(R7,  R8,  R9,  R10,  4, 0xa4beea44,  4);
	ROUND3(R10, R7,  R8,  R9,   7, 0x4bdecfa9, 11);
	ROUND3(R9,  R10, R7,  R8,  10, 0xf6bb4b60, 16);
	ROUND3(R8,  R9,  R10, R7,  13, 0xbebfbc70, 23);
	ROUND3(R7,  R8,  R9,  R10,  0, 0x289b7ec6,  4);
	ROUND3(R10, R7,  R8,  R9,   3, 0xeaa127fa, 11);
	ROUND3(R9,  R10, R7,  R8,   6, 0xd4ef3085, 16);
	ROUND3(R8,  R9,  R10, R7,   9,  0x4881d05, 23);
	ROUND3(R7,  R8,  R9,  R10, 12, 0xd9d4d039,  4);
	ROUND3(R10, R7,  R8,  R9,  15, 0xe6db99e5, 11);
	ROUND3(R9,  R10, R7,  R8,   2, 0x1fa27cf8, 16);
	ROUND3(R8,  R9,  R10, R7,   0, 0xc4ac5665, 23);

	MOVW	(0*4)(R5), R11
	MOVV	$0xffffffff, REGTMP2
	XOR	R10, REGTMP2, REGTMP1	// REGTMP1 = ~d

// F = c ^ (b | (~d))
#define ROUND4(a, b, c, d, index, const, shift) \
	ADDV	$const, a; \
	ADD	R11, a; \
	MOVW	(index*4)(R5), R11; \
	OR	b, REGTMP1; \
	XOR	c, REGTMP1; \
	ADD	REGTMP1, a; \
	ROTR	$(32-shift), a; \
	MOVV	$0xffffffff, REGTMP2; \
	XOR	c, REGTMP2, REGTMP1; \
	ADD	b, a

	ROUND4(R7,  R8,  R9,  R10,  7, 0xf4292244,  6);
	ROUND4(R10, R7,  R8,  R9,  14, 0x432aff97, 10);
	ROUND4(R9,  R10, R7,  R8,   5, 0xab9423a7, 15);
	ROUND4(R8,  R9,  R10, R7,  12, 0xfc93a039, 21);
	ROUND4(R7,  R8,  R9,  R10,  3, 0x655b59c3,  6);
	ROUND4(R10, R7,  R8,  R9,  10, 0x8f0ccc92, 10);
	ROUND4(R9,  R10, R7,  R8,   1, 0xffeff47d, 15);
	ROUND4(R8,  R9,  R10, R7,   8, 0x85845dd1, 21);
	ROUND4(R7,  R8,  R9,  R10, 15, 0x6fa87e4f,  6);
	ROUND4(R10, R7,  R8,  R9,   6, 0xfe2ce6e0, 10);
	ROUND4(R9,  R10, R7,  R8,  13, 0xa3014314, 15);
	ROUND4(R8,  R9,  R10, R7,   4, 0x4e0811a1, 21);
	ROUND4(R7,  R8,  R9,  R10, 11, 0xf7537e82,  6);
	ROUND4(R10, R7,  R8,  R9,   2, 0xbd3af235, 10);
	ROUND4(R9,  R10, R7,  R8,   9, 0x2ad7d2bb, 15);
	ROUND4(R8,  R9,  R10, R7,   0, 0xeb86d391, 21);

	ADD	R14, R7
	ADD	R15, R8
	ADD	R16, R9
	ADD	R17, R10

	ADDV	$64, R5
	BNE	R5, R24, loop

	MOVW	R7, (0*4)(R4)
	MOVW	R8, (1*4)(R4)
	MOVW	R9, (2*4)(R4)
	MOVW	R10, (3*4)(R4)
zero:
	RET
