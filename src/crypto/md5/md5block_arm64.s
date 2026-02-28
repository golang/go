// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// ARM64 version of md5block.go
// derived from crypto/md5/md5block_amd64.s

//go:build !purego

#include "textflag.h"

TEXT	·block(SB),NOSPLIT,$0-32
	MOVD	dig+0(FP), R0
	MOVD	p+8(FP), R1
	MOVD	p_len+16(FP), R2
	AND	$~63, R2
	CBZ	R2, zero

	ADD	R1, R2, R21
	LDPW	(0*8)(R0), (R4, R5)
	LDPW	(1*8)(R0), (R6, R7)

loop:
	MOVW	R4, R12
	MOVW	R5, R13
	MOVW	R6, R14
	MOVW	R7, R15

	MOVW	(0*4)(R1), R8
	MOVW	R7, R9

#define ROUND1(a, b, c, d, index, const, shift) \
	ADDW	$const, a; \
	ADDW	R8, a; \
	MOVW	(index*4)(R1), R8; \
	EORW	c, R9; \
	ANDW	b, R9; \
	EORW	d, R9; \
	ADDW	R9, a; \
	RORW	$(32-shift), a; \
	MOVW	c, R9; \
	ADDW	b, a

	ROUND1(R4,R5,R6,R7, 1,0xd76aa478, 7);
	ROUND1(R7,R4,R5,R6, 2,0xe8c7b756,12);
	ROUND1(R6,R7,R4,R5, 3,0x242070db,17);
	ROUND1(R5,R6,R7,R4, 4,0xc1bdceee,22);
	ROUND1(R4,R5,R6,R7, 5,0xf57c0faf, 7);
	ROUND1(R7,R4,R5,R6, 6,0x4787c62a,12);
	ROUND1(R6,R7,R4,R5, 7,0xa8304613,17);
	ROUND1(R5,R6,R7,R4, 8,0xfd469501,22);
	ROUND1(R4,R5,R6,R7, 9,0x698098d8, 7);
	ROUND1(R7,R4,R5,R6,10,0x8b44f7af,12);
	ROUND1(R6,R7,R4,R5,11,0xffff5bb1,17);
	ROUND1(R5,R6,R7,R4,12,0x895cd7be,22);
	ROUND1(R4,R5,R6,R7,13,0x6b901122, 7);
	ROUND1(R7,R4,R5,R6,14,0xfd987193,12);
	ROUND1(R6,R7,R4,R5,15,0xa679438e,17);
	ROUND1(R5,R6,R7,R4, 0,0x49b40821,22);

	MOVW	(1*4)(R1), R8
	MOVW	R7, R9
	MOVW	R7, R10

#define ROUND2(a, b, c, d, index, const, shift) \
	ADDW	$const, a; \
	ADDW	R8, a; \
	MOVW	(index*4)(R1), R8; \
	ANDW	b, R10; \
	BICW	R9, c, R9; \
	ORRW	R9, R10; \
	MOVW	c, R9; \
	ADDW	R10, a; \
	MOVW	c, R10; \
	RORW	$(32-shift), a; \
	ADDW	b, a

	ROUND2(R4,R5,R6,R7, 6,0xf61e2562, 5);
	ROUND2(R7,R4,R5,R6,11,0xc040b340, 9);
	ROUND2(R6,R7,R4,R5, 0,0x265e5a51,14);
	ROUND2(R5,R6,R7,R4, 5,0xe9b6c7aa,20);
	ROUND2(R4,R5,R6,R7,10,0xd62f105d, 5);
	ROUND2(R7,R4,R5,R6,15, 0x2441453, 9);
	ROUND2(R6,R7,R4,R5, 4,0xd8a1e681,14);
	ROUND2(R5,R6,R7,R4, 9,0xe7d3fbc8,20);
	ROUND2(R4,R5,R6,R7,14,0x21e1cde6, 5);
	ROUND2(R7,R4,R5,R6, 3,0xc33707d6, 9);
	ROUND2(R6,R7,R4,R5, 8,0xf4d50d87,14);
	ROUND2(R5,R6,R7,R4,13,0x455a14ed,20);
	ROUND2(R4,R5,R6,R7, 2,0xa9e3e905, 5);
	ROUND2(R7,R4,R5,R6, 7,0xfcefa3f8, 9);
	ROUND2(R6,R7,R4,R5,12,0x676f02d9,14);
	ROUND2(R5,R6,R7,R4, 0,0x8d2a4c8a,20);

	MOVW	(5*4)(R1), R8
	MOVW	R6, R9

#define ROUND3(a, b, c, d, index, const, shift) \
	ADDW	$const, a; \
	ADDW	R8, a; \
	MOVW	(index*4)(R1), R8; \
	EORW	d, R9; \
	EORW	b, R9; \
	ADDW	R9, a; \
	RORW	$(32-shift), a; \
	MOVW	b, R9; \
	ADDW	b, a

	ROUND3(R4,R5,R6,R7, 8,0xfffa3942, 4);
	ROUND3(R7,R4,R5,R6,11,0x8771f681,11);
	ROUND3(R6,R7,R4,R5,14,0x6d9d6122,16);
	ROUND3(R5,R6,R7,R4, 1,0xfde5380c,23);
	ROUND3(R4,R5,R6,R7, 4,0xa4beea44, 4);
	ROUND3(R7,R4,R5,R6, 7,0x4bdecfa9,11);
	ROUND3(R6,R7,R4,R5,10,0xf6bb4b60,16);
	ROUND3(R5,R6,R7,R4,13,0xbebfbc70,23);
	ROUND3(R4,R5,R6,R7, 0,0x289b7ec6, 4);
	ROUND3(R7,R4,R5,R6, 3,0xeaa127fa,11);
	ROUND3(R6,R7,R4,R5, 6,0xd4ef3085,16);
	ROUND3(R5,R6,R7,R4, 9, 0x4881d05,23);
	ROUND3(R4,R5,R6,R7,12,0xd9d4d039, 4);
	ROUND3(R7,R4,R5,R6,15,0xe6db99e5,11);
	ROUND3(R6,R7,R4,R5, 2,0x1fa27cf8,16);
	ROUND3(R5,R6,R7,R4, 0,0xc4ac5665,23);

	MOVW	(0*4)(R1), R8
	MVNW	R7, R9

#define ROUND4(a, b, c, d, index, const, shift) \
	ADDW	$const, a; \
	ADDW	R8, a; \
	MOVW	(index*4)(R1), R8; \
	ORRW	b, R9; \
	EORW	c, R9; \
	ADDW	R9, a; \
	RORW	$(32-shift), a; \
	MVNW	c, R9; \
	ADDW	b, a

	ROUND4(R4,R5,R6,R7, 7,0xf4292244, 6);
	ROUND4(R7,R4,R5,R6,14,0x432aff97,10);
	ROUND4(R6,R7,R4,R5, 5,0xab9423a7,15);
	ROUND4(R5,R6,R7,R4,12,0xfc93a039,21);
	ROUND4(R4,R5,R6,R7, 3,0x655b59c3, 6);
	ROUND4(R7,R4,R5,R6,10,0x8f0ccc92,10);
	ROUND4(R6,R7,R4,R5, 1,0xffeff47d,15);
	ROUND4(R5,R6,R7,R4, 8,0x85845dd1,21);
	ROUND4(R4,R5,R6,R7,15,0x6fa87e4f, 6);
	ROUND4(R7,R4,R5,R6, 6,0xfe2ce6e0,10);
	ROUND4(R6,R7,R4,R5,13,0xa3014314,15);
	ROUND4(R5,R6,R7,R4, 4,0x4e0811a1,21);
	ROUND4(R4,R5,R6,R7,11,0xf7537e82, 6);
	ROUND4(R7,R4,R5,R6, 2,0xbd3af235,10);
	ROUND4(R6,R7,R4,R5, 9,0x2ad7d2bb,15);
	ROUND4(R5,R6,R7,R4, 0,0xeb86d391,21);

	ADDW	R12, R4
	ADDW	R13, R5
	ADDW	R14, R6
	ADDW	R15, R7

	ADD	$64, R1
	CMP	R1, R21
	BNE	loop

	STPW	(R4, R5), (0*8)(R0)
	STPW	(R6, R7), (1*8)(R0)
zero:
	RET
