// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// RISCV64 version of md5block.go
// derived from crypto/md5/md5block_arm64.s and crypto/md5/md5block.go

//go:build !purego

#include "textflag.h"

#define LOAD32U(base, offset, tmp, dest) \
	MOVBU	(offset+0*1)(base), dest; \
	MOVBU	(offset+1*1)(base), tmp; \
	SLL	$8, tmp; \
	OR	tmp, dest; \
	MOVBU	(offset+2*1)(base), tmp; \
	SLL	$16, tmp; \
	OR	tmp, dest; \
	MOVBU	(offset+3*1)(base), tmp; \
	SLL	$24, tmp; \
	OR	tmp, dest

#define LOAD64U(base, offset, tmp1, tmp2, dst) \
	LOAD32U(base, offset, tmp1, dst); \
	LOAD32U(base, offset+4, tmp1, tmp2); \
	SLL	$32, tmp2; \
	OR	tmp2, dst

#define ROUND1EVN(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	x, a; \
	ADDW	X23, a; \
	XOR	c, d, X23; \
	AND	b, X23; \
	XOR	d, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND1ODD(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	X23, a; \
	SRL	$32, x, X23; \
	ADDW	X23, a; \
	XOR	c, d, X23; \
	AND	b, X23; \
	XOR	d, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND2EVN(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	x, a; \
	ADDW	X23, a; \
	XOR	b, c, X23; \
	AND	d, X23; \
	XOR	c, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND2ODD(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	X23, a; \
	SRL	$32, x, X23; \
	ADDW	X23, a; \
	XOR	b, c, X23; \
	AND	d, X23; \
	XOR	c, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND3EVN(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	x, a; \
	ADDW	X23, a; \
	XOR	c, d, X23; \
	XOR	b, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND3ODD(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	X23, a; \
	SRL	$32, x, X23; \
	ADDW	X23, a; \
	XOR	c, d, X23; \
	XOR	b, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND4EVN(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	x, a; \
	ADDW	X23, a; \
	ORN	d, b, X23; \
	XOR	c, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

#define ROUND4ODD(a, b, c, d, x, const, shift) \
	MOV	$const, X23; \
	ADDW	X23, a; \
	SRL	$32, x, X23; \
	ADDW	X23, a; \
	ORN	d, b, X23; \
	XOR	c, X23; \
	ADDW	X23, a; \
	RORIW	$(32-shift), a; \
	ADDW	b, a

// Register use for the block function
//
// X5 - X12	: contain the 16 32 bit data items in the block we're
//		  processing.  Odd numbered values, e.g., x1, x3 are stored in
//		  the upper 32 bits of the register.
// X13 - X16	: a, b, c, d
// X17 - X20	: used to store the old values of a, b, c, d, i.e., aa, bb, cc,
//		  dd.  X17 and X18 are also used as temporary registers when
//		  loading unaligned data.
// X22		: pointer to dig.s
// X23		: temporary register
// X28		: pointer to the first byte beyond the end of p
// X29		: pointer to current 64 byte block of data, initially set to
//		  &p[0]
// X30		: temporary register

TEXT	·block(SB),NOSPLIT,$0-32
	MOV	p+8(FP), X29
	MOV	p_len+16(FP), X30
	SRL	$6, X30
	SLL	$6, X30
	BEQZ	X30, zero

	ADD	X29, X30, X28

	MOV	dig+0(FP), X22
	MOVWU	(0*4)(X22), X13	// a = s[0]
	MOVWU	(1*4)(X22), X14	// b = s[1]
	MOVWU	(2*4)(X22), X15	// c = s[2]
	MOVWU	(3*4)(X22), X16	// d = s[3]

loop:

	// Load the 64 bytes of data in x0-15 into 8 64 bit registers, X5-X12.
	// Different paths are taken to load the values depending on whether the
	// buffer is 8 byte aligned or not.  We load all the values up front
	// here at the start of the loop to avoid multiple alignment checks and
	// to reduce code size.  It takes 10 instructions to load an unaligned
	// 32 bit value and this value will be used 4 times in the main body
	// of the loop below.

	AND	$7, X29, X30
	BEQZ	X30, aligned

	LOAD64U(X29,0, X17, X18, X5)
	LOAD64U(X29,8, X17, X18, X6)
	LOAD64U(X29,16, X17, X18, X7)
	LOAD64U(X29,24, X17, X18, X8)
	LOAD64U(X29,32, X17, X18, X9)
	LOAD64U(X29,40, X17, X18, X10)
	LOAD64U(X29,48, X17, X18, X11)
	LOAD64U(X29,56, X17, X18, X12)
	JMP block_loaded

aligned:
	MOV	(0*8)(X29), X5
	MOV	(1*8)(X29), X6
	MOV	(2*8)(X29), X7
	MOV	(3*8)(X29), X8
	MOV	(4*8)(X29), X9
	MOV	(5*8)(X29), X10
	MOV	(6*8)(X29), X11
	MOV	(7*8)(X29), X12

block_loaded:
	MOV	X13, X17
	MOV	X14, X18
	MOV	X15, X19
	MOV	X16, X20

	// Some of the hex constants below are too large to fit into a
	// signed 32 bit value.  The assembler will handle these
	// constants in a special way to ensure that they are
	// zero extended.  Our algorithm is only interested in the
	// bottom 32 bits and doesn't care whether constants are
	// sign or zero extended when moved into 64 bit registers.
	// So we use signed constants instead of hex when bit 31 is
	// set so all constants can be loaded by lui+addi.

	ROUND1EVN(X13,X14,X15,X16,X5,  -680876936, 7); // 0xd76aa478
	ROUND1ODD(X16,X13,X14,X15,X5,  -389564586,12); // 0xe8c7b756
	ROUND1EVN(X15,X16,X13,X14,X6,  0x242070db,17); // 0x242070db
	ROUND1ODD(X14,X15,X16,X13,X6, -1044525330,22); // 0xc1bdceee
	ROUND1EVN(X13,X14,X15,X16,X7,  -176418897, 7); // 0xf57c0faf
	ROUND1ODD(X16,X13,X14,X15,X7,  0x4787c62a,12); // 0x4787c62a
	ROUND1EVN(X15,X16,X13,X14,X8, -1473231341,17); // 0xa8304613
	ROUND1ODD(X14,X15,X16,X13,X8,   -45705983,22); // 0xfd469501
	ROUND1EVN(X13,X14,X15,X16,X9,  0x698098d8, 7); // 0x698098d8
	ROUND1ODD(X16,X13,X14,X15,X9, -1958414417,12); // 0x8b44f7af
	ROUND1EVN(X15,X16,X13,X14,X10,     -42063,17); // 0xffff5bb1
	ROUND1ODD(X14,X15,X16,X13,X10,-1990404162,22); // 0x895cd7be
	ROUND1EVN(X13,X14,X15,X16,X11, 0x6b901122, 7); // 0x6b901122
	ROUND1ODD(X16,X13,X14,X15,X11,  -40341101,12); // 0xfd987193
	ROUND1EVN(X15,X16,X13,X14,X12,-1502002290,17); // 0xa679438e
	ROUND1ODD(X14,X15,X16,X13,X12, 0x49b40821,22); // 0x49b40821

	ROUND2ODD(X13,X14,X15,X16,X5,  -165796510, 5); // f61e2562
	ROUND2EVN(X16,X13,X14,X15,X8, -1069501632, 9); // c040b340
	ROUND2ODD(X15,X16,X13,X14,X10, 0x265e5a51,14); // 265e5a51
	ROUND2EVN(X14,X15,X16,X13,X5,  -373897302,20); // e9b6c7aa
	ROUND2ODD(X13,X14,X15,X16,X7,  -701558691, 5); // d62f105d
	ROUND2EVN(X16,X13,X14,X15,X10,  0x2441453, 9); // 2441453
	ROUND2ODD(X15,X16,X13,X14,X12, -660478335,14); // d8a1e681
	ROUND2EVN(X14,X15,X16,X13,X7,  -405537848,20); // e7d3fbc8
	ROUND2ODD(X13,X14,X15,X16,X9,  0x21e1cde6, 5); // 21e1cde6
	ROUND2EVN(X16,X13,X14,X15,X12,-1019803690, 9); // c33707d6
	ROUND2ODD(X15,X16,X13,X14,X6,  -187363961,14); // f4d50d87
	ROUND2EVN(X14,X15,X16,X13,X9,  0x455a14ed,20); // 455a14ed
	ROUND2ODD(X13,X14,X15,X16,X11,-1444681467, 5); // a9e3e905
	ROUND2EVN(X16,X13,X14,X15,X6,   -51403784, 9); // fcefa3f8
	ROUND2ODD(X15,X16,X13,X14,X8,  0x676f02d9,14); // 676f02d9
	ROUND2EVN(X14,X15,X16,X13,X11,-1926607734,20); // 8d2a4c8a

	ROUND3ODD(X13,X14,X15,X16,X7,     -378558, 4); // fffa3942
	ROUND3EVN(X16,X13,X14,X15,X9, -2022574463,11); // 8771f681
	ROUND3ODD(X15,X16,X13,X14,X10, 0x6d9d6122,16); // 6d9d6122
	ROUND3EVN(X14,X15,X16,X13,X12,  -35309556,23); // fde5380c
	ROUND3ODD(X13,X14,X15,X16,X5, -1530992060, 4); // a4beea44
	ROUND3EVN(X16,X13,X14,X15,X7,  0x4bdecfa9,11); // 4bdecfa9
	ROUND3ODD(X15,X16,X13,X14,X8,  -155497632,16); // f6bb4b60
	ROUND3EVN(X14,X15,X16,X13,X10,-1094730640,23); // bebfbc70
	ROUND3ODD(X13,X14,X15,X16,X11, 0x289b7ec6, 4); // 289b7ec6
	ROUND3EVN(X16,X13,X14,X15,X5,  -358537222,11); // eaa127fa
	ROUND3ODD(X15,X16,X13,X14,X6,  -722521979,16); // d4ef3085
	ROUND3EVN(X14,X15,X16,X13,X8,   0x4881d05,23); // 4881d05
	ROUND3ODD(X13,X14,X15,X16,X9,  -640364487, 4); // d9d4d039
	ROUND3EVN(X16,X13,X14,X15,X11, -421815835,11); // e6db99e5
	ROUND3ODD(X15,X16,X13,X14,X12, 0x1fa27cf8,16); // 1fa27cf8
	ROUND3EVN(X14,X15,X16,X13,X6,  -995338651,23); // c4ac5665

	ROUND4EVN(X13,X14,X15,X16,X5,  -198630844, 6); // f4292244
	ROUND4ODD(X16,X13,X14,X15,X8,  0x432aff97,10); // 432aff97
	ROUND4EVN(X15,X16,X13,X14,X12,-1416354905,15); // ab9423a7
	ROUND4ODD(X14,X15,X16,X13,X7,   -57434055,21); // fc93a039
	ROUND4EVN(X13,X14,X15,X16,X11, 0x655b59c3, 6); // 655b59c3
	ROUND4ODD(X16,X13,X14,X15,X6, -1894986606,10); // 8f0ccc92
	ROUND4EVN(X15,X16,X13,X14,X10   ,-1051523,15); // ffeff47d
	ROUND4ODD(X14,X15,X16,X13,X5, -2054922799,21); // 85845dd1
	ROUND4EVN(X13,X14,X15,X16,X9,  0x6fa87e4f, 6); // 6fa87e4f
	ROUND4ODD(X16,X13,X14,X15,X12,  -30611744,10); // fe2ce6e0
	ROUND4EVN(X15,X16,X13,X14,X8, -1560198380,15); // a3014314
	ROUND4ODD(X14,X15,X16,X13,X11, 0x4e0811a1,21); // 4e0811a1
	ROUND4EVN(X13,X14,X15,X16,X7,  -145523070, 6); // f7537e82
	ROUND4ODD(X16,X13,X14,X15,X10,-1120210379,10); // bd3af235
	ROUND4EVN(X15,X16,X13,X14,X6,  0x2ad7d2bb,15); // 2ad7d2bb
	ROUND4ODD(X14,X15,X16,X13,X9,  -343485551,21); // eb86d391

	ADDW	X17, X13
	ADDW	X18, X14
	ADDW	X19, X15
	ADDW	X20, X16

	ADD	$64, X29
	BNE	X28, X29, loop

	MOVW	X13, (0*4)(X22)
	MOVW	X14, (1*4)(X22)
	MOVW	X15, (2*4)(X22)
	MOVW	X16, (3*4)(X22)

zero:
	RET
