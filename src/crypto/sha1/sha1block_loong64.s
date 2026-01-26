// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// SHA-1 block routine. See sha1block.go for Go equivalent.
//
// There are 80 rounds of 4 types:
//   - rounds 0-15 are type 1 and load data (ROUND1 macro).
//   - rounds 16-19 are type 1 and do not load data (ROUND1x macro).
//   - rounds 20-39 are type 2 and do not load data (ROUND2 macro).
//   - rounds 40-59 are type 3 and do not load data (ROUND3 macro).
//   - rounds 60-79 are type 4 and do not load data (ROUND4 macro).
//
// Each round loads or shuffles the data, then computes a per-round
// function of b, c, d, and then mixes the result into and rotates the
// five registers a, b, c, d, e holding the intermediate results.
//
// The register rotation is implemented by rotating the arguments to
// the round macros instead of by explicit move instructions.

#define REGTMP	R30
#define REGTMP1	R17
#define REGTMP2	R18
#define REGTMP3	R19

#define LOAD1(index) \
	MOVW	(index*4)(R5), REGTMP3; \
	REVB2W	REGTMP3, REGTMP3; \
	MOVW	REGTMP3, (index*4)(R3)

#define LOAD(index) \
	MOVW	(((index)&0xf)*4)(R3), REGTMP3; \
	MOVW	(((index-3)&0xf)*4)(R3), REGTMP; \
	MOVW	(((index-8)&0xf)*4)(R3), REGTMP1; \
	MOVW	(((index-14)&0xf)*4)(R3), REGTMP2; \
	XOR	REGTMP, REGTMP3; \
	XOR	REGTMP1, REGTMP3; \
	XOR	REGTMP2, REGTMP3; \
	ROTR	$31, REGTMP3; \
	MOVW	REGTMP3, (((index)&0xf)*4)(R3)

// f = d ^ (b & (c ^ d))
#define FUNC1(a, b, c, d, e) \
	XOR	c, d, REGTMP1; \
	AND	b, REGTMP1; \
	XOR	d, REGTMP1

// f = b ^ c ^ d
#define FUNC2(a, b, c, d, e) \
	XOR	b, c, REGTMP1; \
	XOR	d, REGTMP1

// f = (b & c) | ((b | c) & d)
#define FUNC3(a, b, c, d, e) \
	OR	b, c, REGTMP2; \
	AND	b, c, REGTMP; \
	AND	d, REGTMP2; \
	OR	REGTMP, REGTMP2, REGTMP1

#define FUNC4 FUNC2

#define MIX(a, b, c, d, e, const) \
	ROTR	$2, b; \	// b << 30
	ADD	REGTMP1, e; \	// e = e + f
	ROTR	$27, a, REGTMP2; \	// a << 5
	ADD	REGTMP3, e; \	// e = e + w[i]
	ADDV	$const, e; \	// e = e + k
	ADD	REGTMP2, e	// e = e + a<<5

#define ROUND1(a, b, c, d, e, index) \
	LOAD1(index); \
	FUNC1(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x5A827999)

#define ROUND1x(a, b, c, d, e, index) \
	LOAD(index); \
	FUNC1(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x5A827999)

#define ROUND2(a, b, c, d, e, index) \
	LOAD(index); \
	FUNC2(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x6ED9EBA1)

#define ROUND3(a, b, c, d, e, index) \
	LOAD(index); \
	FUNC3(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x8F1BBCDC)

#define ROUND4(a, b, c, d, e, index) \
	LOAD(index); \
	FUNC4(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0xCA62C1D6)

// A stack frame size of 64 bytes is required here, because
// the frame size used for data expansion is 64 bytes.
// See the definition of the macro LOAD above, and the definition
// of the local variable w in the general implementation (sha1block.go).
TEXT ·block(SB),NOSPLIT,$64-32
	MOVV	dig+0(FP),	R4
	MOVV	p_base+8(FP),	R5
	MOVV	p_len+16(FP),	R6
	AND	$~63, R6
	BEQ	R6, zero

	// p_len >= 64
	ADDV    R5, R6, R24
	MOVW	(0*4)(R4), R7
	MOVW	(1*4)(R4), R8
	MOVW	(2*4)(R4), R9
	MOVW	(3*4)(R4), R10
	MOVW	(4*4)(R4), R11

loop:
	MOVW	R7,	R12
	MOVW	R8,	R13
	MOVW	R9,	R14
	MOVW	R10,	R15
	MOVW	R11,	R16

	ROUND1(R7,  R8,  R9,  R10, R11, 0)
	ROUND1(R11, R7,  R8,  R9,  R10, 1)
	ROUND1(R10, R11, R7,  R8,  R9,  2)
	ROUND1(R9,  R10, R11, R7,  R8,  3)
	ROUND1(R8,  R9,  R10, R11, R7,  4)
	ROUND1(R7,  R8,  R9,  R10, R11, 5)
	ROUND1(R11, R7,  R8,  R9,  R10, 6)
	ROUND1(R10, R11, R7,  R8,  R9,  7)
	ROUND1(R9,  R10, R11, R7,  R8,  8)
	ROUND1(R8,  R9,  R10, R11, R7,  9)
	ROUND1(R7,  R8,  R9,  R10, R11, 10)
	ROUND1(R11, R7,  R8,  R9,  R10, 11)
	ROUND1(R10, R11, R7,  R8,  R9,  12)
	ROUND1(R9,  R10, R11, R7,  R8,  13)
	ROUND1(R8,  R9,  R10, R11, R7,  14)
	ROUND1(R7,  R8,  R9,  R10, R11, 15)

	ROUND1x(R11, R7,  R8,  R9,  R10, 16)
	ROUND1x(R10, R11, R7,  R8,  R9,  17)
	ROUND1x(R9,  R10, R11, R7,  R8,  18)
	ROUND1x(R8,  R9,  R10, R11, R7,  19)

	ROUND2(R7,  R8,  R9,  R10, R11, 20)
	ROUND2(R11, R7,  R8,  R9,  R10, 21)
	ROUND2(R10, R11, R7,  R8,  R9,  22)
	ROUND2(R9,  R10, R11, R7,  R8,  23)
	ROUND2(R8,  R9,  R10, R11, R7,  24)
	ROUND2(R7,  R8,  R9,  R10, R11, 25)
	ROUND2(R11, R7,  R8,  R9,  R10, 26)
	ROUND2(R10, R11, R7,  R8,  R9,  27)
	ROUND2(R9,  R10, R11, R7,  R8,  28)
	ROUND2(R8,  R9,  R10, R11, R7,  29)
	ROUND2(R7,  R8,  R9,  R10, R11, 30)
	ROUND2(R11, R7,  R8,  R9,  R10, 31)
	ROUND2(R10, R11, R7,  R8,  R9,  32)
	ROUND2(R9,  R10, R11, R7,  R8,  33)
	ROUND2(R8,  R9,  R10, R11, R7,  34)
	ROUND2(R7,  R8,  R9,  R10, R11, 35)
	ROUND2(R11, R7,  R8,  R9,  R10, 36)
	ROUND2(R10, R11, R7,  R8,  R9,  37)
	ROUND2(R9,  R10, R11, R7,  R8,  38)
	ROUND2(R8,  R9,  R10, R11, R7,  39)

	ROUND3(R7,  R8,  R9,  R10, R11, 40)
	ROUND3(R11, R7,  R8,  R9,  R10, 41)
	ROUND3(R10, R11, R7,  R8,  R9,  42)
	ROUND3(R9,  R10, R11, R7,  R8,  43)
	ROUND3(R8,  R9,  R10, R11, R7,  44)
	ROUND3(R7,  R8,  R9,  R10, R11, 45)
	ROUND3(R11, R7,  R8,  R9,  R10, 46)
	ROUND3(R10, R11, R7,  R8,  R9,  47)
	ROUND3(R9,  R10, R11, R7,  R8,  48)
	ROUND3(R8,  R9,  R10, R11, R7,  49)
	ROUND3(R7,  R8,  R9,  R10, R11, 50)
	ROUND3(R11, R7,  R8,  R9,  R10, 51)
	ROUND3(R10, R11, R7,  R8,  R9,  52)
	ROUND3(R9,  R10, R11, R7,  R8,  53)
	ROUND3(R8,  R9,  R10, R11, R7,  54)
	ROUND3(R7,  R8,  R9,  R10, R11, 55)
	ROUND3(R11, R7,  R8,  R9,  R10, 56)
	ROUND3(R10, R11, R7,  R8,  R9,  57)
	ROUND3(R9,  R10, R11, R7,  R8,  58)
	ROUND3(R8,  R9,  R10, R11, R7,  59)

	ROUND4(R7,  R8,  R9,  R10, R11, 60)
	ROUND4(R11, R7,  R8,  R9,  R10, 61)
	ROUND4(R10, R11, R7,  R8,  R9,  62)
	ROUND4(R9,  R10, R11, R7,  R8,  63)
	ROUND4(R8,  R9,  R10, R11, R7,  64)
	ROUND4(R7,  R8,  R9,  R10, R11, 65)
	ROUND4(R11, R7,  R8,  R9,  R10, 66)
	ROUND4(R10, R11, R7,  R8,  R9,  67)
	ROUND4(R9,  R10, R11, R7,  R8,  68)
	ROUND4(R8,  R9,  R10, R11, R7,  69)
	ROUND4(R7,  R8,  R9,  R10, R11, 70)
	ROUND4(R11, R7,  R8,  R9,  R10, 71)
	ROUND4(R10, R11, R7,  R8,  R9,  72)
	ROUND4(R9,  R10, R11, R7,  R8,  73)
	ROUND4(R8,  R9,  R10, R11, R7,  74)
	ROUND4(R7,  R8,  R9,  R10, R11, 75)
	ROUND4(R11, R7,  R8,  R9,  R10, 76)
	ROUND4(R10, R11, R7,  R8,  R9,  77)
	ROUND4(R9,  R10, R11, R7,  R8,  78)
	ROUND4(R8,  R9,  R10, R11, R7,  79)

	ADD	R12, R7
	ADD	R13, R8
	ADD	R14, R9
	ADD	R15, R10
	ADD	R16, R11

	ADDV	$64, R5
	BNE	R5, R24, loop

end:
	MOVW	R7, (0*4)(R4)
	MOVW	R8, (1*4)(R4)
	MOVW	R9, (2*4)(R4)
	MOVW	R10, (3*4)(R4)
	MOVW	R11, (4*4)(R4)
zero:
	RET
