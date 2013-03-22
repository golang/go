// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SHA1 block routine. See sha1block.go for Go equivalent.
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

#define LOAD(index) \
	MOVL	(index*4)(SI), R10; \
	BSWAPL	R10; \
	MOVL	R10, (index*4)(SP)

#define SHUFFLE(index) \
	MOVL	(((index)&0xf)*4)(SP), R10; \
	XORL	(((index-3)&0xf)*4)(SP), R10; \
	XORL	(((index-8)&0xf)*4)(SP), R10; \
	XORL	(((index-14)&0xf)*4)(SP), R10; \
	ROLL	$1, R10; \
	MOVL	R10, (((index)&0xf)*4)(SP)

#define FUNC1(a, b, c, d, e) \
	MOVL	b, R8; \
	ANDL	c, R8; \
	MOVL	b, R9; \
	NOTL	R9; \
	ANDL	d, R9; \
	ORL	R8, R9

#define FUNC2(a, b, c, d, e) \
	MOVL	b, R9; \
	XORL	c, R9; \
	XORL	d, R9

#define FUNC3(a, b, c, d, e) \
	MOVL	b, R8; \
	ORL	c, R8; \
	ANDL	d, R8; \
	MOVL	b, R9; \
	ANDL	c, R9; \
	ORL	R8, R9
	
#define FUNC4 FUNC2

#define MIX(a, b, c, d, e, const) \
	ROLL	$30, b; \
	ADDL	R9, e; \
	MOVL	a, R8; \
	ROLL	$5, R8; \
	LEAL	const(e)(R10*1), e; \
	ADDL	R8, e

#define ROUND1(a, b, c, d, e, index) \
	LOAD(index); \
	FUNC1(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x5A827999)

#define ROUND1x(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC1(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x5A827999)

#define ROUND2(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC2(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x6ED9EBA1)

#define ROUND3(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC3(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0x8F1BBCDC)

#define ROUND4(a, b, c, d, e, index) \
	SHUFFLE(index); \
	FUNC4(a, b, c, d, e); \
	MIX(a, b, c, d, e, 0xCA62C1D6)

TEXT ·block(SB),7,$64-32
	MOVQ	dig+0(FP),	BP
	MOVQ	p_base+8(FP),	SI
	MOVQ	p_len+16(FP),	DX
	SHRQ	$6,		DX
	SHLQ	$6,		DX
	
	LEAQ	(SI)(DX*1),	DI
	MOVL	(0*4)(BP),	AX
	MOVL	(1*4)(BP),	BX
	MOVL	(2*4)(BP),	CX
	MOVL	(3*4)(BP),	DX
	MOVL	(4*4)(BP),	BP

	CMPQ	SI,		DI
	JEQ	end

loop:
	MOVL	AX,	R11
	MOVL	BX,	R12
	MOVL	CX,	R13
	MOVL	DX,	R14
	MOVL	BP,	R15

	ROUND1(AX, BX, CX, DX, BP, 0)
	ROUND1(BP, AX, BX, CX, DX, 1)
	ROUND1(DX, BP, AX, BX, CX, 2)
	ROUND1(CX, DX, BP, AX, BX, 3)
	ROUND1(BX, CX, DX, BP, AX, 4)
	ROUND1(AX, BX, CX, DX, BP, 5)
	ROUND1(BP, AX, BX, CX, DX, 6)
	ROUND1(DX, BP, AX, BX, CX, 7)
	ROUND1(CX, DX, BP, AX, BX, 8)
	ROUND1(BX, CX, DX, BP, AX, 9)
	ROUND1(AX, BX, CX, DX, BP, 10)
	ROUND1(BP, AX, BX, CX, DX, 11)
	ROUND1(DX, BP, AX, BX, CX, 12)
	ROUND1(CX, DX, BP, AX, BX, 13)
	ROUND1(BX, CX, DX, BP, AX, 14)
	ROUND1(AX, BX, CX, DX, BP, 15)

	ROUND1x(BP, AX, BX, CX, DX, 16)
	ROUND1x(DX, BP, AX, BX, CX, 17)
	ROUND1x(CX, DX, BP, AX, BX, 18)
	ROUND1x(BX, CX, DX, BP, AX, 19)
	
	ROUND2(AX, BX, CX, DX, BP, 20)
	ROUND2(BP, AX, BX, CX, DX, 21)
	ROUND2(DX, BP, AX, BX, CX, 22)
	ROUND2(CX, DX, BP, AX, BX, 23)
	ROUND2(BX, CX, DX, BP, AX, 24)
	ROUND2(AX, BX, CX, DX, BP, 25)
	ROUND2(BP, AX, BX, CX, DX, 26)
	ROUND2(DX, BP, AX, BX, CX, 27)
	ROUND2(CX, DX, BP, AX, BX, 28)
	ROUND2(BX, CX, DX, BP, AX, 29)
	ROUND2(AX, BX, CX, DX, BP, 30)
	ROUND2(BP, AX, BX, CX, DX, 31)
	ROUND2(DX, BP, AX, BX, CX, 32)
	ROUND2(CX, DX, BP, AX, BX, 33)
	ROUND2(BX, CX, DX, BP, AX, 34)
	ROUND2(AX, BX, CX, DX, BP, 35)
	ROUND2(BP, AX, BX, CX, DX, 36)
	ROUND2(DX, BP, AX, BX, CX, 37)
	ROUND2(CX, DX, BP, AX, BX, 38)
	ROUND2(BX, CX, DX, BP, AX, 39)
	
	ROUND3(AX, BX, CX, DX, BP, 40)
	ROUND3(BP, AX, BX, CX, DX, 41)
	ROUND3(DX, BP, AX, BX, CX, 42)
	ROUND3(CX, DX, BP, AX, BX, 43)
	ROUND3(BX, CX, DX, BP, AX, 44)
	ROUND3(AX, BX, CX, DX, BP, 45)
	ROUND3(BP, AX, BX, CX, DX, 46)
	ROUND3(DX, BP, AX, BX, CX, 47)
	ROUND3(CX, DX, BP, AX, BX, 48)
	ROUND3(BX, CX, DX, BP, AX, 49)
	ROUND3(AX, BX, CX, DX, BP, 50)
	ROUND3(BP, AX, BX, CX, DX, 51)
	ROUND3(DX, BP, AX, BX, CX, 52)
	ROUND3(CX, DX, BP, AX, BX, 53)
	ROUND3(BX, CX, DX, BP, AX, 54)
	ROUND3(AX, BX, CX, DX, BP, 55)
	ROUND3(BP, AX, BX, CX, DX, 56)
	ROUND3(DX, BP, AX, BX, CX, 57)
	ROUND3(CX, DX, BP, AX, BX, 58)
	ROUND3(BX, CX, DX, BP, AX, 59)
	
	ROUND4(AX, BX, CX, DX, BP, 60)
	ROUND4(BP, AX, BX, CX, DX, 61)
	ROUND4(DX, BP, AX, BX, CX, 62)
	ROUND4(CX, DX, BP, AX, BX, 63)
	ROUND4(BX, CX, DX, BP, AX, 64)
	ROUND4(AX, BX, CX, DX, BP, 65)
	ROUND4(BP, AX, BX, CX, DX, 66)
	ROUND4(DX, BP, AX, BX, CX, 67)
	ROUND4(CX, DX, BP, AX, BX, 68)
	ROUND4(BX, CX, DX, BP, AX, 69)
	ROUND4(AX, BX, CX, DX, BP, 70)
	ROUND4(BP, AX, BX, CX, DX, 71)
	ROUND4(DX, BP, AX, BX, CX, 72)
	ROUND4(CX, DX, BP, AX, BX, 73)
	ROUND4(BX, CX, DX, BP, AX, 74)
	ROUND4(AX, BX, CX, DX, BP, 75)
	ROUND4(BP, AX, BX, CX, DX, 76)
	ROUND4(DX, BP, AX, BX, CX, 77)
	ROUND4(CX, DX, BP, AX, BX, 78)
	ROUND4(BX, CX, DX, BP, AX, 79)

	ADDL	R11, AX
	ADDL	R12, BX
	ADDL	R13, CX
	ADDL	R14, DX
	ADDL	R15, BP

	ADDQ	$64, SI
	CMPQ	SI, DI
	JB	loop

end:
	MOVQ	dig+0(FP), DI
	MOVL	AX, (0*4)(DI)
	MOVL	BX, (1*4)(DI)
	MOVL	CX, (2*4)(DI)
	MOVL	DX, (3*4)(DI)
	MOVL	BP, (4*4)(DI)
	RET
