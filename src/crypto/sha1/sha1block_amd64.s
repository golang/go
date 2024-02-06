// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// AVX2 version by Intel, same algorithm as code in Linux kernel:
// https://github.com/torvalds/linux/blob/master/arch/x86/crypto/sha1_avx2_x86_64_asm.S
// Authors:
// Ilya Albrekht <ilya.albrekht@intel.com>
// Maxim Locktyukhin <maxim.locktyukhin@intel.com>
// Ronen Zohar <ronen.zohar@intel.com>
// Chandramouli Narayanan <mouli@linux.intel.com>

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
	MOVL	d, R9; \
	XORL	c, R9; \
	ANDL	b, R9; \
	XORL	d, R9

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

TEXT ·blockAMD64(SB),NOSPLIT,$64-32
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


// This is the implementation using AVX2, BMI1 and BMI2. It is based on:
// "SHA-1 implementation with Intel(R) AVX2 instruction set extensions"
// From http://software.intel.com/en-us/articles
// (look for improving-the-performance-of-the-secure-hash-algorithm-1)
// This implementation is 2x unrolled, and interleaves vector instructions,
// used to precompute W, with scalar computation of current round
// for optimal scheduling.

// Trivial helper macros.
#define UPDATE_HASH(A,TB,C,D,E) \
	ADDL	(R9), A \
	MOVL	A, (R9) \
	ADDL	4(R9), TB \
	MOVL	TB, 4(R9) \
	ADDL	8(R9), C \
	MOVL	C, 8(R9) \
	ADDL	12(R9), D \
	MOVL	D, 12(R9) \
	ADDL	16(R9), E \
	MOVL	E, 16(R9)



// Helper macros for PRECALC, which does precomputations
#define PRECALC_0(OFFSET) \
	VMOVDQU   OFFSET(R10),X0

#define PRECALC_1(OFFSET) \
	VINSERTI128 $1, OFFSET(R13), Y0, Y0

#define PRECALC_2(YREG) \
	VPSHUFB Y10, Y0, YREG

#define PRECALC_4(YREG,K_OFFSET) \
	VPADDD K_OFFSET(R8), YREG, Y0

#define PRECALC_7(OFFSET) \
	VMOVDQU Y0, (OFFSET*2)(R14)


// Message scheduling pre-compute for rounds 0-15
// R13 is a pointer to even 64-byte block
// R10 is a pointer to odd 64-byte block
// R14 is a pointer to temp buffer
// X0 is used as temp register
// YREG is clobbered as part of computation
// OFFSET chooses 16 byte chunk within a block
// R8 is a pointer to constants block
// K_OFFSET chooses K constants relevant to this round
// X10 holds swap mask
#define PRECALC_00_15(OFFSET,YREG) \
	PRECALC_0(OFFSET) \
	PRECALC_1(OFFSET) \
	PRECALC_2(YREG) \
	PRECALC_4(YREG,0x0) \
	PRECALC_7(OFFSET)


// Helper macros for PRECALC_16_31
#define PRECALC_16(REG_SUB_16,REG_SUB_12,REG_SUB_4,REG) \
	VPALIGNR $8, REG_SUB_16, REG_SUB_12, REG \  // w[i-14]
	VPSRLDQ $4, REG_SUB_4, Y0 // w[i-3]

#define PRECALC_17(REG_SUB_16,REG_SUB_8,REG) \
	VPXOR  REG_SUB_8, REG, REG \
	VPXOR  REG_SUB_16, Y0, Y0

#define PRECALC_18(REG) \
	VPXOR Y0, REG, REG \
	VPSLLDQ $12, REG, Y9

#define PRECALC_19(REG) \
	VPSLLD $1, REG, Y0 \
	VPSRLD $31, REG, REG

#define PRECALC_20(REG) \
	VPOR REG, Y0, Y0 \
	VPSLLD $2, Y9,  REG

#define PRECALC_21(REG) \
	VPSRLD $30, Y9, Y9 \
	VPXOR REG, Y0, Y0

#define PRECALC_23(REG,K_OFFSET,OFFSET) \
	VPXOR Y9, Y0, REG \
	VPADDD K_OFFSET(R8), REG, Y0 \
	VMOVDQU Y0, (OFFSET)(R14)

// Message scheduling pre-compute for rounds 16-31
// calculating last 32 w[i] values in 8 XMM registers
// pre-calculate K+w[i] values and store to mem
// for later load by ALU add instruction.
// "brute force" vectorization for rounds 16-31 only
// due to w[i]->w[i-3] dependency.
// clobbers 5 input ymm registers REG_SUB*
// uses X0 and X9 as temp registers
// As always, R8 is a pointer to constants block
// and R14 is a pointer to temp buffer
#define PRECALC_16_31(REG,REG_SUB_4,REG_SUB_8,REG_SUB_12,REG_SUB_16,K_OFFSET,OFFSET) \
	PRECALC_16(REG_SUB_16,REG_SUB_12,REG_SUB_4,REG) \
	PRECALC_17(REG_SUB_16,REG_SUB_8,REG) \
	PRECALC_18(REG) \
	PRECALC_19(REG) \
	PRECALC_20(REG) \
	PRECALC_21(REG) \
	PRECALC_23(REG,K_OFFSET,OFFSET)


// Helper macros for PRECALC_32_79
#define PRECALC_32(REG_SUB_8,REG_SUB_4) \
	VPALIGNR $8, REG_SUB_8, REG_SUB_4, Y0

#define PRECALC_33(REG_SUB_28,REG) \
	VPXOR REG_SUB_28, REG, REG

#define PRECALC_34(REG_SUB_16) \
	VPXOR REG_SUB_16, Y0, Y0

#define PRECALC_35(REG) \
	VPXOR Y0, REG, REG

#define PRECALC_36(REG) \
	VPSLLD $2, REG, Y0

#define PRECALC_37(REG) \
	VPSRLD $30, REG, REG \
	VPOR REG, Y0, REG

#define PRECALC_39(REG,K_OFFSET,OFFSET) \
	VPADDD K_OFFSET(R8), REG, Y0 \
	VMOVDQU Y0, (OFFSET)(R14)

// Message scheduling pre-compute for rounds 32-79
// In SHA-1 specification we have:
// w[i] = (w[i-3] ^ w[i-8]  ^ w[i-14] ^ w[i-16]) rol 1
// Which is the same as:
// w[i] = (w[i-6] ^ w[i-16] ^ w[i-28] ^ w[i-32]) rol 2
// This allows for more efficient vectorization,
// since w[i]->w[i-3] dependency is broken
#define PRECALC_32_79(REG,REG_SUB_4,REG_SUB_8,REG_SUB_16,REG_SUB_28,K_OFFSET,OFFSET) \
	PRECALC_32(REG_SUB_8,REG_SUB_4) \
	PRECALC_33(REG_SUB_28,REG) \
	PRECALC_34(REG_SUB_16) \
	PRECALC_35(REG) \
	PRECALC_36(REG) \
	PRECALC_37(REG) \
	PRECALC_39(REG,K_OFFSET,OFFSET)

#define PRECALC \
	PRECALC_00_15(0,Y15) \
	PRECALC_00_15(0x10,Y14) \
	PRECALC_00_15(0x20,Y13) \
	PRECALC_00_15(0x30,Y12) \
	PRECALC_16_31(Y8,Y12,Y13,Y14,Y15,0,0x80) \
	PRECALC_16_31(Y7,Y8,Y12,Y13,Y14,0x20,0xa0) \
	PRECALC_16_31(Y5,Y7,Y8,Y12,Y13,0x20,0xc0) \
	PRECALC_16_31(Y3,Y5,Y7,Y8,Y12,0x20,0xe0) \
	PRECALC_32_79(Y15,Y3,Y5,Y8,Y14,0x20,0x100) \
	PRECALC_32_79(Y14,Y15,Y3,Y7,Y13,0x20,0x120) \
	PRECALC_32_79(Y13,Y14,Y15,Y5,Y12,0x40,0x140) \
	PRECALC_32_79(Y12,Y13,Y14,Y3,Y8,0x40,0x160) \
	PRECALC_32_79(Y8,Y12,Y13,Y15,Y7,0x40,0x180) \
	PRECALC_32_79(Y7,Y8,Y12,Y14,Y5,0x40,0x1a0) \
	PRECALC_32_79(Y5,Y7,Y8,Y13,Y3,0x40,0x1c0) \
	PRECALC_32_79(Y3,Y5,Y7,Y12,Y15,0x60,0x1e0) \
	PRECALC_32_79(Y15,Y3,Y5,Y8,Y14,0x60,0x200) \
	PRECALC_32_79(Y14,Y15,Y3,Y7,Y13,0x60,0x220) \
	PRECALC_32_79(Y13,Y14,Y15,Y5,Y12,0x60,0x240) \
	PRECALC_32_79(Y12,Y13,Y14,Y3,Y8,0x60,0x260)

// Macros calculating individual rounds have general form
// CALC_ROUND_PRE + PRECALC_ROUND + CALC_ROUND_POST
// CALC_ROUND_{PRE,POST} macros follow

#define CALC_F1_PRE(OFFSET,REG_A,REG_B,REG_C,REG_E) \
	ADDL OFFSET(R15),REG_E \
	ANDNL REG_C,REG_A,BP \
	LEAL (REG_E)(REG_B*1), REG_E \ // Add F from the previous round
	RORXL $0x1b, REG_A, R12 \
	RORXL $2, REG_A, REG_B         // for next round

// Calculate F for the next round
#define CALC_F1_POST(REG_A,REG_B,REG_E) \
	ANDL REG_B,REG_A \             // b&c
	XORL BP, REG_A \               // F1 = (b&c) ^ (~b&d)
	LEAL (REG_E)(R12*1), REG_E     // E += A >>> 5


// Registers are cyclically rotated DX -> AX -> DI -> SI -> BX -> CX
#define CALC_0 \
	MOVL SI, BX \ // Precalculating first round
	RORXL $2, SI, SI \
	ANDNL AX, BX, BP \
	ANDL DI, BX \
	XORL BP, BX \
	CALC_F1_PRE(0x0,CX,BX,DI,DX) \
	PRECALC_0(0x80) \
	CALC_F1_POST(CX,SI,DX)

#define CALC_1 \
	CALC_F1_PRE(0x4,DX,CX,SI,AX) \
	PRECALC_1(0x80) \
	CALC_F1_POST(DX,BX,AX)

#define CALC_2 \
	CALC_F1_PRE(0x8,AX,DX,BX,DI) \
	PRECALC_2(Y15) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_3 \
	CALC_F1_PRE(0xc,DI,AX,CX,SI) \
	CALC_F1_POST(DI,DX,SI)

#define CALC_4 \
	CALC_F1_PRE(0x20,SI,DI,DX,BX) \
	PRECALC_4(Y15,0x0) \
	CALC_F1_POST(SI,AX,BX)

#define CALC_5 \
	CALC_F1_PRE(0x24,BX,SI,AX,CX) \
	CALC_F1_POST(BX,DI,CX)

#define CALC_6 \
	CALC_F1_PRE(0x28,CX,BX,DI,DX) \
	CALC_F1_POST(CX,SI,DX)

#define CALC_7 \
	CALC_F1_PRE(0x2c,DX,CX,SI,AX) \
	PRECALC_7(0x0) \
	CALC_F1_POST(DX,BX,AX)

#define CALC_8 \
	CALC_F1_PRE(0x40,AX,DX,BX,DI) \
	PRECALC_0(0x90) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_9 \
	CALC_F1_PRE(0x44,DI,AX,CX,SI) \
	PRECALC_1(0x90) \
	CALC_F1_POST(DI,DX,SI)

#define CALC_10 \
	CALC_F1_PRE(0x48,SI,DI,DX,BX) \
	PRECALC_2(Y14) \
	CALC_F1_POST(SI,AX,BX)

#define CALC_11 \
	CALC_F1_PRE(0x4c,BX,SI,AX,CX) \
	CALC_F1_POST(BX,DI,CX)

#define CALC_12 \
	CALC_F1_PRE(0x60,CX,BX,DI,DX) \
	PRECALC_4(Y14,0x0) \
	CALC_F1_POST(CX,SI,DX)

#define CALC_13 \
	CALC_F1_PRE(0x64,DX,CX,SI,AX) \
	CALC_F1_POST(DX,BX,AX)

#define CALC_14 \
	CALC_F1_PRE(0x68,AX,DX,BX,DI) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_15 \
	CALC_F1_PRE(0x6c,DI,AX,CX,SI) \
	PRECALC_7(0x10) \
	CALC_F1_POST(DI,DX,SI)

#define CALC_16 \
	CALC_F1_PRE(0x80,SI,DI,DX,BX) \
	PRECALC_0(0xa0) \
	CALC_F1_POST(SI,AX,BX)

#define CALC_17 \
	CALC_F1_PRE(0x84,BX,SI,AX,CX) \
	PRECALC_1(0xa0) \
	CALC_F1_POST(BX,DI,CX)

#define CALC_18 \
	CALC_F1_PRE(0x88,CX,BX,DI,DX) \
	PRECALC_2(Y13) \
	CALC_F1_POST(CX,SI,DX)


#define CALC_F2_PRE(OFFSET,REG_A,REG_B,REG_E) \
	ADDL OFFSET(R15),REG_E \
	LEAL (REG_E)(REG_B*1), REG_E \ // Add F from the previous round
	RORXL $0x1b, REG_A, R12 \
	RORXL $2, REG_A, REG_B         // for next round

#define CALC_F2_POST(REG_A,REG_B,REG_C,REG_E) \
	XORL REG_B, REG_A \
	ADDL R12, REG_E \
	XORL REG_C, REG_A

#define CALC_19 \
	CALC_F2_PRE(0x8c,DX,CX,AX) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_20 \
	CALC_F2_PRE(0xa0,AX,DX,DI) \
	PRECALC_4(Y13,0x0) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_21 \
	CALC_F2_PRE(0xa4,DI,AX,SI) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_22 \
	CALC_F2_PRE(0xa8,SI,DI,BX) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_23 \
	CALC_F2_PRE(0xac,BX,SI,CX) \
	PRECALC_7(0x20) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_24 \
	CALC_F2_PRE(0xc0,CX,BX,DX) \
	PRECALC_0(0xb0) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_25 \
	CALC_F2_PRE(0xc4,DX,CX,AX) \
	PRECALC_1(0xb0) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_26 \
	CALC_F2_PRE(0xc8,AX,DX,DI) \
	PRECALC_2(Y12) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_27 \
	CALC_F2_PRE(0xcc,DI,AX,SI) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_28 \
	CALC_F2_PRE(0xe0,SI,DI,BX) \
	PRECALC_4(Y12,0x0) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_29 \
	CALC_F2_PRE(0xe4,BX,SI,CX) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_30 \
	CALC_F2_PRE(0xe8,CX,BX,DX) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_31 \
	CALC_F2_PRE(0xec,DX,CX,AX) \
	PRECALC_7(0x30) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_32 \
	CALC_F2_PRE(0x100,AX,DX,DI) \
	PRECALC_16(Y15,Y14,Y12,Y8) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_33 \
	CALC_F2_PRE(0x104,DI,AX,SI) \
	PRECALC_17(Y15,Y13,Y8) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_34 \
	CALC_F2_PRE(0x108,SI,DI,BX) \
	PRECALC_18(Y8) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_35 \
	CALC_F2_PRE(0x10c,BX,SI,CX) \
	PRECALC_19(Y8) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_36 \
	CALC_F2_PRE(0x120,CX,BX,DX) \
	PRECALC_20(Y8) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_37 \
	CALC_F2_PRE(0x124,DX,CX,AX) \
	PRECALC_21(Y8) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_38 \
	CALC_F2_PRE(0x128,AX,DX,DI) \
	CALC_F2_POST(AX,CX,BX,DI)


#define CALC_F3_PRE(OFFSET,REG_E) \
	ADDL OFFSET(R15),REG_E

#define CALC_F3_POST(REG_A,REG_B,REG_C,REG_E,REG_TB) \
	LEAL (REG_E)(REG_TB*1), REG_E \ // Add F from the previous round
	MOVL REG_B, BP \
	ORL  REG_A, BP \
	RORXL $0x1b, REG_A, R12 \
	RORXL $2, REG_A, REG_TB \
	ANDL REG_C, BP \		// Calculate F for the next round
	ANDL REG_B, REG_A \
	ORL  BP, REG_A \
	ADDL R12, REG_E

#define CALC_39 \
	CALC_F3_PRE(0x12c,SI) \
	PRECALC_23(Y8,0x0,0x80) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_40 \
	CALC_F3_PRE(0x140,BX) \
	PRECALC_16(Y14,Y13,Y8,Y7) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_41 \
	CALC_F3_PRE(0x144,CX) \
	PRECALC_17(Y14,Y12,Y7) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_42 \
	CALC_F3_PRE(0x148,DX) \
	PRECALC_18(Y7) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_43 \
	CALC_F3_PRE(0x14c,AX) \
	PRECALC_19(Y7) \
	CALC_F3_POST(DX,BX,SI,AX,CX)

#define CALC_44 \
	CALC_F3_PRE(0x160,DI) \
	PRECALC_20(Y7) \
	CALC_F3_POST(AX,CX,BX,DI,DX)

#define CALC_45 \
	CALC_F3_PRE(0x164,SI) \
	PRECALC_21(Y7) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_46 \
	CALC_F3_PRE(0x168,BX) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_47 \
	CALC_F3_PRE(0x16c,CX) \
	VPXOR Y9, Y0, Y7 \
	VPADDD 0x20(R8), Y7, Y0 \
	VMOVDQU Y0, 0xa0(R14) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_48 \
	CALC_F3_PRE(0x180,DX) \
	PRECALC_16(Y13,Y12,Y7,Y5) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_49 \
	CALC_F3_PRE(0x184,AX) \
	PRECALC_17(Y13,Y8,Y5) \
	CALC_F3_POST(DX,BX,SI,AX,CX)

#define CALC_50 \
	CALC_F3_PRE(0x188,DI) \
	PRECALC_18(Y5) \
	CALC_F3_POST(AX,CX,BX,DI,DX)

#define CALC_51 \
	CALC_F3_PRE(0x18c,SI) \
	PRECALC_19(Y5) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_52 \
	CALC_F3_PRE(0x1a0,BX) \
	PRECALC_20(Y5) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_53 \
	CALC_F3_PRE(0x1a4,CX) \
	PRECALC_21(Y5) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_54 \
	CALC_F3_PRE(0x1a8,DX) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_55 \
	CALC_F3_PRE(0x1ac,AX) \
	PRECALC_23(Y5,0x20,0xc0) \
	CALC_F3_POST(DX,BX,SI,AX,CX)

#define CALC_56 \
	CALC_F3_PRE(0x1c0,DI) \
	PRECALC_16(Y12,Y8,Y5,Y3) \
	CALC_F3_POST(AX,CX,BX,DI,DX)

#define CALC_57 \
	CALC_F3_PRE(0x1c4,SI) \
	PRECALC_17(Y12,Y7,Y3) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_58 \
	CALC_F3_PRE(0x1c8,BX) \
	PRECALC_18(Y3) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_59 \
	CALC_F2_PRE(0x1cc,BX,SI,CX) \
	PRECALC_19(Y3) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_60 \
	CALC_F2_PRE(0x1e0,CX,BX,DX) \
	PRECALC_20(Y3) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_61 \
	CALC_F2_PRE(0x1e4,DX,CX,AX) \
	PRECALC_21(Y3) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_62 \
	CALC_F2_PRE(0x1e8,AX,DX,DI) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_63 \
	CALC_F2_PRE(0x1ec,DI,AX,SI) \
	PRECALC_23(Y3,0x20,0xe0) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_64 \
	CALC_F2_PRE(0x200,SI,DI,BX) \
	PRECALC_32(Y5,Y3) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_65 \
	CALC_F2_PRE(0x204,BX,SI,CX) \
	PRECALC_33(Y14,Y15) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_66 \
	CALC_F2_PRE(0x208,CX,BX,DX) \
	PRECALC_34(Y8) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_67 \
	CALC_F2_PRE(0x20c,DX,CX,AX) \
	PRECALC_35(Y15) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_68 \
	CALC_F2_PRE(0x220,AX,DX,DI) \
	PRECALC_36(Y15) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_69 \
	CALC_F2_PRE(0x224,DI,AX,SI) \
	PRECALC_37(Y15) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_70 \
	CALC_F2_PRE(0x228,SI,DI,BX) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_71 \
	CALC_F2_PRE(0x22c,BX,SI,CX) \
	PRECALC_39(Y15,0x20,0x100) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_72 \
	CALC_F2_PRE(0x240,CX,BX,DX) \
	PRECALC_32(Y3,Y15) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_73 \
	CALC_F2_PRE(0x244,DX,CX,AX) \
	PRECALC_33(Y13,Y14) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_74 \
	CALC_F2_PRE(0x248,AX,DX,DI) \
	PRECALC_34(Y7) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_75 \
	CALC_F2_PRE(0x24c,DI,AX,SI) \
	PRECALC_35(Y14) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_76 \
	CALC_F2_PRE(0x260,SI,DI,BX) \
	PRECALC_36(Y14) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_77 \
	CALC_F2_PRE(0x264,BX,SI,CX) \
	PRECALC_37(Y14) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_78 \
	CALC_F2_PRE(0x268,CX,BX,DX) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_79 \
	ADDL 0x26c(R15), AX \
	LEAL (AX)(CX*1), AX \
	RORXL $0x1b, DX, R12 \
	PRECALC_39(Y14,0x20,0x120) \
	ADDL R12, AX

// Similar to CALC_0
#define CALC_80 \
	MOVL CX, DX \
	RORXL $2, CX, CX \
	ANDNL SI, DX, BP \
	ANDL BX, DX \
	XORL BP, DX \
	CALC_F1_PRE(0x10,AX,DX,BX,DI) \
	PRECALC_32(Y15,Y14) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_81 \
	CALC_F1_PRE(0x14,DI,AX,CX,SI) \
	PRECALC_33(Y12,Y13) \
	CALC_F1_POST(DI,DX,SI)

#define CALC_82 \
	CALC_F1_PRE(0x18,SI,DI,DX,BX) \
	PRECALC_34(Y5) \
	CALC_F1_POST(SI,AX,BX)

#define CALC_83 \
	CALC_F1_PRE(0x1c,BX,SI,AX,CX) \
	PRECALC_35(Y13) \
	CALC_F1_POST(BX,DI,CX)

#define CALC_84 \
	CALC_F1_PRE(0x30,CX,BX,DI,DX) \
	PRECALC_36(Y13) \
	CALC_F1_POST(CX,SI,DX)

#define CALC_85 \
	CALC_F1_PRE(0x34,DX,CX,SI,AX) \
	PRECALC_37(Y13) \
	CALC_F1_POST(DX,BX,AX)

#define CALC_86 \
	CALC_F1_PRE(0x38,AX,DX,BX,DI) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_87 \
	CALC_F1_PRE(0x3c,DI,AX,CX,SI) \
	PRECALC_39(Y13,0x40,0x140) \
	CALC_F1_POST(DI,DX,SI)

#define CALC_88 \
	CALC_F1_PRE(0x50,SI,DI,DX,BX) \
	PRECALC_32(Y14,Y13) \
	CALC_F1_POST(SI,AX,BX)

#define CALC_89 \
	CALC_F1_PRE(0x54,BX,SI,AX,CX) \
	PRECALC_33(Y8,Y12) \
	CALC_F1_POST(BX,DI,CX)

#define CALC_90 \
	CALC_F1_PRE(0x58,CX,BX,DI,DX) \
	PRECALC_34(Y3) \
	CALC_F1_POST(CX,SI,DX)

#define CALC_91 \
	CALC_F1_PRE(0x5c,DX,CX,SI,AX) \
	PRECALC_35(Y12) \
	CALC_F1_POST(DX,BX,AX)

#define CALC_92 \
	CALC_F1_PRE(0x70,AX,DX,BX,DI) \
	PRECALC_36(Y12) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_93 \
	CALC_F1_PRE(0x74,DI,AX,CX,SI) \
	PRECALC_37(Y12) \
	CALC_F1_POST(DI,DX,SI)

#define CALC_94 \
	CALC_F1_PRE(0x78,SI,DI,DX,BX) \
	CALC_F1_POST(SI,AX,BX)

#define CALC_95 \
	CALC_F1_PRE(0x7c,BX,SI,AX,CX) \
	PRECALC_39(Y12,0x40,0x160) \
	CALC_F1_POST(BX,DI,CX)

#define CALC_96 \
	CALC_F1_PRE(0x90,CX,BX,DI,DX) \
	PRECALC_32(Y13,Y12) \
	CALC_F1_POST(CX,SI,DX)

#define CALC_97 \
	CALC_F1_PRE(0x94,DX,CX,SI,AX) \
	PRECALC_33(Y7,Y8) \
	CALC_F1_POST(DX,BX,AX)

#define CALC_98 \
	CALC_F1_PRE(0x98,AX,DX,BX,DI) \
	PRECALC_34(Y15) \
	CALC_F1_POST(AX,CX,DI)

#define CALC_99 \
	CALC_F2_PRE(0x9c,DI,AX,SI) \
	PRECALC_35(Y8) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_100 \
	CALC_F2_PRE(0xb0,SI,DI,BX) \
	PRECALC_36(Y8) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_101 \
	CALC_F2_PRE(0xb4,BX,SI,CX) \
	PRECALC_37(Y8) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_102 \
	CALC_F2_PRE(0xb8,CX,BX,DX) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_103 \
	CALC_F2_PRE(0xbc,DX,CX,AX) \
	PRECALC_39(Y8,0x40,0x180) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_104 \
	CALC_F2_PRE(0xd0,AX,DX,DI) \
	PRECALC_32(Y12,Y8) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_105 \
	CALC_F2_PRE(0xd4,DI,AX,SI) \
	PRECALC_33(Y5,Y7) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_106 \
	CALC_F2_PRE(0xd8,SI,DI,BX) \
	PRECALC_34(Y14) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_107 \
	CALC_F2_PRE(0xdc,BX,SI,CX) \
	PRECALC_35(Y7) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_108 \
	CALC_F2_PRE(0xf0,CX,BX,DX) \
	PRECALC_36(Y7) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_109 \
	CALC_F2_PRE(0xf4,DX,CX,AX) \
	PRECALC_37(Y7) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_110 \
	CALC_F2_PRE(0xf8,AX,DX,DI) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_111 \
	CALC_F2_PRE(0xfc,DI,AX,SI) \
	PRECALC_39(Y7,0x40,0x1a0) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_112 \
	CALC_F2_PRE(0x110,SI,DI,BX) \
	PRECALC_32(Y8,Y7) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_113 \
	CALC_F2_PRE(0x114,BX,SI,CX) \
	PRECALC_33(Y3,Y5) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_114 \
	CALC_F2_PRE(0x118,CX,BX,DX) \
	PRECALC_34(Y13) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_115 \
	CALC_F2_PRE(0x11c,DX,CX,AX) \
	PRECALC_35(Y5) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_116 \
	CALC_F2_PRE(0x130,AX,DX,DI) \
	PRECALC_36(Y5) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_117 \
	CALC_F2_PRE(0x134,DI,AX,SI) \
	PRECALC_37(Y5) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_118 \
	CALC_F2_PRE(0x138,SI,DI,BX) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_119 \
	CALC_F3_PRE(0x13c,CX) \
	PRECALC_39(Y5,0x40,0x1c0) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_120 \
	CALC_F3_PRE(0x150,DX) \
	PRECALC_32(Y7,Y5) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_121 \
	CALC_F3_PRE(0x154,AX) \
	PRECALC_33(Y15,Y3) \
	CALC_F3_POST(DX,BX,SI,AX,CX)

#define CALC_122 \
	CALC_F3_PRE(0x158,DI) \
	PRECALC_34(Y12) \
	CALC_F3_POST(AX,CX,BX,DI,DX)

#define CALC_123 \
	CALC_F3_PRE(0x15c,SI) \
	PRECALC_35(Y3) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_124 \
	CALC_F3_PRE(0x170,BX) \
	PRECALC_36(Y3) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_125 \
	CALC_F3_PRE(0x174,CX) \
	PRECALC_37(Y3) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_126 \
	CALC_F3_PRE(0x178,DX) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_127 \
	CALC_F3_PRE(0x17c,AX) \
	PRECALC_39(Y3,0x60,0x1e0) \
	CALC_F3_POST(DX,BX,SI,AX,CX)

#define CALC_128 \
	CALC_F3_PRE(0x190,DI) \
	PRECALC_32(Y5,Y3) \
	CALC_F3_POST(AX,CX,BX,DI,DX)

#define CALC_129 \
	CALC_F3_PRE(0x194,SI) \
	PRECALC_33(Y14,Y15) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_130 \
	CALC_F3_PRE(0x198,BX) \
	PRECALC_34(Y8) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_131 \
	CALC_F3_PRE(0x19c,CX) \
	PRECALC_35(Y15) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_132 \
	CALC_F3_PRE(0x1b0,DX) \
	PRECALC_36(Y15) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_133 \
	CALC_F3_PRE(0x1b4,AX) \
	PRECALC_37(Y15) \
	CALC_F3_POST(DX,BX,SI,AX,CX)

#define CALC_134 \
	CALC_F3_PRE(0x1b8,DI) \
	CALC_F3_POST(AX,CX,BX,DI,DX)

#define CALC_135 \
	CALC_F3_PRE(0x1bc,SI) \
	PRECALC_39(Y15,0x60,0x200) \
	CALC_F3_POST(DI,DX,CX,SI,AX)

#define CALC_136 \
	CALC_F3_PRE(0x1d0,BX) \
	PRECALC_32(Y3,Y15) \
	CALC_F3_POST(SI,AX,DX,BX,DI)

#define CALC_137 \
	CALC_F3_PRE(0x1d4,CX) \
	PRECALC_33(Y13,Y14) \
	CALC_F3_POST(BX,DI,AX,CX,SI)

#define CALC_138 \
	CALC_F3_PRE(0x1d8,DX) \
	PRECALC_34(Y7) \
	CALC_F3_POST(CX,SI,DI,DX,BX)

#define CALC_139 \
	CALC_F2_PRE(0x1dc,DX,CX,AX) \
	PRECALC_35(Y14) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_140 \
	CALC_F2_PRE(0x1f0,AX,DX,DI) \
	PRECALC_36(Y14) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_141 \
	CALC_F2_PRE(0x1f4,DI,AX,SI) \
	PRECALC_37(Y14) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_142 \
	CALC_F2_PRE(0x1f8,SI,DI,BX) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_143 \
	CALC_F2_PRE(0x1fc,BX,SI,CX) \
	PRECALC_39(Y14,0x60,0x220) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_144 \
	CALC_F2_PRE(0x210,CX,BX,DX) \
	PRECALC_32(Y15,Y14) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_145 \
	CALC_F2_PRE(0x214,DX,CX,AX) \
	PRECALC_33(Y12,Y13) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_146 \
	CALC_F2_PRE(0x218,AX,DX,DI) \
	PRECALC_34(Y5) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_147 \
	CALC_F2_PRE(0x21c,DI,AX,SI) \
	PRECALC_35(Y13) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_148 \
	CALC_F2_PRE(0x230,SI,DI,BX) \
	PRECALC_36(Y13) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_149 \
	CALC_F2_PRE(0x234,BX,SI,CX) \
	PRECALC_37(Y13) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_150 \
	CALC_F2_PRE(0x238,CX,BX,DX) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_151 \
	CALC_F2_PRE(0x23c,DX,CX,AX) \
	PRECALC_39(Y13,0x60,0x240) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_152 \
	CALC_F2_PRE(0x250,AX,DX,DI) \
	PRECALC_32(Y14,Y13) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_153 \
	CALC_F2_PRE(0x254,DI,AX,SI) \
	PRECALC_33(Y8,Y12) \
	CALC_F2_POST(DI,DX,CX,SI)

#define CALC_154 \
	CALC_F2_PRE(0x258,SI,DI,BX) \
	PRECALC_34(Y3) \
	CALC_F2_POST(SI,AX,DX,BX)

#define CALC_155 \
	CALC_F2_PRE(0x25c,BX,SI,CX) \
	PRECALC_35(Y12) \
	CALC_F2_POST(BX,DI,AX,CX)

#define CALC_156 \
	CALC_F2_PRE(0x270,CX,BX,DX) \
	PRECALC_36(Y12) \
	CALC_F2_POST(CX,SI,DI,DX)

#define CALC_157 \
	CALC_F2_PRE(0x274,DX,CX,AX) \
	PRECALC_37(Y12) \
	CALC_F2_POST(DX,BX,SI,AX)

#define CALC_158 \
	CALC_F2_PRE(0x278,AX,DX,DI) \
	CALC_F2_POST(AX,CX,BX,DI)

#define CALC_159 \
	ADDL 0x27c(R15),SI \
	LEAL (SI)(AX*1), SI \
	RORXL $0x1b, DI, R12 \
	PRECALC_39(Y12,0x60,0x260) \
	ADDL R12, SI



#define CALC \
	MOVL	(R9), CX \
	MOVL	4(R9), SI \
	MOVL	8(R9), DI \
	MOVL	12(R9), AX \
	MOVL	16(R9), DX \
	MOVQ    SP, R14 \
	LEAQ    (2*4*80+32)(SP), R15 \
	PRECALC \ // Precalc WK for first 2 blocks
	XCHGQ   R15, R14 \
loop: \  // this loops is unrolled
	CMPQ    R10, R8 \ // we use R8 value (set below) as a signal of a last block
	JNE	begin \
	VZEROUPPER \
	RET \
begin: \
	CALC_0 \
	CALC_1 \
	CALC_2 \
	CALC_3 \
	CALC_4 \
	CALC_5 \
	CALC_6 \
	CALC_7 \
	CALC_8 \
	CALC_9 \
	CALC_10 \
	CALC_11 \
	CALC_12 \
	CALC_13 \
	CALC_14 \
	CALC_15 \
	CALC_16 \
	CALC_17 \
	CALC_18 \
	CALC_19 \
	CALC_20 \
	CALC_21 \
	CALC_22 \
	CALC_23 \
	CALC_24 \
	CALC_25 \
	CALC_26 \
	CALC_27 \
	CALC_28 \
	CALC_29 \
	CALC_30 \
	CALC_31 \
	CALC_32 \
	CALC_33 \
	CALC_34 \
	CALC_35 \
	CALC_36 \
	CALC_37 \
	CALC_38 \
	CALC_39 \
	CALC_40 \
	CALC_41 \
	CALC_42 \
	CALC_43 \
	CALC_44 \
	CALC_45 \
	CALC_46 \
	CALC_47 \
	CALC_48 \
	CALC_49 \
	CALC_50 \
	CALC_51 \
	CALC_52 \
	CALC_53 \
	CALC_54 \
	CALC_55 \
	CALC_56 \
	CALC_57 \
	CALC_58 \
	CALC_59 \
	ADDQ $128, R10 \ // move to next even-64-byte block
	CMPQ R10, R11 \ // is current block the last one?
	CMOVQCC R8, R10 \ // signal the last iteration smartly
	CALC_60 \
	CALC_61 \
	CALC_62 \
	CALC_63 \
	CALC_64 \
	CALC_65 \
	CALC_66 \
	CALC_67 \
	CALC_68 \
	CALC_69 \
	CALC_70 \
	CALC_71 \
	CALC_72 \
	CALC_73 \
	CALC_74 \
	CALC_75 \
	CALC_76 \
	CALC_77 \
	CALC_78 \
	CALC_79 \
	UPDATE_HASH(AX,DX,BX,SI,DI) \
	CMPQ R10, R8 \ // is current block the last one?
	JE loop\
	MOVL DX, CX \
	CALC_80 \
	CALC_81 \
	CALC_82 \
	CALC_83 \
	CALC_84 \
	CALC_85 \
	CALC_86 \
	CALC_87 \
	CALC_88 \
	CALC_89 \
	CALC_90 \
	CALC_91 \
	CALC_92 \
	CALC_93 \
	CALC_94 \
	CALC_95 \
	CALC_96 \
	CALC_97 \
	CALC_98 \
	CALC_99 \
	CALC_100 \
	CALC_101 \
	CALC_102 \
	CALC_103 \
	CALC_104 \
	CALC_105 \
	CALC_106 \
	CALC_107 \
	CALC_108 \
	CALC_109 \
	CALC_110 \
	CALC_111 \
	CALC_112 \
	CALC_113 \
	CALC_114 \
	CALC_115 \
	CALC_116 \
	CALC_117 \
	CALC_118 \
	CALC_119 \
	CALC_120 \
	CALC_121 \
	CALC_122 \
	CALC_123 \
	CALC_124 \
	CALC_125 \
	CALC_126 \
	CALC_127 \
	CALC_128 \
	CALC_129 \
	CALC_130 \
	CALC_131 \
	CALC_132 \
	CALC_133 \
	CALC_134 \
	CALC_135 \
	CALC_136 \
	CALC_137 \
	CALC_138 \
	CALC_139 \
	ADDQ $128, R13 \ //move to next even-64-byte block
	CMPQ R13, R11 \ //is current block the last one?
	CMOVQCC R8, R10 \
	CALC_140 \
	CALC_141 \
	CALC_142 \
	CALC_143 \
	CALC_144 \
	CALC_145 \
	CALC_146 \
	CALC_147 \
	CALC_148 \
	CALC_149 \
	CALC_150 \
	CALC_151 \
	CALC_152 \
	CALC_153 \
	CALC_154 \
	CALC_155 \
	CALC_156 \
	CALC_157 \
	CALC_158 \
	CALC_159 \
	UPDATE_HASH(SI,DI,DX,CX,BX) \
	MOVL	SI, R12 \ //Reset state for  AVX2 reg permutation
	MOVL	DI, SI \
	MOVL	DX, DI \
	MOVL	BX, DX \
	MOVL	CX, AX \
	MOVL	R12, CX \
	XCHGQ   R15, R14 \
	JMP     loop



TEXT ·blockAVX2(SB),$1408-32

	MOVQ	dig+0(FP),	DI
	MOVQ	p_base+8(FP),	SI
	MOVQ	p_len+16(FP),	DX
	SHRQ	$6,		DX
	SHLQ	$6,		DX

	MOVQ	$K_XMM_AR<>(SB), R8

	MOVQ	DI, R9
	MOVQ	SI, R10
	LEAQ	64(SI), R13

	ADDQ	SI, DX
	ADDQ	$64, DX
	MOVQ	DX, R11

	CMPQ	R13, R11
	CMOVQCC	R8, R13

	VMOVDQU	BSWAP_SHUFB_CTL<>(SB), Y10

	CALC // RET is inside macros

DATA K_XMM_AR<>+0x00(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x04(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x08(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x0c(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x10(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x14(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x18(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x1c(SB)/4,$0x5a827999
DATA K_XMM_AR<>+0x20(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x24(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x28(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x2c(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x30(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x34(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x38(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x3c(SB)/4,$0x6ed9eba1
DATA K_XMM_AR<>+0x40(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x44(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x48(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x4c(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x50(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x54(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x58(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x5c(SB)/4,$0x8f1bbcdc
DATA K_XMM_AR<>+0x60(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x64(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x68(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x6c(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x70(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x74(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x78(SB)/4,$0xca62c1d6
DATA K_XMM_AR<>+0x7c(SB)/4,$0xca62c1d6
GLOBL K_XMM_AR<>(SB),RODATA,$128

DATA BSWAP_SHUFB_CTL<>+0x00(SB)/4,$0x00010203
DATA BSWAP_SHUFB_CTL<>+0x04(SB)/4,$0x04050607
DATA BSWAP_SHUFB_CTL<>+0x08(SB)/4,$0x08090a0b
DATA BSWAP_SHUFB_CTL<>+0x0c(SB)/4,$0x0c0d0e0f
DATA BSWAP_SHUFB_CTL<>+0x10(SB)/4,$0x00010203
DATA BSWAP_SHUFB_CTL<>+0x14(SB)/4,$0x04050607
DATA BSWAP_SHUFB_CTL<>+0x18(SB)/4,$0x08090a0b
DATA BSWAP_SHUFB_CTL<>+0x1c(SB)/4,$0x0c0d0e0f
GLOBL BSWAP_SHUFB_CTL<>(SB),RODATA,$32
