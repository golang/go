// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

// This is an optimized implementation of AES-GCM using AES-NI and CLMUL-NI
// The implementation uses some optimization as described in:
// [1] Gueron, S., Kounavis, M.E.: Intel® Carry-Less Multiplication
//     Instruction and its Usage for Computing the GCM Mode rev. 2.02
// [2] Gueron, S., Krasnov, V.: Speeding up Counter Mode in Software and
//     Hardware

#include "textflag.h"

#define B0 X0
#define B1 X1
#define B2 X2
#define B3 X3
#define B4 X4
#define B5 X5
#define B6 X6
#define B7 X7

#define ACC0 X8
#define ACC1 X9
#define ACCM X10

#define T0 X11
#define T1 X12
#define T2 X13
#define POLY X14
#define BSWAP X15

DATA bswapMask<>+0x00(SB)/8, $0x08090a0b0c0d0e0f
DATA bswapMask<>+0x08(SB)/8, $0x0001020304050607

DATA gcmPoly<>+0x00(SB)/8, $0x0000000000000001
DATA gcmPoly<>+0x08(SB)/8, $0xc200000000000000

DATA andMask<>+0x00(SB)/8, $0x00000000000000ff
DATA andMask<>+0x08(SB)/8, $0x0000000000000000
DATA andMask<>+0x10(SB)/8, $0x000000000000ffff
DATA andMask<>+0x18(SB)/8, $0x0000000000000000
DATA andMask<>+0x20(SB)/8, $0x0000000000ffffff
DATA andMask<>+0x28(SB)/8, $0x0000000000000000
DATA andMask<>+0x30(SB)/8, $0x00000000ffffffff
DATA andMask<>+0x38(SB)/8, $0x0000000000000000
DATA andMask<>+0x40(SB)/8, $0x000000ffffffffff
DATA andMask<>+0x48(SB)/8, $0x0000000000000000
DATA andMask<>+0x50(SB)/8, $0x0000ffffffffffff
DATA andMask<>+0x58(SB)/8, $0x0000000000000000
DATA andMask<>+0x60(SB)/8, $0x00ffffffffffffff
DATA andMask<>+0x68(SB)/8, $0x0000000000000000
DATA andMask<>+0x70(SB)/8, $0xffffffffffffffff
DATA andMask<>+0x78(SB)/8, $0x0000000000000000
DATA andMask<>+0x80(SB)/8, $0xffffffffffffffff
DATA andMask<>+0x88(SB)/8, $0x00000000000000ff
DATA andMask<>+0x90(SB)/8, $0xffffffffffffffff
DATA andMask<>+0x98(SB)/8, $0x000000000000ffff
DATA andMask<>+0xa0(SB)/8, $0xffffffffffffffff
DATA andMask<>+0xa8(SB)/8, $0x0000000000ffffff
DATA andMask<>+0xb0(SB)/8, $0xffffffffffffffff
DATA andMask<>+0xb8(SB)/8, $0x00000000ffffffff
DATA andMask<>+0xc0(SB)/8, $0xffffffffffffffff
DATA andMask<>+0xc8(SB)/8, $0x000000ffffffffff
DATA andMask<>+0xd0(SB)/8, $0xffffffffffffffff
DATA andMask<>+0xd8(SB)/8, $0x0000ffffffffffff
DATA andMask<>+0xe0(SB)/8, $0xffffffffffffffff
DATA andMask<>+0xe8(SB)/8, $0x00ffffffffffffff

GLOBL bswapMask<>(SB), (NOPTR+RODATA), $16
GLOBL gcmPoly<>(SB), (NOPTR+RODATA), $16
GLOBL andMask<>(SB), (NOPTR+RODATA), $240

// func gcmAesFinish(productTable *[256]byte, tagMask, T *[16]byte, pLen, dLen uint64)
TEXT ·gcmAesFinish(SB),NOSPLIT,$0
#define pTbl DI
#define tMsk SI
#define tPtr DX
#define plen AX
#define dlen CX

	MOVQ productTable+0(FP), pTbl
	MOVQ tagMask+8(FP), tMsk
	MOVQ T+16(FP), tPtr
	MOVQ pLen+24(FP), plen
	MOVQ dLen+32(FP), dlen

	MOVOU (tPtr), ACC0
	MOVOU (tMsk), T2

	MOVOU bswapMask<>(SB), BSWAP
	MOVOU gcmPoly<>(SB), POLY

	SHLQ $3, plen
	SHLQ $3, dlen

	MOVQ plen, B0
	PINSRQ $1, dlen, B0

	PXOR ACC0, B0

	MOVOU (16*14)(pTbl), ACC0
	MOVOU (16*15)(pTbl), ACCM
	MOVOU ACC0, ACC1

	PCLMULQDQ $0x00, B0, ACC0
	PCLMULQDQ $0x11, B0, ACC1
	PSHUFD $78, B0, T0
	PXOR B0, T0
	PCLMULQDQ $0x00, T0, ACCM

	PXOR ACC0, ACCM
	PXOR ACC1, ACCM
	MOVOU ACCM, T0
	PSRLDQ $8, ACCM
	PSLLDQ $8, T0
	PXOR ACCM, ACC1
	PXOR T0, ACC0

	MOVOU POLY, T0
	PCLMULQDQ $0x01, ACC0, T0
	PSHUFD $78, ACC0, ACC0
	PXOR T0, ACC0

	MOVOU POLY, T0
	PCLMULQDQ $0x01, ACC0, T0
	PSHUFD $78, ACC0, ACC0
	PXOR T0, ACC0

	PXOR ACC1, ACC0

	PSHUFB BSWAP, ACC0
	PXOR T2, ACC0
	MOVOU ACC0, (tPtr)

	RET
#undef pTbl
#undef tMsk
#undef tPtr
#undef plen
#undef dlen

// func gcmAesInit(productTable *[256]byte, ks []uint32)
TEXT ·gcmAesInit(SB),NOSPLIT,$0
#define dst DI
#define KS SI
#define NR DX

	MOVQ productTable+0(FP), dst
	MOVQ ks_base+8(FP), KS
	MOVQ ks_len+16(FP), NR

	SHRQ $2, NR
	DECQ NR

	MOVOU bswapMask<>(SB), BSWAP
	MOVOU gcmPoly<>(SB), POLY

	// Encrypt block 0, with the AES key to generate the hash key H
	MOVOU (16*0)(KS), B0
	MOVOU (16*1)(KS), T0
	AESENC T0, B0
	MOVOU (16*2)(KS), T0
	AESENC T0, B0
	MOVOU (16*3)(KS), T0
	AESENC T0, B0
	MOVOU (16*4)(KS), T0
	AESENC T0, B0
	MOVOU (16*5)(KS), T0
	AESENC T0, B0
	MOVOU (16*6)(KS), T0
	AESENC T0, B0
	MOVOU (16*7)(KS), T0
	AESENC T0, B0
	MOVOU (16*8)(KS), T0
	AESENC T0, B0
	MOVOU (16*9)(KS), T0
	AESENC T0, B0
	MOVOU (16*10)(KS), T0
	CMPQ NR, $12
	JB initEncLast
	AESENC T0, B0
	MOVOU (16*11)(KS), T0
	AESENC T0, B0
	MOVOU (16*12)(KS), T0
	JE initEncLast
	AESENC T0, B0
	MOVOU (16*13)(KS), T0
	AESENC T0, B0
	MOVOU (16*14)(KS), T0
initEncLast:
	AESENCLAST T0, B0

	PSHUFB BSWAP, B0
	// H * 2
	PSHUFD $0xff, B0, T0
	MOVOU B0, T1
	PSRAL $31, T0
	PAND POLY, T0
	PSRLL $31, T1
	PSLLDQ $4, T1
	PSLLL $1, B0
	PXOR T0, B0
	PXOR T1, B0
	// Karatsuba pre-computations
	MOVOU B0, (16*14)(dst)
	PSHUFD $78, B0, B1
	PXOR B0, B1
	MOVOU B1, (16*15)(dst)

	MOVOU B0, B2
	MOVOU B1, B3
	// Now prepare powers of H and pre-computations for them
	MOVQ $7, AX

initLoop:
		MOVOU B2, T0
		MOVOU B2, T1
		MOVOU B3, T2
		PCLMULQDQ $0x00, B0, T0
		PCLMULQDQ $0x11, B0, T1
		PCLMULQDQ $0x00, B1, T2

		PXOR T0, T2
		PXOR T1, T2
		MOVOU T2, B4
		PSLLDQ $8, B4
		PSRLDQ $8, T2
		PXOR B4, T0
		PXOR T2, T1

		MOVOU POLY, B2
		PCLMULQDQ $0x01, T0, B2
		PSHUFD $78, T0, T0
		PXOR B2, T0
		MOVOU POLY, B2
		PCLMULQDQ $0x01, T0, B2
		PSHUFD $78, T0, T0
		PXOR T0, B2
		PXOR T1, B2

		MOVOU B2, (16*12)(dst)
		PSHUFD $78, B2, B3
		PXOR B2, B3
		MOVOU B3, (16*13)(dst)

		DECQ AX
		LEAQ (-16*2)(dst), dst
	JNE initLoop

	RET
#undef NR
#undef KS
#undef dst

// func gcmAesData(productTable *[256]byte, data []byte, T *[16]byte)
TEXT ·gcmAesData(SB),NOSPLIT,$0
#define pTbl DI
#define aut SI
#define tPtr CX
#define autLen DX

#define reduceRound(a) 	MOVOU POLY, T0;	PCLMULQDQ $0x01, a, T0; PSHUFD $78, a, a; PXOR T0, a
#define mulRoundAAD(X ,i) \
	MOVOU (16*(i*2))(pTbl), T1;\
	MOVOU T1, T2;\
	PCLMULQDQ $0x00, X, T1;\
	PXOR T1, ACC0;\
	PCLMULQDQ $0x11, X, T2;\
	PXOR T2, ACC1;\
	PSHUFD $78, X, T1;\
	PXOR T1, X;\
	MOVOU (16*(i*2+1))(pTbl), T1;\
	PCLMULQDQ $0x00, X, T1;\
	PXOR T1, ACCM

	MOVQ productTable+0(FP), pTbl
	MOVQ data_base+8(FP), aut
	MOVQ data_len+16(FP), autLen
	MOVQ T+32(FP), tPtr

	PXOR ACC0, ACC0
	MOVOU bswapMask<>(SB), BSWAP
	MOVOU gcmPoly<>(SB), POLY

	TESTQ autLen, autLen
	JEQ dataBail

	CMPQ autLen, $13	// optimize the TLS case
	JE dataTLS
	CMPQ autLen, $128
	JB startSinglesLoop
	JMP dataOctaLoop

dataTLS:
	MOVOU (16*14)(pTbl), T1
	MOVOU (16*15)(pTbl), T2
	PXOR B0, B0
	MOVQ (aut), B0
	PINSRD $2, 8(aut), B0
	PINSRB $12, 12(aut), B0
	XORQ autLen, autLen
	JMP dataMul

dataOctaLoop:
		CMPQ autLen, $128
		JB startSinglesLoop
		SUBQ $128, autLen

		MOVOU (16*0)(aut), X0
		MOVOU (16*1)(aut), X1
		MOVOU (16*2)(aut), X2
		MOVOU (16*3)(aut), X3
		MOVOU (16*4)(aut), X4
		MOVOU (16*5)(aut), X5
		MOVOU (16*6)(aut), X6
		MOVOU (16*7)(aut), X7
		LEAQ (16*8)(aut), aut
		PSHUFB BSWAP, X0
		PSHUFB BSWAP, X1
		PSHUFB BSWAP, X2
		PSHUFB BSWAP, X3
		PSHUFB BSWAP, X4
		PSHUFB BSWAP, X5
		PSHUFB BSWAP, X6
		PSHUFB BSWAP, X7
		PXOR ACC0, X0

		MOVOU (16*0)(pTbl), ACC0
		MOVOU (16*1)(pTbl), ACCM
		MOVOU ACC0, ACC1
		PSHUFD $78, X0, T1
		PXOR X0, T1
		PCLMULQDQ $0x00, X0, ACC0
		PCLMULQDQ $0x11, X0, ACC1
		PCLMULQDQ $0x00, T1, ACCM

		mulRoundAAD(X1, 1)
		mulRoundAAD(X2, 2)
		mulRoundAAD(X3, 3)
		mulRoundAAD(X4, 4)
		mulRoundAAD(X5, 5)
		mulRoundAAD(X6, 6)
		mulRoundAAD(X7, 7)

		PXOR ACC0, ACCM
		PXOR ACC1, ACCM
		MOVOU ACCM, T0
		PSRLDQ $8, ACCM
		PSLLDQ $8, T0
		PXOR ACCM, ACC1
		PXOR T0, ACC0
		reduceRound(ACC0)
		reduceRound(ACC0)
		PXOR ACC1, ACC0
	JMP dataOctaLoop

startSinglesLoop:
	MOVOU (16*14)(pTbl), T1
	MOVOU (16*15)(pTbl), T2

dataSinglesLoop:

		CMPQ autLen, $16
		JB dataEnd
		SUBQ $16, autLen

		MOVOU (aut), B0
dataMul:
		PSHUFB BSWAP, B0
		PXOR ACC0, B0

		MOVOU T1, ACC0
		MOVOU T2, ACCM
		MOVOU T1, ACC1

		PSHUFD $78, B0, T0
		PXOR B0, T0
		PCLMULQDQ $0x00, B0, ACC0
		PCLMULQDQ $0x11, B0, ACC1
		PCLMULQDQ $0x00, T0, ACCM

		PXOR ACC0, ACCM
		PXOR ACC1, ACCM
		MOVOU ACCM, T0
		PSRLDQ $8, ACCM
		PSLLDQ $8, T0
		PXOR ACCM, ACC1
		PXOR T0, ACC0

		MOVOU POLY, T0
		PCLMULQDQ $0x01, ACC0, T0
		PSHUFD $78, ACC0, ACC0
		PXOR T0, ACC0

		MOVOU POLY, T0
		PCLMULQDQ $0x01, ACC0, T0
		PSHUFD $78, ACC0, ACC0
		PXOR T0, ACC0
		PXOR ACC1, ACC0

		LEAQ 16(aut), aut

	JMP dataSinglesLoop

dataEnd:

	TESTQ autLen, autLen
	JEQ dataBail

	PXOR B0, B0
	LEAQ -1(aut)(autLen*1), aut

dataLoadLoop:

		PSLLDQ $1, B0
		PINSRB $0, (aut), B0

		LEAQ -1(aut), aut
		DECQ autLen
		JNE dataLoadLoop

	JMP dataMul

dataBail:
	MOVOU ACC0, (tPtr)
	RET
#undef pTbl
#undef aut
#undef tPtr
#undef autLen

// func gcmAesEnc(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)
TEXT ·gcmAesEnc(SB),0,$256-96
#define pTbl DI
#define ctx DX
#define ctrPtr CX
#define ptx SI
#define ks AX
#define tPtr R8
#define ptxLen R9
#define aluCTR R10
#define aluTMP R11
#define aluK R12
#define NR R13

#define increment(i) ADDL $1, aluCTR; MOVL aluCTR, aluTMP; XORL aluK, aluTMP; BSWAPL aluTMP; MOVL aluTMP, (3*4 + 8*16 + i*16)(SP)
#define aesRnd(k) AESENC k, B0; AESENC k, B1; AESENC k, B2; AESENC k, B3; AESENC k, B4; AESENC k, B5; AESENC k, B6; AESENC k, B7
#define aesRound(i) MOVOU (16*i)(ks), T0;AESENC T0, B0; AESENC T0, B1; AESENC T0, B2; AESENC T0, B3; AESENC T0, B4; AESENC T0, B5; AESENC T0, B6; AESENC T0, B7
#define aesRndLast(k) AESENCLAST k, B0; AESENCLAST k, B1; AESENCLAST k, B2; AESENCLAST k, B3; AESENCLAST k, B4; AESENCLAST k, B5; AESENCLAST k, B6; AESENCLAST k, B7
#define combinedRound(i) \
	MOVOU (16*i)(ks), T0;\
	AESENC T0, B0;\
	AESENC T0, B1;\
	AESENC T0, B2;\
	AESENC T0, B3;\
	 MOVOU (16*(i*2))(pTbl), T1;\
	 MOVOU T1, T2;\
	AESENC T0, B4;\
	AESENC T0, B5;\
	AESENC T0, B6;\
	AESENC T0, B7;\
	 MOVOU (16*i)(SP), T0;\
	 PCLMULQDQ $0x00, T0, T1;\
	 PXOR T1, ACC0;\
	 PSHUFD $78, T0, T1;\
	 PCLMULQDQ $0x11, T0, T2;\
	 PXOR T1, T0;\
	 PXOR T2, ACC1;\
	 MOVOU (16*(i*2+1))(pTbl), T2;\
	 PCLMULQDQ $0x00, T2, T0;\
	 PXOR T0, ACCM
#define mulRound(i) \
	MOVOU (16*i)(SP), T0;\
	MOVOU (16*(i*2))(pTbl), T1;\
	MOVOU T1, T2;\
	PCLMULQDQ $0x00, T0, T1;\
	PXOR T1, ACC0;\
	PCLMULQDQ $0x11, T0, T2;\
	PXOR T2, ACC1;\
	PSHUFD $78, T0, T1;\
	PXOR T1, T0;\
	MOVOU (16*(i*2+1))(pTbl), T1;\
	PCLMULQDQ $0x00, T0, T1;\
	PXOR T1, ACCM

	MOVQ productTable+0(FP), pTbl
	MOVQ dst+8(FP), ctx
	MOVQ src_base+32(FP), ptx
	MOVQ src_len+40(FP), ptxLen
	MOVQ ctr+56(FP), ctrPtr
	MOVQ T+64(FP), tPtr
	MOVQ ks_base+72(FP), ks
	MOVQ ks_len+80(FP), NR

	SHRQ $2, NR
	DECQ NR

	MOVOU bswapMask<>(SB), BSWAP
	MOVOU gcmPoly<>(SB), POLY

	MOVOU (tPtr), ACC0
	PXOR ACC1, ACC1
	PXOR ACCM, ACCM
	MOVOU (ctrPtr), B0
	MOVL (3*4)(ctrPtr), aluCTR
	MOVOU (ks), T0
	MOVL (3*4)(ks), aluK
	BSWAPL aluCTR
	BSWAPL aluK

	PXOR B0, T0
	MOVOU T0, (8*16 + 0*16)(SP)
	increment(0)

	CMPQ ptxLen, $128
	JB gcmAesEncSingles
	SUBQ $128, ptxLen

	// We have at least 8 blocks to encrypt, prepare the rest of the counters
	MOVOU T0, (8*16 + 1*16)(SP)
	increment(1)
	MOVOU T0, (8*16 + 2*16)(SP)
	increment(2)
	MOVOU T0, (8*16 + 3*16)(SP)
	increment(3)
	MOVOU T0, (8*16 + 4*16)(SP)
	increment(4)
	MOVOU T0, (8*16 + 5*16)(SP)
	increment(5)
	MOVOU T0, (8*16 + 6*16)(SP)
	increment(6)
	MOVOU T0, (8*16 + 7*16)(SP)
	increment(7)

	MOVOU (8*16 + 0*16)(SP), B0
	MOVOU (8*16 + 1*16)(SP), B1
	MOVOU (8*16 + 2*16)(SP), B2
	MOVOU (8*16 + 3*16)(SP), B3
	MOVOU (8*16 + 4*16)(SP), B4
	MOVOU (8*16 + 5*16)(SP), B5
	MOVOU (8*16 + 6*16)(SP), B6
	MOVOU (8*16 + 7*16)(SP), B7

	aesRound(1)
	increment(0)
	aesRound(2)
	increment(1)
	aesRound(3)
	increment(2)
	aesRound(4)
	increment(3)
	aesRound(5)
	increment(4)
	aesRound(6)
	increment(5)
	aesRound(7)
	increment(6)
	aesRound(8)
	increment(7)
	aesRound(9)
	MOVOU (16*10)(ks), T0
	CMPQ NR, $12
	JB encLast1
	aesRnd(T0)
	aesRound(11)
	MOVOU (16*12)(ks), T0
	JE encLast1
	aesRnd(T0)
	aesRound(13)
	MOVOU (16*14)(ks), T0
encLast1:
	aesRndLast(T0)

	MOVOU (16*0)(ptx), T0
	PXOR T0, B0
	MOVOU (16*1)(ptx), T0
	PXOR T0, B1
	MOVOU (16*2)(ptx), T0
	PXOR T0, B2
	MOVOU (16*3)(ptx), T0
	PXOR T0, B3
	MOVOU (16*4)(ptx), T0
	PXOR T0, B4
	MOVOU (16*5)(ptx), T0
	PXOR T0, B5
	MOVOU (16*6)(ptx), T0
	PXOR T0, B6
	MOVOU (16*7)(ptx), T0
	PXOR T0, B7

	MOVOU B0, (16*0)(ctx)
	PSHUFB BSWAP, B0
	PXOR ACC0, B0
	MOVOU B1, (16*1)(ctx)
	PSHUFB BSWAP, B1
	MOVOU B2, (16*2)(ctx)
	PSHUFB BSWAP, B2
	MOVOU B3, (16*3)(ctx)
	PSHUFB BSWAP, B3
	MOVOU B4, (16*4)(ctx)
	PSHUFB BSWAP, B4
	MOVOU B5, (16*5)(ctx)
	PSHUFB BSWAP, B5
	MOVOU B6, (16*6)(ctx)
	PSHUFB BSWAP, B6
	MOVOU B7, (16*7)(ctx)
	PSHUFB BSWAP, B7

	MOVOU B0, (16*0)(SP)
	MOVOU B1, (16*1)(SP)
	MOVOU B2, (16*2)(SP)
	MOVOU B3, (16*3)(SP)
	MOVOU B4, (16*4)(SP)
	MOVOU B5, (16*5)(SP)
	MOVOU B6, (16*6)(SP)
	MOVOU B7, (16*7)(SP)

	LEAQ 128(ptx), ptx
	LEAQ 128(ctx), ctx

gcmAesEncOctetsLoop:

		CMPQ ptxLen, $128
		JB gcmAesEncOctetsEnd
		SUBQ $128, ptxLen

		MOVOU (8*16 + 0*16)(SP), B0
		MOVOU (8*16 + 1*16)(SP), B1
		MOVOU (8*16 + 2*16)(SP), B2
		MOVOU (8*16 + 3*16)(SP), B3
		MOVOU (8*16 + 4*16)(SP), B4
		MOVOU (8*16 + 5*16)(SP), B5
		MOVOU (8*16 + 6*16)(SP), B6
		MOVOU (8*16 + 7*16)(SP), B7

		MOVOU (16*0)(SP), T0
		PSHUFD $78, T0, T1
		PXOR T0, T1

		MOVOU (16*0)(pTbl), ACC0
		MOVOU (16*1)(pTbl), ACCM
		MOVOU ACC0, ACC1

		PCLMULQDQ $0x00, T1, ACCM
		PCLMULQDQ $0x00, T0, ACC0
		PCLMULQDQ $0x11, T0, ACC1

		combinedRound(1)
		increment(0)
		combinedRound(2)
		increment(1)
		combinedRound(3)
		increment(2)
		combinedRound(4)
		increment(3)
		combinedRound(5)
		increment(4)
		combinedRound(6)
		increment(5)
		combinedRound(7)
		increment(6)

		aesRound(8)
		increment(7)

		PXOR ACC0, ACCM
		PXOR ACC1, ACCM
		MOVOU ACCM, T0
		PSRLDQ $8, ACCM
		PSLLDQ $8, T0
		PXOR ACCM, ACC1
		PXOR T0, ACC0

		reduceRound(ACC0)
		aesRound(9)

		reduceRound(ACC0)
		PXOR ACC1, ACC0

		MOVOU (16*10)(ks), T0
		CMPQ NR, $12
		JB encLast2
		aesRnd(T0)
		aesRound(11)
		MOVOU (16*12)(ks), T0
		JE encLast2
		aesRnd(T0)
		aesRound(13)
		MOVOU (16*14)(ks), T0
encLast2:
		aesRndLast(T0)

		MOVOU (16*0)(ptx), T0
		PXOR T0, B0
		MOVOU (16*1)(ptx), T0
		PXOR T0, B1
		MOVOU (16*2)(ptx), T0
		PXOR T0, B2
		MOVOU (16*3)(ptx), T0
		PXOR T0, B3
		MOVOU (16*4)(ptx), T0
		PXOR T0, B4
		MOVOU (16*5)(ptx), T0
		PXOR T0, B5
		MOVOU (16*6)(ptx), T0
		PXOR T0, B6
		MOVOU (16*7)(ptx), T0
		PXOR T0, B7

		MOVOU B0, (16*0)(ctx)
		PSHUFB BSWAP, B0
		PXOR ACC0, B0
		MOVOU B1, (16*1)(ctx)
		PSHUFB BSWAP, B1
		MOVOU B2, (16*2)(ctx)
		PSHUFB BSWAP, B2
		MOVOU B3, (16*3)(ctx)
		PSHUFB BSWAP, B3
		MOVOU B4, (16*4)(ctx)
		PSHUFB BSWAP, B4
		MOVOU B5, (16*5)(ctx)
		PSHUFB BSWAP, B5
		MOVOU B6, (16*6)(ctx)
		PSHUFB BSWAP, B6
		MOVOU B7, (16*7)(ctx)
		PSHUFB BSWAP, B7

		MOVOU B0, (16*0)(SP)
		MOVOU B1, (16*1)(SP)
		MOVOU B2, (16*2)(SP)
		MOVOU B3, (16*3)(SP)
		MOVOU B4, (16*4)(SP)
		MOVOU B5, (16*5)(SP)
		MOVOU B6, (16*6)(SP)
		MOVOU B7, (16*7)(SP)

		LEAQ 128(ptx), ptx
		LEAQ 128(ctx), ctx

		JMP gcmAesEncOctetsLoop

gcmAesEncOctetsEnd:

	MOVOU (16*0)(SP), T0
	MOVOU (16*0)(pTbl), ACC0
	MOVOU (16*1)(pTbl), ACCM
	MOVOU ACC0, ACC1
	PSHUFD $78, T0, T1
	PXOR T0, T1
	PCLMULQDQ $0x00, T0, ACC0
	PCLMULQDQ $0x11, T0, ACC1
	PCLMULQDQ $0x00, T1, ACCM

	mulRound(1)
	mulRound(2)
	mulRound(3)
	mulRound(4)
	mulRound(5)
	mulRound(6)
	mulRound(7)

	PXOR ACC0, ACCM
	PXOR ACC1, ACCM
	MOVOU ACCM, T0
	PSRLDQ $8, ACCM
	PSLLDQ $8, T0
	PXOR ACCM, ACC1
	PXOR T0, ACC0

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR ACC1, ACC0

	TESTQ ptxLen, ptxLen
	JE gcmAesEncDone

	SUBQ $7, aluCTR

gcmAesEncSingles:

	MOVOU (16*1)(ks), B1
	MOVOU (16*2)(ks), B2
	MOVOU (16*3)(ks), B3
	MOVOU (16*4)(ks), B4
	MOVOU (16*5)(ks), B5
	MOVOU (16*6)(ks), B6
	MOVOU (16*7)(ks), B7

	MOVOU (16*14)(pTbl), T2

gcmAesEncSinglesLoop:

		CMPQ ptxLen, $16
		JB gcmAesEncTail
		SUBQ $16, ptxLen

		MOVOU (8*16 + 0*16)(SP), B0
		increment(0)

		AESENC B1, B0
		AESENC B2, B0
		AESENC B3, B0
		AESENC B4, B0
		AESENC B5, B0
		AESENC B6, B0
		AESENC B7, B0
		MOVOU (16*8)(ks), T0
		AESENC T0, B0
		MOVOU (16*9)(ks), T0
		AESENC T0, B0
		MOVOU (16*10)(ks), T0
		CMPQ NR, $12
		JB encLast3
		AESENC T0, B0
		MOVOU (16*11)(ks), T0
		AESENC T0, B0
		MOVOU (16*12)(ks), T0
		JE encLast3
		AESENC T0, B0
		MOVOU (16*13)(ks), T0
		AESENC T0, B0
		MOVOU (16*14)(ks), T0
encLast3:
		AESENCLAST T0, B0

		MOVOU (ptx), T0
		PXOR T0, B0
		MOVOU B0, (ctx)

		PSHUFB BSWAP, B0
		PXOR ACC0, B0

		MOVOU T2, ACC0
		MOVOU T2, ACC1
		MOVOU (16*15)(pTbl), ACCM

		PSHUFD $78, B0, T0
		PXOR B0, T0
		PCLMULQDQ $0x00, B0, ACC0
		PCLMULQDQ $0x11, B0, ACC1
		PCLMULQDQ $0x00, T0, ACCM

		PXOR ACC0, ACCM
		PXOR ACC1, ACCM
		MOVOU ACCM, T0
		PSRLDQ $8, ACCM
		PSLLDQ $8, T0
		PXOR ACCM, ACC1
		PXOR T0, ACC0

		reduceRound(ACC0)
		reduceRound(ACC0)
		PXOR ACC1, ACC0

		LEAQ (16*1)(ptx), ptx
		LEAQ (16*1)(ctx), ctx

	JMP gcmAesEncSinglesLoop

gcmAesEncTail:
	TESTQ ptxLen, ptxLen
	JE gcmAesEncDone

	MOVOU (8*16 + 0*16)(SP), B0
	AESENC B1, B0
	AESENC B2, B0
	AESENC B3, B0
	AESENC B4, B0
	AESENC B5, B0
	AESENC B6, B0
	AESENC B7, B0
	MOVOU (16*8)(ks), T0
	AESENC T0, B0
	MOVOU (16*9)(ks), T0
	AESENC T0, B0
	MOVOU (16*10)(ks), T0
	CMPQ NR, $12
	JB encLast4
	AESENC T0, B0
	MOVOU (16*11)(ks), T0
	AESENC T0, B0
	MOVOU (16*12)(ks), T0
	JE encLast4
	AESENC T0, B0
	MOVOU (16*13)(ks), T0
	AESENC T0, B0
	MOVOU (16*14)(ks), T0
encLast4:
	AESENCLAST T0, B0
	MOVOU B0, T0

	LEAQ -1(ptx)(ptxLen*1), ptx

	MOVQ ptxLen, aluTMP
	SHLQ $4, aluTMP

	LEAQ andMask<>(SB), aluCTR
	MOVOU -16(aluCTR)(aluTMP*1), T1

	PXOR B0, B0
ptxLoadLoop:
		PSLLDQ $1, B0
		PINSRB $0, (ptx), B0
		LEAQ -1(ptx), ptx
		DECQ ptxLen
	JNE ptxLoadLoop

	PXOR T0, B0
	PAND T1, B0
	MOVOU B0, (ctx)	// I assume there is always space, due to TAG in the end of the CT

	PSHUFB BSWAP, B0
	PXOR ACC0, B0

	MOVOU T2, ACC0
	MOVOU T2, ACC1
	MOVOU (16*15)(pTbl), ACCM

	PSHUFD $78, B0, T0
	PXOR B0, T0
	PCLMULQDQ $0x00, B0, ACC0
	PCLMULQDQ $0x11, B0, ACC1
	PCLMULQDQ $0x00, T0, ACCM

	PXOR ACC0, ACCM
	PXOR ACC1, ACCM
	MOVOU ACCM, T0
	PSRLDQ $8, ACCM
	PSLLDQ $8, T0
	PXOR ACCM, ACC1
	PXOR T0, ACC0

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR ACC1, ACC0

gcmAesEncDone:
	MOVOU ACC0, (tPtr)
	RET
#undef increment

// func gcmAesDec(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)
TEXT ·gcmAesDec(SB),0,$128-96
#define increment(i) ADDL $1, aluCTR; MOVL aluCTR, aluTMP; XORL aluK, aluTMP; BSWAPL aluTMP; MOVL aluTMP, (3*4 + i*16)(SP)
#define combinedDecRound(i) \
	MOVOU (16*i)(ks), T0;\
	AESENC T0, B0;\
	AESENC T0, B1;\
	AESENC T0, B2;\
	AESENC T0, B3;\
	MOVOU (16*(i*2))(pTbl), T1;\
	MOVOU T1, T2;\
	AESENC T0, B4;\
	AESENC T0, B5;\
	AESENC T0, B6;\
	AESENC T0, B7;\
	MOVOU (16*i)(ctx), T0;\
	PSHUFB BSWAP, T0;\
	PCLMULQDQ $0x00, T0, T1;\
	PXOR T1, ACC0;\
	PSHUFD $78, T0, T1;\
	PCLMULQDQ $0x11, T0, T2;\
	PXOR T1, T0;\
	PXOR T2, ACC1;\
	MOVOU (16*(i*2+1))(pTbl), T2;\
	PCLMULQDQ $0x00, T2, T0;\
	PXOR T0, ACCM

	MOVQ productTable+0(FP), pTbl
	MOVQ dst+8(FP), ptx
	MOVQ src_base+32(FP), ctx
	MOVQ src_len+40(FP), ptxLen
	MOVQ ctr+56(FP), ctrPtr
	MOVQ T+64(FP), tPtr
	MOVQ ks_base+72(FP), ks
	MOVQ ks_len+80(FP), NR

	SHRQ $2, NR
	DECQ NR

	MOVOU bswapMask<>(SB), BSWAP
	MOVOU gcmPoly<>(SB), POLY

	MOVOU (tPtr), ACC0
	PXOR ACC1, ACC1
	PXOR ACCM, ACCM
	MOVOU (ctrPtr), B0
	MOVL (3*4)(ctrPtr), aluCTR
	MOVOU (ks), T0
	MOVL (3*4)(ks), aluK
	BSWAPL aluCTR
	BSWAPL aluK

	PXOR B0, T0
	MOVOU T0, (0*16)(SP)
	increment(0)

	CMPQ ptxLen, $128
	JB gcmAesDecSingles

	MOVOU T0, (1*16)(SP)
	increment(1)
	MOVOU T0, (2*16)(SP)
	increment(2)
	MOVOU T0, (3*16)(SP)
	increment(3)
	MOVOU T0, (4*16)(SP)
	increment(4)
	MOVOU T0, (5*16)(SP)
	increment(5)
	MOVOU T0, (6*16)(SP)
	increment(6)
	MOVOU T0, (7*16)(SP)
	increment(7)

gcmAesDecOctetsLoop:

		CMPQ ptxLen, $128
		JB gcmAesDecEndOctets
		SUBQ $128, ptxLen

		MOVOU (0*16)(SP), B0
		MOVOU (1*16)(SP), B1
		MOVOU (2*16)(SP), B2
		MOVOU (3*16)(SP), B3
		MOVOU (4*16)(SP), B4
		MOVOU (5*16)(SP), B5
		MOVOU (6*16)(SP), B6
		MOVOU (7*16)(SP), B7

		MOVOU (16*0)(ctx), T0
		PSHUFB BSWAP, T0
		PXOR ACC0, T0
		PSHUFD $78, T0, T1
		PXOR T0, T1

		MOVOU (16*0)(pTbl), ACC0
		MOVOU (16*1)(pTbl), ACCM
		MOVOU ACC0, ACC1

		PCLMULQDQ $0x00, T1, ACCM
		PCLMULQDQ $0x00, T0, ACC0
		PCLMULQDQ $0x11, T0, ACC1

		combinedDecRound(1)
		increment(0)
		combinedDecRound(2)
		increment(1)
		combinedDecRound(3)
		increment(2)
		combinedDecRound(4)
		increment(3)
		combinedDecRound(5)
		increment(4)
		combinedDecRound(6)
		increment(5)
		combinedDecRound(7)
		increment(6)

		aesRound(8)
		increment(7)

		PXOR ACC0, ACCM
		PXOR ACC1, ACCM
		MOVOU ACCM, T0
		PSRLDQ $8, ACCM
		PSLLDQ $8, T0
		PXOR ACCM, ACC1
		PXOR T0, ACC0

		reduceRound(ACC0)
		aesRound(9)

		reduceRound(ACC0)
		PXOR ACC1, ACC0

		MOVOU (16*10)(ks), T0
		CMPQ NR, $12
		JB decLast1
		aesRnd(T0)
		aesRound(11)
		MOVOU (16*12)(ks), T0
		JE decLast1
		aesRnd(T0)
		aesRound(13)
		MOVOU (16*14)(ks), T0
decLast1:
		aesRndLast(T0)

		MOVOU (16*0)(ctx), T0
		PXOR T0, B0
		MOVOU (16*1)(ctx), T0
		PXOR T0, B1
		MOVOU (16*2)(ctx), T0
		PXOR T0, B2
		MOVOU (16*3)(ctx), T0
		PXOR T0, B3
		MOVOU (16*4)(ctx), T0
		PXOR T0, B4
		MOVOU (16*5)(ctx), T0
		PXOR T0, B5
		MOVOU (16*6)(ctx), T0
		PXOR T0, B6
		MOVOU (16*7)(ctx), T0
		PXOR T0, B7

		MOVOU B0, (16*0)(ptx)
		MOVOU B1, (16*1)(ptx)
		MOVOU B2, (16*2)(ptx)
		MOVOU B3, (16*3)(ptx)
		MOVOU B4, (16*4)(ptx)
		MOVOU B5, (16*5)(ptx)
		MOVOU B6, (16*6)(ptx)
		MOVOU B7, (16*7)(ptx)

		LEAQ 128(ptx), ptx
		LEAQ 128(ctx), ctx

		JMP gcmAesDecOctetsLoop

gcmAesDecEndOctets:

	SUBQ $7, aluCTR

gcmAesDecSingles:

	MOVOU (16*1)(ks), B1
	MOVOU (16*2)(ks), B2
	MOVOU (16*3)(ks), B3
	MOVOU (16*4)(ks), B4
	MOVOU (16*5)(ks), B5
	MOVOU (16*6)(ks), B6
	MOVOU (16*7)(ks), B7

	MOVOU (16*14)(pTbl), T2

gcmAesDecSinglesLoop:

		CMPQ ptxLen, $16
		JB gcmAesDecTail
		SUBQ $16, ptxLen

		MOVOU (ctx), B0
		MOVOU B0, T1
		PSHUFB BSWAP, B0
		PXOR ACC0, B0

		MOVOU T2, ACC0
		MOVOU T2, ACC1
		MOVOU (16*15)(pTbl), ACCM

		PCLMULQDQ $0x00, B0, ACC0
		PCLMULQDQ $0x11, B0, ACC1
		PSHUFD $78, B0, T0
		PXOR B0, T0
		PCLMULQDQ $0x00, T0, ACCM

		PXOR ACC0, ACCM
		PXOR ACC1, ACCM
		MOVOU ACCM, T0
		PSRLDQ $8, ACCM
		PSLLDQ $8, T0
		PXOR ACCM, ACC1
		PXOR T0, ACC0

		reduceRound(ACC0)
		reduceRound(ACC0)
		PXOR ACC1, ACC0

		MOVOU (0*16)(SP), B0
		increment(0)
		AESENC B1, B0
		AESENC B2, B0
		AESENC B3, B0
		AESENC B4, B0
		AESENC B5, B0
		AESENC B6, B0
		AESENC B7, B0
		MOVOU (16*8)(ks), T0
		AESENC T0, B0
		MOVOU (16*9)(ks), T0
		AESENC T0, B0
		MOVOU (16*10)(ks), T0
		CMPQ NR, $12
		JB decLast2
		AESENC T0, B0
		MOVOU (16*11)(ks), T0
		AESENC T0, B0
		MOVOU (16*12)(ks), T0
		JE decLast2
		AESENC T0, B0
		MOVOU (16*13)(ks), T0
		AESENC T0, B0
		MOVOU (16*14)(ks), T0
decLast2:
		AESENCLAST T0, B0

		PXOR T1, B0
		MOVOU B0, (ptx)

		LEAQ (16*1)(ptx), ptx
		LEAQ (16*1)(ctx), ctx

	JMP gcmAesDecSinglesLoop

gcmAesDecTail:

	TESTQ ptxLen, ptxLen
	JE gcmAesDecDone

	MOVQ ptxLen, aluTMP
	SHLQ $4, aluTMP
	LEAQ andMask<>(SB), aluCTR
	MOVOU -16(aluCTR)(aluTMP*1), T1

	MOVOU (ctx), B0	// I assume there is TAG attached to the ctx, and there is no read overflow
	PAND T1, B0

	MOVOU B0, T1
	PSHUFB BSWAP, B0
	PXOR ACC0, B0

	MOVOU (16*14)(pTbl), ACC0
	MOVOU (16*15)(pTbl), ACCM
	MOVOU ACC0, ACC1

	PCLMULQDQ $0x00, B0, ACC0
	PCLMULQDQ $0x11, B0, ACC1
	PSHUFD $78, B0, T0
	PXOR B0, T0
	PCLMULQDQ $0x00, T0, ACCM

	PXOR ACC0, ACCM
	PXOR ACC1, ACCM
	MOVOU ACCM, T0
	PSRLDQ $8, ACCM
	PSLLDQ $8, T0
	PXOR ACCM, ACC1
	PXOR T0, ACC0

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR ACC1, ACC0

	MOVOU (0*16)(SP), B0
	increment(0)
	AESENC B1, B0
	AESENC B2, B0
	AESENC B3, B0
	AESENC B4, B0
	AESENC B5, B0
	AESENC B6, B0
	AESENC B7, B0
	MOVOU (16*8)(ks), T0
	AESENC T0, B0
	MOVOU (16*9)(ks), T0
	AESENC T0, B0
	MOVOU (16*10)(ks), T0
	CMPQ NR, $12
	JB decLast3
	AESENC T0, B0
	MOVOU (16*11)(ks), T0
	AESENC T0, B0
	MOVOU (16*12)(ks), T0
	JE decLast3
	AESENC T0, B0
	MOVOU (16*13)(ks), T0
	AESENC T0, B0
	MOVOU (16*14)(ks), T0
decLast3:
	AESENCLAST T0, B0
	PXOR T1, B0

ptxStoreLoop:
		PEXTRB $0, B0, (ptx)
		PSRLDQ $1, B0
		LEAQ 1(ptx), ptx
		DECQ ptxLen

	JNE ptxStoreLoop

gcmAesDecDone:

	MOVOU ACC0, (tPtr)
	RET
