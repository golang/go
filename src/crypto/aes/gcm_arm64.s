// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define B0 V0
#define B1 V1
#define B2 V2
#define B3 V3
#define B4 V4
#define B5 V5
#define B6 V6
#define B7 V7

#define ACC0 V8
#define ACC1 V9
#define ACCM V10

#define T0 V11
#define T1 V12
#define T2 V13
#define T3 V14

#define POLY V15
#define ZERO V16
#define INC V17
#define CTR V18

#define K0 V19
#define K1 V20
#define K2 V21
#define K3 V22
#define K4 V23
#define K5 V24
#define K6 V25
#define K7 V26
#define K8 V27
#define K9 V28
#define K10 V29
#define K11 V30
#define KLAST V31

#define reduce() \
	VEOR	ACC0.B16, ACCM.B16, ACCM.B16     \
	VEOR	ACC1.B16, ACCM.B16, ACCM.B16     \
	VEXT	$8, ZERO.B16, ACCM.B16, T0.B16   \
	VEXT	$8, ACCM.B16, ZERO.B16, ACCM.B16 \
	VEOR	ACCM.B16, ACC0.B16, ACC0.B16     \
	VEOR	T0.B16, ACC1.B16, ACC1.B16       \
	VPMULL	POLY.D1, ACC0.D1, T0.Q1          \
	VEXT	$8, ACC0.B16, ACC0.B16, ACC0.B16 \
	VEOR	T0.B16, ACC0.B16, ACC0.B16       \
	VPMULL	POLY.D1, ACC0.D1, T0.Q1          \
	VEOR	T0.B16, ACC1.B16, ACC1.B16       \
	VEXT	$8, ACC1.B16, ACC1.B16, ACC1.B16 \
	VEOR	ACC1.B16, ACC0.B16, ACC0.B16     \

// func gcmAesFinish(productTable *[256]byte, tagMask, T *[16]byte, pLen, dLen uint64)
TEXT ·gcmAesFinish(SB),NOSPLIT,$0
#define pTbl R0
#define tMsk R1
#define tPtr R2
#define plen R3
#define dlen R4

	MOVD	$0xC2, R1
	LSL	$56, R1
	MOVD	$1, R0
	VMOV	R1, POLY.D[0]
	VMOV	R0, POLY.D[1]
	VEOR	ZERO.B16, ZERO.B16, ZERO.B16

	MOVD	productTable+0(FP), pTbl
	MOVD	tagMask+8(FP), tMsk
	MOVD	T+16(FP), tPtr
	MOVD	pLen+24(FP), plen
	MOVD	dLen+32(FP), dlen

	VLD1	(tPtr), [ACC0.B16]
	VLD1	(tMsk), [B1.B16]

	LSL	$3, plen
	LSL	$3, dlen

	VMOV	dlen, B0.D[0]
	VMOV	plen, B0.D[1]

	ADD	$14*16, pTbl
	VLD1.P	(pTbl), [T1.B16, T2.B16]

	VEOR	ACC0.B16, B0.B16, B0.B16

	VEXT	$8, B0.B16, B0.B16, T0.B16
	VEOR	B0.B16, T0.B16, T0.B16
	VPMULL	B0.D1, T1.D1, ACC1.Q1
	VPMULL2	B0.D2, T1.D2, ACC0.Q1
	VPMULL	T0.D1, T2.D1, ACCM.Q1

	reduce()

	VREV64	ACC0.B16, ACC0.B16
	VEOR	B1.B16, ACC0.B16, ACC0.B16

	VST1	[ACC0.B16], (tPtr)
	RET
#undef pTbl
#undef tMsk
#undef tPtr
#undef plen
#undef dlen

// func gcmAesInit(productTable *[256]byte, ks []uint32)
TEXT ·gcmAesInit(SB),NOSPLIT,$0
#define pTbl R0
#define KS R1
#define NR R2
#define I R3
	MOVD	productTable+0(FP), pTbl
	MOVD	ks_base+8(FP), KS
	MOVD	ks_len+16(FP), NR

	MOVD	$0xC2, I
	LSL	$56, I
	VMOV	I, POLY.D[0]
	MOVD	$1, I
	VMOV	I, POLY.D[1]
	VEOR	ZERO.B16, ZERO.B16, ZERO.B16

	// Encrypt block 0 with the AES key to generate the hash key H
	VLD1.P	64(KS), [T0.B16, T1.B16, T2.B16, T3.B16]
	VEOR	B0.B16, B0.B16, B0.B16
	AESE	T0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T1.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T2.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T3.B16, B0.B16
	AESMC	B0.B16, B0.B16
	VLD1.P	64(KS), [T0.B16, T1.B16, T2.B16, T3.B16]
	AESE	T0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T1.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T2.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T3.B16, B0.B16
	AESMC	B0.B16, B0.B16
	TBZ	$4, NR, initEncFinish
	VLD1.P	32(KS), [T0.B16, T1.B16]
	AESE	T0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T1.B16, B0.B16
	AESMC	B0.B16, B0.B16
	TBZ	$3, NR, initEncFinish
	VLD1.P	32(KS), [T0.B16, T1.B16]
	AESE	T0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T1.B16, B0.B16
	AESMC	B0.B16, B0.B16
initEncFinish:
	VLD1	(KS), [T0.B16, T1.B16, T2.B16]
	AESE	T0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	T1.B16, B0.B16
 	VEOR	T2.B16, B0.B16, B0.B16

	VREV64	B0.B16, B0.B16

	// Multiply by 2 modulo P
	VMOV	B0.D[0], I
	ASR	$63, I
	VMOV	I, T1.D[0]
	VMOV	I, T1.D[1]
	VAND	POLY.B16, T1.B16, T1.B16
	VUSHR	$63, B0.D2, T2.D2
	VEXT	$8, ZERO.B16, T2.B16, T2.B16
	VSHL	$1, B0.D2, B0.D2
	VEOR	T1.B16, B0.B16, B0.B16
	VEOR	T2.B16, B0.B16, B0.B16 // Can avoid this when VSLI is available

	// Karatsuba pre-computation
	VEXT	$8, B0.B16, B0.B16, B1.B16
	VEOR	B0.B16, B1.B16, B1.B16

	ADD	$14*16, pTbl
	VST1	[B0.B16, B1.B16], (pTbl)
	SUB	$2*16, pTbl

	VMOV	B0.B16, B2.B16
	VMOV	B1.B16, B3.B16

	MOVD	$7, I

initLoop:
	// Compute powers of H
	SUBS	$1, I

	VPMULL	B0.D1, B2.D1, T1.Q1
	VPMULL2	B0.D2, B2.D2, T0.Q1
	VPMULL	B1.D1, B3.D1, T2.Q1
	VEOR	T0.B16, T2.B16, T2.B16
	VEOR	T1.B16, T2.B16, T2.B16
	VEXT	$8, ZERO.B16, T2.B16, T3.B16
	VEXT	$8, T2.B16, ZERO.B16, T2.B16
	VEOR	T2.B16, T0.B16, T0.B16
	VEOR	T3.B16, T1.B16, T1.B16
	VPMULL	POLY.D1, T0.D1, T2.Q1
	VEXT	$8, T0.B16, T0.B16, T0.B16
	VEOR	T2.B16, T0.B16, T0.B16
	VPMULL	POLY.D1, T0.D1, T2.Q1
	VEXT	$8, T0.B16, T0.B16, T0.B16
	VEOR	T2.B16, T0.B16, T0.B16
	VEOR	T1.B16, T0.B16, B2.B16
	VMOV	B2.B16, B3.B16
	VEXT	$8, B2.B16, B2.B16, B2.B16
	VEOR	B2.B16, B3.B16, B3.B16

	VST1	[B2.B16, B3.B16], (pTbl)
	SUB	$2*16, pTbl

	BNE	initLoop
	RET
#undef I
#undef NR
#undef KS
#undef pTbl

// func gcmAesData(productTable *[256]byte, data []byte, T *[16]byte)
TEXT ·gcmAesData(SB),NOSPLIT,$0
#define pTbl R0
#define aut R1
#define tPtr R2
#define autLen R3
#define H0 R4
#define pTblSave R5

#define mulRound(X) \
	VLD1.P	32(pTbl), [T1.B16, T2.B16] \
	VREV64	X.B16, X.B16               \
	VEXT	$8, X.B16, X.B16, T0.B16   \
	VEOR	X.B16, T0.B16, T0.B16      \
	VPMULL	X.D1, T1.D1, T3.Q1         \
	VEOR	T3.B16, ACC1.B16, ACC1.B16 \
	VPMULL2	X.D2, T1.D2, T3.Q1         \
	VEOR	T3.B16, ACC0.B16, ACC0.B16 \
	VPMULL	T0.D1, T2.D1, T3.Q1        \
	VEOR	T3.B16, ACCM.B16, ACCM.B16

	MOVD	productTable+0(FP), pTbl
	MOVD	data_base+8(FP), aut
	MOVD	data_len+16(FP), autLen
	MOVD	T+32(FP), tPtr

	VEOR	ACC0.B16, ACC0.B16, ACC0.B16
	CBZ	autLen, dataBail

	MOVD	$0xC2, H0
	LSL	$56, H0
	VMOV	H0, POLY.D[0]
	MOVD	$1, H0
	VMOV	H0, POLY.D[1]
	VEOR	ZERO.B16, ZERO.B16, ZERO.B16
	MOVD	pTbl, pTblSave

	CMP	$13, autLen
	BEQ	dataTLS
	CMP	$128, autLen
	BLT	startSinglesLoop
	B	octetsLoop

dataTLS:
	ADD	$14*16, pTbl
	VLD1.P	(pTbl), [T1.B16, T2.B16]
	VEOR	B0.B16, B0.B16, B0.B16

	MOVD	(aut), H0
	VMOV	H0, B0.D[0]
	MOVW	8(aut), H0
	VMOV	H0, B0.S[2]
	MOVB	12(aut), H0
	VMOV	H0, B0.B[12]

	MOVD	$0, autLen
	B	dataMul

octetsLoop:
		CMP	$128, autLen
		BLT	startSinglesLoop
		SUB	$128, autLen

		VLD1.P	32(aut), [B0.B16, B1.B16]

		VLD1.P	32(pTbl), [T1.B16, T2.B16]
		VREV64	B0.B16, B0.B16
		VEOR	ACC0.B16, B0.B16, B0.B16
		VEXT	$8, B0.B16, B0.B16, T0.B16
		VEOR	B0.B16, T0.B16, T0.B16
		VPMULL	B0.D1, T1.D1, ACC1.Q1
		VPMULL2	B0.D2, T1.D2, ACC0.Q1
		VPMULL	T0.D1, T2.D1, ACCM.Q1

		mulRound(B1)
		VLD1.P  32(aut), [B2.B16, B3.B16]
		mulRound(B2)
		mulRound(B3)
		VLD1.P  32(aut), [B4.B16, B5.B16]
		mulRound(B4)
		mulRound(B5)
		VLD1.P  32(aut), [B6.B16, B7.B16]
		mulRound(B6)
		mulRound(B7)

		MOVD	pTblSave, pTbl
		reduce()
	B	octetsLoop

startSinglesLoop:

	ADD	$14*16, pTbl
	VLD1.P	(pTbl), [T1.B16, T2.B16]

singlesLoop:

		CMP	$16, autLen
		BLT	dataEnd
		SUB	$16, autLen

		VLD1.P	16(aut), [B0.B16]
dataMul:
		VREV64	B0.B16, B0.B16
		VEOR	ACC0.B16, B0.B16, B0.B16

		VEXT	$8, B0.B16, B0.B16, T0.B16
		VEOR	B0.B16, T0.B16, T0.B16
		VPMULL	B0.D1, T1.D1, ACC1.Q1
		VPMULL2	B0.D2, T1.D2, ACC0.Q1
		VPMULL	T0.D1, T2.D1, ACCM.Q1

		reduce()

	B	singlesLoop

dataEnd:

	CBZ	autLen, dataBail
	VEOR	B0.B16, B0.B16, B0.B16
	ADD	autLen, aut

dataLoadLoop:
		MOVB.W	-1(aut), H0
		VEXT	$15, B0.B16, ZERO.B16, B0.B16
		VMOV	H0, B0.B[0]
		SUBS	$1, autLen
		BNE	dataLoadLoop
	B	dataMul

dataBail:
	VST1	[ACC0.B16], (tPtr)
	RET

#undef pTbl
#undef aut
#undef tPtr
#undef autLen
#undef H0
#undef pTblSave

// func gcmAesEnc(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)
TEXT ·gcmAesEnc(SB),NOSPLIT,$0
#define pTbl R0
#define dstPtr R1
#define ctrPtr R2
#define srcPtr R3
#define ks R4
#define tPtr R5
#define srcPtrLen R6
#define aluCTR R7
#define aluTMP R8
#define aluK R9
#define NR R10
#define H0 R11
#define H1 R12
#define curK R13
#define pTblSave R14

#define aesrndx8(K) \
	AESE	K.B16, B0.B16    \
	AESMC	B0.B16, B0.B16   \
	AESE	K.B16, B1.B16    \
	AESMC	B1.B16, B1.B16   \
	AESE	K.B16, B2.B16    \
	AESMC	B2.B16, B2.B16   \
	AESE	K.B16, B3.B16    \
	AESMC	B3.B16, B3.B16   \
	AESE	K.B16, B4.B16    \
	AESMC	B4.B16, B4.B16   \
	AESE	K.B16, B5.B16    \
	AESMC	B5.B16, B5.B16   \
	AESE	K.B16, B6.B16    \
	AESMC	B6.B16, B6.B16   \
	AESE	K.B16, B7.B16    \
	AESMC	B7.B16, B7.B16

#define aesrndlastx8(K) \
	AESE	K.B16, B0.B16    \
	AESE	K.B16, B1.B16    \
	AESE	K.B16, B2.B16    \
	AESE	K.B16, B3.B16    \
	AESE	K.B16, B4.B16    \
	AESE	K.B16, B5.B16    \
	AESE	K.B16, B6.B16    \
	AESE	K.B16, B7.B16

	MOVD	productTable+0(FP), pTbl
	MOVD	dst+8(FP), dstPtr
	MOVD	src_base+32(FP), srcPtr
	MOVD	src_len+40(FP), srcPtrLen
	MOVD	ctr+56(FP), ctrPtr
	MOVD	T+64(FP), tPtr
	MOVD	ks_base+72(FP), ks
	MOVD	ks_len+80(FP), NR

	MOVD	$0xC2, H1
	LSL	$56, H1
	MOVD	$1, H0
	VMOV	H1, POLY.D[0]
	VMOV	H0, POLY.D[1]
	VEOR	ZERO.B16, ZERO.B16, ZERO.B16
	// Compute NR from len(ks)
	MOVD	pTbl, pTblSave
	// Current tag, after AAD
	VLD1	(tPtr), [ACC0.B16]
	VEOR	ACC1.B16, ACC1.B16, ACC1.B16
	VEOR	ACCM.B16, ACCM.B16, ACCM.B16
	// Prepare initial counter, and the increment vector
	VLD1	(ctrPtr), [CTR.B16]
	VEOR	INC.B16, INC.B16, INC.B16
	MOVD	$1, H0
	VMOV	H0, INC.S[3]
	VREV32	CTR.B16, CTR.B16
	VADD	CTR.S4, INC.S4, CTR.S4
	// Skip to <8 blocks loop
	CMP	$128, srcPtrLen

	MOVD	ks, H0
	// For AES-128 round keys are stored in: K0 .. K10, KLAST
	VLD1.P	64(H0), [K0.B16, K1.B16, K2.B16, K3.B16]
	VLD1.P	64(H0), [K4.B16, K5.B16, K6.B16, K7.B16]
	VLD1.P	48(H0), [K8.B16, K9.B16, K10.B16]
	VMOV	K10.B16, KLAST.B16

	BLT	startSingles
	// There are at least 8 blocks to encrypt
	TBZ	$4, NR, octetsLoop

	// For AES-192 round keys occupy: K0 .. K7, K10, K11, K8, K9, KLAST
	VMOV	K8.B16, K10.B16
	VMOV	K9.B16, K11.B16
	VMOV	KLAST.B16, K8.B16
	VLD1.P	16(H0), [K9.B16]
	VLD1.P  16(H0), [KLAST.B16]
	TBZ	$3, NR, octetsLoop
	// For AES-256 round keys occupy: K0 .. K7, K10, K11, mem, mem, K8, K9, KLAST
	VMOV	KLAST.B16, K8.B16
	VLD1.P	16(H0), [K9.B16]
	VLD1.P  16(H0), [KLAST.B16]
	ADD	$10*16, ks, H0
	MOVD	H0, curK

octetsLoop:
		SUB	$128, srcPtrLen

		VMOV	CTR.B16, B0.B16
		VADD	B0.S4, INC.S4, B1.S4
		VREV32	B0.B16, B0.B16
		VADD	B1.S4, INC.S4, B2.S4
		VREV32	B1.B16, B1.B16
		VADD	B2.S4, INC.S4, B3.S4
		VREV32	B2.B16, B2.B16
		VADD	B3.S4, INC.S4, B4.S4
		VREV32	B3.B16, B3.B16
		VADD	B4.S4, INC.S4, B5.S4
		VREV32	B4.B16, B4.B16
		VADD	B5.S4, INC.S4, B6.S4
		VREV32	B5.B16, B5.B16
		VADD	B6.S4, INC.S4, B7.S4
		VREV32	B6.B16, B6.B16
		VADD	B7.S4, INC.S4, CTR.S4
		VREV32	B7.B16, B7.B16

		aesrndx8(K0)
		aesrndx8(K1)
		aesrndx8(K2)
		aesrndx8(K3)
		aesrndx8(K4)
		aesrndx8(K5)
		aesrndx8(K6)
		aesrndx8(K7)
		TBZ	$4, NR, octetsFinish
		aesrndx8(K10)
		aesrndx8(K11)
		TBZ	$3, NR, octetsFinish
		VLD1.P	32(curK), [T1.B16, T2.B16]
		aesrndx8(T1)
		aesrndx8(T2)
		MOVD	H0, curK
octetsFinish:
		aesrndx8(K8)
		aesrndlastx8(K9)

		VEOR	KLAST.B16, B0.B16, B0.B16
		VEOR	KLAST.B16, B1.B16, B1.B16
		VEOR	KLAST.B16, B2.B16, B2.B16
		VEOR	KLAST.B16, B3.B16, B3.B16
		VEOR	KLAST.B16, B4.B16, B4.B16
		VEOR	KLAST.B16, B5.B16, B5.B16
		VEOR	KLAST.B16, B6.B16, B6.B16
		VEOR	KLAST.B16, B7.B16, B7.B16

		VLD1.P	32(srcPtr), [T1.B16, T2.B16]
		VEOR	B0.B16, T1.B16, B0.B16
		VEOR	B1.B16, T2.B16, B1.B16
		VST1.P  [B0.B16, B1.B16], 32(dstPtr)
		VLD1.P	32(srcPtr), [T1.B16, T2.B16]
		VEOR	B2.B16, T1.B16, B2.B16
		VEOR	B3.B16, T2.B16, B3.B16
		VST1.P  [B2.B16, B3.B16], 32(dstPtr)
		VLD1.P	32(srcPtr), [T1.B16, T2.B16]
		VEOR	B4.B16, T1.B16, B4.B16
		VEOR	B5.B16, T2.B16, B5.B16
		VST1.P  [B4.B16, B5.B16], 32(dstPtr)
		VLD1.P	32(srcPtr), [T1.B16, T2.B16]
		VEOR	B6.B16, T1.B16, B6.B16
		VEOR	B7.B16, T2.B16, B7.B16
		VST1.P  [B6.B16, B7.B16], 32(dstPtr)

		VLD1.P	32(pTbl), [T1.B16, T2.B16]
		VREV64	B0.B16, B0.B16
		VEOR	ACC0.B16, B0.B16, B0.B16
		VEXT	$8, B0.B16, B0.B16, T0.B16
		VEOR	B0.B16, T0.B16, T0.B16
		VPMULL	B0.D1, T1.D1, ACC1.Q1
		VPMULL2	B0.D2, T1.D2, ACC0.Q1
		VPMULL	T0.D1, T2.D1, ACCM.Q1

		mulRound(B1)
		mulRound(B2)
		mulRound(B3)
		mulRound(B4)
		mulRound(B5)
		mulRound(B6)
		mulRound(B7)
		MOVD	pTblSave, pTbl
		reduce()

		CMP	$128, srcPtrLen
		BGE	octetsLoop

startSingles:
	CBZ	srcPtrLen, done
	ADD	$14*16, pTbl
	// Preload H and its Karatsuba precomp
	VLD1.P	(pTbl), [T1.B16, T2.B16]
	// Preload AES round keys
	ADD	$128, ks
	VLD1.P	48(ks), [K8.B16, K9.B16, K10.B16]
	VMOV	K10.B16, KLAST.B16
	TBZ	$4, NR, singlesLoop
	VLD1.P	32(ks), [B1.B16, B2.B16]
	VMOV	B2.B16, KLAST.B16
	TBZ	$3, NR, singlesLoop
	VLD1.P	32(ks), [B3.B16, B4.B16]
	VMOV	B4.B16, KLAST.B16

singlesLoop:
		CMP	$16, srcPtrLen
		BLT	tail
		SUB	$16, srcPtrLen

		VLD1.P	16(srcPtr), [T0.B16]
		VEOR	KLAST.B16, T0.B16, T0.B16

		VREV32	CTR.B16, B0.B16
		VADD	CTR.S4, INC.S4, CTR.S4

		AESE	K0.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K1.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K2.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K3.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K4.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K5.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K6.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K7.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K8.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K9.B16, B0.B16
		TBZ	$4, NR, singlesLast
		AESMC	B0.B16, B0.B16
		AESE	K10.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	B1.B16, B0.B16
		TBZ	$3, NR, singlesLast
		AESMC	B0.B16, B0.B16
		AESE	B2.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	B3.B16, B0.B16
singlesLast:
		VEOR	T0.B16, B0.B16, B0.B16
encReduce:
		VST1.P	[B0.B16], 16(dstPtr)

		VREV64	B0.B16, B0.B16
		VEOR	ACC0.B16, B0.B16, B0.B16

		VEXT	$8, B0.B16, B0.B16, T0.B16
		VEOR	B0.B16, T0.B16, T0.B16
		VPMULL	B0.D1, T1.D1, ACC1.Q1
		VPMULL2	B0.D2, T1.D2, ACC0.Q1
		VPMULL	T0.D1, T2.D1, ACCM.Q1

		reduce()

	B	singlesLoop
tail:
	CBZ	srcPtrLen, done

	VEOR	T0.B16, T0.B16, T0.B16
	VEOR	T3.B16, T3.B16, T3.B16
	MOVD	$0, H1
	SUB	$1, H1
	ADD	srcPtrLen, srcPtr

	TBZ	$3, srcPtrLen, ld4
	MOVD.W	-8(srcPtr), H0
	VMOV	H0, T0.D[0]
	VMOV	H1, T3.D[0]
ld4:
	TBZ	$2, srcPtrLen, ld2
	MOVW.W	-4(srcPtr), H0
	VEXT	$12, T0.B16, ZERO.B16, T0.B16
	VEXT	$12, T3.B16, ZERO.B16, T3.B16
	VMOV	H0, T0.S[0]
	VMOV	H1, T3.S[0]
ld2:
	TBZ	$1, srcPtrLen, ld1
	MOVH.W	-2(srcPtr), H0
	VEXT	$14, T0.B16, ZERO.B16, T0.B16
	VEXT	$14, T3.B16, ZERO.B16, T3.B16
	VMOV	H0, T0.H[0]
	VMOV	H1, T3.H[0]
ld1:
	TBZ	$0, srcPtrLen, ld0
	MOVB.W	-1(srcPtr), H0
	VEXT	$15, T0.B16, ZERO.B16, T0.B16
	VEXT	$15, T3.B16, ZERO.B16, T3.B16
	VMOV	H0, T0.B[0]
	VMOV	H1, T3.B[0]
ld0:

	MOVD	ZR, srcPtrLen
	VEOR	KLAST.B16, T0.B16, T0.B16
	VREV32	CTR.B16, B0.B16

	AESE	K0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K1.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K2.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K3.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K4.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K5.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K6.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K7.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K8.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K9.B16, B0.B16
	TBZ	$4, NR, tailLast
	AESMC	B0.B16, B0.B16
	AESE	K10.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	B1.B16, B0.B16
	TBZ	$3, NR, tailLast
	AESMC	B0.B16, B0.B16
	AESE	B2.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	B3.B16, B0.B16

tailLast:
	VEOR	T0.B16, B0.B16, B0.B16
	VAND	T3.B16, B0.B16, B0.B16
	B	encReduce

done:
	VST1	[ACC0.B16], (tPtr)
	RET

// func gcmAesDec(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)
TEXT ·gcmAesDec(SB),NOSPLIT,$0
	MOVD	productTable+0(FP), pTbl
	MOVD	dst+8(FP), dstPtr
	MOVD	src_base+32(FP), srcPtr
	MOVD	src_len+40(FP), srcPtrLen
	MOVD	ctr+56(FP), ctrPtr
	MOVD	T+64(FP), tPtr
	MOVD	ks_base+72(FP), ks
	MOVD	ks_len+80(FP), NR

	MOVD	$0xC2, H1
	LSL	$56, H1
	MOVD	$1, H0
	VMOV	H1, POLY.D[0]
	VMOV	H0, POLY.D[1]
	VEOR	ZERO.B16, ZERO.B16, ZERO.B16
	// Compute NR from len(ks)
	MOVD	pTbl, pTblSave
	// Current tag, after AAD
	VLD1	(tPtr), [ACC0.B16]
	VEOR	ACC1.B16, ACC1.B16, ACC1.B16
	VEOR	ACCM.B16, ACCM.B16, ACCM.B16
	// Prepare initial counter, and the increment vector
	VLD1	(ctrPtr), [CTR.B16]
	VEOR	INC.B16, INC.B16, INC.B16
	MOVD	$1, H0
	VMOV	H0, INC.S[3]
	VREV32	CTR.B16, CTR.B16
	VADD	CTR.S4, INC.S4, CTR.S4

	MOVD	ks, H0
	// For AES-128 round keys are stored in: K0 .. K10, KLAST
	VLD1.P	64(H0), [K0.B16, K1.B16, K2.B16, K3.B16]
	VLD1.P	64(H0), [K4.B16, K5.B16, K6.B16, K7.B16]
	VLD1.P	48(H0), [K8.B16, K9.B16, K10.B16]
	VMOV	K10.B16, KLAST.B16

	// Skip to <8 blocks loop
	CMP	$128, srcPtrLen
	BLT	startSingles
	// There are at least 8 blocks to encrypt
	TBZ	$4, NR, octetsLoop

	// For AES-192 round keys occupy: K0 .. K7, K10, K11, K8, K9, KLAST
	VMOV	K8.B16, K10.B16
	VMOV	K9.B16, K11.B16
	VMOV	KLAST.B16, K8.B16
	VLD1.P	16(H0), [K9.B16]
	VLD1.P  16(H0), [KLAST.B16]
	TBZ	$3, NR, octetsLoop
	// For AES-256 round keys occupy: K0 .. K7, K10, K11, mem, mem, K8, K9, KLAST
	VMOV	KLAST.B16, K8.B16
	VLD1.P	16(H0), [K9.B16]
	VLD1.P  16(H0), [KLAST.B16]
	ADD	$10*16, ks, H0
	MOVD	H0, curK

octetsLoop:
		SUB	$128, srcPtrLen

		VMOV	CTR.B16, B0.B16
		VADD	B0.S4, INC.S4, B1.S4
		VREV32	B0.B16, B0.B16
		VADD	B1.S4, INC.S4, B2.S4
		VREV32	B1.B16, B1.B16
		VADD	B2.S4, INC.S4, B3.S4
		VREV32	B2.B16, B2.B16
		VADD	B3.S4, INC.S4, B4.S4
		VREV32	B3.B16, B3.B16
		VADD	B4.S4, INC.S4, B5.S4
		VREV32	B4.B16, B4.B16
		VADD	B5.S4, INC.S4, B6.S4
		VREV32	B5.B16, B5.B16
		VADD	B6.S4, INC.S4, B7.S4
		VREV32	B6.B16, B6.B16
		VADD	B7.S4, INC.S4, CTR.S4
		VREV32	B7.B16, B7.B16

		aesrndx8(K0)
		aesrndx8(K1)
		aesrndx8(K2)
		aesrndx8(K3)
		aesrndx8(K4)
		aesrndx8(K5)
		aesrndx8(K6)
		aesrndx8(K7)
		TBZ	$4, NR, octetsFinish
		aesrndx8(K10)
		aesrndx8(K11)
		TBZ	$3, NR, octetsFinish
		VLD1.P	32(curK), [T1.B16, T2.B16]
		aesrndx8(T1)
		aesrndx8(T2)
		MOVD	H0, curK
octetsFinish:
		aesrndx8(K8)
		aesrndlastx8(K9)

		VEOR	KLAST.B16, B0.B16, T1.B16
		VEOR	KLAST.B16, B1.B16, T2.B16
		VEOR	KLAST.B16, B2.B16, B2.B16
		VEOR	KLAST.B16, B3.B16, B3.B16
		VEOR	KLAST.B16, B4.B16, B4.B16
		VEOR	KLAST.B16, B5.B16, B5.B16
		VEOR	KLAST.B16, B6.B16, B6.B16
		VEOR	KLAST.B16, B7.B16, B7.B16

		VLD1.P	32(srcPtr), [B0.B16, B1.B16]
		VEOR	B0.B16, T1.B16, T1.B16
		VEOR	B1.B16, T2.B16, T2.B16
		VST1.P  [T1.B16, T2.B16], 32(dstPtr)

		VLD1.P	32(pTbl), [T1.B16, T2.B16]
		VREV64	B0.B16, B0.B16
		VEOR	ACC0.B16, B0.B16, B0.B16
		VEXT	$8, B0.B16, B0.B16, T0.B16
		VEOR	B0.B16, T0.B16, T0.B16
		VPMULL	B0.D1, T1.D1, ACC1.Q1
		VPMULL2	B0.D2, T1.D2, ACC0.Q1
		VPMULL	T0.D1, T2.D1, ACCM.Q1
		mulRound(B1)

		VLD1.P	32(srcPtr), [B0.B16, B1.B16]
		VEOR	B2.B16, B0.B16, T1.B16
		VEOR	B3.B16, B1.B16, T2.B16
		VST1.P  [T1.B16, T2.B16], 32(dstPtr)
		mulRound(B0)
		mulRound(B1)

		VLD1.P	32(srcPtr), [B0.B16, B1.B16]
		VEOR	B4.B16, B0.B16, T1.B16
		VEOR	B5.B16, B1.B16, T2.B16
		VST1.P  [T1.B16, T2.B16], 32(dstPtr)
		mulRound(B0)
		mulRound(B1)

		VLD1.P	32(srcPtr), [B0.B16, B1.B16]
		VEOR	B6.B16, B0.B16, T1.B16
		VEOR	B7.B16, B1.B16, T2.B16
		VST1.P  [T1.B16, T2.B16], 32(dstPtr)
		mulRound(B0)
		mulRound(B1)

		MOVD	pTblSave, pTbl
		reduce()

		CMP	$128, srcPtrLen
		BGE	octetsLoop

startSingles:
	CBZ	srcPtrLen, done
	ADD	$14*16, pTbl
	// Preload H and its Karatsuba precomp
	VLD1.P	(pTbl), [T1.B16, T2.B16]
	// Preload AES round keys
	ADD	$128, ks
	VLD1.P	48(ks), [K8.B16, K9.B16, K10.B16]
	VMOV	K10.B16, KLAST.B16
	TBZ	$4, NR, singlesLoop
	VLD1.P	32(ks), [B1.B16, B2.B16]
	VMOV	B2.B16, KLAST.B16
	TBZ	$3, NR, singlesLoop
	VLD1.P	32(ks), [B3.B16, B4.B16]
	VMOV	B4.B16, KLAST.B16

singlesLoop:
		CMP	$16, srcPtrLen
		BLT	tail
		SUB	$16, srcPtrLen

		VLD1.P	16(srcPtr), [T0.B16]
		VREV64	T0.B16, B5.B16
		VEOR	KLAST.B16, T0.B16, T0.B16

		VREV32	CTR.B16, B0.B16
		VADD	CTR.S4, INC.S4, CTR.S4

		AESE	K0.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K1.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K2.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K3.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K4.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K5.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K6.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K7.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K8.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	K9.B16, B0.B16
		TBZ	$4, NR, singlesLast
		AESMC	B0.B16, B0.B16
		AESE	K10.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	B1.B16, B0.B16
		TBZ	$3, NR, singlesLast
		AESMC	B0.B16, B0.B16
		AESE	B2.B16, B0.B16
		AESMC	B0.B16, B0.B16
		AESE	B3.B16, B0.B16
singlesLast:
		VEOR	T0.B16, B0.B16, B0.B16

		VST1.P	[B0.B16], 16(dstPtr)

		VEOR	ACC0.B16, B5.B16, B5.B16
		VEXT	$8, B5.B16, B5.B16, T0.B16
		VEOR	B5.B16, T0.B16, T0.B16
		VPMULL	B5.D1, T1.D1, ACC1.Q1
		VPMULL2	B5.D2, T1.D2, ACC0.Q1
		VPMULL	T0.D1, T2.D1, ACCM.Q1
		reduce()

	B	singlesLoop
tail:
	CBZ	srcPtrLen, done

	VREV32	CTR.B16, B0.B16
	VADD	CTR.S4, INC.S4, CTR.S4

	AESE	K0.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K1.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K2.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K3.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K4.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K5.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K6.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K7.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K8.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	K9.B16, B0.B16
	TBZ	$4, NR, tailLast
	AESMC	B0.B16, B0.B16
	AESE	K10.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	B1.B16, B0.B16
	TBZ	$3, NR, tailLast
	AESMC	B0.B16, B0.B16
	AESE	B2.B16, B0.B16
	AESMC	B0.B16, B0.B16
	AESE	B3.B16, B0.B16
tailLast:
	VEOR	KLAST.B16, B0.B16, B0.B16

	// Assuming it is safe to load past dstPtr due to the presence of the tag
	VLD1	(srcPtr), [B5.B16]

	VEOR	B5.B16, B0.B16, B0.B16

	VEOR	T3.B16, T3.B16, T3.B16
	MOVD	$0, H1
	SUB	$1, H1

	TBZ	$3, srcPtrLen, ld4
	VMOV	B0.D[0], H0
	MOVD.P	H0, 8(dstPtr)
	VMOV	H1, T3.D[0]
	VEXT	$8, ZERO.B16, B0.B16, B0.B16
ld4:
	TBZ	$2, srcPtrLen, ld2
	VMOV	B0.S[0], H0
	MOVW.P	H0, 4(dstPtr)
	VEXT	$12, T3.B16, ZERO.B16, T3.B16
	VMOV	H1, T3.S[0]
	VEXT	$4, ZERO.B16, B0.B16, B0.B16
ld2:
	TBZ	$1, srcPtrLen, ld1
	VMOV	B0.H[0], H0
	MOVH.P	H0, 2(dstPtr)
	VEXT	$14, T3.B16, ZERO.B16, T3.B16
	VMOV	H1, T3.H[0]
	VEXT	$2, ZERO.B16, B0.B16, B0.B16
ld1:
	TBZ	$0, srcPtrLen, ld0
	VMOV	B0.B[0], H0
	MOVB.P	H0, 1(dstPtr)
	VEXT	$15, T3.B16, ZERO.B16, T3.B16
	VMOV	H1, T3.B[0]
ld0:

	VAND	T3.B16, B5.B16, B5.B16
	VREV64	B5.B16, B5.B16

	VEOR	ACC0.B16, B5.B16, B5.B16
	VEXT	$8, B5.B16, B5.B16, T0.B16
	VEOR	B5.B16, T0.B16, T0.B16
	VPMULL	B5.D1, T1.D1, ACC1.Q1
	VPMULL2	B5.D2, T1.D2, ACC0.Q1
	VPMULL	T0.D1, T2.D1, ACCM.Q1
	reduce()
done:
	VST1	[ACC0.B16], (tPtr)

	RET
