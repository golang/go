// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is an optimized implementation of AES-GCM using AES-NI and CLMUL-NI
// The implementation uses some optimization as described in:
// [1] Gueron, S., Kounavis, M.E.: Intel® Carry-Less Multiplication
//     Instruction and its Usage for Computing the GCM Mode rev. 2.02
// [2] Gueron, S., Krasnov, V.: Speeding up Counter Mode in Software and
//     Hardware

package main

import (
	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/ir"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../../gcm_amd64.s

var (
	B0 VecPhysical = X0
	B1             = X1
	B2             = X2
	B3             = X3
	B4             = X4
	B5             = X5
	B6             = X6
	B7             = X7

	ACC0 VecPhysical = X8
	ACC1             = X9
	ACCM             = X10

	T0    VecPhysical = X11
	T1                = X12
	T2                = X13
	POLY              = X14
	BSWAP             = X15
)

func main() {
	Package("crypto/aes")
	ConstraintExpr("!purego")

	gcmAesFinish()
	gcmAesInit()
	gcmAesData()
	gcmAesEnc()
	gcmAesDec()

	Generate()
}

func gcmAesFinish() {
	Implement("gcmAesFinish")
	Attributes(NOSPLIT)
	AllocLocal(0)

	var (
		pTbl GPPhysical = RDI
		tMsk            = RSI
		tPtr            = RDX
		plen            = RAX
		dlen            = RCX
	)

	Load(Param("productTable"), pTbl)
	Load(Param("tagMask"), tMsk)
	Load(Param("T"), tPtr)
	Load(Param("pLen"), plen)
	Load(Param("dLen"), dlen)

	MOVOU(Mem{Base: tPtr}, ACC0)
	MOVOU(Mem{Base: tMsk}, T2)

	bswapMask := bswapMask_DATA()
	gcmPoly := gcmPoly_DATA()
	MOVOU(bswapMask, BSWAP)
	MOVOU(gcmPoly, POLY)

	SHLQ(Imm(3), plen)
	SHLQ(Imm(3), dlen)

	MOVQ(plen, B0)
	PINSRQ(Imm(1), dlen, B0)

	PXOR(ACC0, B0)

	MOVOU(Mem{Base: pTbl}.Offset(16*14), ACC0)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), ACCM)
	MOVOU(ACC0, ACC1)

	PCLMULQDQ(Imm(0x00), B0, ACC0)
	PCLMULQDQ(Imm(0x11), B0, ACC1)
	PSHUFD(Imm(78), B0, T0)
	PXOR(B0, T0)
	PCLMULQDQ(Imm(0x00), T0, ACCM)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	MOVOU(POLY, T0)
	PCLMULQDQ(Imm(0x01), ACC0, T0)
	PSHUFD(Imm(78), ACC0, ACC0)
	PXOR(T0, ACC0)

	MOVOU(POLY, T0)
	PCLMULQDQ(Imm(0x01), ACC0, T0)
	PSHUFD(Imm(78), ACC0, ACC0)
	PXOR(T0, ACC0)

	PXOR(ACC1, ACC0)

	PSHUFB(BSWAP, ACC0)
	PXOR(T2, ACC0)
	MOVOU(ACC0, Mem{Base: tPtr})

	RET()
}

func gcmAesInit() {
	Implement("gcmAesInit")
	Attributes(NOSPLIT)
	AllocLocal(0)

	var (
		dst GPPhysical = RDI
		KS             = RSI
		NR             = RDX
	)

	Load(Param("productTable"), dst)
	Load(Param("ks").Base(), KS)
	Load(Param("ks").Len(), NR)

	SHRQ(Imm(2), NR)
	DECQ(NR)

	bswapMask := bswapMask_DATA()
	gcmPoly := gcmPoly_DATA()
	MOVOU(bswapMask, BSWAP)
	MOVOU(gcmPoly, POLY)

	Comment("Encrypt block 0, with the AES key to generate the hash key H")
	MOVOU(Mem{Base: KS}.Offset(16*0), B0)
	MOVOU(Mem{Base: KS}.Offset(16*1), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*2), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*3), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*4), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*5), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*6), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*7), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*8), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*9), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("initEncLast"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*11), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*12), T0)
	JE(LabelRef("initEncLast"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*13), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: KS}.Offset(16*14), T0)

	initEncLast(dst)
	initLoop(dst)

	RET()
}

func initEncLast(dst GPPhysical) {
	Label("initEncLast")
	AESENCLAST(T0, B0)

	PSHUFB(BSWAP, B0)
	Comment("H * 2")
	PSHUFD(Imm(0xff), B0, T0)
	MOVOU(B0, T1)
	PSRAL(Imm(31), T0)
	PAND(POLY, T0)
	PSRLL(Imm(31), T1)
	PSLLDQ(Imm(4), T1)
	PSLLL(Imm(1), B0)
	PXOR(T0, B0)
	PXOR(T1, B0)
	Comment("Karatsuba pre-computations")
	MOVOU(B0, Mem{Base: dst}.Offset(16*14))
	PSHUFD(Imm(78), B0, B1)
	PXOR(B0, B1)
	MOVOU(B1, Mem{Base: dst}.Offset(16*15))

	MOVOU(B0, B2)
	MOVOU(B1, B3)
	Comment("Now prepare powers of H and pre-computations for them")
	MOVQ(U32(7), RAX)
}

func initLoop(dst GPPhysical) {
	Label("initLoop")
	MOVOU(B2, T0)
	MOVOU(B2, T1)
	MOVOU(B3, T2)
	PCLMULQDQ(Imm(0x00), B0, T0)
	PCLMULQDQ(Imm(0x11), B0, T1)
	PCLMULQDQ(Imm(0x00), B1, T2)

	PXOR(T0, T2)
	PXOR(T1, T2)
	MOVOU(T2, B4)
	PSLLDQ(Imm(8), B4)
	PSRLDQ(Imm(8), T2)
	PXOR(B4, T0)
	PXOR(T2, T1)

	MOVOU(POLY, B2)
	PCLMULQDQ(Imm(0x01), T0, B2)
	PSHUFD(Imm(78), T0, T0)
	PXOR(B2, T0)
	MOVOU(POLY, B2)
	PCLMULQDQ(Imm(0x01), T0, B2)
	PSHUFD(Imm(78), T0, T0)
	PXOR(T0, B2)
	PXOR(T1, B2)

	MOVOU(B2, Mem{Base: dst}.Offset(16*12))
	PSHUFD(Imm(78), B2, B3)
	PXOR(B2, B3)
	MOVOU(B3, Mem{Base: dst}.Offset(16*13))

	DECQ(RAX)
	LEAQ(Mem{Base: dst}.Offset(-16*2), dst)
	JNE(LabelRef("initLoop"))
}

func gcmAesData() {
	Implement("gcmAesData")
	Attributes(NOSPLIT)
	AllocLocal(0)

	var (
		pTbl   GPPhysical = RDI
		aut               = RSI
		tPtr              = RCX
		autLen            = RDX
	)

	Load(Param("productTable"), pTbl)
	Load(Param("data").Base(), aut)
	Load(Param("data").Len(), autLen)
	Load(Param("T"), tPtr)

	bswapMask := bswapMask_DATA()
	gcmPoly := gcmPoly_DATA()
	PXOR(ACC0, ACC0)
	MOVOU(bswapMask, BSWAP)
	MOVOU(gcmPoly, POLY)

	TESTQ(autLen, autLen)
	JEQ(LabelRef("dataBail"))

	CMPQ(autLen, Imm(13)) // optimize the TLS case
	JE(LabelRef("dataTLS"))
	CMPQ(autLen, Imm(128))
	JB(LabelRef("startSinglesLoop"))
	JMP(LabelRef("dataOctaLoop"))

	dataTLS(pTbl, aut, autLen)
	dataOctaLoop(pTbl, aut, autLen)
	startSinglesLoop(pTbl)
	dataSinglesLoop(aut, autLen)
	dataMul(aut)
	dataEnd(aut, autLen)
	dataLoadLoop(aut, autLen)
	dataBail(tPtr)
}

func reduceRound(a VecPhysical) {
	MOVOU(POLY, T0)
	PCLMULQDQ(Imm(0x01), a, T0)
	PSHUFD(Imm(78), a, a)
	PXOR(T0, a)
}

func mulRoundAAD(X VecPhysical, i int, pTbl GPPhysical) {
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2)), T1)
	MOVOU(T1, T2)
	PCLMULQDQ(Imm(0x00), X, T1)
	PXOR(T1, ACC0)
	PCLMULQDQ(Imm(0x11), X, T2)
	PXOR(T2, ACC1)
	PSHUFD(Imm(78), X, T1)
	PXOR(T1, X)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2+1)), T1)
	PCLMULQDQ(Imm(0x00), X, T1)
	PXOR(T1, ACCM)
}

func dataTLS(pTbl, aut, autLen GPPhysical) {
	Label("dataTLS")
	MOVOU(Mem{Base: pTbl}.Offset(16*14), T1)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), T2)
	PXOR(B0, B0)
	MOVQ(Mem{Base: aut}, B0)
	PINSRD(Imm(2), Mem{Base: aut}.Offset(8), B0)
	PINSRB(Imm(12), Mem{Base: aut}.Offset(12), B0)
	XORQ(autLen, autLen)
	JMP(LabelRef("dataMul"))
}

func dataOctaLoop(pTbl, aut, autLen GPPhysical) {
	Label("dataOctaLoop")
	CMPQ(autLen, Imm(128))
	JB(LabelRef("startSinglesLoop"))
	SUBQ(Imm(128), autLen)

	MOVOU(Mem{Base: aut}.Offset(16*0), X0)
	MOVOU(Mem{Base: aut}.Offset(16*1), X1)
	MOVOU(Mem{Base: aut}.Offset(16*2), X2)
	MOVOU(Mem{Base: aut}.Offset(16*3), X3)
	MOVOU(Mem{Base: aut}.Offset(16*4), X4)
	MOVOU(Mem{Base: aut}.Offset(16*5), X5)
	MOVOU(Mem{Base: aut}.Offset(16*6), X6)
	MOVOU(Mem{Base: aut}.Offset(16*7), X7)
	LEAQ(Mem{Base: aut}.Offset(16*8), aut)
	PSHUFB(BSWAP, X0)
	PSHUFB(BSWAP, X1)
	PSHUFB(BSWAP, X2)
	PSHUFB(BSWAP, X3)
	PSHUFB(BSWAP, X4)
	PSHUFB(BSWAP, X5)
	PSHUFB(BSWAP, X6)
	PSHUFB(BSWAP, X7)
	PXOR(ACC0, X0)

	MOVOU(Mem{Base: pTbl}.Offset(16*0), ACC0)
	MOVOU(Mem{Base: pTbl}.Offset(16*1), ACCM)
	MOVOU(ACC0, ACC1)
	PSHUFD(Imm(78), X0, T1)
	PXOR(X0, T1)
	PCLMULQDQ(Imm(0x00), X0, ACC0)
	PCLMULQDQ(Imm(0x11), X0, ACC1)
	PCLMULQDQ(Imm(0x00), T1, ACCM)

	mulRoundAAD(X1, 1, pTbl)
	mulRoundAAD(X2, 2, pTbl)
	mulRoundAAD(X3, 3, pTbl)
	mulRoundAAD(X4, 4, pTbl)
	mulRoundAAD(X5, 5, pTbl)
	mulRoundAAD(X6, 6, pTbl)
	mulRoundAAD(X7, 7, pTbl)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)
	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR(ACC1, ACC0)
	JMP(LabelRef("dataOctaLoop"))
}

func startSinglesLoop(pTbl GPPhysical) {
	Label("startSinglesLoop")
	MOVOU(Mem{Base: pTbl}.Offset(16*14), T1)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), T2)

}

func dataSinglesLoop(aut, autLen GPPhysical) {
	Label("dataSinglesLoop")

	CMPQ(autLen, Imm(16))
	JB(LabelRef("dataEnd"))
	SUBQ(Imm(16), autLen)

	MOVOU(Mem{Base: aut}, B0)
}

func dataMul(aut GPPhysical) {
	Label("dataMul")
	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)

	MOVOU(T1, ACC0)
	MOVOU(T2, ACCM)
	MOVOU(T1, ACC1)

	PSHUFD(Imm(78), B0, T0)
	PXOR(B0, T0)
	PCLMULQDQ(Imm(0x00), B0, ACC0)
	PCLMULQDQ(Imm(0x11), B0, ACC1)
	PCLMULQDQ(Imm(0x00), T0, ACCM)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	MOVOU(POLY, T0)
	PCLMULQDQ(Imm(0x01), ACC0, T0)
	PSHUFD(Imm(78), ACC0, ACC0)
	PXOR(T0, ACC0)

	MOVOU(POLY, T0)
	PCLMULQDQ(Imm(0x01), ACC0, T0)
	PSHUFD(Imm(78), ACC0, ACC0)
	PXOR(T0, ACC0)
	PXOR(ACC1, ACC0)

	LEAQ(Mem{Base: aut}.Offset(16), aut)

	JMP(LabelRef("dataSinglesLoop"))
}

func dataEnd(aut, autLen GPPhysical) {
	Label("dataEnd")

	TESTQ(autLen, autLen)
	JEQ(LabelRef("dataBail"))

	PXOR(B0, B0)
	// LEAQ -1(aut)(autLen*1), aut
	LEAQ(Mem{Base: aut, Index: autLen, Scale: 1}.Offset(-1), aut)
}

func dataLoadLoop(aut, autLen GPPhysical) {
	Label("dataLoadLoop")

	PSLLDQ(Imm(1), B0)
	PINSRB(Imm(0), Mem{Base: aut}, B0)

	LEAQ(Mem{Base: aut}.Offset(-1), aut)
	DECQ(autLen)
	JNE(LabelRef("dataLoadLoop"))

	JMP(LabelRef("dataMul"))
}

func dataBail(tPtr GPPhysical) {
	Label("dataBail")
	MOVOU(ACC0, Mem{Base: tPtr})
	RET()
}

func gcmAesEnc() {
	Implement("gcmAesEnc")
	Attributes(0)
	AllocLocal(256)

	var (
		pTbl   GPPhysical = RDI
		ctx               = RDX
		ctrPtr            = RCX
		ptx               = RSI
		ks                = RAX
		tPtr              = R8
		ptxLen            = R9
		aluCTR            = R10L
		aluTMP            = R11L
		aluK              = R12L
		NR                = R13
	)

	Load(Param("productTable"), pTbl)
	Load(Param("dst").Base(), ctx)
	Load(Param("src").Base(), ptx)
	Load(Param("src").Len(), ptxLen)
	Load(Param("ctr"), ctrPtr)
	Load(Param("T"), tPtr)
	Load(Param("ks").Base(), ks)
	Load(Param("ks").Len(), NR)

	SHRQ(Imm(2), NR)
	DECQ(NR)

	bswapMask := bswapMask_DATA()
	gcmPoly := gcmPoly_DATA()
	MOVOU(bswapMask, BSWAP)
	MOVOU(gcmPoly, POLY)

	MOVOU(Mem{Base: tPtr}, ACC0)
	PXOR(ACC1, ACC1)
	PXOR(ACCM, ACCM)
	MOVOU(Mem{Base: ctrPtr}, B0)
	MOVL(Mem{Base: ctrPtr}.Offset(3*4), aluCTR)
	MOVOU(Mem{Base: ks}, T0)
	MOVL(Mem{Base: ks}.Offset(3*4), aluK)
	BSWAPL(aluCTR)
	BSWAPL(aluK)

	PXOR(B0, T0)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+0*16))
	incrementEnc(0, aluCTR, aluTMP, aluK)

	CMPQ(ptxLen, Imm(128))
	JB(LabelRef("gcmAesEncSingles"))
	SUBQ(Imm(128), ptxLen)

	Comment("We have at least 8 blocks to encrypt, prepare the rest of the counters")
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+1*16))
	incrementEnc(1, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+2*16))
	incrementEnc(2, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+3*16))
	incrementEnc(3, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+4*16))
	incrementEnc(4, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+5*16))
	incrementEnc(5, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+6*16))
	incrementEnc(6, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(8*16+7*16))
	incrementEnc(7, aluCTR, aluTMP, aluK)

	MOVOU(Mem{Base: SP}.Offset(8*16+0*16), B0)
	MOVOU(Mem{Base: SP}.Offset(8*16+1*16), B1)
	MOVOU(Mem{Base: SP}.Offset(8*16+2*16), B2)
	MOVOU(Mem{Base: SP}.Offset(8*16+3*16), B3)
	MOVOU(Mem{Base: SP}.Offset(8*16+4*16), B4)
	MOVOU(Mem{Base: SP}.Offset(8*16+5*16), B5)
	MOVOU(Mem{Base: SP}.Offset(8*16+6*16), B6)
	MOVOU(Mem{Base: SP}.Offset(8*16+7*16), B7)

	aesRound(1, ks)
	incrementEnc(0, aluCTR, aluTMP, aluK)
	aesRound(2, ks)
	incrementEnc(1, aluCTR, aluTMP, aluK)
	aesRound(3, ks)
	incrementEnc(2, aluCTR, aluTMP, aluK)
	aesRound(4, ks)
	incrementEnc(3, aluCTR, aluTMP, aluK)
	aesRound(5, ks)
	incrementEnc(4, aluCTR, aluTMP, aluK)
	aesRound(6, ks)
	incrementEnc(5, aluCTR, aluTMP, aluK)
	aesRound(7, ks)
	incrementEnc(6, aluCTR, aluTMP, aluK)
	aesRound(8, ks)
	incrementEnc(7, aluCTR, aluTMP, aluK)
	aesRound(9, ks)
	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("encLast1"))
	aesRnd(T0)
	aesRound(11, ks)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("encLast1"))
	aesRnd(T0)
	aesRound(13, ks)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)

	encLast1(ctx, ptx)
	gcmAesEncOctetsLoop(pTbl, ks, ptxLen, aluCTR, aluTMP, aluK, NR)
	encLast2(ctx, ptx)
	gcmAesEncOctetsEnd(pTbl, ptxLen, aluCTR)
	gcmAesEncSingles(pTbl, ks)
	gcmAesEncSinglesLoop(ks, ptxLen, aluCTR, aluTMP, aluK, NR)
	encLast3(pTbl, ctx, ptx)
	gcmAesEncTail(ks, ptxLen, NR)
	encLast4(ptx, ptxLen, aluCTR, aluTMP)
	ptxLoadLoop(pTbl, ctx, ptx, ptxLen)
	gcmAesEncDone(tPtr)
}

func incrementEnc(i int, aluCTR, aluTMP, aluK GPPhysical) {
	ADDL(Imm(1), aluCTR)
	MOVL(aluCTR, aluTMP)
	XORL(aluK, aluTMP)
	BSWAPL(aluTMP)
	MOVL(aluTMP, Mem{Base: SP}.Offset(3*4+8*16+i*16))
}

func aesRnd(k VecPhysical) {
	AESENC(k, B0)
	AESENC(k, B1)
	AESENC(k, B2)
	AESENC(k, B3)
	AESENC(k, B4)
	AESENC(k, B5)
	AESENC(k, B6)
	AESENC(k, B7)
}

func aesRound(i int, ks GPPhysical) {
	// MOVOU (16*i)(ks), T0
	MOVOU(Mem{Base: ks}.Offset(16*i), T0)
	AESENC(T0, B0)
	AESENC(T0, B1)
	AESENC(T0, B2)
	AESENC(T0, B3)
	AESENC(T0, B4)
	AESENC(T0, B5)
	AESENC(T0, B6)
	AESENC(T0, B7)
}

func aesRndLast(k VecPhysical) {
	AESENCLAST(k, B0)
	AESENCLAST(k, B1)
	AESENCLAST(k, B2)
	AESENCLAST(k, B3)
	AESENCLAST(k, B4)
	AESENCLAST(k, B5)
	AESENCLAST(k, B6)
	AESENCLAST(k, B7)
}

func combinedRound(i int, pTbl, ks GPPhysical) {
	MOVOU(Mem{Base: ks}.Offset(16*i), T0)
	AESENC(T0, B0)
	AESENC(T0, B1)
	AESENC(T0, B2)
	AESENC(T0, B3)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2)), T1)
	MOVOU(T1, T2)
	AESENC(T0, B4)
	AESENC(T0, B5)
	AESENC(T0, B6)
	AESENC(T0, B7)
	MOVOU(Mem{Base: SP}.Offset(16*i), T0)
	PCLMULQDQ(Imm(0x00), T0, T1)
	PXOR(T1, ACC0)
	PSHUFD(Imm(78), T0, T1)
	PCLMULQDQ(Imm(0x11), T0, T2)
	PXOR(T1, T0)
	PXOR(T2, ACC1)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2+1)), T2)
	PCLMULQDQ(Imm(0x00), T2, T0)
	PXOR(T0, ACCM)
}

func mulRound(i int, pTbl GPPhysical) {
	MOVOU(Mem{Base: SP}.Offset(16*i), T0)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2)), T1)
	MOVOU(T1, T2)
	PCLMULQDQ(Imm(0x00), T0, T1)
	PXOR(T1, ACC0)
	PCLMULQDQ(Imm(0x11), T0, T2)
	PXOR(T2, ACC1)
	PSHUFD(Imm(78), T0, T1)
	PXOR(T1, T0)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2+1)), T1)
	PCLMULQDQ(Imm(0x00), T0, T1)
	PXOR(T1, ACCM)
}

func encLast1(ctx, ptx GPPhysical) {
	Label("encLast1")
	aesRndLast(T0)

	MOVOU(Mem{Base: ptx}.Offset(16*0), T0)
	PXOR(T0, B0)
	MOVOU(Mem{Base: ptx}.Offset(16*1), T0)
	PXOR(T0, B1)
	MOVOU(Mem{Base: ptx}.Offset(16*2), T0)
	PXOR(T0, B2)
	MOVOU(Mem{Base: ptx}.Offset(16*3), T0)
	PXOR(T0, B3)
	MOVOU(Mem{Base: ptx}.Offset(16*4), T0)
	PXOR(T0, B4)
	MOVOU(Mem{Base: ptx}.Offset(16*5), T0)
	PXOR(T0, B5)
	MOVOU(Mem{Base: ptx}.Offset(16*6), T0)
	PXOR(T0, B6)
	MOVOU(Mem{Base: ptx}.Offset(16*7), T0)
	PXOR(T0, B7)

	MOVOU(B0, Mem{Base: ctx}.Offset(16*0))
	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)
	MOVOU(B1, Mem{Base: ctx}.Offset(16*1))
	PSHUFB(BSWAP, B1)
	MOVOU(B2, Mem{Base: ctx}.Offset(16*2))
	PSHUFB(BSWAP, B2)
	MOVOU(B3, Mem{Base: ctx}.Offset(16*3))
	PSHUFB(BSWAP, B3)
	MOVOU(B4, Mem{Base: ctx}.Offset(16*4))
	PSHUFB(BSWAP, B4)
	MOVOU(B5, Mem{Base: ctx}.Offset(16*5))
	PSHUFB(BSWAP, B5)
	MOVOU(B6, Mem{Base: ctx}.Offset(16*6))
	PSHUFB(BSWAP, B6)
	MOVOU(B7, Mem{Base: ctx}.Offset(16*7))
	PSHUFB(BSWAP, B7)

	MOVOU(B0, Mem{Base: SP}.Offset(16*0))
	MOVOU(B1, Mem{Base: SP}.Offset(16*1))
	MOVOU(B2, Mem{Base: SP}.Offset(16*2))
	MOVOU(B3, Mem{Base: SP}.Offset(16*3))
	MOVOU(B4, Mem{Base: SP}.Offset(16*4))
	MOVOU(B5, Mem{Base: SP}.Offset(16*5))
	MOVOU(B6, Mem{Base: SP}.Offset(16*6))
	MOVOU(B7, Mem{Base: SP}.Offset(16*7))

	LEAQ(Mem{Base: ptx}.Offset(128), ptx)
	LEAQ(Mem{Base: ctx}.Offset(128), ctx)
}

func gcmAesEncOctetsLoop(pTbl, ks, ptxLen, aluCTR, aluTMP, aluK, NR GPPhysical) {
	Label("gcmAesEncOctetsLoop")

	CMPQ(ptxLen, Imm(128))
	JB(LabelRef("gcmAesEncOctetsEnd"))
	SUBQ(Imm(128), ptxLen)

	MOVOU(Mem{Base: SP}.Offset(8*16+0*16), B0)
	MOVOU(Mem{Base: SP}.Offset(8*16+1*16), B1)
	MOVOU(Mem{Base: SP}.Offset(8*16+2*16), B2)
	MOVOU(Mem{Base: SP}.Offset(8*16+3*16), B3)
	MOVOU(Mem{Base: SP}.Offset(8*16+4*16), B4)
	MOVOU(Mem{Base: SP}.Offset(8*16+5*16), B5)
	MOVOU(Mem{Base: SP}.Offset(8*16+6*16), B6)
	MOVOU(Mem{Base: SP}.Offset(8*16+7*16), B7)

	MOVOU(Mem{Base: SP}.Offset(16*0), T0)
	PSHUFD(Imm(78), T0, T1)
	PXOR(T0, T1)

	MOVOU(Mem{Base: pTbl}.Offset(16*0), ACC0)
	MOVOU(Mem{Base: pTbl}.Offset(16*1), ACCM)
	MOVOU(ACC0, ACC1)

	PCLMULQDQ(Imm(0x00), T1, ACCM)
	PCLMULQDQ(Imm(0x00), T0, ACC0)
	PCLMULQDQ(Imm(0x11), T0, ACC1)

	combinedRound(1, pTbl, ks)
	incrementEnc(0, aluCTR, aluTMP, aluK)
	combinedRound(2, pTbl, ks)
	incrementEnc(1, aluCTR, aluTMP, aluK)
	combinedRound(3, pTbl, ks)
	incrementEnc(2, aluCTR, aluTMP, aluK)
	combinedRound(4, pTbl, ks)
	incrementEnc(3, aluCTR, aluTMP, aluK)
	combinedRound(5, pTbl, ks)
	incrementEnc(4, aluCTR, aluTMP, aluK)
	combinedRound(6, pTbl, ks)
	incrementEnc(5, aluCTR, aluTMP, aluK)
	combinedRound(7, pTbl, ks)
	incrementEnc(6, aluCTR, aluTMP, aluK)

	aesRound(8, ks)
	incrementEnc(7, aluCTR, aluTMP, aluK)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	aesRound(9, ks)

	reduceRound(ACC0)
	PXOR(ACC1, ACC0)

	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("encLast2"))
	aesRnd(T0)
	aesRound(11, ks)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("encLast2"))
	aesRnd(T0)
	aesRound(13, ks)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)
}

func encLast2(ctx, ptx GPPhysical) {
	Label("encLast2")
	aesRndLast(T0)

	MOVOU(Mem{Base: ptx}.Offset(16*0), T0)
	PXOR(T0, B0)
	MOVOU(Mem{Base: ptx}.Offset(16*1), T0)
	PXOR(T0, B1)
	MOVOU(Mem{Base: ptx}.Offset(16*2), T0)
	PXOR(T0, B2)
	MOVOU(Mem{Base: ptx}.Offset(16*3), T0)
	PXOR(T0, B3)
	MOVOU(Mem{Base: ptx}.Offset(16*4), T0)
	PXOR(T0, B4)
	MOVOU(Mem{Base: ptx}.Offset(16*5), T0)
	PXOR(T0, B5)
	MOVOU(Mem{Base: ptx}.Offset(16*6), T0)
	PXOR(T0, B6)
	MOVOU(Mem{Base: ptx}.Offset(16*7), T0)
	PXOR(T0, B7)

	MOVOU(B0, Mem{Base: ctx}.Offset(16*0))
	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)
	MOVOU(B1, Mem{Base: ctx}.Offset(16*1))
	PSHUFB(BSWAP, B1)
	MOVOU(B2, Mem{Base: ctx}.Offset(16*2))
	PSHUFB(BSWAP, B2)
	MOVOU(B3, Mem{Base: ctx}.Offset(16*3))
	PSHUFB(BSWAP, B3)
	MOVOU(B4, Mem{Base: ctx}.Offset(16*4))
	PSHUFB(BSWAP, B4)
	MOVOU(B5, Mem{Base: ctx}.Offset(16*5))
	PSHUFB(BSWAP, B5)
	MOVOU(B6, Mem{Base: ctx}.Offset(16*6))
	PSHUFB(BSWAP, B6)
	MOVOU(B7, Mem{Base: ctx}.Offset(16*7))
	PSHUFB(BSWAP, B7)

	MOVOU(B0, Mem{Base: SP}.Offset(16*0))
	MOVOU(B1, Mem{Base: SP}.Offset(16*1))
	MOVOU(B2, Mem{Base: SP}.Offset(16*2))
	MOVOU(B3, Mem{Base: SP}.Offset(16*3))
	MOVOU(B4, Mem{Base: SP}.Offset(16*4))
	MOVOU(B5, Mem{Base: SP}.Offset(16*5))
	MOVOU(B6, Mem{Base: SP}.Offset(16*6))
	MOVOU(B7, Mem{Base: SP}.Offset(16*7))

	LEAQ(Mem{Base: ptx}.Offset(128), ptx)
	LEAQ(Mem{Base: ctx}.Offset(128), ctx)

	JMP(LabelRef("gcmAesEncOctetsLoop"))
}

func gcmAesEncOctetsEnd(pTbl, ptxLen, aluCTR GPPhysical) {
	Label("gcmAesEncOctetsEnd")

	MOVOU(Mem{Base: SP}.Offset(16*0), T0)
	MOVOU(Mem{Base: pTbl}.Offset(16*0), ACC0)
	MOVOU(Mem{Base: pTbl}.Offset(16*1), ACCM)
	MOVOU(ACC0, ACC1)
	PSHUFD(Imm(78), T0, T1)
	PXOR(T0, T1)
	PCLMULQDQ(Imm(0x00), T0, ACC0)
	PCLMULQDQ(Imm(0x11), T0, ACC1)
	PCLMULQDQ(Imm(0x00), T1, ACCM)

	mulRound(1, pTbl)
	mulRound(2, pTbl)
	mulRound(3, pTbl)
	mulRound(4, pTbl)
	mulRound(5, pTbl)
	mulRound(6, pTbl)
	mulRound(7, pTbl)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR(ACC1, ACC0)

	TESTQ(ptxLen, ptxLen)
	JE(LabelRef("gcmAesEncDone"))

	// Hack to get Avo to emit:
	// 	SUBQ $7, aluCTR`
	Instruction(&ir.Instruction{Opcode: "SUBQ", Operands: []Op{Imm(7), aluCTR}})
}

func gcmAesEncSingles(pTbl, ks GPPhysical) {
	Label("gcmAesEncSingles")

	MOVOU(Mem{Base: ks}.Offset(16*1), B1)
	MOVOU(Mem{Base: ks}.Offset(16*2), B2)
	MOVOU(Mem{Base: ks}.Offset(16*3), B3)
	MOVOU(Mem{Base: ks}.Offset(16*4), B4)
	MOVOU(Mem{Base: ks}.Offset(16*5), B5)
	MOVOU(Mem{Base: ks}.Offset(16*6), B6)
	MOVOU(Mem{Base: ks}.Offset(16*7), B7)

	MOVOU(Mem{Base: pTbl}.Offset(16*14), T2)
}

func gcmAesEncSinglesLoop(ks, ptxLen, aluCTR, aluTMP, aluK, NR GPPhysical) {
	Label("gcmAesEncSinglesLoop")

	CMPQ(ptxLen, Imm(16))
	JB(LabelRef("gcmAesEncTail"))
	SUBQ(Imm(16), ptxLen)

	MOVOU(Mem{Base: SP}.Offset(8*16+0*16), B0)
	incrementEnc(0, aluCTR, aluTMP, aluK)

	AESENC(B1, B0)
	AESENC(B2, B0)
	AESENC(B3, B0)
	AESENC(B4, B0)
	AESENC(B5, B0)
	AESENC(B6, B0)
	AESENC(B7, B0)
	MOVOU(Mem{Base: ks}.Offset(16*8), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*9), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("encLast3"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*11), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("encLast3"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*13), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)
}

func encLast3(pTbl, ctx, ptx GPPhysical) {
	Label("encLast3")
	AESENCLAST(T0, B0)

	MOVOU(Mem{Base: ptx}, T0)
	PXOR(T0, B0)
	MOVOU(B0, Mem{Base: ctx})

	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)

	MOVOU(T2, ACC0)
	MOVOU(T2, ACC1)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), ACCM)

	PSHUFD(Imm(78), B0, T0)
	PXOR(B0, T0)
	PCLMULQDQ(Imm(0x00), B0, ACC0)
	PCLMULQDQ(Imm(0x11), B0, ACC1)
	PCLMULQDQ(Imm(0x00), T0, ACCM)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR(ACC1, ACC0)

	LEAQ(Mem{Base: ptx}.Offset(16*1), ptx)
	LEAQ(Mem{Base: ctx}.Offset(16*1), ctx)

	JMP(LabelRef("gcmAesEncSinglesLoop"))
}

func gcmAesEncTail(ks, ptxLen, NR GPPhysical) {
	Label("gcmAesEncTail")
	TESTQ(ptxLen, ptxLen)
	JE(LabelRef("gcmAesEncDone"))

	MOVOU(Mem{Base: SP}.Offset(8*16+0*16), B0)
	AESENC(B1, B0)
	AESENC(B2, B0)
	AESENC(B3, B0)
	AESENC(B4, B0)
	AESENC(B5, B0)
	AESENC(B6, B0)
	AESENC(B7, B0)
	MOVOU(Mem{Base: ks}.Offset(16*8), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*9), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("encLast4"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*11), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("encLast4"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*13), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)
}

func encLast4(ptx, ptxLen, aluCTR, aluTMP GPPhysical) {
	Label("encLast4")
	AESENCLAST(T0, B0)
	MOVOU(B0, T0)

	LEAQ(Mem{Base: ptx, Index: ptxLen, Scale: 1}.Offset(-1), ptx)

	// Hack to get Avo to emit:
	// 	MOVQ ptxLen, aluTMP
	Instruction(&ir.Instruction{Opcode: "MOVQ", Operands: []Op{ptxLen, aluTMP}})
	// Hack to get Avo to emit:
	// 	SHLQ $4, aluTMP
	Instruction(&ir.Instruction{Opcode: "SHLQ", Operands: []Op{Imm(4), aluTMP}})

	andMask := andMask_DATA()
	// Hack to get Avo to emit:
	// 	LEAQ andMask<>(SB), aluCTR
	Instruction(&ir.Instruction{Opcode: "LEAQ", Operands: []Op{andMask, aluCTR}})
	MOVOU(Mem{Base: aluCTR, Index: aluTMP, Scale: 1}.Offset(-16), T1)

	PXOR(B0, B0)
}

func ptxLoadLoop(pTbl, ctx, ptx, ptxLen GPPhysical) {
	Label("ptxLoadLoop")
	PSLLDQ(Imm(1), B0)
	PINSRB(Imm(0), Mem{Base: ptx}, B0)
	LEAQ(Mem{Base: ptx}.Offset(-1), ptx)
	DECQ(ptxLen)
	JNE(LabelRef("ptxLoadLoop"))

	PXOR(T0, B0)
	PAND(T1, B0)
	MOVOU(B0, Mem{Base: ctx})

	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)

	MOVOU(T2, ACC0)
	MOVOU(T2, ACC1)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), ACCM)

	PSHUFD(Imm(78), B0, T0)
	PXOR(B0, T0)
	PCLMULQDQ(Imm(0x00), B0, ACC0)
	PCLMULQDQ(Imm(0x11), B0, ACC1)
	PCLMULQDQ(Imm(0x00), T0, ACCM)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR(ACC1, ACC0)
}

func gcmAesEncDone(tPtr GPPhysical) {
	Label("gcmAesEncDone")
	MOVOU(ACC0, Mem{Base: tPtr})
	RET()
}

func gcmAesDec() {
	Implement("gcmAesDec")
	Attributes(0)
	AllocLocal(128)

	var (
		pTbl   GPPhysical = RDI
		ctx               = RDX
		ctrPtr            = RCX
		ptx               = RSI
		ks                = RAX
		tPtr              = R8
		ptxLen            = R9
		aluCTR            = R10L
		aluTMP            = R11L
		aluK              = R12L
		NR                = R13
	)

	Load(Param("productTable"), pTbl)
	Load(Param("dst").Base(), ptx)
	Load(Param("src").Base(), ctx)
	Load(Param("src").Len(), ptxLen)
	Load(Param("ctr"), ctrPtr)
	Load(Param("T"), tPtr)
	Load(Param("ks").Base(), ks)
	Load(Param("ks").Len(), NR)

	SHRQ(Imm(2), NR)
	DECQ(NR)

	bswapMask := bswapMask_DATA()
	gcmPoly := gcmPoly_DATA()
	MOVOU(bswapMask, BSWAP)
	MOVOU(gcmPoly, POLY)

	MOVOU(Mem{Base: tPtr}, ACC0)
	PXOR(ACC1, ACC1)
	PXOR(ACCM, ACCM)
	MOVOU(Mem{Base: ctrPtr}, B0)
	MOVL(Mem{Base: ctrPtr}.Offset(3*4), aluCTR)
	MOVOU(Mem{Base: ks}, T0)
	MOVL(Mem{Base: ks}.Offset(3*4), aluK)
	BSWAPL(aluCTR)
	BSWAPL(aluK)

	PXOR(B0, T0)
	MOVOU(T0, Mem{Base: SP}.Offset(0*16))
	incrementDec(0, aluCTR, aluTMP, aluK)

	CMPQ(ptxLen, Imm(128))
	JB(LabelRef("gcmAesDecSingles"))

	MOVOU(T0, Mem{Base: SP}.Offset(1*16))
	incrementDec(1, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(2*16))
	incrementDec(2, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(3*16))
	incrementDec(3, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(4*16))
	incrementDec(4, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(5*16))
	incrementDec(5, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(6*16))
	incrementDec(6, aluCTR, aluTMP, aluK)
	MOVOU(T0, Mem{Base: SP}.Offset(7*16))
	incrementDec(7, aluCTR, aluTMP, aluK)

	gcmAesDecOctetsLoop(pTbl, ctx, ks, ptxLen, aluCTR, aluTMP, aluK, NR)
	decLast1(ctx, ptx)
	gcmAesDecEndOctets(aluCTR)
	gcmAesDecSingles(pTbl, ks)
	gcmAesDecSinglesLoop(pTbl, ctx, ks, ptxLen, aluCTR, aluTMP, aluK, NR)
	decLast2(ctx, ptx)
	gcmAesDecTail(pTbl, ctx, ks, ptxLen, aluCTR, aluTMP, aluK, NR)
	decLast3()
	ptxStoreLoop(ptx, ptxLen)
	gcmAesDecDone(tPtr)
}

func incrementDec(i int, aluCTR, aluTMP, aluK GPPhysical) {
	ADDL(Imm(1), aluCTR)
	MOVL(aluCTR, aluTMP)
	XORL(aluK, aluTMP)
	BSWAPL(aluTMP)
	MOVL(aluTMP, Mem{Base: SP}.Offset(3*4+i*16))
}

func combinedDecRound(i int, pTbl, ctx, ks GPPhysical) {
	MOVOU(Mem{Base: ks}.Offset(16*i), T0)
	AESENC(T0, B0)
	AESENC(T0, B1)
	AESENC(T0, B2)
	AESENC(T0, B3)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2)), T1)
	MOVOU(T1, T2)
	AESENC(T0, B4)
	AESENC(T0, B5)
	AESENC(T0, B6)
	AESENC(T0, B7)
	MOVOU(Mem{Base: ctx}.Offset(16*i), T0)
	PSHUFB(BSWAP, T0)
	PCLMULQDQ(Imm(0x00), T0, T1)
	PXOR(T1, ACC0)
	PSHUFD(Imm(78), T0, T1)
	PCLMULQDQ(Imm(0x11), T0, T2)
	PXOR(T1, T0)
	PXOR(T2, ACC1)
	MOVOU(Mem{Base: pTbl}.Offset(16*(i*2+1)), T2)
	PCLMULQDQ(Imm(0x00), T2, T0)
	PXOR(T0, ACCM)
}

func gcmAesDecOctetsLoop(pTbl, ctx, ks, ptxLen, aluCTR, aluTMP, aluK, NR GPPhysical) {
	Label("gcmAesDecOctetsLoop")

	CMPQ(ptxLen, Imm(128))
	JB(LabelRef("gcmAesDecEndOctets"))
	SUBQ(Imm(128), ptxLen)

	MOVOU(Mem{Base: SP}.Offset(0*16), B0)
	MOVOU(Mem{Base: SP}.Offset(1*16), B1)
	MOVOU(Mem{Base: SP}.Offset(2*16), B2)
	MOVOU(Mem{Base: SP}.Offset(3*16), B3)
	MOVOU(Mem{Base: SP}.Offset(4*16), B4)
	MOVOU(Mem{Base: SP}.Offset(5*16), B5)
	MOVOU(Mem{Base: SP}.Offset(6*16), B6)
	MOVOU(Mem{Base: SP}.Offset(7*16), B7)

	MOVOU(Mem{Base: ctx}.Offset(16*0), T0)
	PSHUFB(BSWAP, T0)
	PXOR(ACC0, T0)
	PSHUFD(Imm(78), T0, T1)
	PXOR(T0, T1)

	MOVOU(Mem{Base: pTbl}.Offset(16*0), ACC0)
	MOVOU(Mem{Base: pTbl}.Offset(16*1), ACCM)
	MOVOU(ACC0, ACC1)

	PCLMULQDQ(Imm(0x00), T1, ACCM)
	PCLMULQDQ(Imm(0x00), T0, ACC0)
	PCLMULQDQ(Imm(0x11), T0, ACC1)

	combinedDecRound(1, pTbl, ctx, ks)
	incrementDec(0, aluCTR, aluTMP, aluK)
	combinedDecRound(2, pTbl, ctx, ks)
	incrementDec(1, aluCTR, aluTMP, aluK)
	combinedDecRound(3, pTbl, ctx, ks)
	incrementDec(2, aluCTR, aluTMP, aluK)
	combinedDecRound(4, pTbl, ctx, ks)
	incrementDec(3, aluCTR, aluTMP, aluK)
	combinedDecRound(5, pTbl, ctx, ks)
	incrementDec(4, aluCTR, aluTMP, aluK)
	combinedDecRound(6, pTbl, ctx, ks)
	incrementDec(5, aluCTR, aluTMP, aluK)
	combinedDecRound(7, pTbl, ctx, ks)
	incrementDec(6, aluCTR, aluTMP, aluK)

	aesRound(8, ks)
	incrementDec(7, aluCTR, aluTMP, aluK)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	aesRound(9, ks)

	reduceRound(ACC0)
	PXOR(ACC1, ACC0)

	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("decLast1"))
	aesRnd(T0)
	aesRound(11, ks)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("decLast1"))
	aesRnd(T0)
	aesRound(13, ks)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)
}

func decLast1(ctx, ptx GPPhysical) {
	Label("decLast1")
	aesRndLast(T0)

	MOVOU(Mem{Base: ctx}.Offset(16*0), T0)
	PXOR(T0, B0)
	MOVOU(Mem{Base: ctx}.Offset(16*1), T0)
	PXOR(T0, B1)
	MOVOU(Mem{Base: ctx}.Offset(16*2), T0)
	PXOR(T0, B2)
	MOVOU(Mem{Base: ctx}.Offset(16*3), T0)
	PXOR(T0, B3)
	MOVOU(Mem{Base: ctx}.Offset(16*4), T0)
	PXOR(T0, B4)
	MOVOU(Mem{Base: ctx}.Offset(16*5), T0)
	PXOR(T0, B5)
	MOVOU(Mem{Base: ctx}.Offset(16*6), T0)
	PXOR(T0, B6)
	MOVOU(Mem{Base: ctx}.Offset(16*7), T0)
	PXOR(T0, B7)

	MOVOU(B0, Mem{Base: ptx}.Offset(16*0))
	MOVOU(B1, Mem{Base: ptx}.Offset(16*1))
	MOVOU(B2, Mem{Base: ptx}.Offset(16*2))
	MOVOU(B3, Mem{Base: ptx}.Offset(16*3))
	MOVOU(B4, Mem{Base: ptx}.Offset(16*4))
	MOVOU(B5, Mem{Base: ptx}.Offset(16*5))
	MOVOU(B6, Mem{Base: ptx}.Offset(16*6))
	MOVOU(B7, Mem{Base: ptx}.Offset(16*7))

	LEAQ(Mem{Base: ptx}.Offset(128), ptx)
	LEAQ(Mem{Base: ctx}.Offset(128), ctx)

	JMP(LabelRef("gcmAesDecOctetsLoop"))
}

func gcmAesDecEndOctets(aluCTR GPPhysical) {
	Label("gcmAesDecEndOctets")
	// Hack to make Avo emit:
	// 	SUBQ $7, aluCTR
	Instruction(&ir.Instruction{Opcode: "SUBQ", Operands: []Op{Imm(7), aluCTR}})
}

func gcmAesDecSingles(pTbl, ks GPPhysical) {
	Label("gcmAesDecSingles")

	MOVOU(Mem{Base: ks}.Offset(16*1), B1)
	MOVOU(Mem{Base: ks}.Offset(16*2), B2)
	MOVOU(Mem{Base: ks}.Offset(16*3), B3)
	MOVOU(Mem{Base: ks}.Offset(16*4), B4)
	MOVOU(Mem{Base: ks}.Offset(16*5), B5)
	MOVOU(Mem{Base: ks}.Offset(16*6), B6)
	MOVOU(Mem{Base: ks}.Offset(16*7), B7)

	MOVOU(Mem{Base: pTbl}.Offset(16*14), T2)
}

func gcmAesDecSinglesLoop(pTbl, ctx, ks, ptxLen, aluCTR, aluTMP, aluK, NR GPPhysical) {
	Label("gcmAesDecSinglesLoop")

	CMPQ(ptxLen, Imm(16))
	JB(LabelRef("gcmAesDecTail"))
	SUBQ(Imm(16), ptxLen)

	MOVOU(Mem{Base: ctx}, B0)
	MOVOU(B0, T1)
	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)

	MOVOU(T2, ACC0)
	MOVOU(T2, ACC1)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), ACCM)

	PCLMULQDQ(Imm(0x00), B0, ACC0)
	PCLMULQDQ(Imm(0x11), B0, ACC1)
	PSHUFD(Imm(78), B0, T0)
	PXOR(B0, T0)
	PCLMULQDQ(Imm(0x00), T0, ACCM)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR(ACC1, ACC0)

	MOVOU(Mem{Base: SP}.Offset(0*16), B0)
	incrementDec(0, aluCTR, aluTMP, aluK)
	AESENC(B1, B0)
	AESENC(B2, B0)
	AESENC(B3, B0)
	AESENC(B4, B0)
	AESENC(B5, B0)
	AESENC(B6, B0)
	AESENC(B7, B0)
	MOVOU(Mem{Base: ks}.Offset(16*8), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*9), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("decLast2"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*11), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("decLast2"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*13), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)
}

func decLast2(ctx, ptx GPPhysical) {
	Label("decLast2")
	AESENCLAST(T0, B0)

	PXOR(T1, B0)
	MOVOU(B0, Mem{Base: ptx})

	LEAQ(Mem{Base: ptx}.Offset(16*1), ptx)
	LEAQ(Mem{Base: ctx}.Offset(16*1), ctx)

	JMP(LabelRef("gcmAesDecSinglesLoop"))
}

func gcmAesDecTail(pTbl, ctx, ks, ptxLen, aluCTR, aluTMP, aluK, NR GPPhysical) {
	Label("gcmAesDecTail")

	TESTQ(ptxLen, ptxLen)
	JE(LabelRef("gcmAesDecDone"))

	// Hack to get Avo to emit:
	// 	MOVQ ptxLen, aluTMP
	Instruction(&ir.Instruction{Opcode: "MOVQ", Operands: []Op{ptxLen, aluTMP}})
	// Hack to get Avo to emit:
	// 	SHLQ $4, aluTMP
	Instruction(&ir.Instruction{Opcode: "SHLQ", Operands: []Op{Imm(4), aluTMP}})

	andMask := andMask_DATA()
	// Hack to get Avo to emit:
	// 	LEAQ andMask<>(SB), aluCTR
	Instruction(&ir.Instruction{Opcode: "LEAQ", Operands: []Op{andMask, aluCTR}})
	MOVOU(Mem{Base: aluCTR, Index: aluTMP, Scale: 1}.Offset(-16), T1)

	MOVOU(Mem{Base: ctx}, B0)
	PAND(T1, B0)

	MOVOU(B0, T1)
	PSHUFB(BSWAP, B0)
	PXOR(ACC0, B0)

	MOVOU(Mem{Base: pTbl}.Offset(16*14), ACC0)
	MOVOU(Mem{Base: pTbl}.Offset(16*15), ACCM)
	MOVOU(ACC0, ACC1)

	PCLMULQDQ(Imm(0x00), B0, ACC0)
	PCLMULQDQ(Imm(0x11), B0, ACC1)
	PSHUFD(Imm(78), B0, T0)
	PXOR(B0, T0)
	PCLMULQDQ(Imm(0x00), T0, ACCM)

	PXOR(ACC0, ACCM)
	PXOR(ACC1, ACCM)
	MOVOU(ACCM, T0)
	PSRLDQ(Imm(8), ACCM)
	PSLLDQ(Imm(8), T0)
	PXOR(ACCM, ACC1)
	PXOR(T0, ACC0)

	reduceRound(ACC0)
	reduceRound(ACC0)
	PXOR(ACC1, ACC0)

	MOVOU(Mem{Base: SP}.Offset(0*16), B0)
	incrementDec(0, aluCTR, aluTMP, aluK)
	AESENC(B1, B0)
	AESENC(B2, B0)
	AESENC(B3, B0)
	AESENC(B4, B0)
	AESENC(B5, B0)
	AESENC(B6, B0)
	AESENC(B7, B0)
	MOVOU(Mem{Base: ks}.Offset(16*8), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*9), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*10), T0)
	CMPQ(NR, Imm(12))
	JB(LabelRef("decLast3"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*11), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*12), T0)
	JE(LabelRef("decLast3"))
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*13), T0)
	AESENC(T0, B0)
	MOVOU(Mem{Base: ks}.Offset(16*14), T0)
}

func decLast3() {
	Label("decLast3")
	AESENCLAST(T0, B0)
	PXOR(T1, B0)
}

func ptxStoreLoop(ptx, ptxLen GPPhysical) {
	Label("ptxStoreLoop")
	PEXTRB(Imm(0), B0, Mem{Base: ptx})
	PSRLDQ(Imm(1), B0)
	LEAQ(Mem{Base: ptx}.Offset(1), ptx)
	DECQ(ptxLen)

	JNE(LabelRef("ptxStoreLoop"))
}

func gcmAesDecDone(tPtr GPPhysical) {
	Label("gcmAesDecDone")
	MOVOU(ACC0, Mem{Base: tPtr})
	RET()
}

// ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA SECTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

var bswapMask_DATA_ptr, gcmPoly_DATA_ptr, andMask_DATA_ptr *Mem

func bswapMask_DATA() Mem {
	if bswapMask_DATA_ptr != nil {
		return *bswapMask_DATA_ptr
	}

	bswapMask := GLOBL("bswapMask", NOPTR|RODATA)
	bswapMask_DATA_ptr = &bswapMask
	DATA(0x00, U64(0x08090a0b0c0d0e0f))
	DATA(0x08, U64(0x0001020304050607))

	return bswapMask
}

func gcmPoly_DATA() Mem {
	if gcmPoly_DATA_ptr != nil {
		return *gcmPoly_DATA_ptr
	}

	gcmPoly := GLOBL("gcmPoly", NOPTR|RODATA)
	gcmPoly_DATA_ptr = &gcmPoly
	DATA(0x00, U64(0x0000000000000001))
	DATA(0x08, U64(0xc200000000000000))

	return gcmPoly
}

var andMask_K = [30]uint64{
	0x00000000000000ff,
	0x0000000000000000,
	0x000000000000ffff,
	0x0000000000000000,
	0x0000000000ffffff,
	0x0000000000000000,
	0x00000000ffffffff,
	0x0000000000000000,
	0x000000ffffffffff,
	0x0000000000000000,
	0x0000ffffffffffff,
	0x0000000000000000,
	0x00ffffffffffffff,
	0x0000000000000000,
	0xffffffffffffffff,
	0x0000000000000000,
	0xffffffffffffffff,
	0x00000000000000ff,
	0xffffffffffffffff,
	0x000000000000ffff,
	0xffffffffffffffff,
	0x0000000000ffffff,
	0xffffffffffffffff,
	0x00000000ffffffff,
	0xffffffffffffffff,
	0x000000ffffffffff,
	0xffffffffffffffff,
	0x0000ffffffffffff,
	0xffffffffffffffff,
	0x00ffffffffffffff,
}

func andMask_DATA() Mem {
	if andMask_DATA_ptr != nil {
		return *andMask_DATA_ptr
	}
	andMask := GLOBL("andMask", NOPTR|RODATA)
	andMask_DATA_ptr = &andMask

	for i, k := range andMask_K {
		DATA(i*8, U64(k))
	}

	return andMask
}
