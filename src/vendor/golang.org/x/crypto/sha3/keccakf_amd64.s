// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !purego && gc

// This code was translated into a form compatible with 6a from the public
// domain sources at https://github.com/gvanas/KeccakCodePackage

// Offsets in state
#define _ba  (0*8)
#define _be  (1*8)
#define _bi  (2*8)
#define _bo  (3*8)
#define _bu  (4*8)
#define _ga  (5*8)
#define _ge  (6*8)
#define _gi  (7*8)
#define _go  (8*8)
#define _gu  (9*8)
#define _ka (10*8)
#define _ke (11*8)
#define _ki (12*8)
#define _ko (13*8)
#define _ku (14*8)
#define _ma (15*8)
#define _me (16*8)
#define _mi (17*8)
#define _mo (18*8)
#define _mu (19*8)
#define _sa (20*8)
#define _se (21*8)
#define _si (22*8)
#define _so (23*8)
#define _su (24*8)

// Temporary registers
#define rT1  AX

// Round vars
#define rpState DI
#define rpStack SP

#define rDa BX
#define rDe CX
#define rDi DX
#define rDo R8
#define rDu R9

#define rBa R10
#define rBe R11
#define rBi R12
#define rBo R13
#define rBu R14

#define rCa SI
#define rCe BP
#define rCi rBi
#define rCo rBo
#define rCu R15

#define MOVQ_RBI_RCE MOVQ rBi, rCe
#define XORQ_RT1_RCA XORQ rT1, rCa
#define XORQ_RT1_RCE XORQ rT1, rCe
#define XORQ_RBA_RCU XORQ rBa, rCu
#define XORQ_RBE_RCU XORQ rBe, rCu
#define XORQ_RDU_RCU XORQ rDu, rCu
#define XORQ_RDA_RCA XORQ rDa, rCa
#define XORQ_RDE_RCE XORQ rDe, rCe

#define mKeccakRound(iState, oState, rc, B_RBI_RCE, G_RT1_RCA, G_RT1_RCE, G_RBA_RCU, K_RT1_RCA, K_RT1_RCE, K_RBA_RCU, M_RT1_RCA, M_RT1_RCE, M_RBE_RCU, S_RDU_RCU, S_RDA_RCA, S_RDE_RCE) \
	/* Prepare round */    \
	MOVQ rCe, rDa;         \
	ROLQ $1, rDa;          \
	                       \
	MOVQ _bi(iState), rCi; \
	XORQ _gi(iState), rDi; \
	XORQ rCu, rDa;         \
	XORQ _ki(iState), rCi; \
	XORQ _mi(iState), rDi; \
	XORQ rDi, rCi;         \
	                       \
	MOVQ rCi, rDe;         \
	ROLQ $1, rDe;          \
	                       \
	MOVQ _bo(iState), rCo; \
	XORQ _go(iState), rDo; \
	XORQ rCa, rDe;         \
	XORQ _ko(iState), rCo; \
	XORQ _mo(iState), rDo; \
	XORQ rDo, rCo;         \
	                       \
	MOVQ rCo, rDi;         \
	ROLQ $1, rDi;          \
	                       \
	MOVQ rCu, rDo;         \
	XORQ rCe, rDi;         \
	ROLQ $1, rDo;          \
	                       \
	MOVQ rCa, rDu;         \
	XORQ rCi, rDo;         \
	ROLQ $1, rDu;          \
	                       \
	/* Result b */         \
	MOVQ _ba(iState), rBa; \
	MOVQ _ge(iState), rBe; \
	XORQ rCo, rDu;         \
	MOVQ _ki(iState), rBi; \
	MOVQ _mo(iState), rBo; \
	MOVQ _su(iState), rBu; \
	XORQ rDe, rBe;         \
	ROLQ $44, rBe;         \
	XORQ rDi, rBi;         \
	XORQ rDa, rBa;         \
	ROLQ $43, rBi;         \
	                       \
	MOVQ rBe, rCa;         \
	MOVQ rc, rT1;          \
	ORQ  rBi, rCa;         \
	XORQ rBa, rT1;         \
	XORQ rT1, rCa;         \
	MOVQ rCa, _ba(oState); \
	                       \
	XORQ rDu, rBu;         \
	ROLQ $14, rBu;         \
	MOVQ rBa, rCu;         \
	ANDQ rBe, rCu;         \
	XORQ rBu, rCu;         \
	MOVQ rCu, _bu(oState); \
	                       \
	XORQ rDo, rBo;         \
	ROLQ $21, rBo;         \
	MOVQ rBo, rT1;         \
	ANDQ rBu, rT1;         \
	XORQ rBi, rT1;         \
	MOVQ rT1, _bi(oState); \
	                       \
	NOTQ rBi;              \
	ORQ  rBa, rBu;         \
	ORQ  rBo, rBi;         \
	XORQ rBo, rBu;         \
	XORQ rBe, rBi;         \
	MOVQ rBu, _bo(oState); \
	MOVQ rBi, _be(oState); \
	B_RBI_RCE;             \
	                       \
	/* Result g */         \
	MOVQ _gu(iState), rBe; \
	XORQ rDu, rBe;         \
	MOVQ _ka(iState), rBi; \
	ROLQ $20, rBe;         \
	XORQ rDa, rBi;         \
	ROLQ $3, rBi;          \
	MOVQ _bo(iState), rBa; \
	MOVQ rBe, rT1;         \
	ORQ  rBi, rT1;         \
	XORQ rDo, rBa;         \
	MOVQ _me(iState), rBo; \
	MOVQ _si(iState), rBu; \
	ROLQ $28, rBa;         \
	XORQ rBa, rT1;         \
	MOVQ rT1, _ga(oState); \
	G_RT1_RCA;             \
	                       \
	XORQ rDe, rBo;         \
	ROLQ $45, rBo;         \
	MOVQ rBi, rT1;         \
	ANDQ rBo, rT1;         \
	XORQ rBe, rT1;         \
	MOVQ rT1, _ge(oState); \
	G_RT1_RCE;             \
	                       \
	XORQ rDi, rBu;         \
	ROLQ $61, rBu;         \
	MOVQ rBu, rT1;         \
	ORQ  rBa, rT1;         \
	XORQ rBo, rT1;         \
	MOVQ rT1, _go(oState); \
	                       \
	ANDQ rBe, rBa;         \
	XORQ rBu, rBa;         \
	MOVQ rBa, _gu(oState); \
	NOTQ rBu;              \
	G_RBA_RCU;             \
	                       \
	ORQ  rBu, rBo;         \
	XORQ rBi, rBo;         \
	MOVQ rBo, _gi(oState); \
	                       \
	/* Result k */         \
	MOVQ _be(iState), rBa; \
	MOVQ _gi(iState), rBe; \
	MOVQ _ko(iState), rBi; \
	MOVQ _mu(iState), rBo; \
	MOVQ _sa(iState), rBu; \
	XORQ rDi, rBe;         \
	ROLQ $6, rBe;          \
	XORQ rDo, rBi;         \
	ROLQ $25, rBi;         \
	MOVQ rBe, rT1;         \
	ORQ  rBi, rT1;         \
	XORQ rDe, rBa;         \
	ROLQ $1, rBa;          \
	XORQ rBa, rT1;         \
	MOVQ rT1, _ka(oState); \
	K_RT1_RCA;             \
	                       \
	XORQ rDu, rBo;         \
	ROLQ $8, rBo;          \
	MOVQ rBi, rT1;         \
	ANDQ rBo, rT1;         \
	XORQ rBe, rT1;         \
	MOVQ rT1, _ke(oState); \
	K_RT1_RCE;             \
	                       \
	XORQ rDa, rBu;         \
	ROLQ $18, rBu;         \
	NOTQ rBo;              \
	MOVQ rBo, rT1;         \
	ANDQ rBu, rT1;         \
	XORQ rBi, rT1;         \
	MOVQ rT1, _ki(oState); \
	                       \
	MOVQ rBu, rT1;         \
	ORQ  rBa, rT1;         \
	XORQ rBo, rT1;         \
	MOVQ rT1, _ko(oState); \
	                       \
	ANDQ rBe, rBa;         \
	XORQ rBu, rBa;         \
	MOVQ rBa, _ku(oState); \
	K_RBA_RCU;             \
	                       \
	/* Result m */         \
	MOVQ _ga(iState), rBe; \
	XORQ rDa, rBe;         \
	MOVQ _ke(iState), rBi; \
	ROLQ $36, rBe;         \
	XORQ rDe, rBi;         \
	MOVQ _bu(iState), rBa; \
	ROLQ $10, rBi;         \
	MOVQ rBe, rT1;         \
	MOVQ _mi(iState), rBo; \
	ANDQ rBi, rT1;         \
	XORQ rDu, rBa;         \
	MOVQ _so(iState), rBu; \
	ROLQ $27, rBa;         \
	XORQ rBa, rT1;         \
	MOVQ rT1, _ma(oState); \
	M_RT1_RCA;             \
	                       \
	XORQ rDi, rBo;         \
	ROLQ $15, rBo;         \
	MOVQ rBi, rT1;         \
	ORQ  rBo, rT1;         \
	XORQ rBe, rT1;         \
	MOVQ rT1, _me(oState); \
	M_RT1_RCE;             \
	                       \
	XORQ rDo, rBu;         \
	ROLQ $56, rBu;         \
	NOTQ rBo;              \
	MOVQ rBo, rT1;         \
	ORQ  rBu, rT1;         \
	XORQ rBi, rT1;         \
	MOVQ rT1, _mi(oState); \
	                       \
	ORQ  rBa, rBe;         \
	XORQ rBu, rBe;         \
	MOVQ rBe, _mu(oState); \
	                       \
	ANDQ rBa, rBu;         \
	XORQ rBo, rBu;         \
	MOVQ rBu, _mo(oState); \
	M_RBE_RCU;             \
	                       \
	/* Result s */         \
	MOVQ _bi(iState), rBa; \
	MOVQ _go(iState), rBe; \
	MOVQ _ku(iState), rBi; \
	XORQ rDi, rBa;         \
	MOVQ _ma(iState), rBo; \
	ROLQ $62, rBa;         \
	XORQ rDo, rBe;         \
	MOVQ _se(iState), rBu; \
	ROLQ $55, rBe;         \
	                       \
	XORQ rDu, rBi;         \
	MOVQ rBa, rDu;         \
	XORQ rDe, rBu;         \
	ROLQ $2, rBu;          \
	ANDQ rBe, rDu;         \
	XORQ rBu, rDu;         \
	MOVQ rDu, _su(oState); \
	                       \
	ROLQ $39, rBi;         \
	S_RDU_RCU;             \
	NOTQ rBe;              \
	XORQ rDa, rBo;         \
	MOVQ rBe, rDa;         \
	ANDQ rBi, rDa;         \
	XORQ rBa, rDa;         \
	MOVQ rDa, _sa(oState); \
	S_RDA_RCA;             \
	                       \
	ROLQ $41, rBo;         \
	MOVQ rBi, rDe;         \
	ORQ  rBo, rDe;         \
	XORQ rBe, rDe;         \
	MOVQ rDe, _se(oState); \
	S_RDE_RCE;             \
	                       \
	MOVQ rBo, rDi;         \
	MOVQ rBu, rDo;         \
	ANDQ rBu, rDi;         \
	ORQ  rBa, rDo;         \
	XORQ rBi, rDi;         \
	XORQ rBo, rDo;         \
	MOVQ rDi, _si(oState); \
	MOVQ rDo, _so(oState)  \

// func keccakF1600(a *[25]uint64)
TEXT ·keccakF1600(SB), 0, $200-8
	MOVQ a+0(FP), rpState

	// Convert the user state into an internal state
	NOTQ _be(rpState)
	NOTQ _bi(rpState)
	NOTQ _go(rpState)
	NOTQ _ki(rpState)
	NOTQ _mi(rpState)
	NOTQ _sa(rpState)

	// Execute the KeccakF permutation
	MOVQ _ba(rpState), rCa
	MOVQ _be(rpState), rCe
	MOVQ _bu(rpState), rCu

	XORQ _ga(rpState), rCa
	XORQ _ge(rpState), rCe
	XORQ _gu(rpState), rCu

	XORQ _ka(rpState), rCa
	XORQ _ke(rpState), rCe
	XORQ _ku(rpState), rCu

	XORQ _ma(rpState), rCa
	XORQ _me(rpState), rCe
	XORQ _mu(rpState), rCu

	XORQ _sa(rpState), rCa
	XORQ _se(rpState), rCe
	MOVQ _si(rpState), rDi
	MOVQ _so(rpState), rDo
	XORQ _su(rpState), rCu

	mKeccakRound(rpState, rpStack, $0x0000000000000001, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x0000000000008082, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x800000000000808a, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x8000000080008000, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x000000000000808b, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x0000000080000001, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x8000000080008081, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x8000000000008009, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x000000000000008a, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x0000000000000088, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x0000000080008009, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x000000008000000a, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x000000008000808b, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x800000000000008b, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x8000000000008089, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x8000000000008003, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x8000000000008002, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x8000000000000080, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x000000000000800a, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x800000008000000a, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x8000000080008081, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x8000000000008080, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpState, rpStack, $0x0000000080000001, MOVQ_RBI_RCE, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBA_RCU, XORQ_RT1_RCA, XORQ_RT1_RCE, XORQ_RBE_RCU, XORQ_RDU_RCU, XORQ_RDA_RCA, XORQ_RDE_RCE)
	mKeccakRound(rpStack, rpState, $0x8000000080008008, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP, NOP)

	// Revert the internal state to the user state
	NOTQ _be(rpState)
	NOTQ _bi(rpState)
	NOTQ _go(rpState)
	NOTQ _ki(rpState)
	NOTQ _mi(rpState)
	NOTQ _sa(rpState)

	RET
