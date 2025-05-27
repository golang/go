// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

//go:generate go run . -out ../sha1block_amd64.s -pkg sha1

// AVX2 version by Intel, same algorithm as code in Linux kernel:
// https://github.com/torvalds/linux/blob/master/arch/x86/crypto/sha1_avx2_x86_64_asm.S
// Authors:
// Ilya Albrekht <ilya.albrekht@intel.com>
// Maxim Locktyukhin <maxim.locktyukhin@intel.com>
// Ronen Zohar <ronen.zohar@intel.com>
// Chandramouli Narayanan <mouli@linux.intel.com>

func main() {
	Package("crypto/sha1")
	ConstraintExpr("!purego")
	blockAVX2()
	blockSHANI()
	Generate()
}

// This is the implementation using AVX2, BMI1 and BMI2. It is based on:
// "SHA-1 implementation with Intel(R) AVX2 instruction set extensions"
// From http://software.intel.com/en-us/articles
// (look for improving-the-performance-of-the-secure-hash-algorithm-1)
// This implementation is 2x unrolled, and interleaves vector instructions,
// used to precompute W, with scalar computation of current round
// for optimal scheduling.

// Trivial helper macros.

func UPDATE_HASH(A, TB, C, D, E GPPhysical) {
	ADDL(Mem{Base: R9}, A)
	MOVL(A, Mem{Base: R9})
	ADDL(Mem{Base: R9}.Offset(4), TB)
	MOVL(TB, Mem{Base: R9}.Offset(4))
	ADDL(Mem{Base: R9}.Offset(8), C)
	MOVL(C, Mem{Base: R9}.Offset(8))
	ADDL(Mem{Base: R9}.Offset(12), D)
	MOVL(D, Mem{Base: R9}.Offset(12))
	ADDL(Mem{Base: R9}.Offset(16), E)
	MOVL(E, Mem{Base: R9}.Offset(16))
}

// Helper macros for PRECALC, which does precomputations

func PRECALC_0(OFFSET int) {
	VMOVDQU(Mem{Base: R10}.Offset(OFFSET), X0)
}

func PRECALC_1(OFFSET int) {
	VINSERTI128(Imm(1), Mem{Base: R13}.Offset(OFFSET), Y0, Y0)
}

func PRECALC_2(YREG VecPhysical) {
	VPSHUFB(Y10, Y0, YREG)
}

func PRECALC_4(YREG VecPhysical, K_OFFSET int) {
	VPADDD(Mem{Base: R8}.Offset(K_OFFSET), YREG, Y0)
}

func PRECALC_7(OFFSET int) {
	VMOVDQU(Y0, Mem{Base: R14}.Offset(OFFSET*2))
}

// Message scheduling pre-compute for rounds 0-15
//
//   - R13 is a pointer to even 64-byte block
//   - R10 is a pointer to odd 64-byte block
//   - R14 is a pointer to temp buffer
//   - X0 is used as temp register
//   - YREG is clobbered as part of computation
//   - OFFSET chooses 16 byte chunk within a block
//   - R8 is a pointer to constants block
//   - K_OFFSET chooses K constants relevant to this round
//   - X10 holds swap mask
func PRECALC_00_15(OFFSET int, YREG VecPhysical) {
	PRECALC_0(OFFSET)
	PRECALC_1(OFFSET)
	PRECALC_2(YREG)
	PRECALC_4(YREG, 0x0)
	PRECALC_7(OFFSET)
}

// Helper macros for PRECALC_16_31

func PRECALC_16(REG_SUB_16, REG_SUB_12, REG_SUB_4, REG VecPhysical) {
	VPALIGNR(Imm(8), REG_SUB_16, REG_SUB_12, REG) // w[i-14]
	VPSRLDQ(Imm(4), REG_SUB_4, Y0)                // w[i-3]
}

func PRECALC_17(REG_SUB_16, REG_SUB_8, REG VecPhysical) {
	VPXOR(REG_SUB_8, REG, REG)
	VPXOR(REG_SUB_16, Y0, Y0)
}

func PRECALC_18(REG VecPhysical) {
	VPXOR(Y0, REG, REG)
	VPSLLDQ(Imm(12), REG, Y9)
}

func PRECALC_19(REG VecPhysical) {
	VPSLLD(Imm(1), REG, Y0)
	VPSRLD(Imm(31), REG, REG)
}

func PRECALC_20(REG VecPhysical) {
	VPOR(REG, Y0, Y0)
	VPSLLD(Imm(2), Y9, REG)
}

func PRECALC_21(REG VecPhysical) {
	VPSRLD(Imm(30), Y9, Y9)
	VPXOR(REG, Y0, Y0)
}

func PRECALC_23(REG VecPhysical, K_OFFSET, OFFSET int) {
	VPXOR(Y9, Y0, REG)
	VPADDD(Mem{Base: R8}.Offset(K_OFFSET), REG, Y0)
	VMOVDQU(Y0, Mem{Base: R14}.Offset(OFFSET))
}

// Message scheduling pre-compute for rounds 16-31
//   - calculating last 32 w[i] values in 8 XMM registers
//   - pre-calculate K+w[i] values and store to mem
//   - for later load by ALU add instruction.
//   - "brute force" vectorization for rounds 16-31 only
//   - due to w[i]->w[i-3] dependency.
//   - clobbers 5 input ymm registers REG_SUB*
//   - uses X0 and X9 as temp registers
//   - As always, R8 is a pointer to constants block
//   - and R14 is a pointer to temp buffer
func PRECALC_16_31(REG, REG_SUB_4, REG_SUB_8, REG_SUB_12, REG_SUB_16 VecPhysical, K_OFFSET, OFFSET int) {
	PRECALC_16(REG_SUB_16, REG_SUB_12, REG_SUB_4, REG)
	PRECALC_17(REG_SUB_16, REG_SUB_8, REG)
	PRECALC_18(REG)
	PRECALC_19(REG)
	PRECALC_20(REG)
	PRECALC_21(REG)
	PRECALC_23(REG, K_OFFSET, OFFSET)
}

// Helper macros for PRECALC_32_79

func PRECALC_32(REG_SUB_8, REG_SUB_4 VecPhysical) {
	VPALIGNR(Imm(8), REG_SUB_8, REG_SUB_4, Y0)
}

func PRECALC_33(REG_SUB_28, REG VecPhysical) {
	VPXOR(REG_SUB_28, REG, REG)
}

func PRECALC_34(REG_SUB_16 VecPhysical) {
	VPXOR(REG_SUB_16, Y0, Y0)
}

func PRECALC_35(REG VecPhysical) {
	VPXOR(Y0, REG, REG)
}

func PRECALC_36(REG VecPhysical) {
	VPSLLD(Imm(2), REG, Y0)
}

func PRECALC_37(REG VecPhysical) {
	VPSRLD(Imm(30), REG, REG)
	VPOR(REG, Y0, REG)
}

func PRECALC_39(REG VecPhysical, K_OFFSET, OFFSET int) {
	VPADDD(Mem{Base: R8}.Offset(K_OFFSET), REG, Y0)
	VMOVDQU(Y0, Mem{Base: R14}.Offset(OFFSET))
}

// Message scheduling pre-compute for rounds 32-79
// In SHA-1 specification we have:
// w[i] = (w[i-3] ^ w[i-8]  ^ w[i-14] ^ w[i-16]) rol 1
// Which is the same as:
// w[i] = (w[i-6] ^ w[i-16] ^ w[i-28] ^ w[i-32]) rol 2
// This allows for more efficient vectorization,
// since w[i]->w[i-3] dependency is broken

func PRECALC_32_79(REG, REG_SUB_4, REG_SUB_8, REG_SUB_16, REG_SUB_28 VecPhysical, K_OFFSET, OFFSET int) {
	PRECALC_32(REG_SUB_8, REG_SUB_4)
	PRECALC_33(REG_SUB_28, REG)
	PRECALC_34(REG_SUB_16)
	PRECALC_35(REG)
	PRECALC_36(REG)
	PRECALC_37(REG)
	PRECALC_39(REG, K_OFFSET, OFFSET)
}

func PRECALC() {
	PRECALC_00_15(0, Y15)
	PRECALC_00_15(0x10, Y14)
	PRECALC_00_15(0x20, Y13)
	PRECALC_00_15(0x30, Y12)
	PRECALC_16_31(Y8, Y12, Y13, Y14, Y15, 0, 0x80)
	PRECALC_16_31(Y7, Y8, Y12, Y13, Y14, 0x20, 0xa0)
	PRECALC_16_31(Y5, Y7, Y8, Y12, Y13, 0x20, 0xc0)
	PRECALC_16_31(Y3, Y5, Y7, Y8, Y12, 0x20, 0xe0)
	PRECALC_32_79(Y15, Y3, Y5, Y8, Y14, 0x20, 0x100)
	PRECALC_32_79(Y14, Y15, Y3, Y7, Y13, 0x20, 0x120)
	PRECALC_32_79(Y13, Y14, Y15, Y5, Y12, 0x40, 0x140)
	PRECALC_32_79(Y12, Y13, Y14, Y3, Y8, 0x40, 0x160)
	PRECALC_32_79(Y8, Y12, Y13, Y15, Y7, 0x40, 0x180)
	PRECALC_32_79(Y7, Y8, Y12, Y14, Y5, 0x40, 0x1a0)
	PRECALC_32_79(Y5, Y7, Y8, Y13, Y3, 0x40, 0x1c0)
	PRECALC_32_79(Y3, Y5, Y7, Y12, Y15, 0x60, 0x1e0)
	PRECALC_32_79(Y15, Y3, Y5, Y8, Y14, 0x60, 0x200)
	PRECALC_32_79(Y14, Y15, Y3, Y7, Y13, 0x60, 0x220)
	PRECALC_32_79(Y13, Y14, Y15, Y5, Y12, 0x60, 0x240)
	PRECALC_32_79(Y12, Y13, Y14, Y3, Y8, 0x60, 0x260)
}

// Macros calculating individual rounds have general form
// CALC_ROUND_PRE + PRECALC_ROUND + CALC_ROUND_POST
// CALC_ROUND_{PRE,POST} macros follow

func CALC_F1_PRE(OFFSET int, REG_A, REG_B, REG_C, REG_E GPPhysical) {
	ADDL(Mem{Base: R15}.Offset(OFFSET), REG_E)
	ANDNL(REG_C, REG_A, EBP)
	LEAL(Mem{Base: REG_E, Index: REG_B, Scale: 1}, REG_E) // Add F from the previous round
	RORXL(Imm(0x1b), REG_A, R12L)
	RORXL(Imm(2), REG_A, REG_B) //                           for next round
}

func CALC_F1_POST(REG_A, REG_B, REG_E GPPhysical) {
	ANDL(REG_B, REG_A)                                  // b&c
	XORL(EBP, REG_A)                                    // F1 = (b&c) ^ (~b&d)
	LEAL(Mem{Base: REG_E, Index: R12, Scale: 1}, REG_E) // E += A >>> 5
}

// Registers are cyclically rotated DX -> AX -> DI -> SI -> BX -> CX

func CALC_0() {
	MOVL(ESI, EBX) // Precalculating first round
	RORXL(Imm(2), ESI, ESI)
	ANDNL(EAX, EBX, EBP)
	ANDL(EDI, EBX)
	XORL(EBP, EBX)
	CALC_F1_PRE(0x0, ECX, EBX, EDI, EDX)
	PRECALC_0(0x80)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_1() {
	CALC_F1_PRE(0x4, EDX, ECX, ESI, EAX)
	PRECALC_1(0x80)
	CALC_F1_POST(EDX, EBX, EAX)
}

func CALC_2() {
	CALC_F1_PRE(0x8, EAX, EDX, EBX, EDI)
	PRECALC_2(Y15)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_3() {
	CALC_F1_PRE(0xc, EDI, EAX, ECX, ESI)
	CALC_F1_POST(EDI, EDX, ESI)
}

func CALC_4() {
	CALC_F1_PRE(0x20, ESI, EDI, EDX, EBX)
	PRECALC_4(Y15, 0x0)
	CALC_F1_POST(ESI, EAX, EBX)
}

func CALC_5() {
	CALC_F1_PRE(0x24, EBX, ESI, EAX, ECX)
	CALC_F1_POST(EBX, EDI, ECX)
}

func CALC_6() {
	CALC_F1_PRE(0x28, ECX, EBX, EDI, EDX)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_7() {
	CALC_F1_PRE(0x2c, EDX, ECX, ESI, EAX)
	PRECALC_7(0x0)
	CALC_F1_POST(EDX, EBX, EAX)
}

func CALC_8() {
	CALC_F1_PRE(0x40, EAX, EDX, EBX, EDI)
	PRECALC_0(0x90)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_9() {
	CALC_F1_PRE(0x44, EDI, EAX, ECX, ESI)
	PRECALC_1(0x90)
	CALC_F1_POST(EDI, EDX, ESI)
}

func CALC_10() {
	CALC_F1_PRE(0x48, ESI, EDI, EDX, EBX)
	PRECALC_2(Y14)
	CALC_F1_POST(ESI, EAX, EBX)
}

func CALC_11() {
	CALC_F1_PRE(0x4c, EBX, ESI, EAX, ECX)
	CALC_F1_POST(EBX, EDI, ECX)
}

func CALC_12() {
	CALC_F1_PRE(0x60, ECX, EBX, EDI, EDX)
	PRECALC_4(Y14, 0x0)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_13() {
	CALC_F1_PRE(0x64, EDX, ECX, ESI, EAX)
	CALC_F1_POST(EDX, EBX, EAX)
}

func CALC_14() {
	CALC_F1_PRE(0x68, EAX, EDX, EBX, EDI)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_15() {
	CALC_F1_PRE(0x6c, EDI, EAX, ECX, ESI)
	PRECALC_7(0x10)
	CALC_F1_POST(EDI, EDX, ESI)
}

func CALC_16() {
	CALC_F1_PRE(0x80, ESI, EDI, EDX, EBX)
	PRECALC_0(0xa0)
	CALC_F1_POST(ESI, EAX, EBX)
}

func CALC_17() {
	CALC_F1_PRE(0x84, EBX, ESI, EAX, ECX)
	PRECALC_1(0xa0)
	CALC_F1_POST(EBX, EDI, ECX)
}

func CALC_18() {
	CALC_F1_PRE(0x88, ECX, EBX, EDI, EDX)
	PRECALC_2(Y13)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_F2_PRE(OFFSET int, REG_A, REG_B, REG_E GPPhysical) {
	ADDL(Mem{Base: R15}.Offset(OFFSET), REG_E)
	LEAL(Mem{Base: REG_E, Index: REG_B, Scale: 1}, REG_E) // Add F from the previous round
	RORXL(Imm(0x1b), REG_A, R12L)
	RORXL(Imm(2), REG_A, REG_B) //                           for next round
}

func CALC_F2_POST(REG_A, REG_B, REG_C, REG_E GPPhysical) {
	XORL(REG_B, REG_A)
	ADDL(R12L, REG_E)
	XORL(REG_C, REG_A)
}

func CALC_19() {
	CALC_F2_PRE(0x8c, EDX, ECX, EAX)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_20() {
	CALC_F2_PRE(0xa0, EAX, EDX, EDI)
	PRECALC_4(Y13, 0x0)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_21() {
	CALC_F2_PRE(0xa4, EDI, EAX, ESI)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_22() {
	CALC_F2_PRE(0xa8, ESI, EDI, EBX)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_23() {
	CALC_F2_PRE(0xac, EBX, ESI, ECX)
	PRECALC_7(0x20)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_24() {
	CALC_F2_PRE(0xc0, ECX, EBX, EDX)
	PRECALC_0(0xb0)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_25() {
	CALC_F2_PRE(0xc4, EDX, ECX, EAX)
	PRECALC_1(0xb0)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_26() {
	CALC_F2_PRE(0xc8, EAX, EDX, EDI)
	PRECALC_2(Y12)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_27() {
	CALC_F2_PRE(0xcc, EDI, EAX, ESI)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_28() {
	CALC_F2_PRE(0xe0, ESI, EDI, EBX)
	PRECALC_4(Y12, 0x0)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_29() {
	CALC_F2_PRE(0xe4, EBX, ESI, ECX)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_30() {
	CALC_F2_PRE(0xe8, ECX, EBX, EDX)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_31() {
	CALC_F2_PRE(0xec, EDX, ECX, EAX)
	PRECALC_7(0x30)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_32() {
	CALC_F2_PRE(0x100, EAX, EDX, EDI)
	PRECALC_16(Y15, Y14, Y12, Y8)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_33() {
	CALC_F2_PRE(0x104, EDI, EAX, ESI)
	PRECALC_17(Y15, Y13, Y8)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_34() {
	CALC_F2_PRE(0x108, ESI, EDI, EBX)
	PRECALC_18(Y8)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_35() {
	CALC_F2_PRE(0x10c, EBX, ESI, ECX)
	PRECALC_19(Y8)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_36() {
	CALC_F2_PRE(0x120, ECX, EBX, EDX)
	PRECALC_20(Y8)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_37() {
	CALC_F2_PRE(0x124, EDX, ECX, EAX)
	PRECALC_21(Y8)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_38() {
	CALC_F2_PRE(0x128, EAX, EDX, EDI)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_F3_PRE(OFFSET int, REG_E GPPhysical) {
	ADDL(Mem{Base: R15}.Offset(OFFSET), REG_E)
}

func CALC_F3_POST(REG_A, REG_B, REG_C, REG_E, REG_TB GPPhysical) {
	LEAL(Mem{Base: REG_E, Index: REG_TB, Scale: 1}, REG_E) // Add F from the previous round
	MOVL(REG_B, EBP)
	ORL(REG_A, EBP)
	RORXL(Imm(0x1b), REG_A, R12L)
	RORXL(Imm(2), REG_A, REG_TB)
	ANDL(REG_C, EBP)
	ANDL(REG_B, REG_A)
	ORL(EBP, REG_A)
	ADDL(R12L, REG_E)
}

func CALC_39() {
	CALC_F3_PRE(0x12c, ESI)
	PRECALC_23(Y8, 0x0, 0x80)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_40() {
	CALC_F3_PRE(0x140, EBX)
	PRECALC_16(Y14, Y13, Y8, Y7)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_41() {
	CALC_F3_PRE(0x144, ECX)
	PRECALC_17(Y14, Y12, Y7)
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_42() {
	CALC_F3_PRE(0x148, EDX)
	PRECALC_18(Y7)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_43() {
	CALC_F3_PRE(0x14c, EAX)
	PRECALC_19(Y7)
	CALC_F3_POST(EDX, EBX, ESI, EAX, ECX)
}

func CALC_44() {
	CALC_F3_PRE(0x160, EDI)
	PRECALC_20(Y7)
	CALC_F3_POST(EAX, ECX, EBX, EDI, EDX)
}

func CALC_45() {
	CALC_F3_PRE(0x164, ESI)
	PRECALC_21(Y7)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_46() {
	CALC_F3_PRE(0x168, EBX)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_47() {
	CALC_F3_PRE(0x16c, ECX)
	VPXOR(Y9, Y0, Y7)
	VPADDD(Mem{Base: R8}.Offset(0x20), Y7, Y0)
	VMOVDQU(Y0, Mem{Base: R14}.Offset(0xa0))
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_48() {
	CALC_F3_PRE(0x180, EDX)
	PRECALC_16(Y13, Y12, Y7, Y5)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_49() {
	CALC_F3_PRE(0x184, EAX)
	PRECALC_17(Y13, Y8, Y5)
	CALC_F3_POST(EDX, EBX, ESI, EAX, ECX)
}

func CALC_50() {
	CALC_F3_PRE(0x188, EDI)
	PRECALC_18(Y5)
	CALC_F3_POST(EAX, ECX, EBX, EDI, EDX)
}

func CALC_51() {
	CALC_F3_PRE(0x18c, ESI)
	PRECALC_19(Y5)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_52() {
	CALC_F3_PRE(0x1a0, EBX)
	PRECALC_20(Y5)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_53() {
	CALC_F3_PRE(0x1a4, ECX)
	PRECALC_21(Y5)
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_54() {
	CALC_F3_PRE(0x1a8, EDX)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_55() {
	CALC_F3_PRE(0x1ac, EAX)
	PRECALC_23(Y5, 0x20, 0xc0)
	CALC_F3_POST(EDX, EBX, ESI, EAX, ECX)
}

func CALC_56() {
	CALC_F3_PRE(0x1c0, EDI)
	PRECALC_16(Y12, Y8, Y5, Y3)
	CALC_F3_POST(EAX, ECX, EBX, EDI, EDX)
}

func CALC_57() {
	CALC_F3_PRE(0x1c4, ESI)
	PRECALC_17(Y12, Y7, Y3)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_58() {
	CALC_F3_PRE(0x1c8, EBX)
	PRECALC_18(Y3)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_59() {
	CALC_F2_PRE(0x1cc, EBX, ESI, ECX)
	PRECALC_19(Y3)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_60() {
	CALC_F2_PRE(0x1e0, ECX, EBX, EDX)
	PRECALC_20(Y3)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_61() {
	CALC_F2_PRE(0x1e4, EDX, ECX, EAX)
	PRECALC_21(Y3)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_62() {
	CALC_F2_PRE(0x1e8, EAX, EDX, EDI)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_63() {
	CALC_F2_PRE(0x1ec, EDI, EAX, ESI)
	PRECALC_23(Y3, 0x20, 0xe0)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_64() {
	CALC_F2_PRE(0x200, ESI, EDI, EBX)
	PRECALC_32(Y5, Y3)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_65() {
	CALC_F2_PRE(0x204, EBX, ESI, ECX)
	PRECALC_33(Y14, Y15)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_66() {
	CALC_F2_PRE(0x208, ECX, EBX, EDX)
	PRECALC_34(Y8)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_67() {
	CALC_F2_PRE(0x20c, EDX, ECX, EAX)
	PRECALC_35(Y15)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_68() {
	CALC_F2_PRE(0x220, EAX, EDX, EDI)
	PRECALC_36(Y15)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_69() {
	CALC_F2_PRE(0x224, EDI, EAX, ESI)
	PRECALC_37(Y15)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_70() {
	CALC_F2_PRE(0x228, ESI, EDI, EBX)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_71() {
	CALC_F2_PRE(0x22c, EBX, ESI, ECX)
	PRECALC_39(Y15, 0x20, 0x100)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_72() {
	CALC_F2_PRE(0x240, ECX, EBX, EDX)
	PRECALC_32(Y3, Y15)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_73() {
	CALC_F2_PRE(0x244, EDX, ECX, EAX)
	PRECALC_33(Y13, Y14)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_74() {
	CALC_F2_PRE(0x248, EAX, EDX, EDI)
	PRECALC_34(Y7)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_75() {
	CALC_F2_PRE(0x24c, EDI, EAX, ESI)
	PRECALC_35(Y14)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_76() {
	CALC_F2_PRE(0x260, ESI, EDI, EBX)
	PRECALC_36(Y14)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_77() {
	CALC_F2_PRE(0x264, EBX, ESI, ECX)
	PRECALC_37(Y14)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_78() {
	CALC_F2_PRE(0x268, ECX, EBX, EDX)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_79() {
	ADDL(Mem{Base: R15}.Offset(0x26c), EAX)
	LEAL(Mem{Base: AX, Index: CX, Scale: 1}, EAX)
	RORXL(Imm(0x1b), EDX, R12L)
	PRECALC_39(Y14, 0x20, 0x120)
	ADDL(R12L, EAX)
}

// Similar to CALC_0
func CALC_80() {
	MOVL(ECX, EDX)
	RORXL(Imm(2), ECX, ECX)
	ANDNL(ESI, EDX, EBP)
	ANDL(EBX, EDX)
	XORL(EBP, EDX)
	CALC_F1_PRE(0x10, EAX, EDX, EBX, EDI)
	PRECALC_32(Y15, Y14)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_81() {
	CALC_F1_PRE(0x14, EDI, EAX, ECX, ESI)
	PRECALC_33(Y12, Y13)
	CALC_F1_POST(EDI, EDX, ESI)
}

func CALC_82() {
	CALC_F1_PRE(0x18, ESI, EDI, EDX, EBX)
	PRECALC_34(Y5)
	CALC_F1_POST(ESI, EAX, EBX)
}

func CALC_83() {
	CALC_F1_PRE(0x1c, EBX, ESI, EAX, ECX)
	PRECALC_35(Y13)
	CALC_F1_POST(EBX, EDI, ECX)
}

func CALC_84() {
	CALC_F1_PRE(0x30, ECX, EBX, EDI, EDX)
	PRECALC_36(Y13)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_85() {
	CALC_F1_PRE(0x34, EDX, ECX, ESI, EAX)
	PRECALC_37(Y13)
	CALC_F1_POST(EDX, EBX, EAX)
}

func CALC_86() {
	CALC_F1_PRE(0x38, EAX, EDX, EBX, EDI)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_87() {
	CALC_F1_PRE(0x3c, EDI, EAX, ECX, ESI)
	PRECALC_39(Y13, 0x40, 0x140)
	CALC_F1_POST(EDI, EDX, ESI)
}

func CALC_88() {
	CALC_F1_PRE(0x50, ESI, EDI, EDX, EBX)
	PRECALC_32(Y14, Y13)
	CALC_F1_POST(ESI, EAX, EBX)
}

func CALC_89() {
	CALC_F1_PRE(0x54, EBX, ESI, EAX, ECX)
	PRECALC_33(Y8, Y12)
	CALC_F1_POST(EBX, EDI, ECX)
}

func CALC_90() {
	CALC_F1_PRE(0x58, ECX, EBX, EDI, EDX)
	PRECALC_34(Y3)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_91() {
	CALC_F1_PRE(0x5c, EDX, ECX, ESI, EAX)
	PRECALC_35(Y12)
	CALC_F1_POST(EDX, EBX, EAX)
}

func CALC_92() {
	CALC_F1_PRE(0x70, EAX, EDX, EBX, EDI)
	PRECALC_36(Y12)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_93() {
	CALC_F1_PRE(0x74, EDI, EAX, ECX, ESI)
	PRECALC_37(Y12)
	CALC_F1_POST(EDI, EDX, ESI)
}

func CALC_94() {
	CALC_F1_PRE(0x78, ESI, EDI, EDX, EBX)
	CALC_F1_POST(ESI, EAX, EBX)
}

func CALC_95() {
	CALC_F1_PRE(0x7c, EBX, ESI, EAX, ECX)
	PRECALC_39(Y12, 0x40, 0x160)
	CALC_F1_POST(EBX, EDI, ECX)
}

func CALC_96() {
	CALC_F1_PRE(0x90, ECX, EBX, EDI, EDX)
	PRECALC_32(Y13, Y12)
	CALC_F1_POST(ECX, ESI, EDX)
}

func CALC_97() {
	CALC_F1_PRE(0x94, EDX, ECX, ESI, EAX)
	PRECALC_33(Y7, Y8)
	CALC_F1_POST(EDX, EBX, EAX)
}

func CALC_98() {
	CALC_F1_PRE(0x98, EAX, EDX, EBX, EDI)
	PRECALC_34(Y15)
	CALC_F1_POST(EAX, ECX, EDI)
}

func CALC_99() {
	CALC_F2_PRE(0x9c, EDI, EAX, ESI)
	PRECALC_35(Y8)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_100() {
	CALC_F2_PRE(0xb0, ESI, EDI, EBX)
	PRECALC_36(Y8)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_101() {
	CALC_F2_PRE(0xb4, EBX, ESI, ECX)
	PRECALC_37(Y8)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_102() {
	CALC_F2_PRE(0xb8, ECX, EBX, EDX)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_103() {
	CALC_F2_PRE(0xbc, EDX, ECX, EAX)
	PRECALC_39(Y8, 0x40, 0x180)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_104() {
	CALC_F2_PRE(0xd0, EAX, EDX, EDI)
	PRECALC_32(Y12, Y8)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_105() {
	CALC_F2_PRE(0xd4, EDI, EAX, ESI)
	PRECALC_33(Y5, Y7)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_106() {
	CALC_F2_PRE(0xd8, ESI, EDI, EBX)
	PRECALC_34(Y14)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_107() {
	CALC_F2_PRE(0xdc, EBX, ESI, ECX)
	PRECALC_35(Y7)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_108() {
	CALC_F2_PRE(0xf0, ECX, EBX, EDX)
	PRECALC_36(Y7)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_109() {
	CALC_F2_PRE(0xf4, EDX, ECX, EAX)
	PRECALC_37(Y7)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_110() {
	CALC_F2_PRE(0xf8, EAX, EDX, EDI)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_111() {
	CALC_F2_PRE(0xfc, EDI, EAX, ESI)
	PRECALC_39(Y7, 0x40, 0x1a0)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_112() {
	CALC_F2_PRE(0x110, ESI, EDI, EBX)
	PRECALC_32(Y8, Y7)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_113() {
	CALC_F2_PRE(0x114, EBX, ESI, ECX)
	PRECALC_33(Y3, Y5)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_114() {
	CALC_F2_PRE(0x118, ECX, EBX, EDX)
	PRECALC_34(Y13)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_115() {
	CALC_F2_PRE(0x11c, EDX, ECX, EAX)
	PRECALC_35(Y5)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_116() {
	CALC_F2_PRE(0x130, EAX, EDX, EDI)
	PRECALC_36(Y5)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_117() {
	CALC_F2_PRE(0x134, EDI, EAX, ESI)
	PRECALC_37(Y5)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_118() {
	CALC_F2_PRE(0x138, ESI, EDI, EBX)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_119() {
	CALC_F3_PRE(0x13c, ECX)
	PRECALC_39(Y5, 0x40, 0x1c0)
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_120() {
	CALC_F3_PRE(0x150, EDX)
	PRECALC_32(Y7, Y5)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_121() {
	CALC_F3_PRE(0x154, EAX)
	PRECALC_33(Y15, Y3)
	CALC_F3_POST(EDX, EBX, ESI, EAX, ECX)
}

func CALC_122() {
	CALC_F3_PRE(0x158, EDI)
	PRECALC_34(Y12)
	CALC_F3_POST(EAX, ECX, EBX, EDI, EDX)
}

func CALC_123() {
	CALC_F3_PRE(0x15c, ESI)
	PRECALC_35(Y3)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_124() {
	CALC_F3_PRE(0x170, EBX)
	PRECALC_36(Y3)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_125() {
	CALC_F3_PRE(0x174, ECX)
	PRECALC_37(Y3)
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_126() {
	CALC_F3_PRE(0x178, EDX)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_127() {
	CALC_F3_PRE(0x17c, EAX)
	PRECALC_39(Y3, 0x60, 0x1e0)
	CALC_F3_POST(EDX, EBX, ESI, EAX, ECX)
}

func CALC_128() {
	CALC_F3_PRE(0x190, EDI)
	PRECALC_32(Y5, Y3)
	CALC_F3_POST(EAX, ECX, EBX, EDI, EDX)
}

func CALC_129() {
	CALC_F3_PRE(0x194, ESI)
	PRECALC_33(Y14, Y15)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_130() {
	CALC_F3_PRE(0x198, EBX)
	PRECALC_34(Y8)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_131() {
	CALC_F3_PRE(0x19c, ECX)
	PRECALC_35(Y15)
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_132() {
	CALC_F3_PRE(0x1b0, EDX)
	PRECALC_36(Y15)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_133() {
	CALC_F3_PRE(0x1b4, EAX)
	PRECALC_37(Y15)
	CALC_F3_POST(EDX, EBX, ESI, EAX, ECX)
}

func CALC_134() {
	CALC_F3_PRE(0x1b8, EDI)
	CALC_F3_POST(EAX, ECX, EBX, EDI, EDX)
}

func CALC_135() {
	CALC_F3_PRE(0x1bc, ESI)
	PRECALC_39(Y15, 0x60, 0x200)
	CALC_F3_POST(EDI, EDX, ECX, ESI, EAX)
}

func CALC_136() {
	CALC_F3_PRE(0x1d0, EBX)
	PRECALC_32(Y3, Y15)
	CALC_F3_POST(ESI, EAX, EDX, EBX, EDI)
}

func CALC_137() {
	CALC_F3_PRE(0x1d4, ECX)
	PRECALC_33(Y13, Y14)
	CALC_F3_POST(EBX, EDI, EAX, ECX, ESI)
}

func CALC_138() {
	CALC_F3_PRE(0x1d8, EDX)
	PRECALC_34(Y7)
	CALC_F3_POST(ECX, ESI, EDI, EDX, EBX)
}

func CALC_139() {
	CALC_F2_PRE(0x1dc, EDX, ECX, EAX)
	PRECALC_35(Y14)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_140() {
	CALC_F2_PRE(0x1f0, EAX, EDX, EDI)
	PRECALC_36(Y14)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_141() {
	CALC_F2_PRE(0x1f4, EDI, EAX, ESI)
	PRECALC_37(Y14)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_142() {
	CALC_F2_PRE(0x1f8, ESI, EDI, EBX)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_143() {
	CALC_F2_PRE(0x1fc, EBX, ESI, ECX)
	PRECALC_39(Y14, 0x60, 0x220)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_144() {
	CALC_F2_PRE(0x210, ECX, EBX, EDX)
	PRECALC_32(Y15, Y14)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_145() {
	CALC_F2_PRE(0x214, EDX, ECX, EAX)
	PRECALC_33(Y12, Y13)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_146() {
	CALC_F2_PRE(0x218, EAX, EDX, EDI)
	PRECALC_34(Y5)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_147() {
	CALC_F2_PRE(0x21c, EDI, EAX, ESI)
	PRECALC_35(Y13)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_148() {
	CALC_F2_PRE(0x230, ESI, EDI, EBX)
	PRECALC_36(Y13)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_149() {
	CALC_F2_PRE(0x234, EBX, ESI, ECX)
	PRECALC_37(Y13)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_150() {
	CALC_F2_PRE(0x238, ECX, EBX, EDX)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_151() {
	CALC_F2_PRE(0x23c, EDX, ECX, EAX)
	PRECALC_39(Y13, 0x60, 0x240)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_152() {
	CALC_F2_PRE(0x250, EAX, EDX, EDI)
	PRECALC_32(Y14, Y13)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_153() {
	CALC_F2_PRE(0x254, EDI, EAX, ESI)
	PRECALC_33(Y8, Y12)
	CALC_F2_POST(EDI, EDX, ECX, ESI)
}

func CALC_154() {
	CALC_F2_PRE(0x258, ESI, EDI, EBX)
	PRECALC_34(Y3)
	CALC_F2_POST(ESI, EAX, EDX, EBX)
}

func CALC_155() {
	CALC_F2_PRE(0x25c, EBX, ESI, ECX)
	PRECALC_35(Y12)
	CALC_F2_POST(EBX, EDI, EAX, ECX)
}

func CALC_156() {
	CALC_F2_PRE(0x270, ECX, EBX, EDX)
	PRECALC_36(Y12)
	CALC_F2_POST(ECX, ESI, EDI, EDX)
}

func CALC_157() {
	CALC_F2_PRE(0x274, EDX, ECX, EAX)
	PRECALC_37(Y12)
	CALC_F2_POST(EDX, EBX, ESI, EAX)
}

func CALC_158() {
	CALC_F2_PRE(0x278, EAX, EDX, EDI)
	CALC_F2_POST(EAX, ECX, EBX, EDI)
}

func CALC_159() {
	ADDL(Mem{Base: R15}.Offset(0x27c), ESI)
	LEAL(Mem{Base: SI, Index: AX, Scale: 1}, ESI)
	RORXL(Imm(0x1b), EDI, R12L)
	PRECALC_39(Y12, 0x60, 0x260)
	ADDL(R12L, ESI)
}

func CALC() {
	MOVL(Mem{Base: R9}, ECX)
	MOVL(Mem{Base: R9}.Offset(4), ESI)
	MOVL(Mem{Base: R9}.Offset(8), EDI)
	MOVL(Mem{Base: R9}.Offset(12), EAX)
	MOVL(Mem{Base: R9}.Offset(16), EDX)
	MOVQ(RSP, R14)
	LEAQ(Mem{Base: SP}.Offset(2*4*80+32), R15)
	PRECALC() // Precalc WK for first 2 blocks
	XCHGQ(R15, R14)
	loop_avx2()
	begin()
}

// this loops is unrolled
func loop_avx2() {
	Label("loop")
	CMPQ(R10, R8) // we use R8 value (set below) as a signal of a last block
	JNE(LabelRef("begin"))
	VZEROUPPER()
	RET()
}

func begin() {
	Label("begin")
	CALC_0()
	CALC_1()
	CALC_2()
	CALC_3()
	CALC_4()
	CALC_5()
	CALC_6()
	CALC_7()
	CALC_8()
	CALC_9()
	CALC_10()
	CALC_11()
	CALC_12()
	CALC_13()
	CALC_14()
	CALC_15()
	CALC_16()
	CALC_17()
	CALC_18()
	CALC_19()
	CALC_20()
	CALC_21()
	CALC_22()
	CALC_23()
	CALC_24()
	CALC_25()
	CALC_26()
	CALC_27()
	CALC_28()
	CALC_29()
	CALC_30()
	CALC_31()
	CALC_32()
	CALC_33()
	CALC_34()
	CALC_35()
	CALC_36()
	CALC_37()
	CALC_38()
	CALC_39()
	CALC_40()
	CALC_41()
	CALC_42()
	CALC_43()
	CALC_44()
	CALC_45()
	CALC_46()
	CALC_47()
	CALC_48()
	CALC_49()
	CALC_50()
	CALC_51()
	CALC_52()
	CALC_53()
	CALC_54()
	CALC_55()
	CALC_56()
	CALC_57()
	CALC_58()
	CALC_59()
	ADDQ(Imm(128), R10) // move to next even-64-byte block
	CMPQ(R10, R11)      // is current block the last one?
	CMOVQCC(R8, R10)    // signal the last iteration smartly
	CALC_60()
	CALC_61()
	CALC_62()
	CALC_63()
	CALC_64()
	CALC_65()
	CALC_66()
	CALC_67()
	CALC_68()
	CALC_69()
	CALC_70()
	CALC_71()
	CALC_72()
	CALC_73()
	CALC_74()
	CALC_75()
	CALC_76()
	CALC_77()
	CALC_78()
	CALC_79()
	UPDATE_HASH(EAX, EDX, EBX, ESI, EDI)
	CMPQ(R10, R8) // is current block the last one?
	JE(LabelRef("loop"))
	MOVL(EDX, ECX)
	CALC_80()
	CALC_81()
	CALC_82()
	CALC_83()
	CALC_84()
	CALC_85()
	CALC_86()
	CALC_87()
	CALC_88()
	CALC_89()
	CALC_90()
	CALC_91()
	CALC_92()
	CALC_93()
	CALC_94()
	CALC_95()
	CALC_96()
	CALC_97()
	CALC_98()
	CALC_99()
	CALC_100()
	CALC_101()
	CALC_102()
	CALC_103()
	CALC_104()
	CALC_105()
	CALC_106()
	CALC_107()
	CALC_108()
	CALC_109()
	CALC_110()
	CALC_111()
	CALC_112()
	CALC_113()
	CALC_114()
	CALC_115()
	CALC_116()
	CALC_117()
	CALC_118()
	CALC_119()
	CALC_120()
	CALC_121()
	CALC_122()
	CALC_123()
	CALC_124()
	CALC_125()
	CALC_126()
	CALC_127()
	CALC_128()
	CALC_129()
	CALC_130()
	CALC_131()
	CALC_132()
	CALC_133()
	CALC_134()
	CALC_135()
	CALC_136()
	CALC_137()
	CALC_138()
	CALC_139()
	ADDQ(Imm(128), R13) //move to next even-64-byte block
	CMPQ(R13, R11)      //is current block the last one?
	CMOVQCC(R8, R10)
	CALC_140()
	CALC_141()
	CALC_142()
	CALC_143()
	CALC_144()
	CALC_145()
	CALC_146()
	CALC_147()
	CALC_148()
	CALC_149()
	CALC_150()
	CALC_151()
	CALC_152()
	CALC_153()
	CALC_154()
	CALC_155()
	CALC_156()
	CALC_157()
	CALC_158()
	CALC_159()
	UPDATE_HASH(ESI, EDI, EDX, ECX, EBX)
	MOVL(ESI, R12L)
	MOVL(EDI, ESI)
	MOVL(EDX, EDI)
	MOVL(EBX, EDX)
	MOVL(ECX, EAX)
	MOVL(R12L, ECX)
	XCHGQ(R15, R14)
	JMP(LabelRef("loop"))
}

func blockAVX2() {
	Implement("blockAVX2")
	AllocLocal(1408)

	Load(Param("dig"), RDI)
	Load(Param("p").Base(), RSI)
	Load(Param("p").Len(), RDX)
	SHRQ(Imm(6), RDX)
	SHLQ(Imm(6), RDX)

	K_XMM_AR := K_XMM_AR_DATA()
	LEAQ(K_XMM_AR, R8)

	MOVQ(RDI, R9)
	MOVQ(RSI, R10)
	LEAQ(Mem{Base: SI}.Offset(64), R13)

	ADDQ(RSI, RDX)
	ADDQ(Imm(64), RDX)
	MOVQ(RDX, R11)

	CMPQ(R13, R11)
	CMOVQCC(R8, R13)

	BSWAP_SHUFB_CTL := BSWAP_SHUFB_CTL_DATA()
	VMOVDQU(BSWAP_SHUFB_CTL, Y10)
	CALC()
}

// ##~~~~~~~~~~~~~~~~~~~~~~~~~~DATA SECTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

// Pointers for memoizing Data section symbols
var (
	K_XMM_AR_ptr, BSWAP_SHUFB_CTL_ptr *Mem
)

// To hold Round Constants for K_XMM_AR_DATA

var _K = []uint32{
	0x5A827999,
	0x6ED9EBA1,
	0x8F1BBCDC,
	0xCA62C1D6,
}

func K_XMM_AR_DATA() Mem {
	if K_XMM_AR_ptr != nil {
		return *K_XMM_AR_ptr
	}

	K_XMM_AR := GLOBL("K_XMM_AR", RODATA)
	K_XMM_AR_ptr = &K_XMM_AR

	offset_idx := 0
	for _, v := range _K {
		DATA((offset_idx+0)*4, U32(v))
		DATA((offset_idx+1)*4, U32(v))
		DATA((offset_idx+2)*4, U32(v))
		DATA((offset_idx+3)*4, U32(v))
		DATA((offset_idx+4)*4, U32(v))
		DATA((offset_idx+5)*4, U32(v))
		DATA((offset_idx+6)*4, U32(v))
		DATA((offset_idx+7)*4, U32(v))
		offset_idx += 8
	}
	return K_XMM_AR
}

var BSWAP_SHUFB_CTL_CONSTANTS = [8]uint32{
	0x00010203,
	0x04050607,
	0x08090a0b,
	0x0c0d0e0f,
	0x00010203,
	0x04050607,
	0x08090a0b,
	0x0c0d0e0f,
}

func BSWAP_SHUFB_CTL_DATA() Mem {
	if BSWAP_SHUFB_CTL_ptr != nil {
		return *BSWAP_SHUFB_CTL_ptr
	}

	BSWAP_SHUFB_CTL := GLOBL("BSWAP_SHUFB_CTL", RODATA)
	BSWAP_SHUFB_CTL_ptr = &BSWAP_SHUFB_CTL
	for i, v := range BSWAP_SHUFB_CTL_CONSTANTS {

		DATA(i*4, U32(v))
	}
	return BSWAP_SHUFB_CTL
}
