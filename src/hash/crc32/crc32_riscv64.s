// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// RISC-V 64-bit CRC32 using carry-less multiplication (Zbc extension)
// with Barrett reduction. Algorithm adapted from ISA-L's RISC-V CRC
// implementation and Intel's PCLMULQDQ CRC paper.
//
// The algorithm works as follows:
//   1. Load first 16 bytes of data, XOR the CRC seed into the low word
//   2. Fold each subsequent 16-byte block using carry-less multiplication
//   3. Final fold: reduce 128-bit accumulator to 64 bits
//   4. Barrett reduction: reduce 64 bits to 32-bit CRC

#include "textflag.h"

// IEEE CRC32 constants (reflected polynomial 0xEDB88320)

// Fold-by-128-bit (16-byte) constants for IEEE CRC32 reflected.
// K1 and K2 are used to fold a 128-bit accumulator with new 16-byte data.
DATA ieee_fold<>+0(SB)/8, $0xccaa009e   // K2 (fold high)
DATA ieee_fold<>+8(SB)/8, $0xae689191   // K1 (fold low)
GLOBL ieee_fold<>(SB), RODATA, $16

// Final reduction constants for IEEE CRC32 reflected.
// const_low:  128→64 bit fold constant (low)
// const_high: 128→64 bit fold constant (high)
// const_quo:  Barrett reduction quotient μ = floor(x^64 / P(x))
// const_poly: CRC32-IEEE polynomial P(x) with implicit x^32 term
DATA ieee_reduce<>+0(SB)/8, $0xb8bc6765     // const_low
DATA ieee_reduce<>+8(SB)/8, $0xccaa009e     // const_high
DATA ieee_reduce<>+16(SB)/8, $0x1f7011641   // const_quo (μ)
DATA ieee_reduce<>+24(SB)/8, $0x1db710641   // const_poly (P)
GLOBL ieee_reduce<>(SB), RODATA, $32

// Castagnoli CRC32-C constants (reflected polynomial 0x82F63B78)
// Fold-by-128-bit (16-byte) constants for Castagnoli CRC32-C reflected.
DATA cast_fold<>+0(SB)/8, $0x493c7d27   // K2 (fold high)
DATA cast_fold<>+8(SB)/8, $0xf20c0dfe   // K1 (fold low)
GLOBL cast_fold<>(SB), RODATA, $16

// Final reduction constants for Castagnoli CRC32-C reflected.
DATA cast_reduce<>+0(SB)/8, $0xdd45aab8     // const_low
DATA cast_reduce<>+8(SB)/8, $0x493c7d27     // const_high
DATA cast_reduce<>+16(SB)/8, $0x0dea713f1   // const_quo (μ)
DATA cast_reduce<>+24(SB)/8, $0x105ec76f1   // const_poly (P)
GLOBL cast_reduce<>(SB), RODATA, $32

// 32-bit mask constant
DATA mask32<>+0(SB)/8, $0xffffffff
GLOBL mask32<>(SB), RODATA, $8


// Computes CRC32-IEEE using carry-less multiplication (Zbc extension).
// Expects non-inverted CRC input. Returns non-inverted CRC.
// Requires len(p) >= 64 and len(p) is a multiple of 16.
TEXT ·ieeeUpdateCLMUL(SB), NOSPLIT, $0-36
	MOVWU	crc+0(FP), X5		// CRC value (inverted by caller)
	MOV	p+8(FP), X6		// data pointer
	MOV	p_len+16(FP), X7	// len(p)

	// Load first 16 bytes of data
	MOV	(X6), X8		// t0 = low 64 bits
	MOV	8(X6), X9		// t1 = high 64 bits
	ADD	$16, X6
	ADD	$-16, X7

	// XOR CRC seed into the low word of the first block
	XOR	X5, X8, X8

	// Load fold constants: K2 (high), K1 (low)
	MOV	ieee_fold<>+0(SB), X10		// K2
	MOV	ieee_fold<>+8(SB), X11		// K1

	// Main fold loop: process 16 bytes per iteration.
	// We fold the 128-bit accumulator (X8, X9) with each new 16-byte block.
	MOV	$16, X12
ieee_fold_loop:
	BLT	X7, X12, ieee_fold_done

	// Load next 16 bytes of data
	MOV	(X6), X13		// d0 = new low 64 bits
	MOV	8(X6), X14		// d1 = new high 64 bits
	ADD	$16, X6
	ADD	$-16, X7

	// Carry-less multiply fold:
	//   new_low  = clmul(K1, t0)  XOR clmul(K2, t1)  XOR d0
	//   new_high = clmulh(K1, t0) XOR clmulh(K2, t1) XOR d1
	CLMUL	X11, X8, X15		// clmul(K1, t0) → low result
	CLMULH	X11, X8, X16		// clmulh(K1, t0) → high result
	CLMUL	X10, X9, X17		// clmul(K2, t1) → low result
	CLMULH	X10, X9, X18		// clmulh(K2, t1) → high result

	// Combine fold results with new data
	XOR	X15, X17, X8		// t0 = fold_low
	XOR	X8, X13, X8		// t0 ^= d0
	XOR	X16, X18, X9		// t1 = fold_high
	XOR	X9, X14, X9		// t1 ^= d1

	JMP	ieee_fold_loop

ieee_fold_done:
	// (X8, X9) = (t0, t1) holds the folded 128-bit accumulator.
	// Now reduce 128 bits → 64 bits → 32 bits.

	// Load final reduction constants
	MOV	ieee_reduce<>+0(SB), X10	// const_low
	MOV	ieee_reduce<>+8(SB), X11	// const_high

	// 128-bit → 64-bit folding
	// Formula (from ISA-L reflected CRC):
	//   t4 = clmul(t0, const_high)
	//   t3h = clmulh(t0, const_high)
	//   t1 = t1 XOR t4
	//   low32 = t1 & 0xFFFFFFFF
	//   high32 = t1 >> 32
	//   f = clmul(low32, const_low)
	//   result = (t3h << 32) XOR high32 XOR f
	CLMUL	X11, X8, X12		// t4 = clmul(t0, const_high)
	CLMULH	X11, X8, X13		// t3h = clmulh(t0, const_high)
	XOR	X9, X12, X9		// t1 = t1 XOR t4

	// Extract low 32 and high 32 bits from t1
	MOV	mask32<>(SB), X20	// X20 = 0xFFFFFFFF
	AND	X9, X20, X14		// low32 = t1 & 0xFFFFFFFF
	SRLI	$32, X9, X15		// high32 = t1 >> 32

	CLMUL	X10, X14, X16		// f = clmul(low32, const_low)
	SLLI	$32, X13, X13		// t3h <<= 32
	XOR	X13, X15, X17		// result = (t3h << 32) XOR high32
	XOR	X17, X16, X17		// result ^= f

	// Barrett reduction: 64 bits → 32 bits
	// Formula:
	//   tmp = clmul(result_low32, μ)
	//   tmp = clmul(tmp_low32, P)
	//   crc = (result XOR tmp) >> 32
	AND	X17, X20, X18		// low32 = result & 0xFFFFFFFF
	MOV	ieee_reduce<>+16(SB), X21	// const_quo (μ)
	MOV	ieee_reduce<>+24(SB), X22	// const_poly (P)
	CLMUL	X21, X18, X18		// tmp = clmul(low32, μ)
	AND	X18, X20, X18		// tmp = tmp & 0xFFFFFFFF
	CLMUL	X22, X18, X18		// tmp = clmul(tmp, P)
	XOR	X17, X18, X18		// result = result XOR tmp
	SRLI	$32, X18, X5		// CRC = upper 32 bits

	MOVW	X5, ret+32(FP)
	RET


// Computes CRC32-C (Castagnoli) using carry-less multiplication (Zbc extension).
// Expects non-inverted CRC input. Returns non-inverted CRC.
// Requires len(p) >= 64 and len(p) is a multiple of 16.
TEXT ·castagnoliUpdateCLMUL(SB), NOSPLIT, $0-36
	MOVWU	crc+0(FP), X5		// CRC value (inverted by caller)
	MOV	p+8(FP), X6		// data pointer
	MOV	p_len+16(FP), X7	// len(p)

	// Load first 16 bytes of data
	MOV	(X6), X8		// t0 = low 64 bits
	MOV	8(X6), X9		// t1 = high 64 bits
	ADD	$16, X6
	ADD	$-16, X7

	// XOR CRC seed into the low word of the first block
	XOR	X5, X8, X8

	// Load fold constants: K2 (high), K1 (low)
	MOV	cast_fold<>+0(SB), X10		// K2
	MOV	cast_fold<>+8(SB), X11		// K1

	// Main fold loop: process 16 bytes per iteration.
	MOV	$16, X12
cast_fold_loop:
	BLT	X7, X12, cast_fold_done

	// Load next 16 bytes of data
	MOV	(X6), X13		// d0 = new low 64 bits
	MOV	8(X6), X14		// d1 = new high 64 bits
	ADD	$16, X6
	ADD	$-16, X7

	// Carry-less multiply fold
	CLMUL	X11, X8, X15		// clmul(K1, t0)
	CLMULH	X11, X8, X16		// clmulh(K1, t0)
	CLMUL	X10, X9, X17		// clmul(K2, t1)
	CLMULH	X10, X9, X18		// clmulh(K2, t1)

	// Combine fold results with new data
	XOR	X15, X17, X8		// t0 = fold_low
	XOR	X8, X13, X8		// t0 ^= d0
	XOR	X16, X18, X9		// t1 = fold_high
	XOR	X9, X14, X9		// t1 ^= d1

	JMP	cast_fold_loop

cast_fold_done:
	// Load final reduction constants
	MOV	cast_reduce<>+0(SB), X10	// const_low
	MOV	cast_reduce<>+8(SB), X11	// const_high

	// 128-bit → 64-bit folding
	CLMUL	X11, X8, X12		// t4 = clmul(t0, const_high)
	CLMULH	X11, X8, X13		// t3h = clmulh(t0, const_high)
	XOR	X9, X12, X9		// t1 = t1 XOR t4

	MOV	mask32<>(SB), X20	// X20 = 0xFFFFFFFF
	AND	X9, X20, X14		// low32 = t1 & 0xFFFFFFFF
	SRLI	$32, X9, X15		// high32 = t1 >> 32

	CLMUL	X10, X14, X16		// f = clmul(low32, const_low)
	SLLI	$32, X13, X13		// t3h <<= 32
	XOR	X13, X15, X17		// result = (t3h << 32) XOR high32
	XOR	X17, X16, X17		// result ^= f

	// Barrett reduction: 64 bits → 32 bits
	AND	X17, X20, X18		// low32 = result & 0xFFFFFFFF
	MOV	cast_reduce<>+16(SB), X21	// const_quo (μ)
	MOV	cast_reduce<>+24(SB), X22	// const_poly (P)
	CLMUL	X21, X18, X18		// tmp = clmul(low32, μ)
	AND	X18, X20, X18		// tmp = tmp & 0xFFFFFFFF
	CLMUL	X22, X18, X18		// tmp = clmul(tmp, P)
	XOR	X17, X18, X18		// result = result XOR tmp
	SRLI	$32, X18, X5		// CRC = upper 32 bits

	MOVW	X5, ret+32(FP)
	RET
