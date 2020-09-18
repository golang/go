// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains constant-time, 64-bit assembly implementation of
// P256. The optimizations performed here are described in detail in:
// S.Gueron and V.Krasnov, "Fast prime field elliptic-curve cryptography with
//                          256-bit primes"
// https://link.springer.com/article/10.1007%2Fs13389-014-0090-x
// https://eprint.iacr.org/2013/816.pdf

#include "textflag.h"

#define res_ptr DI
#define x_ptr SI
#define y_ptr CX

#define acc0 R8
#define acc1 R9
#define acc2 R10
#define acc3 R11
#define acc4 R12
#define acc5 R13
#define t0 R14
#define t1 R15

DATA p256const0<>+0x00(SB)/8, $0x00000000ffffffff
DATA p256const1<>+0x00(SB)/8, $0xffffffff00000001
DATA p256ordK0<>+0x00(SB)/8, $0xccd1c8aaee00bc4f
DATA p256ord<>+0x00(SB)/8, $0xf3b9cac2fc632551
DATA p256ord<>+0x08(SB)/8, $0xbce6faada7179e84
DATA p256ord<>+0x10(SB)/8, $0xffffffffffffffff
DATA p256ord<>+0x18(SB)/8, $0xffffffff00000000
DATA p256one<>+0x00(SB)/8, $0x0000000000000001
DATA p256one<>+0x08(SB)/8, $0xffffffff00000000
DATA p256one<>+0x10(SB)/8, $0xffffffffffffffff
DATA p256one<>+0x18(SB)/8, $0x00000000fffffffe
GLOBL p256const0<>(SB), 8, $8
GLOBL p256const1<>(SB), 8, $8
GLOBL p256ordK0<>(SB), 8, $8
GLOBL p256ord<>(SB), 8, $32
GLOBL p256one<>(SB), 8, $32

/* ---------------------------------------*/
// func p256LittleToBig(res []byte, in []uint64)
TEXT ·p256LittleToBig(SB),NOSPLIT,$0
	JMP ·p256BigToLittle(SB)
/* ---------------------------------------*/
// func p256BigToLittle(res []uint64, in []byte)
TEXT ·p256BigToLittle(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ in+24(FP), x_ptr

	MOVQ (8*0)(x_ptr), acc0
	MOVQ (8*1)(x_ptr), acc1
	MOVQ (8*2)(x_ptr), acc2
	MOVQ (8*3)(x_ptr), acc3

	BSWAPQ acc0
	BSWAPQ acc1
	BSWAPQ acc2
	BSWAPQ acc3

	MOVQ acc3, (8*0)(res_ptr)
	MOVQ acc2, (8*1)(res_ptr)
	MOVQ acc1, (8*2)(res_ptr)
	MOVQ acc0, (8*3)(res_ptr)

	RET
/* ---------------------------------------*/
// func p256MovCond(res, a, b []uint64, cond int)
// If cond == 0 res=b, else res=a
TEXT ·p256MovCond(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ a+24(FP), x_ptr
	MOVQ b+48(FP), y_ptr
	MOVQ cond+72(FP), X12

	PXOR X13, X13
	PSHUFD $0, X12, X12
	PCMPEQL X13, X12

	MOVOU X12, X0
	MOVOU (16*0)(x_ptr), X6
	PANDN X6, X0
	MOVOU X12, X1
	MOVOU (16*1)(x_ptr), X7
	PANDN X7, X1
	MOVOU X12, X2
	MOVOU (16*2)(x_ptr), X8
	PANDN X8, X2
	MOVOU X12, X3
	MOVOU (16*3)(x_ptr), X9
	PANDN X9, X3
	MOVOU X12, X4
	MOVOU (16*4)(x_ptr), X10
	PANDN X10, X4
	MOVOU X12, X5
	MOVOU (16*5)(x_ptr), X11
	PANDN X11, X5

	MOVOU (16*0)(y_ptr), X6
	MOVOU (16*1)(y_ptr), X7
	MOVOU (16*2)(y_ptr), X8
	MOVOU (16*3)(y_ptr), X9
	MOVOU (16*4)(y_ptr), X10
	MOVOU (16*5)(y_ptr), X11

	PAND X12, X6
	PAND X12, X7
	PAND X12, X8
	PAND X12, X9
	PAND X12, X10
	PAND X12, X11

	PXOR X6, X0
	PXOR X7, X1
	PXOR X8, X2
	PXOR X9, X3
	PXOR X10, X4
	PXOR X11, X5

	MOVOU X0, (16*0)(res_ptr)
	MOVOU X1, (16*1)(res_ptr)
	MOVOU X2, (16*2)(res_ptr)
	MOVOU X3, (16*3)(res_ptr)
	MOVOU X4, (16*4)(res_ptr)
	MOVOU X5, (16*5)(res_ptr)

	RET
/* ---------------------------------------*/
// func p256NegCond(val []uint64, cond int)
TEXT ·p256NegCond(SB),NOSPLIT,$0
	MOVQ val+0(FP), res_ptr
	MOVQ cond+24(FP), t0
	// acc = poly
	MOVQ $-1, acc0
	MOVQ p256const0<>(SB), acc1
	MOVQ $0, acc2
	MOVQ p256const1<>(SB), acc3
	// Load the original value
	MOVQ (8*0)(res_ptr), acc5
	MOVQ (8*1)(res_ptr), x_ptr
	MOVQ (8*2)(res_ptr), y_ptr
	MOVQ (8*3)(res_ptr), t1
	// Speculatively subtract
	SUBQ acc5, acc0
	SBBQ x_ptr, acc1
	SBBQ y_ptr, acc2
	SBBQ t1, acc3
	// If condition is 0, keep original value
	TESTQ t0, t0
	CMOVQEQ acc5, acc0
	CMOVQEQ x_ptr, acc1
	CMOVQEQ y_ptr, acc2
	CMOVQEQ t1, acc3
	// Store result
	MOVQ acc0, (8*0)(res_ptr)
	MOVQ acc1, (8*1)(res_ptr)
	MOVQ acc2, (8*2)(res_ptr)
	MOVQ acc3, (8*3)(res_ptr)

	RET
/* ---------------------------------------*/
// func p256Sqr(res, in []uint64, n int)
TEXT ·p256Sqr(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ in+24(FP), x_ptr
	MOVQ n+48(FP), BX

sqrLoop:

	// y[1:] * y[0]
	MOVQ (8*0)(x_ptr), t0

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	MOVQ AX, acc1
	MOVQ DX, acc2

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, acc3

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, acc4
	// y[2:] * y[1]
	MOVQ (8*1)(x_ptr), t0

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, acc5
	// y[3] * y[2]
	MOVQ (8*2)(x_ptr), t0

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc5
	ADCQ $0, DX
	MOVQ DX, y_ptr
	XORQ t1, t1
	// *2
	ADDQ acc1, acc1
	ADCQ acc2, acc2
	ADCQ acc3, acc3
	ADCQ acc4, acc4
	ADCQ acc5, acc5
	ADCQ y_ptr, y_ptr
	ADCQ $0, t1
	// Missing products
	MOVQ (8*0)(x_ptr), AX
	MULQ AX
	MOVQ AX, acc0
	MOVQ DX, t0

	MOVQ (8*1)(x_ptr), AX
	MULQ AX
	ADDQ t0, acc1
	ADCQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t0

	MOVQ (8*2)(x_ptr), AX
	MULQ AX
	ADDQ t0, acc3
	ADCQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t0

	MOVQ (8*3)(x_ptr), AX
	MULQ AX
	ADDQ t0, acc5
	ADCQ AX, y_ptr
	ADCQ DX, t1
	MOVQ t1, x_ptr
	// First reduction step
	MOVQ acc0, AX
	MOVQ acc0, t1
	SHLQ $32, acc0
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc0, acc1
	ADCQ t1, acc2
	ADCQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, acc0
	// Second reduction step
	MOVQ acc1, AX
	MOVQ acc1, t1
	SHLQ $32, acc1
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc1, acc2
	ADCQ t1, acc3
	ADCQ AX, acc0
	ADCQ $0, DX
	MOVQ DX, acc1
	// Third reduction step
	MOVQ acc2, AX
	MOVQ acc2, t1
	SHLQ $32, acc2
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc2, acc3
	ADCQ t1, acc0
	ADCQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, acc2
	// Last reduction step
	XORQ t0, t0
	MOVQ acc3, AX
	MOVQ acc3, t1
	SHLQ $32, acc3
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc3, acc0
	ADCQ t1, acc1
	ADCQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, acc3
	// Add bits [511:256] of the sqr result
	ADCQ acc4, acc0
	ADCQ acc5, acc1
	ADCQ y_ptr, acc2
	ADCQ x_ptr, acc3
	ADCQ $0, t0

	MOVQ acc0, acc4
	MOVQ acc1, acc5
	MOVQ acc2, y_ptr
	MOVQ acc3, t1
	// Subtract p256
	SUBQ $-1, acc0
	SBBQ p256const0<>(SB) ,acc1
	SBBQ $0, acc2
	SBBQ p256const1<>(SB), acc3
	SBBQ $0, t0

	CMOVQCS acc4, acc0
	CMOVQCS acc5, acc1
	CMOVQCS y_ptr, acc2
	CMOVQCS t1, acc3

	MOVQ acc0, (8*0)(res_ptr)
	MOVQ acc1, (8*1)(res_ptr)
	MOVQ acc2, (8*2)(res_ptr)
	MOVQ acc3, (8*3)(res_ptr)
	MOVQ res_ptr, x_ptr
	DECQ BX
	JNE  sqrLoop

	RET
/* ---------------------------------------*/
// func p256Mul(res, in1, in2 []uint64)
TEXT ·p256Mul(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ in1+24(FP), x_ptr
	MOVQ in2+48(FP), y_ptr
	// x * y[0]
	MOVQ (8*0)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	MOVQ AX, acc0
	MOVQ DX, acc1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, acc2

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, acc3

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, acc4
	XORQ acc5, acc5
	// First reduction step
	MOVQ acc0, AX
	MOVQ acc0, t1
	SHLQ $32, acc0
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc0, acc1
	ADCQ t1, acc2
	ADCQ AX, acc3
	ADCQ DX, acc4
	ADCQ $0, acc5
	XORQ acc0, acc0
	// x * y[1]
	MOVQ (8*1)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc2
	ADCQ $0, DX
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ DX, acc5
	ADCQ $0, acc0
	// Second reduction step
	MOVQ acc1, AX
	MOVQ acc1, t1
	SHLQ $32, acc1
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc1, acc2
	ADCQ t1, acc3
	ADCQ AX, acc4
	ADCQ DX, acc5
	ADCQ $0, acc0
	XORQ acc1, acc1
	// x * y[2]
	MOVQ (8*2)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc5
	ADCQ $0, DX
	ADDQ AX, acc5
	ADCQ DX, acc0
	ADCQ $0, acc1
	// Third reduction step
	MOVQ acc2, AX
	MOVQ acc2, t1
	SHLQ $32, acc2
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc2, acc3
	ADCQ t1, acc4
	ADCQ AX, acc5
	ADCQ DX, acc0
	ADCQ $0, acc1
	XORQ acc2, acc2
	// x * y[3]
	MOVQ (8*3)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc5
	ADCQ $0, DX
	ADDQ AX, acc5
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc0
	ADCQ $0, DX
	ADDQ AX, acc0
	ADCQ DX, acc1
	ADCQ $0, acc2
	// Last reduction step
	MOVQ acc3, AX
	MOVQ acc3, t1
	SHLQ $32, acc3
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc3, acc4
	ADCQ t1, acc5
	ADCQ AX, acc0
	ADCQ DX, acc1
	ADCQ $0, acc2
	// Copy result [255:0]
	MOVQ acc4, x_ptr
	MOVQ acc5, acc3
	MOVQ acc0, t0
	MOVQ acc1, t1
	// Subtract p256
	SUBQ $-1, acc4
	SBBQ p256const0<>(SB) ,acc5
	SBBQ $0, acc0
	SBBQ p256const1<>(SB), acc1
	SBBQ $0, acc2

	CMOVQCS x_ptr, acc4
	CMOVQCS acc3, acc5
	CMOVQCS t0, acc0
	CMOVQCS t1, acc1

	MOVQ acc4, (8*0)(res_ptr)
	MOVQ acc5, (8*1)(res_ptr)
	MOVQ acc0, (8*2)(res_ptr)
	MOVQ acc1, (8*3)(res_ptr)

	RET
/* ---------------------------------------*/
// func p256FromMont(res, in []uint64)
TEXT ·p256FromMont(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ in+24(FP), x_ptr

	MOVQ (8*0)(x_ptr), acc0
	MOVQ (8*1)(x_ptr), acc1
	MOVQ (8*2)(x_ptr), acc2
	MOVQ (8*3)(x_ptr), acc3
	XORQ acc4, acc4

	// Only reduce, no multiplications are needed
	// First stage
	MOVQ acc0, AX
	MOVQ acc0, t1
	SHLQ $32, acc0
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc0, acc1
	ADCQ t1, acc2
	ADCQ AX, acc3
	ADCQ DX, acc4
	XORQ acc5, acc5
	// Second stage
	MOVQ acc1, AX
	MOVQ acc1, t1
	SHLQ $32, acc1
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc1, acc2
	ADCQ t1, acc3
	ADCQ AX, acc4
	ADCQ DX, acc5
	XORQ acc0, acc0
	// Third stage
	MOVQ acc2, AX
	MOVQ acc2, t1
	SHLQ $32, acc2
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc2, acc3
	ADCQ t1, acc4
	ADCQ AX, acc5
	ADCQ DX, acc0
	XORQ acc1, acc1
	// Last stage
	MOVQ acc3, AX
	MOVQ acc3, t1
	SHLQ $32, acc3
	MULQ p256const1<>(SB)
	SHRQ $32, t1
	ADDQ acc3, acc4
	ADCQ t1, acc5
	ADCQ AX, acc0
	ADCQ DX, acc1

	MOVQ acc4, x_ptr
	MOVQ acc5, acc3
	MOVQ acc0, t0
	MOVQ acc1, t1

	SUBQ $-1, acc4
	SBBQ p256const0<>(SB), acc5
	SBBQ $0, acc0
	SBBQ p256const1<>(SB), acc1

	CMOVQCS x_ptr, acc4
	CMOVQCS acc3, acc5
	CMOVQCS t0, acc0
	CMOVQCS t1, acc1

	MOVQ acc4, (8*0)(res_ptr)
	MOVQ acc5, (8*1)(res_ptr)
	MOVQ acc0, (8*2)(res_ptr)
	MOVQ acc1, (8*3)(res_ptr)

	RET
/* ---------------------------------------*/
// Constant time point access to arbitrary point table.
// Indexed from 1 to 15, with -1 offset
// (index 0 is implicitly point at infinity)
// func p256Select(point, table []uint64, idx int)
TEXT ·p256Select(SB),NOSPLIT,$0
	MOVQ idx+48(FP),AX
	MOVQ table+24(FP),DI
	MOVQ point+0(FP),DX

	PXOR X15, X15	// X15 = 0
	PCMPEQL X14, X14 // X14 = -1
	PSUBL X14, X15   // X15 = 1
	MOVL AX, X14
	PSHUFD $0, X14, X14

	PXOR X0, X0
	PXOR X1, X1
	PXOR X2, X2
	PXOR X3, X3
	PXOR X4, X4
	PXOR X5, X5
	MOVQ $16, AX

	MOVOU X15, X13

loop_select:

		MOVOU X13, X12
		PADDL X15, X13
		PCMPEQL X14, X12

		MOVOU (16*0)(DI), X6
		MOVOU (16*1)(DI), X7
		MOVOU (16*2)(DI), X8
		MOVOU (16*3)(DI), X9
		MOVOU (16*4)(DI), X10
		MOVOU (16*5)(DI), X11
		ADDQ $(16*6), DI

		PAND X12, X6
		PAND X12, X7
		PAND X12, X8
		PAND X12, X9
		PAND X12, X10
		PAND X12, X11

		PXOR X6, X0
		PXOR X7, X1
		PXOR X8, X2
		PXOR X9, X3
		PXOR X10, X4
		PXOR X11, X5

		DECQ AX
		JNE loop_select

	MOVOU X0, (16*0)(DX)
	MOVOU X1, (16*1)(DX)
	MOVOU X2, (16*2)(DX)
	MOVOU X3, (16*3)(DX)
	MOVOU X4, (16*4)(DX)
	MOVOU X5, (16*5)(DX)

	RET
/* ---------------------------------------*/
// Constant time point access to base point table.
// func p256SelectBase(point, table []uint64, idx int)
TEXT ·p256SelectBase(SB),NOSPLIT,$0
	MOVQ idx+48(FP),AX
	MOVQ table+24(FP),DI
	MOVQ point+0(FP),DX

	PXOR X15, X15	// X15 = 0
	PCMPEQL X14, X14 // X14 = -1
	PSUBL X14, X15   // X15 = 1
	MOVL AX, X14
	PSHUFD $0, X14, X14

	PXOR X0, X0
	PXOR X1, X1
	PXOR X2, X2
	PXOR X3, X3
	MOVQ $16, AX

	MOVOU X15, X13

loop_select_base:

		MOVOU X13, X12
		PADDL X15, X13
		PCMPEQL X14, X12

		MOVOU (16*0)(DI), X4
		MOVOU (16*1)(DI), X5
		MOVOU (16*2)(DI), X6
		MOVOU (16*3)(DI), X7

		MOVOU (16*4)(DI), X8
		MOVOU (16*5)(DI), X9
		MOVOU (16*6)(DI), X10
		MOVOU (16*7)(DI), X11

		ADDQ $(16*8), DI

		PAND X12, X4
		PAND X12, X5
		PAND X12, X6
		PAND X12, X7

		MOVOU X13, X12
		PADDL X15, X13
		PCMPEQL X14, X12

		PAND X12, X8
		PAND X12, X9
		PAND X12, X10
		PAND X12, X11

		PXOR X4, X0
		PXOR X5, X1
		PXOR X6, X2
		PXOR X7, X3

		PXOR X8, X0
		PXOR X9, X1
		PXOR X10, X2
		PXOR X11, X3

		DECQ AX
		JNE loop_select_base

	MOVOU X0, (16*0)(DX)
	MOVOU X1, (16*1)(DX)
	MOVOU X2, (16*2)(DX)
	MOVOU X3, (16*3)(DX)

	RET
/* ---------------------------------------*/
// func p256OrdMul(res, in1, in2 []uint64)
TEXT ·p256OrdMul(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ in1+24(FP), x_ptr
	MOVQ in2+48(FP), y_ptr
	// x * y[0]
	MOVQ (8*0)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	MOVQ AX, acc0
	MOVQ DX, acc1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, acc2

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, acc3

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, acc4
	XORQ acc5, acc5
	// First reduction step
	MOVQ acc0, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc0
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc1
	ADCQ $0, DX
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x10(SB), AX
	MULQ t0
	ADDQ t1, acc2
	ADCQ $0, DX
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x18(SB), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ DX, acc4
	ADCQ $0, acc5
	// x * y[1]
	MOVQ (8*1)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc2
	ADCQ $0, DX
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ DX, acc5
	ADCQ $0, acc0
	// Second reduction step
	MOVQ acc1, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc2
	ADCQ $0, DX
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x10(SB), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x18(SB), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ DX, acc5
	ADCQ $0, acc0
	// x * y[2]
	MOVQ (8*2)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc5
	ADCQ $0, DX
	ADDQ AX, acc5
	ADCQ DX, acc0
	ADCQ $0, acc1
	// Third reduction step
	MOVQ acc2, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x10(SB), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x18(SB), AX
	MULQ t0
	ADDQ t1, acc5
	ADCQ $0, DX
	ADDQ AX, acc5
	ADCQ DX, acc0
	ADCQ $0, acc1
	// x * y[3]
	MOVQ (8*3)(y_ptr), t0

	MOVQ (8*0)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc5
	ADCQ $0, DX
	ADDQ AX, acc5
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc0
	ADCQ $0, DX
	ADDQ AX, acc0
	ADCQ DX, acc1
	ADCQ $0, acc2
	// Last reduction step
	MOVQ acc3, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x10(SB), AX
	MULQ t0
	ADDQ t1, acc5
	ADCQ $0, DX
	ADDQ AX, acc5
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x18(SB), AX
	MULQ t0
	ADDQ t1, acc0
	ADCQ $0, DX
	ADDQ AX, acc0
	ADCQ DX, acc1
	ADCQ $0, acc2
	// Copy result [255:0]
	MOVQ acc4, x_ptr
	MOVQ acc5, acc3
	MOVQ acc0, t0
	MOVQ acc1, t1
	// Subtract p256
	SUBQ p256ord<>+0x00(SB), acc4
	SBBQ p256ord<>+0x08(SB) ,acc5
	SBBQ p256ord<>+0x10(SB), acc0
	SBBQ p256ord<>+0x18(SB), acc1
	SBBQ $0, acc2

	CMOVQCS x_ptr, acc4
	CMOVQCS acc3, acc5
	CMOVQCS t0, acc0
	CMOVQCS t1, acc1

	MOVQ acc4, (8*0)(res_ptr)
	MOVQ acc5, (8*1)(res_ptr)
	MOVQ acc0, (8*2)(res_ptr)
	MOVQ acc1, (8*3)(res_ptr)

	RET
/* ---------------------------------------*/
// func p256OrdSqr(res, in []uint64, n int)
TEXT ·p256OrdSqr(SB),NOSPLIT,$0
	MOVQ res+0(FP), res_ptr
	MOVQ in+24(FP), x_ptr
	MOVQ n+48(FP), BX

ordSqrLoop:

	// y[1:] * y[0]
	MOVQ (8*0)(x_ptr), t0

	MOVQ (8*1)(x_ptr), AX
	MULQ t0
	MOVQ AX, acc1
	MOVQ DX, acc2

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, acc3

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, acc4
	// y[2:] * y[1]
	MOVQ (8*1)(x_ptr), t0

	MOVQ (8*2)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ t1, acc4
	ADCQ $0, DX
	ADDQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, acc5
	// y[3] * y[2]
	MOVQ (8*2)(x_ptr), t0

	MOVQ (8*3)(x_ptr), AX
	MULQ t0
	ADDQ AX, acc5
	ADCQ $0, DX
	MOVQ DX, y_ptr
	XORQ t1, t1
	// *2
	ADDQ acc1, acc1
	ADCQ acc2, acc2
	ADCQ acc3, acc3
	ADCQ acc4, acc4
	ADCQ acc5, acc5
	ADCQ y_ptr, y_ptr
	ADCQ $0, t1
	// Missing products
	MOVQ (8*0)(x_ptr), AX
	MULQ AX
	MOVQ AX, acc0
	MOVQ DX, t0

	MOVQ (8*1)(x_ptr), AX
	MULQ AX
	ADDQ t0, acc1
	ADCQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t0

	MOVQ (8*2)(x_ptr), AX
	MULQ AX
	ADDQ t0, acc3
	ADCQ AX, acc4
	ADCQ $0, DX
	MOVQ DX, t0

	MOVQ (8*3)(x_ptr), AX
	MULQ AX
	ADDQ t0, acc5
	ADCQ AX, y_ptr
	ADCQ DX, t1
	MOVQ t1, x_ptr
	// First reduction step
	MOVQ acc0, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc0
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc1
	ADCQ $0, DX
	ADDQ AX, acc1

	MOVQ t0, t1
	ADCQ DX, acc2
	ADCQ $0, t1
	SUBQ t0, acc2
	SBBQ $0, t1

	MOVQ t0, AX
	MOVQ t0, DX
	MOVQ t0, acc0
	SHLQ $32, AX
	SHRQ $32, DX

	ADDQ t1, acc3
	ADCQ $0, acc0
	SUBQ AX, acc3
	SBBQ DX, acc0
	// Second reduction step
	MOVQ acc1, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc1
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc2
	ADCQ $0, DX
	ADDQ AX, acc2

	MOVQ t0, t1
	ADCQ DX, acc3
	ADCQ $0, t1
	SUBQ t0, acc3
	SBBQ $0, t1

	MOVQ t0, AX
	MOVQ t0, DX
	MOVQ t0, acc1
	SHLQ $32, AX
	SHRQ $32, DX

	ADDQ t1, acc0
	ADCQ $0, acc1
	SUBQ AX, acc0
	SBBQ DX, acc1
	// Third reduction step
	MOVQ acc2, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc2
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc3
	ADCQ $0, DX
	ADDQ AX, acc3

	MOVQ t0, t1
	ADCQ DX, acc0
	ADCQ $0, t1
	SUBQ t0, acc0
	SBBQ $0, t1

	MOVQ t0, AX
	MOVQ t0, DX
	MOVQ t0, acc2
	SHLQ $32, AX
	SHRQ $32, DX

	ADDQ t1, acc1
	ADCQ $0, acc2
	SUBQ AX, acc1
	SBBQ DX, acc2
	// Last reduction step
	MOVQ acc3, AX
	MULQ p256ordK0<>(SB)
	MOVQ AX, t0

	MOVQ p256ord<>+0x00(SB), AX
	MULQ t0
	ADDQ AX, acc3
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ p256ord<>+0x08(SB), AX
	MULQ t0
	ADDQ t1, acc0
	ADCQ $0, DX
	ADDQ AX, acc0
	ADCQ $0, DX
	MOVQ DX, t1

	MOVQ t0, t1
	ADCQ DX, acc1
	ADCQ $0, t1
	SUBQ t0, acc1
	SBBQ $0, t1

	MOVQ t0, AX
	MOVQ t0, DX
	MOVQ t0, acc3
	SHLQ $32, AX
	SHRQ $32, DX

	ADDQ t1, acc2
	ADCQ $0, acc3
	SUBQ AX, acc2
	SBBQ DX, acc3
	XORQ t0, t0
	// Add bits [511:256] of the sqr result
	ADCQ acc4, acc0
	ADCQ acc5, acc1
	ADCQ y_ptr, acc2
	ADCQ x_ptr, acc3
	ADCQ $0, t0

	MOVQ acc0, acc4
	MOVQ acc1, acc5
	MOVQ acc2, y_ptr
	MOVQ acc3, t1
	// Subtract p256
	SUBQ p256ord<>+0x00(SB), acc0
	SBBQ p256ord<>+0x08(SB) ,acc1
	SBBQ p256ord<>+0x10(SB), acc2
	SBBQ p256ord<>+0x18(SB), acc3
	SBBQ $0, t0

	CMOVQCS acc4, acc0
	CMOVQCS acc5, acc1
	CMOVQCS y_ptr, acc2
	CMOVQCS t1, acc3

	MOVQ acc0, (8*0)(res_ptr)
	MOVQ acc1, (8*1)(res_ptr)
	MOVQ acc2, (8*2)(res_ptr)
	MOVQ acc3, (8*3)(res_ptr)
	MOVQ res_ptr, x_ptr
	DECQ BX
	JNE ordSqrLoop

	RET
/* ---------------------------------------*/
#undef res_ptr
#undef x_ptr
#undef y_ptr

#undef acc0
#undef acc1
#undef acc2
#undef acc3
#undef acc4
#undef acc5
#undef t0
#undef t1
/* ---------------------------------------*/
#define mul0 AX
#define mul1 DX
#define acc0 BX
#define acc1 CX
#define acc2 R8
#define acc3 R9
#define acc4 R10
#define acc5 R11
#define acc6 R12
#define acc7 R13
#define t0 R14
#define t1 R15
#define t2 DI
#define t3 SI
#define hlp BP
/* ---------------------------------------*/
TEXT p256SubInternal(SB),NOSPLIT,$0
	XORQ mul0, mul0
	SUBQ t0, acc4
	SBBQ t1, acc5
	SBBQ t2, acc6
	SBBQ t3, acc7
	SBBQ $0, mul0

	MOVQ acc4, acc0
	MOVQ acc5, acc1
	MOVQ acc6, acc2
	MOVQ acc7, acc3

	ADDQ $-1, acc4
	ADCQ p256const0<>(SB), acc5
	ADCQ $0, acc6
	ADCQ p256const1<>(SB), acc7
	ANDQ $1, mul0

	CMOVQEQ acc0, acc4
	CMOVQEQ acc1, acc5
	CMOVQEQ acc2, acc6
	CMOVQEQ acc3, acc7

	RET
/* ---------------------------------------*/
TEXT p256MulInternal(SB),NOSPLIT,$8
	MOVQ acc4, mul0
	MULQ t0
	MOVQ mul0, acc0
	MOVQ mul1, acc1

	MOVQ acc4, mul0
	MULQ t1
	ADDQ mul0, acc1
	ADCQ $0, mul1
	MOVQ mul1, acc2

	MOVQ acc4, mul0
	MULQ t2
	ADDQ mul0, acc2
	ADCQ $0, mul1
	MOVQ mul1, acc3

	MOVQ acc4, mul0
	MULQ t3
	ADDQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, acc4

	MOVQ acc5, mul0
	MULQ t0
	ADDQ mul0, acc1
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc5, mul0
	MULQ t1
	ADDQ hlp, acc2
	ADCQ $0, mul1
	ADDQ mul0, acc2
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc5, mul0
	MULQ t2
	ADDQ hlp, acc3
	ADCQ $0, mul1
	ADDQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc5, mul0
	MULQ t3
	ADDQ hlp, acc4
	ADCQ $0, mul1
	ADDQ mul0, acc4
	ADCQ $0, mul1
	MOVQ mul1, acc5

	MOVQ acc6, mul0
	MULQ t0
	ADDQ mul0, acc2
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc6, mul0
	MULQ t1
	ADDQ hlp, acc3
	ADCQ $0, mul1
	ADDQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc6, mul0
	MULQ t2
	ADDQ hlp, acc4
	ADCQ $0, mul1
	ADDQ mul0, acc4
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc6, mul0
	MULQ t3
	ADDQ hlp, acc5
	ADCQ $0, mul1
	ADDQ mul0, acc5
	ADCQ $0, mul1
	MOVQ mul1, acc6

	MOVQ acc7, mul0
	MULQ t0
	ADDQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc7, mul0
	MULQ t1
	ADDQ hlp, acc4
	ADCQ $0, mul1
	ADDQ mul0, acc4
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc7, mul0
	MULQ t2
	ADDQ hlp, acc5
	ADCQ $0, mul1
	ADDQ mul0, acc5
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc7, mul0
	MULQ t3
	ADDQ hlp, acc6
	ADCQ $0, mul1
	ADDQ mul0, acc6
	ADCQ $0, mul1
	MOVQ mul1, acc7
	// First reduction step
	MOVQ acc0, mul0
	MOVQ acc0, hlp
	SHLQ $32, acc0
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc0, acc1
	ADCQ hlp, acc2
	ADCQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, acc0
	// Second reduction step
	MOVQ acc1, mul0
	MOVQ acc1, hlp
	SHLQ $32, acc1
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc1, acc2
	ADCQ hlp, acc3
	ADCQ mul0, acc0
	ADCQ $0, mul1
	MOVQ mul1, acc1
	// Third reduction step
	MOVQ acc2, mul0
	MOVQ acc2, hlp
	SHLQ $32, acc2
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc2, acc3
	ADCQ hlp, acc0
	ADCQ mul0, acc1
	ADCQ $0, mul1
	MOVQ mul1, acc2
	// Last reduction step
	MOVQ acc3, mul0
	MOVQ acc3, hlp
	SHLQ $32, acc3
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc3, acc0
	ADCQ hlp, acc1
	ADCQ mul0, acc2
	ADCQ $0, mul1
	MOVQ mul1, acc3
	MOVQ $0, BP
	// Add bits [511:256] of the result
	ADCQ acc0, acc4
	ADCQ acc1, acc5
	ADCQ acc2, acc6
	ADCQ acc3, acc7
	ADCQ $0, hlp
	// Copy result
	MOVQ acc4, acc0
	MOVQ acc5, acc1
	MOVQ acc6, acc2
	MOVQ acc7, acc3
	// Subtract p256
	SUBQ $-1, acc4
	SBBQ p256const0<>(SB) ,acc5
	SBBQ $0, acc6
	SBBQ p256const1<>(SB), acc7
	SBBQ $0, hlp
	// If the result of the subtraction is negative, restore the previous result
	CMOVQCS acc0, acc4
	CMOVQCS acc1, acc5
	CMOVQCS acc2, acc6
	CMOVQCS acc3, acc7

	RET
/* ---------------------------------------*/
TEXT p256SqrInternal(SB),NOSPLIT,$8

	MOVQ acc4, mul0
	MULQ acc5
	MOVQ mul0, acc1
	MOVQ mul1, acc2

	MOVQ acc4, mul0
	MULQ acc6
	ADDQ mul0, acc2
	ADCQ $0, mul1
	MOVQ mul1, acc3

	MOVQ acc4, mul0
	MULQ acc7
	ADDQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, t0

	MOVQ acc5, mul0
	MULQ acc6
	ADDQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, hlp

	MOVQ acc5, mul0
	MULQ acc7
	ADDQ hlp, t0
	ADCQ $0, mul1
	ADDQ mul0, t0
	ADCQ $0, mul1
	MOVQ mul1, t1

	MOVQ acc6, mul0
	MULQ acc7
	ADDQ mul0, t1
	ADCQ $0, mul1
	MOVQ mul1, t2
	XORQ t3, t3
	// *2
	ADDQ acc1, acc1
	ADCQ acc2, acc2
	ADCQ acc3, acc3
	ADCQ t0, t0
	ADCQ t1, t1
	ADCQ t2, t2
	ADCQ $0, t3
	// Missing products
	MOVQ acc4, mul0
	MULQ mul0
	MOVQ mul0, acc0
	MOVQ DX, acc4

	MOVQ acc5, mul0
	MULQ mul0
	ADDQ acc4, acc1
	ADCQ mul0, acc2
	ADCQ $0, DX
	MOVQ DX, acc4

	MOVQ acc6, mul0
	MULQ mul0
	ADDQ acc4, acc3
	ADCQ mul0, t0
	ADCQ $0, DX
	MOVQ DX, acc4

	MOVQ acc7, mul0
	MULQ mul0
	ADDQ acc4, t1
	ADCQ mul0, t2
	ADCQ DX, t3
	// First reduction step
	MOVQ acc0, mul0
	MOVQ acc0, hlp
	SHLQ $32, acc0
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc0, acc1
	ADCQ hlp, acc2
	ADCQ mul0, acc3
	ADCQ $0, mul1
	MOVQ mul1, acc0
	// Second reduction step
	MOVQ acc1, mul0
	MOVQ acc1, hlp
	SHLQ $32, acc1
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc1, acc2
	ADCQ hlp, acc3
	ADCQ mul0, acc0
	ADCQ $0, mul1
	MOVQ mul1, acc1
	// Third reduction step
	MOVQ acc2, mul0
	MOVQ acc2, hlp
	SHLQ $32, acc2
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc2, acc3
	ADCQ hlp, acc0
	ADCQ mul0, acc1
	ADCQ $0, mul1
	MOVQ mul1, acc2
	// Last reduction step
	MOVQ acc3, mul0
	MOVQ acc3, hlp
	SHLQ $32, acc3
	MULQ p256const1<>(SB)
	SHRQ $32, hlp
	ADDQ acc3, acc0
	ADCQ hlp, acc1
	ADCQ mul0, acc2
	ADCQ $0, mul1
	MOVQ mul1, acc3
	MOVQ $0, BP
	// Add bits [511:256] of the result
	ADCQ acc0, t0
	ADCQ acc1, t1
	ADCQ acc2, t2
	ADCQ acc3, t3
	ADCQ $0, hlp
	// Copy result
	MOVQ t0, acc4
	MOVQ t1, acc5
	MOVQ t2, acc6
	MOVQ t3, acc7
	// Subtract p256
	SUBQ $-1, acc4
	SBBQ p256const0<>(SB) ,acc5
	SBBQ $0, acc6
	SBBQ p256const1<>(SB), acc7
	SBBQ $0, hlp
	// If the result of the subtraction is negative, restore the previous result
	CMOVQCS t0, acc4
	CMOVQCS t1, acc5
	CMOVQCS t2, acc6
	CMOVQCS t3, acc7

	RET
/* ---------------------------------------*/
#define p256MulBy2Inline\
	XORQ mul0, mul0;\
	ADDQ acc4, acc4;\
	ADCQ acc5, acc5;\
	ADCQ acc6, acc6;\
	ADCQ acc7, acc7;\
	ADCQ $0, mul0;\
	MOVQ acc4, t0;\
	MOVQ acc5, t1;\
	MOVQ acc6, t2;\
	MOVQ acc7, t3;\
	SUBQ $-1, t0;\
	SBBQ p256const0<>(SB), t1;\
	SBBQ $0, t2;\
	SBBQ p256const1<>(SB), t3;\
	SBBQ $0, mul0;\
	CMOVQCS acc4, t0;\
	CMOVQCS acc5, t1;\
	CMOVQCS acc6, t2;\
	CMOVQCS acc7, t3;
/* ---------------------------------------*/
#define p256AddInline \
	XORQ mul0, mul0;\
	ADDQ t0, acc4;\
	ADCQ t1, acc5;\
	ADCQ t2, acc6;\
	ADCQ t3, acc7;\
	ADCQ $0, mul0;\
	MOVQ acc4, t0;\
	MOVQ acc5, t1;\
	MOVQ acc6, t2;\
	MOVQ acc7, t3;\
	SUBQ $-1, t0;\
	SBBQ p256const0<>(SB), t1;\
	SBBQ $0, t2;\
	SBBQ p256const1<>(SB), t3;\
	SBBQ $0, mul0;\
	CMOVQCS acc4, t0;\
	CMOVQCS acc5, t1;\
	CMOVQCS acc6, t2;\
	CMOVQCS acc7, t3;
/* ---------------------------------------*/
#define LDacc(src) MOVQ src(8*0), acc4; MOVQ src(8*1), acc5; MOVQ src(8*2), acc6; MOVQ src(8*3), acc7
#define LDt(src)   MOVQ src(8*0), t0; MOVQ src(8*1), t1; MOVQ src(8*2), t2; MOVQ src(8*3), t3
#define ST(dst)    MOVQ acc4, dst(8*0); MOVQ acc5, dst(8*1); MOVQ acc6, dst(8*2); MOVQ acc7, dst(8*3)
#define STt(dst)   MOVQ t0, dst(8*0); MOVQ t1, dst(8*1); MOVQ t2, dst(8*2); MOVQ t3, dst(8*3)
#define acc2t      MOVQ acc4, t0; MOVQ acc5, t1; MOVQ acc6, t2; MOVQ acc7, t3
#define t2acc      MOVQ t0, acc4; MOVQ t1, acc5; MOVQ t2, acc6; MOVQ t3, acc7
/* ---------------------------------------*/
#define x1in(off) (32*0 + off)(SP)
#define y1in(off) (32*1 + off)(SP)
#define z1in(off) (32*2 + off)(SP)
#define x2in(off) (32*3 + off)(SP)
#define y2in(off) (32*4 + off)(SP)
#define xout(off) (32*5 + off)(SP)
#define yout(off) (32*6 + off)(SP)
#define zout(off) (32*7 + off)(SP)
#define s2(off)   (32*8 + off)(SP)
#define z1sqr(off) (32*9 + off)(SP)
#define h(off)	  (32*10 + off)(SP)
#define r(off)	  (32*11 + off)(SP)
#define hsqr(off) (32*12 + off)(SP)
#define rsqr(off) (32*13 + off)(SP)
#define hcub(off) (32*14 + off)(SP)
#define rptr	  (32*15)(SP)
#define sel_save  (32*15 + 8)(SP)
#define zero_save (32*15 + 8 + 4)(SP)

// func p256PointAddAffineAsm(res, in1, in2 []uint64, sign, sel, zero int)
TEXT ·p256PointAddAffineAsm(SB),0,$512-96
	// Move input to stack in order to free registers
	MOVQ res+0(FP), AX
	MOVQ in1+24(FP), BX
	MOVQ in2+48(FP), CX
	MOVQ sign+72(FP), DX
	MOVQ sel+80(FP), t1
	MOVQ zero+88(FP), t2

	MOVOU (16*0)(BX), X0
	MOVOU (16*1)(BX), X1
	MOVOU (16*2)(BX), X2
	MOVOU (16*3)(BX), X3
	MOVOU (16*4)(BX), X4
	MOVOU (16*5)(BX), X5

	MOVOU X0, x1in(16*0)
	MOVOU X1, x1in(16*1)
	MOVOU X2, y1in(16*0)
	MOVOU X3, y1in(16*1)
	MOVOU X4, z1in(16*0)
	MOVOU X5, z1in(16*1)

	MOVOU (16*0)(CX), X0
	MOVOU (16*1)(CX), X1

	MOVOU X0, x2in(16*0)
	MOVOU X1, x2in(16*1)
	// Store pointer to result
	MOVQ mul0, rptr
	MOVL t1, sel_save
	MOVL t2, zero_save
	// Negate y2in based on sign
	MOVQ (16*2 + 8*0)(CX), acc4
	MOVQ (16*2 + 8*1)(CX), acc5
	MOVQ (16*2 + 8*2)(CX), acc6
	MOVQ (16*2 + 8*3)(CX), acc7
	MOVQ $-1, acc0
	MOVQ p256const0<>(SB), acc1
	MOVQ $0, acc2
	MOVQ p256const1<>(SB), acc3
	XORQ mul0, mul0
	// Speculatively subtract
	SUBQ acc4, acc0
	SBBQ acc5, acc1
	SBBQ acc6, acc2
	SBBQ acc7, acc3
	SBBQ $0, mul0
	MOVQ acc0, t0
	MOVQ acc1, t1
	MOVQ acc2, t2
	MOVQ acc3, t3
	// Add in case the operand was > p256
	ADDQ $-1, acc0
	ADCQ p256const0<>(SB), acc1
	ADCQ $0, acc2
	ADCQ p256const1<>(SB), acc3
	ADCQ $0, mul0
	CMOVQNE t0, acc0
	CMOVQNE t1, acc1
	CMOVQNE t2, acc2
	CMOVQNE t3, acc3
	// If condition is 0, keep original value
	TESTQ DX, DX
	CMOVQEQ acc4, acc0
	CMOVQEQ acc5, acc1
	CMOVQEQ acc6, acc2
	CMOVQEQ acc7, acc3
	// Store result
	MOVQ acc0, y2in(8*0)
	MOVQ acc1, y2in(8*1)
	MOVQ acc2, y2in(8*2)
	MOVQ acc3, y2in(8*3)
	// Begin point add
	LDacc (z1in)
	CALL p256SqrInternal(SB)	// z1ˆ2
	ST (z1sqr)

	LDt (x2in)
	CALL p256MulInternal(SB)	// x2 * z1ˆ2

	LDt (x1in)
	CALL p256SubInternal(SB)	// h = u2 - u1
	ST (h)

	LDt (z1in)
	CALL p256MulInternal(SB)	// z3 = h * z1
	ST (zout)

	LDacc (z1sqr)
	CALL p256MulInternal(SB)	// z1ˆ3

	LDt (y2in)
	CALL p256MulInternal(SB)	// s2 = y2 * z1ˆ3
	ST (s2)

	LDt (y1in)
	CALL p256SubInternal(SB)	// r = s2 - s1
	ST (r)

	CALL p256SqrInternal(SB)	// rsqr = rˆ2
	ST (rsqr)

	LDacc (h)
	CALL p256SqrInternal(SB)	// hsqr = hˆ2
	ST (hsqr)

	LDt (h)
	CALL p256MulInternal(SB)	// hcub = hˆ3
	ST (hcub)

	LDt (y1in)
	CALL p256MulInternal(SB)	// y1 * hˆ3
	ST (s2)

	LDacc (x1in)
	LDt (hsqr)
	CALL p256MulInternal(SB)	// u1 * hˆ2
	ST (h)

	p256MulBy2Inline			// u1 * hˆ2 * 2, inline
	LDacc (rsqr)
	CALL p256SubInternal(SB)	// rˆ2 - u1 * hˆ2 * 2

	LDt (hcub)
	CALL p256SubInternal(SB)
	ST (xout)

	MOVQ acc4, t0
	MOVQ acc5, t1
	MOVQ acc6, t2
	MOVQ acc7, t3
	LDacc (h)
	CALL p256SubInternal(SB)

	LDt (r)
	CALL p256MulInternal(SB)

	LDt (s2)
	CALL p256SubInternal(SB)
	ST (yout)
	// Load stored values from stack
	MOVQ rptr, AX
	MOVL sel_save, BX
	MOVL zero_save, CX
	// The result is not valid if (sel == 0), conditional choose
	MOVOU xout(16*0), X0
	MOVOU xout(16*1), X1
	MOVOU yout(16*0), X2
	MOVOU yout(16*1), X3
	MOVOU zout(16*0), X4
	MOVOU zout(16*1), X5

	MOVL BX, X6
	MOVL CX, X7

	PXOR X8, X8
	PCMPEQL X9, X9

	PSHUFD $0, X6, X6
	PSHUFD $0, X7, X7

	PCMPEQL X8, X6
	PCMPEQL X8, X7

	MOVOU X6, X15
	PANDN X9, X15

	MOVOU x1in(16*0), X9
	MOVOU x1in(16*1), X10
	MOVOU y1in(16*0), X11
	MOVOU y1in(16*1), X12
	MOVOU z1in(16*0), X13
	MOVOU z1in(16*1), X14

	PAND X15, X0
	PAND X15, X1
	PAND X15, X2
	PAND X15, X3
	PAND X15, X4
	PAND X15, X5

	PAND X6, X9
	PAND X6, X10
	PAND X6, X11
	PAND X6, X12
	PAND X6, X13
	PAND X6, X14

	PXOR X9, X0
	PXOR X10, X1
	PXOR X11, X2
	PXOR X12, X3
	PXOR X13, X4
	PXOR X14, X5
	// Similarly if zero == 0
	PCMPEQL X9, X9
	MOVOU X7, X15
	PANDN X9, X15

	MOVOU x2in(16*0), X9
	MOVOU x2in(16*1), X10
	MOVOU y2in(16*0), X11
	MOVOU y2in(16*1), X12
	MOVOU p256one<>+0x00(SB), X13
	MOVOU p256one<>+0x10(SB), X14

	PAND X15, X0
	PAND X15, X1
	PAND X15, X2
	PAND X15, X3
	PAND X15, X4
	PAND X15, X5

	PAND X7, X9
	PAND X7, X10
	PAND X7, X11
	PAND X7, X12
	PAND X7, X13
	PAND X7, X14

	PXOR X9, X0
	PXOR X10, X1
	PXOR X11, X2
	PXOR X12, X3
	PXOR X13, X4
	PXOR X14, X5
	// Finally output the result
	MOVOU X0, (16*0)(AX)
	MOVOU X1, (16*1)(AX)
	MOVOU X2, (16*2)(AX)
	MOVOU X3, (16*3)(AX)
	MOVOU X4, (16*4)(AX)
	MOVOU X5, (16*5)(AX)
	MOVQ $0, rptr

	RET
#undef x1in
#undef y1in
#undef z1in
#undef x2in
#undef y2in
#undef xout
#undef yout
#undef zout
#undef s2
#undef z1sqr
#undef h
#undef r
#undef hsqr
#undef rsqr
#undef hcub
#undef rptr
#undef sel_save
#undef zero_save

// p256IsZero returns 1 in AX if [acc4..acc7] represents zero and zero
// otherwise. It writes to [acc4..acc7], t0 and t1.
TEXT p256IsZero(SB),NOSPLIT,$0
	// AX contains a flag that is set if the input is zero.
	XORQ AX, AX
	MOVQ $1, t1

	// Check whether [acc4..acc7] are all zero.
	MOVQ acc4, t0
	ORQ acc5, t0
	ORQ acc6, t0
	ORQ acc7, t0

	// Set the zero flag if so. (CMOV of a constant to a register doesn't
	// appear to be supported in Go. Thus t1 = 1.)
	CMOVQEQ t1, AX

	// XOR [acc4..acc7] with P and compare with zero again.
	XORQ $-1, acc4
	XORQ p256const0<>(SB), acc5
	XORQ p256const1<>(SB), acc7
	ORQ acc5, acc4
	ORQ acc6, acc4
	ORQ acc7, acc4

	// Set the zero flag if so.
	CMOVQEQ t1, AX
	RET

/* ---------------------------------------*/
#define x1in(off) (32*0 + off)(SP)
#define y1in(off) (32*1 + off)(SP)
#define z1in(off) (32*2 + off)(SP)
#define x2in(off) (32*3 + off)(SP)
#define y2in(off) (32*4 + off)(SP)
#define z2in(off) (32*5 + off)(SP)

#define xout(off) (32*6 + off)(SP)
#define yout(off) (32*7 + off)(SP)
#define zout(off) (32*8 + off)(SP)

#define u1(off)    (32*9 + off)(SP)
#define u2(off)    (32*10 + off)(SP)
#define s1(off)    (32*11 + off)(SP)
#define s2(off)    (32*12 + off)(SP)
#define z1sqr(off) (32*13 + off)(SP)
#define z2sqr(off) (32*14 + off)(SP)
#define h(off)     (32*15 + off)(SP)
#define r(off)     (32*16 + off)(SP)
#define hsqr(off)  (32*17 + off)(SP)
#define rsqr(off)  (32*18 + off)(SP)
#define hcub(off)  (32*19 + off)(SP)
#define rptr       (32*20)(SP)
#define points_eq  (32*20+8)(SP)

//func p256PointAddAsm(res, in1, in2 []uint64) int
TEXT ·p256PointAddAsm(SB),0,$680-80
	// See https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-2007-bl
	// Move input to stack in order to free registers
	MOVQ res+0(FP), AX
	MOVQ in1+24(FP), BX
	MOVQ in2+48(FP), CX

	MOVOU (16*0)(BX), X0
	MOVOU (16*1)(BX), X1
	MOVOU (16*2)(BX), X2
	MOVOU (16*3)(BX), X3
	MOVOU (16*4)(BX), X4
	MOVOU (16*5)(BX), X5

	MOVOU X0, x1in(16*0)
	MOVOU X1, x1in(16*1)
	MOVOU X2, y1in(16*0)
	MOVOU X3, y1in(16*1)
	MOVOU X4, z1in(16*0)
	MOVOU X5, z1in(16*1)

	MOVOU (16*0)(CX), X0
	MOVOU (16*1)(CX), X1
	MOVOU (16*2)(CX), X2
	MOVOU (16*3)(CX), X3
	MOVOU (16*4)(CX), X4
	MOVOU (16*5)(CX), X5

	MOVOU X0, x2in(16*0)
	MOVOU X1, x2in(16*1)
	MOVOU X2, y2in(16*0)
	MOVOU X3, y2in(16*1)
	MOVOU X4, z2in(16*0)
	MOVOU X5, z2in(16*1)
	// Store pointer to result
	MOVQ AX, rptr
	// Begin point add
	LDacc (z2in)
	CALL p256SqrInternal(SB)	// z2ˆ2
	ST (z2sqr)
	LDt (z2in)
	CALL p256MulInternal(SB)	// z2ˆ3
	LDt (y1in)
	CALL p256MulInternal(SB)	// s1 = z2ˆ3*y1
	ST (s1)

	LDacc (z1in)
	CALL p256SqrInternal(SB)	// z1ˆ2
	ST (z1sqr)
	LDt (z1in)
	CALL p256MulInternal(SB)	// z1ˆ3
	LDt (y2in)
	CALL p256MulInternal(SB)	// s2 = z1ˆ3*y2
	ST (s2)

	LDt (s1)
	CALL p256SubInternal(SB)	// r = s2 - s1
	ST (r)
	CALL p256IsZero(SB)
	MOVQ AX, points_eq

	LDacc (z2sqr)
	LDt (x1in)
	CALL p256MulInternal(SB)	// u1 = x1 * z2ˆ2
	ST (u1)
	LDacc (z1sqr)
	LDt (x2in)
	CALL p256MulInternal(SB)	// u2 = x2 * z1ˆ2
	ST (u2)

	LDt (u1)
	CALL p256SubInternal(SB)	// h = u2 - u1
	ST (h)
	CALL p256IsZero(SB)
	ANDQ points_eq, AX
	MOVQ AX, points_eq

	LDacc (r)
	CALL p256SqrInternal(SB)	// rsqr = rˆ2
	ST (rsqr)

	LDacc (h)
	CALL p256SqrInternal(SB)	// hsqr = hˆ2
	ST (hsqr)

	LDt (h)
	CALL p256MulInternal(SB)	// hcub = hˆ3
	ST (hcub)

	LDt (s1)
	CALL p256MulInternal(SB)
	ST (s2)

	LDacc (z1in)
	LDt (z2in)
	CALL p256MulInternal(SB)	// z1 * z2
	LDt (h)
	CALL p256MulInternal(SB)	// z1 * z2 * h
	ST (zout)

	LDacc (hsqr)
	LDt (u1)
	CALL p256MulInternal(SB)	// hˆ2 * u1
	ST (u2)

	p256MulBy2Inline	// u1 * hˆ2 * 2, inline
	LDacc (rsqr)
	CALL p256SubInternal(SB)	// rˆ2 - u1 * hˆ2 * 2

	LDt (hcub)
	CALL p256SubInternal(SB)
	ST (xout)

	MOVQ acc4, t0
	MOVQ acc5, t1
	MOVQ acc6, t2
	MOVQ acc7, t3
	LDacc (u2)
	CALL p256SubInternal(SB)

	LDt (r)
	CALL p256MulInternal(SB)

	LDt (s2)
	CALL p256SubInternal(SB)
	ST (yout)

	MOVOU xout(16*0), X0
	MOVOU xout(16*1), X1
	MOVOU yout(16*0), X2
	MOVOU yout(16*1), X3
	MOVOU zout(16*0), X4
	MOVOU zout(16*1), X5
	// Finally output the result
	MOVQ rptr, AX
	MOVQ $0, rptr
	MOVOU X0, (16*0)(AX)
	MOVOU X1, (16*1)(AX)
	MOVOU X2, (16*2)(AX)
	MOVOU X3, (16*3)(AX)
	MOVOU X4, (16*4)(AX)
	MOVOU X5, (16*5)(AX)

	MOVQ points_eq, AX
	MOVQ AX, ret+72(FP)

	RET
#undef x1in
#undef y1in
#undef z1in
#undef x2in
#undef y2in
#undef z2in
#undef xout
#undef yout
#undef zout
#undef s1
#undef s2
#undef u1
#undef u2
#undef z1sqr
#undef z2sqr
#undef h
#undef r
#undef hsqr
#undef rsqr
#undef hcub
#undef rptr
/* ---------------------------------------*/
#define x(off) (32*0 + off)(SP)
#define y(off) (32*1 + off)(SP)
#define z(off) (32*2 + off)(SP)

#define s(off)	(32*3 + off)(SP)
#define m(off)	(32*4 + off)(SP)
#define zsqr(off) (32*5 + off)(SP)
#define tmp(off)  (32*6 + off)(SP)
#define rptr	  (32*7)(SP)

//func p256PointDoubleAsm(res, in []uint64)
TEXT ·p256PointDoubleAsm(SB),NOSPLIT,$256-48
	// Move input to stack in order to free registers
	MOVQ res+0(FP), AX
	MOVQ in+24(FP), BX

	MOVOU (16*0)(BX), X0
	MOVOU (16*1)(BX), X1
	MOVOU (16*2)(BX), X2
	MOVOU (16*3)(BX), X3
	MOVOU (16*4)(BX), X4
	MOVOU (16*5)(BX), X5

	MOVOU X0, x(16*0)
	MOVOU X1, x(16*1)
	MOVOU X2, y(16*0)
	MOVOU X3, y(16*1)
	MOVOU X4, z(16*0)
	MOVOU X5, z(16*1)
	// Store pointer to result
	MOVQ AX, rptr
	// Begin point double
	LDacc (z)
	CALL p256SqrInternal(SB)
	ST (zsqr)

	LDt (x)
	p256AddInline
	STt (m)

	LDacc (z)
	LDt (y)
	CALL p256MulInternal(SB)
	p256MulBy2Inline
	MOVQ rptr, AX
	// Store z
	MOVQ t0, (16*4 + 8*0)(AX)
	MOVQ t1, (16*4 + 8*1)(AX)
	MOVQ t2, (16*4 + 8*2)(AX)
	MOVQ t3, (16*4 + 8*3)(AX)

	LDacc (x)
	LDt (zsqr)
	CALL p256SubInternal(SB)
	LDt (m)
	CALL p256MulInternal(SB)
	ST (m)
	// Multiply by 3
	p256MulBy2Inline
	LDacc (m)
	p256AddInline
	STt (m)
	////////////////////////
	LDacc (y)
	p256MulBy2Inline
	t2acc
	CALL p256SqrInternal(SB)
	ST (s)
	CALL p256SqrInternal(SB)
	// Divide by 2
	XORQ mul0, mul0
	MOVQ acc4, t0
	MOVQ acc5, t1
	MOVQ acc6, t2
	MOVQ acc7, t3

	ADDQ $-1, acc4
	ADCQ p256const0<>(SB), acc5
	ADCQ $0, acc6
	ADCQ p256const1<>(SB), acc7
	ADCQ $0, mul0
	TESTQ $1, t0

	CMOVQEQ t0, acc4
	CMOVQEQ t1, acc5
	CMOVQEQ t2, acc6
	CMOVQEQ t3, acc7
	ANDQ t0, mul0

	SHRQ $1, acc5, acc4
	SHRQ $1, acc6, acc5
	SHRQ $1, acc7, acc6
	SHRQ $1, mul0, acc7
	ST (y)
	/////////////////////////
	LDacc (x)
	LDt (s)
	CALL p256MulInternal(SB)
	ST (s)
	p256MulBy2Inline
	STt (tmp)

	LDacc (m)
	CALL p256SqrInternal(SB)
	LDt (tmp)
	CALL p256SubInternal(SB)

	MOVQ rptr, AX
	// Store x
	MOVQ acc4, (16*0 + 8*0)(AX)
	MOVQ acc5, (16*0 + 8*1)(AX)
	MOVQ acc6, (16*0 + 8*2)(AX)
	MOVQ acc7, (16*0 + 8*3)(AX)

	acc2t
	LDacc (s)
	CALL p256SubInternal(SB)

	LDt (m)
	CALL p256MulInternal(SB)

	LDt (y)
	CALL p256SubInternal(SB)
	MOVQ rptr, AX
	// Store y
	MOVQ acc4, (16*2 + 8*0)(AX)
	MOVQ acc5, (16*2 + 8*1)(AX)
	MOVQ acc6, (16*2 + 8*2)(AX)
	MOVQ acc7, (16*2 + 8*3)(AX)
	///////////////////////
	MOVQ $0, rptr

	RET
/* ---------------------------------------*/
