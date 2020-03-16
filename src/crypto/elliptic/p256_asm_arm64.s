// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains constant-time, 64-bit assembly implementation of
// P256. The optimizations performed here are described in detail in:
// S.Gueron and V.Krasnov, "Fast prime field elliptic-curve cryptography with
//                          256-bit primes"
// http://link.springer.com/article/10.1007%2Fs13389-014-0090-x
// https://eprint.iacr.org/2013/816.pdf

#include "textflag.h"

#define res_ptr R0
#define a_ptr R1
#define b_ptr R2

#define acc0 R3
#define acc1 R4
#define acc2 R5
#define acc3 R6

#define acc4 R7
#define acc5 R8
#define acc6 R9
#define acc7 R10
#define t0 R11
#define t1 R12
#define t2 R13
#define t3 R14
#define const0 R15
#define const1 R16

#define hlp0 R17
#define hlp1 res_ptr

#define x0 R19
#define x1 R20
#define x2 R21
#define x3 R22
#define y0 R23
#define y1 R24
#define y2 R25
#define y3 R26

#define const2 t2
#define const3 t3

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
	JMP	·p256BigToLittle(SB)
/* ---------------------------------------*/
// func p256BigToLittle(res []uint64, in []byte)
TEXT ·p256BigToLittle(SB),NOSPLIT,$0
	MOVD	res+0(FP), res_ptr
	MOVD	in+24(FP), a_ptr

	LDP	0*16(a_ptr), (acc0, acc1)
	LDP	1*16(a_ptr), (acc2, acc3)

	REV	acc0, acc0
	REV	acc1, acc1
	REV	acc2, acc2
	REV	acc3, acc3

	STP	(acc3, acc2), 0*16(res_ptr)
	STP	(acc1, acc0), 1*16(res_ptr)
	RET
/* ---------------------------------------*/
// func p256MovCond(res, a, b []uint64, cond int)
// If cond == 0 res=b, else res=a
TEXT ·p256MovCond(SB),NOSPLIT,$0
	MOVD	res+0(FP), res_ptr
	MOVD	a+24(FP), a_ptr
	MOVD	b+48(FP), b_ptr
	MOVD	cond+72(FP), R3

	CMP	$0, R3
	// Two remarks:
	// 1) Will want to revisit NEON, when support is better
	// 2) CSEL might not be constant time on all ARM processors
	LDP	0*16(a_ptr), (R4, R5)
	LDP	1*16(a_ptr), (R6, R7)
	LDP	2*16(a_ptr), (R8, R9)
	LDP	0*16(b_ptr), (R16, R17)
	LDP	1*16(b_ptr), (R19, R20)
	LDP	2*16(b_ptr), (R21, R22)
	CSEL	EQ, R16, R4, R4
	CSEL	EQ, R17, R5, R5
	CSEL	EQ, R19, R6, R6
	CSEL	EQ, R20, R7, R7
	CSEL	EQ, R21, R8, R8
	CSEL	EQ, R22, R9, R9
	STP	(R4, R5), 0*16(res_ptr)
	STP	(R6, R7), 1*16(res_ptr)
	STP	(R8, R9), 2*16(res_ptr)

	LDP	3*16(a_ptr), (R4, R5)
	LDP	4*16(a_ptr), (R6, R7)
	LDP	5*16(a_ptr), (R8, R9)
	LDP	3*16(b_ptr), (R16, R17)
	LDP	4*16(b_ptr), (R19, R20)
	LDP	5*16(b_ptr), (R21, R22)
	CSEL	EQ, R16, R4, R4
	CSEL	EQ, R17, R5, R5
	CSEL	EQ, R19, R6, R6
	CSEL	EQ, R20, R7, R7
	CSEL	EQ, R21, R8, R8
	CSEL	EQ, R22, R9, R9
	STP	(R4, R5), 3*16(res_ptr)
	STP	(R6, R7), 4*16(res_ptr)
	STP	(R8, R9), 5*16(res_ptr)

	RET
/* ---------------------------------------*/
// func p256NegCond(val []uint64, cond int)
TEXT ·p256NegCond(SB),NOSPLIT,$0
	MOVD	val+0(FP), a_ptr
	MOVD	cond+24(FP), hlp0
	MOVD	a_ptr, res_ptr
	// acc = poly
	MOVD	$-1, acc0
	MOVD	p256const0<>(SB), acc1
	MOVD	$0, acc2
	MOVD	p256const1<>(SB), acc3
	// Load the original value
	LDP	0*16(a_ptr), (t0, t1)
	LDP	1*16(a_ptr), (t2, t3)
	// Speculatively subtract
	SUBS	t0, acc0
	SBCS	t1, acc1
	SBCS	t2, acc2
	SBC	t3, acc3
	// If condition is 0, keep original value
	CMP	$0, hlp0
	CSEL	EQ, t0, acc0, acc0
	CSEL	EQ, t1, acc1, acc1
	CSEL	EQ, t2, acc2, acc2
	CSEL	EQ, t3, acc3, acc3
	// Store result
	STP	(acc0, acc1), 0*16(res_ptr)
	STP	(acc2, acc3), 1*16(res_ptr)

	RET
/* ---------------------------------------*/
// func p256Sqr(res, in []uint64, n int)
TEXT ·p256Sqr(SB),NOSPLIT,$0
	MOVD	res+0(FP), res_ptr
	MOVD	in+24(FP), a_ptr
	MOVD	n+48(FP), b_ptr

	MOVD	p256const0<>(SB), const0
	MOVD	p256const1<>(SB), const1

	LDP	0*16(a_ptr), (x0, x1)
	LDP	1*16(a_ptr), (x2, x3)

sqrLoop:
	SUB	$1, b_ptr
	CALL	p256SqrInternal<>(SB)
	MOVD	y0, x0
	MOVD	y1, x1
	MOVD	y2, x2
	MOVD	y3, x3
	CBNZ	b_ptr, sqrLoop

	STP	(y0, y1), 0*16(res_ptr)
	STP	(y2, y3), 1*16(res_ptr)
	RET
/* ---------------------------------------*/
// func p256Mul(res, in1, in2 []uint64)
TEXT ·p256Mul(SB),NOSPLIT,$0
	MOVD	res+0(FP), res_ptr
	MOVD	in1+24(FP), a_ptr
	MOVD	in2+48(FP), b_ptr

	MOVD	p256const0<>(SB), const0
	MOVD	p256const1<>(SB), const1

	LDP	0*16(a_ptr), (x0, x1)
	LDP	1*16(a_ptr), (x2, x3)

	LDP	0*16(b_ptr), (y0, y1)
	LDP	1*16(b_ptr), (y2, y3)

	CALL	p256MulInternal<>(SB)

	STP	(y0, y1), 0*16(res_ptr)
	STP	(y2, y3), 1*16(res_ptr)
	RET
/* ---------------------------------------*/
// func p256FromMont(res, in []uint64)
TEXT ·p256FromMont(SB),NOSPLIT,$0
	MOVD	res+0(FP), res_ptr
	MOVD	in+24(FP), a_ptr

	MOVD	p256const0<>(SB), const0
	MOVD	p256const1<>(SB), const1

	LDP	0*16(a_ptr), (acc0, acc1)
	LDP	1*16(a_ptr), (acc2, acc3)
	// Only reduce, no multiplications are needed
	// First reduction step
	ADDS	acc0<<32, acc1, acc1
	LSR	$32, acc0, t0
	MUL	acc0, const1, t1
	UMULH	acc0, const1, acc0
	ADCS	t0, acc2
	ADCS	t1, acc3
	ADC	$0, acc0
	// Second reduction step
	ADDS	acc1<<32, acc2, acc2
	LSR	$32, acc1, t0
	MUL	acc1, const1, t1
	UMULH	acc1, const1, acc1
	ADCS	t0, acc3
	ADCS	t1, acc0
	ADC	$0, acc1
	// Third reduction step
	ADDS	acc2<<32, acc3, acc3
	LSR	$32, acc2, t0
	MUL	acc2, const1, t1
	UMULH	acc2, const1, acc2
	ADCS	t0, acc0
	ADCS	t1, acc1
	ADC	$0, acc2
	// Last reduction step
	ADDS	acc3<<32, acc0, acc0
	LSR	$32, acc3, t0
	MUL	acc3, const1, t1
	UMULH	acc3, const1, acc3
	ADCS	t0, acc1
	ADCS	t1, acc2
	ADC	$0, acc3

	SUBS	$-1, acc0, t0
	SBCS	const0, acc1, t1
	SBCS	$0, acc2, t2
	SBCS	const1, acc3, t3

	CSEL	CS, t0, acc0, acc0
	CSEL	CS, t1, acc1, acc1
	CSEL	CS, t2, acc2, acc2
	CSEL	CS, t3, acc3, acc3

	STP	(acc0, acc1), 0*16(res_ptr)
	STP	(acc2, acc3), 1*16(res_ptr)

	RET
/* ---------------------------------------*/
// Constant time point access to arbitrary point table.
// Indexed from 1 to 15, with -1 offset
// (index 0 is implicitly point at infinity)
// func p256Select(point, table []uint64, idx int)
TEXT ·p256Select(SB),NOSPLIT,$0
	MOVD	idx+48(FP), const0
	MOVD	table+24(FP), b_ptr
	MOVD	point+0(FP), res_ptr

	EOR	x0, x0, x0
	EOR	x1, x1, x1
	EOR	x2, x2, x2
	EOR	x3, x3, x3
	EOR	y0, y0, y0
	EOR	y1, y1, y1
	EOR	y2, y2, y2
	EOR	y3, y3, y3
	EOR	t0, t0, t0
	EOR	t1, t1, t1
	EOR	t2, t2, t2
	EOR	t3, t3, t3

	MOVD	$0, const1

loop_select:
		ADD	$1, const1
		CMP	const0, const1
		LDP.P	16(b_ptr), (acc0, acc1)
		CSEL	EQ, acc0, x0, x0
		CSEL	EQ, acc1, x1, x1
		LDP.P	16(b_ptr), (acc2, acc3)
		CSEL	EQ, acc2, x2, x2
		CSEL	EQ, acc3, x3, x3
		LDP.P	16(b_ptr), (acc4, acc5)
		CSEL	EQ, acc4, y0, y0
		CSEL	EQ, acc5, y1, y1
		LDP.P	16(b_ptr), (acc6, acc7)
		CSEL	EQ, acc6, y2, y2
		CSEL	EQ, acc7, y3, y3
		LDP.P	16(b_ptr), (acc0, acc1)
		CSEL	EQ, acc0, t0, t0
		CSEL	EQ, acc1, t1, t1
		LDP.P	16(b_ptr), (acc2, acc3)
		CSEL	EQ, acc2, t2, t2
		CSEL	EQ, acc3, t3, t3

		CMP	$16, const1
		BNE	loop_select

	STP	(x0, x1), 0*16(res_ptr)
	STP	(x2, x3), 1*16(res_ptr)
	STP	(y0, y1), 2*16(res_ptr)
	STP	(y2, y3), 3*16(res_ptr)
	STP	(t0, t1), 4*16(res_ptr)
	STP	(t2, t3), 5*16(res_ptr)
	RET
/* ---------------------------------------*/
// Constant time point access to base point table.
// func p256SelectBase(point, table []uint64, idx int)
TEXT ·p256SelectBase(SB),NOSPLIT,$0
	MOVD	idx+48(FP), t0
	MOVD	table+24(FP), t1
	MOVD	point+0(FP), res_ptr

	EOR	x0, x0, x0
	EOR	x1, x1, x1
	EOR	x2, x2, x2
	EOR	x3, x3, x3
	EOR	y0, y0, y0
	EOR	y1, y1, y1
	EOR	y2, y2, y2
	EOR	y3, y3, y3

	MOVD	$0, t2

loop_select:
		ADD	$1, t2
		CMP	t0, t2
		LDP.P	16(t1), (acc0, acc1)
		CSEL	EQ, acc0, x0, x0
		CSEL	EQ, acc1, x1, x1
		LDP.P	16(t1), (acc2, acc3)
		CSEL	EQ, acc2, x2, x2
		CSEL	EQ, acc3, x3, x3
		LDP.P	16(t1), (acc4, acc5)
		CSEL	EQ, acc4, y0, y0
		CSEL	EQ, acc5, y1, y1
		LDP.P	16(t1), (acc6, acc7)
		CSEL	EQ, acc6, y2, y2
		CSEL	EQ, acc7, y3, y3

		CMP	$32, t2
		BNE	loop_select

	STP	(x0, x1), 0*16(res_ptr)
	STP	(x2, x3), 1*16(res_ptr)
	STP	(y0, y1), 2*16(res_ptr)
	STP	(y2, y3), 3*16(res_ptr)
	RET
/* ---------------------------------------*/
// func p256OrdSqr(res, in []uint64, n int)
TEXT ·p256OrdSqr(SB),NOSPLIT,$0
	MOVD	in+24(FP), a_ptr
	MOVD	n+48(FP), b_ptr

	MOVD	p256ordK0<>(SB), hlp1
	LDP	p256ord<>+0x00(SB), (const0, const1)
	LDP	p256ord<>+0x10(SB), (const2, const3)

	LDP	0*16(a_ptr), (x0, x1)
	LDP	1*16(a_ptr), (x2, x3)

ordSqrLoop:
	SUB	$1, b_ptr

	// x[1:] * x[0]
	MUL	x0, x1, acc1
	UMULH	x0, x1, acc2

	MUL	x0, x2, t0
	ADDS	t0, acc2, acc2
	UMULH	x0, x2, acc3

	MUL	x0, x3, t0
	ADCS	t0, acc3, acc3
	UMULH	x0, x3, acc4
	ADC	$0, acc4, acc4
	// x[2:] * x[1]
	MUL	x1, x2, t0
	ADDS	t0, acc3
	UMULH	x1, x2, t1
	ADCS	t1, acc4
	ADC	$0, ZR, acc5

	MUL	x1, x3, t0
	ADDS	t0, acc4
	UMULH	x1, x3, t1
	ADC	t1, acc5
	// x[3] * x[2]
	MUL	x2, x3, t0
	ADDS	t0, acc5
	UMULH	x2, x3, acc6
	ADC	$0, acc6

	MOVD	$0, acc7
	// *2
	ADDS	acc1, acc1
	ADCS	acc2, acc2
	ADCS	acc3, acc3
	ADCS	acc4, acc4
	ADCS	acc5, acc5
	ADCS	acc6, acc6
	ADC	$0, acc7
	// Missing products
	MUL	x0, x0, acc0
	UMULH	x0, x0, t0
	ADDS	t0, acc1, acc1

	MUL	x1, x1, t0
	ADCS	t0, acc2, acc2
	UMULH	x1, x1, t1
	ADCS	t1, acc3, acc3

	MUL	x2, x2, t0
	ADCS	t0, acc4, acc4
	UMULH	x2, x2, t1
	ADCS	t1, acc5, acc5

	MUL	x3, x3, t0
	ADCS	t0, acc6, acc6
	UMULH	x3, x3, t1
	ADC	t1, acc7, acc7
	// First reduction step
	MUL	acc0, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc0, acc0
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc1, acc1
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc2, acc2
	UMULH	const2, hlp0, acc0

	MUL	const3, hlp0, t0
	ADCS	t0, acc3, acc3

	UMULH	const3, hlp0, hlp0
	ADC	$0, hlp0

	ADDS	t1, acc1, acc1
	ADCS	y0, acc2, acc2
	ADCS	acc0, acc3, acc3
	ADC	$0, hlp0, acc0
	// Second reduction step
	MUL	acc1, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc1, acc1
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc2, acc2
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc3, acc3
	UMULH	const2, hlp0, acc1

	MUL	const3, hlp0, t0
	ADCS	t0, acc0, acc0

	UMULH	const3, hlp0, hlp0
	ADC	$0, hlp0

	ADDS	t1, acc2, acc2
	ADCS	y0, acc3, acc3
	ADCS	acc1, acc0, acc0
	ADC	$0, hlp0, acc1
	// Third reduction step
	MUL	acc2, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc2, acc2
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc3, acc3
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc0, acc0
	UMULH	const2, hlp0, acc2

	MUL	const3, hlp0, t0
	ADCS	t0, acc1, acc1

	UMULH	const3, hlp0, hlp0
	ADC	$0, hlp0

	ADDS	t1, acc3, acc3
	ADCS	y0, acc0, acc0
	ADCS	acc2, acc1, acc1
	ADC	$0, hlp0, acc2

	// Last reduction step
	MUL	acc3, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc3, acc3
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc0, acc0
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc1, acc1
	UMULH	const2, hlp0, acc3

	MUL	const3, hlp0, t0
	ADCS	t0, acc2, acc2

	UMULH	const3, hlp0, hlp0
	ADC	$0, acc7

	ADDS	t1, acc0, acc0
	ADCS	y0, acc1, acc1
	ADCS	acc3, acc2, acc2
	ADC	$0, hlp0, acc3

	ADDS	acc4, acc0, acc0
	ADCS	acc5, acc1, acc1
	ADCS	acc6, acc2, acc2
	ADCS	acc7, acc3, acc3
	ADC	$0, ZR, acc4

	SUBS	const0, acc0, y0
	SBCS	const1, acc1, y1
	SBCS	const2, acc2, y2
	SBCS	const3, acc3, y3
	SBCS	$0, acc4, acc4

	CSEL	CS, y0, acc0, x0
	CSEL	CS, y1, acc1, x1
	CSEL	CS, y2, acc2, x2
	CSEL	CS, y3, acc3, x3

	CBNZ	b_ptr, ordSqrLoop

	MOVD	res+0(FP), res_ptr
	STP	(x0, x1), 0*16(res_ptr)
	STP	(x2, x3), 1*16(res_ptr)

	RET
/* ---------------------------------------*/
// func p256OrdMul(res, in1, in2 []uint64)
TEXT ·p256OrdMul(SB),NOSPLIT,$0
	MOVD	in1+24(FP), a_ptr
	MOVD	in2+48(FP), b_ptr

	MOVD	p256ordK0<>(SB), hlp1
	LDP	p256ord<>+0x00(SB), (const0, const1)
	LDP	p256ord<>+0x10(SB), (const2, const3)

	LDP	0*16(a_ptr), (x0, x1)
	LDP	1*16(a_ptr), (x2, x3)
	LDP	0*16(b_ptr), (y0, y1)
	LDP	1*16(b_ptr), (y2, y3)

	// y[0] * x
	MUL	y0, x0, acc0
	UMULH	y0, x0, acc1

	MUL	y0, x1, t0
	ADDS	t0, acc1
	UMULH	y0, x1, acc2

	MUL	y0, x2, t0
	ADCS	t0, acc2
	UMULH	y0, x2, acc3

	MUL	y0, x3, t0
	ADCS	t0, acc3
	UMULH	y0, x3, acc4
	ADC	$0, acc4
	// First reduction step
	MUL	acc0, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc0, acc0
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc1, acc1
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc2, acc2
	UMULH	const2, hlp0, acc0

	MUL	const3, hlp0, t0
	ADCS	t0, acc3, acc3

	UMULH	const3, hlp0, hlp0
	ADC	$0, acc4

	ADDS	t1, acc1, acc1
	ADCS	y0, acc2, acc2
	ADCS	acc0, acc3, acc3
	ADC	$0, hlp0, acc0
	// y[1] * x
	MUL	y1, x0, t0
	ADDS	t0, acc1
	UMULH	y1, x0, t1

	MUL	y1, x1, t0
	ADCS	t0, acc2
	UMULH	y1, x1, hlp0

	MUL	y1, x2, t0
	ADCS	t0, acc3
	UMULH	y1, x2, y0

	MUL	y1, x3, t0
	ADCS	t0, acc4
	UMULH	y1, x3, y1
	ADC	$0, ZR, acc5

	ADDS	t1, acc2
	ADCS	hlp0, acc3
	ADCS	y0, acc4
	ADC	y1, acc5
	// Second reduction step
	MUL	acc1, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc1, acc1
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc2, acc2
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc3, acc3
	UMULH	const2, hlp0, acc1

	MUL	const3, hlp0, t0
	ADCS	t0, acc0, acc0

	UMULH	const3, hlp0, hlp0
	ADC	$0, acc5

	ADDS	t1, acc2, acc2
	ADCS	y0, acc3, acc3
	ADCS	acc1, acc0, acc0
	ADC	$0, hlp0, acc1
	// y[2] * x
	MUL	y2, x0, t0
	ADDS	t0, acc2
	UMULH	y2, x0, t1

	MUL	y2, x1, t0
	ADCS	t0, acc3
	UMULH	y2, x1, hlp0

	MUL	y2, x2, t0
	ADCS	t0, acc4
	UMULH	y2, x2, y0

	MUL	y2, x3, t0
	ADCS	t0, acc5
	UMULH	y2, x3, y1
	ADC	$0, ZR, acc6

	ADDS	t1, acc3
	ADCS	hlp0, acc4
	ADCS	y0, acc5
	ADC	y1, acc6
	// Third reduction step
	MUL	acc2, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc2, acc2
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc3, acc3
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc0, acc0
	UMULH	const2, hlp0, acc2

	MUL	const3, hlp0, t0
	ADCS	t0, acc1, acc1

	UMULH	const3, hlp0, hlp0
	ADC	$0, acc6

	ADDS	t1, acc3, acc3
	ADCS	y0, acc0, acc0
	ADCS	acc2, acc1, acc1
	ADC	$0, hlp0, acc2
	// y[3] * x
	MUL	y3, x0, t0
	ADDS	t0, acc3
	UMULH	y3, x0, t1

	MUL	y3, x1, t0
	ADCS	t0, acc4
	UMULH	y3, x1, hlp0

	MUL	y3, x2, t0
	ADCS	t0, acc5
	UMULH	y3, x2, y0

	MUL	y3, x3, t0
	ADCS	t0, acc6
	UMULH	y3, x3, y1
	ADC	$0, ZR, acc7

	ADDS	t1, acc4
	ADCS	hlp0, acc5
	ADCS	y0, acc6
	ADC	y1, acc7
	// Last reduction step
	MUL	acc3, hlp1, hlp0

	MUL	const0, hlp1, t0
	ADDS	t0, acc3, acc3
	UMULH	const0, hlp0, t1

	MUL	const1, hlp0, t0
	ADCS	t0, acc0, acc0
	UMULH	const1, hlp0, y0

	MUL	const2, hlp0, t0
	ADCS	t0, acc1, acc1
	UMULH	const2, hlp0, acc3

	MUL	const3, hlp0, t0
	ADCS	t0, acc2, acc2

	UMULH	const3, hlp0, hlp0
	ADC	$0, acc7

	ADDS	t1, acc0, acc0
	ADCS	y0, acc1, acc1
	ADCS	acc3, acc2, acc2
	ADC	$0, hlp0, acc3

	ADDS	acc4, acc0, acc0
	ADCS	acc5, acc1, acc1
	ADCS	acc6, acc2, acc2
	ADCS	acc7, acc3, acc3
	ADC	$0, ZR, acc4

	SUBS	const0, acc0, t0
	SBCS	const1, acc1, t1
	SBCS	const2, acc2, t2
	SBCS	const3, acc3, t3
	SBCS	$0, acc4, acc4

	CSEL	CS, t0, acc0, acc0
	CSEL	CS, t1, acc1, acc1
	CSEL	CS, t2, acc2, acc2
	CSEL	CS, t3, acc3, acc3

	MOVD	res+0(FP), res_ptr
	STP	(acc0, acc1), 0*16(res_ptr)
	STP	(acc2, acc3), 1*16(res_ptr)

	RET
/* ---------------------------------------*/
TEXT p256SubInternal<>(SB),NOSPLIT,$0
	SUBS	x0, y0, acc0
	SBCS	x1, y1, acc1
	SBCS	x2, y2, acc2
	SBCS	x3, y3, acc3
	SBC	$0, ZR, t0

	ADDS	$-1, acc0, acc4
	ADCS	const0, acc1, acc5
	ADCS	$0, acc2, acc6
	ADC	const1, acc3, acc7

	ANDS	$1, t0
	CSEL	EQ, acc0, acc4, x0
	CSEL	EQ, acc1, acc5, x1
	CSEL	EQ, acc2, acc6, x2
	CSEL	EQ, acc3, acc7, x3

	RET
/* ---------------------------------------*/
TEXT p256SqrInternal<>(SB),NOSPLIT,$0
	// x[1:] * x[0]
	MUL	x0, x1, acc1
	UMULH	x0, x1, acc2

	MUL	x0, x2, t0
	ADDS	t0, acc2, acc2
	UMULH	x0, x2, acc3

	MUL	x0, x3, t0
	ADCS	t0, acc3, acc3
	UMULH	x0, x3, acc4
	ADC	$0, acc4, acc4
	// x[2:] * x[1]
	MUL	x1, x2, t0
	ADDS	t0, acc3
	UMULH	x1, x2, t1
	ADCS	t1, acc4
	ADC	$0, ZR, acc5

	MUL	x1, x3, t0
	ADDS	t0, acc4
	UMULH	x1, x3, t1
	ADC	t1, acc5
	// x[3] * x[2]
	MUL	x2, x3, t0
	ADDS	t0, acc5
	UMULH	x2, x3, acc6
	ADC	$0, acc6

	MOVD	$0, acc7
	// *2
	ADDS	acc1, acc1
	ADCS	acc2, acc2
	ADCS	acc3, acc3
	ADCS	acc4, acc4
	ADCS	acc5, acc5
	ADCS	acc6, acc6
	ADC	$0, acc7
	// Missing products
	MUL	x0, x0, acc0
	UMULH	x0, x0, t0
	ADDS	t0, acc1, acc1

	MUL	x1, x1, t0
	ADCS	t0, acc2, acc2
	UMULH	x1, x1, t1
	ADCS	t1, acc3, acc3

	MUL	x2, x2, t0
	ADCS	t0, acc4, acc4
	UMULH	x2, x2, t1
	ADCS	t1, acc5, acc5

	MUL	x3, x3, t0
	ADCS	t0, acc6, acc6
	UMULH	x3, x3, t1
	ADCS	t1, acc7, acc7
	// First reduction step
	ADDS	acc0<<32, acc1, acc1
	LSR	$32, acc0, t0
	MUL	acc0, const1, t1
	UMULH	acc0, const1, acc0
	ADCS	t0, acc2, acc2
	ADCS	t1, acc3, acc3
	ADC	$0, acc0, acc0
	// Second reduction step
	ADDS	acc1<<32, acc2, acc2
	LSR	$32, acc1, t0
	MUL	acc1, const1, t1
	UMULH	acc1, const1, acc1
	ADCS	t0, acc3, acc3
	ADCS	t1, acc0, acc0
	ADC	$0, acc1, acc1
	// Third reduction step
	ADDS	acc2<<32, acc3, acc3
	LSR	$32, acc2, t0
	MUL	acc2, const1, t1
	UMULH	acc2, const1, acc2
	ADCS	t0, acc0, acc0
	ADCS	t1, acc1, acc1
	ADC	$0, acc2, acc2
	// Last reduction step
	ADDS	acc3<<32, acc0, acc0
	LSR	$32, acc3, t0
	MUL	acc3, const1, t1
	UMULH	acc3, const1, acc3
	ADCS	t0, acc1, acc1
	ADCS	t1, acc2, acc2
	ADC	$0, acc3, acc3
	// Add bits [511:256] of the sqr result
	ADDS	acc4, acc0, acc0
	ADCS	acc5, acc1, acc1
	ADCS	acc6, acc2, acc2
	ADCS	acc7, acc3, acc3
	ADC	$0, ZR, acc4

	SUBS	$-1, acc0, t0
	SBCS	const0, acc1, t1
	SBCS	$0, acc2, t2
	SBCS	const1, acc3, t3
	SBCS	$0, acc4, acc4

	CSEL	CS, t0, acc0, y0
	CSEL	CS, t1, acc1, y1
	CSEL	CS, t2, acc2, y2
	CSEL	CS, t3, acc3, y3
	RET
/* ---------------------------------------*/
TEXT p256MulInternal<>(SB),NOSPLIT,$0
	// y[0] * x
	MUL	y0, x0, acc0
	UMULH	y0, x0, acc1

	MUL	y0, x1, t0
	ADDS	t0, acc1
	UMULH	y0, x1, acc2

	MUL	y0, x2, t0
	ADCS	t0, acc2
	UMULH	y0, x2, acc3

	MUL	y0, x3, t0
	ADCS	t0, acc3
	UMULH	y0, x3, acc4
	ADC	$0, acc4
	// First reduction step
	ADDS	acc0<<32, acc1, acc1
	LSR	$32, acc0, t0
	MUL	acc0, const1, t1
	UMULH	acc0, const1, acc0
	ADCS	t0, acc2
	ADCS	t1, acc3
	ADC	$0, acc0
	// y[1] * x
	MUL	y1, x0, t0
	ADDS	t0, acc1
	UMULH	y1, x0, t1

	MUL	y1, x1, t0
	ADCS	t0, acc2
	UMULH	y1, x1, t2

	MUL	y1, x2, t0
	ADCS	t0, acc3
	UMULH	y1, x2, t3

	MUL	y1, x3, t0
	ADCS	t0, acc4
	UMULH	y1, x3, hlp0
	ADC	$0, ZR, acc5

	ADDS	t1, acc2
	ADCS	t2, acc3
	ADCS	t3, acc4
	ADC	hlp0, acc5
	// Second reduction step
	ADDS	acc1<<32, acc2, acc2
	LSR	$32, acc1, t0
	MUL	acc1, const1, t1
	UMULH	acc1, const1, acc1
	ADCS	t0, acc3
	ADCS	t1, acc0
	ADC	$0, acc1
	// y[2] * x
	MUL	y2, x0, t0
	ADDS	t0, acc2
	UMULH	y2, x0, t1

	MUL	y2, x1, t0
	ADCS	t0, acc3
	UMULH	y2, x1, t2

	MUL	y2, x2, t0
	ADCS	t0, acc4
	UMULH	y2, x2, t3

	MUL	y2, x3, t0
	ADCS	t0, acc5
	UMULH	y2, x3, hlp0
	ADC	$0, ZR, acc6

	ADDS	t1, acc3
	ADCS	t2, acc4
	ADCS	t3, acc5
	ADC	hlp0, acc6
	// Third reduction step
	ADDS	acc2<<32, acc3, acc3
	LSR	$32, acc2, t0
	MUL	acc2, const1, t1
	UMULH	acc2, const1, acc2
	ADCS	t0, acc0
	ADCS	t1, acc1
	ADC	$0, acc2
	// y[3] * x
	MUL	y3, x0, t0
	ADDS	t0, acc3
	UMULH	y3, x0, t1

	MUL	y3, x1, t0
	ADCS	t0, acc4
	UMULH	y3, x1, t2

	MUL	y3, x2, t0
	ADCS	t0, acc5
	UMULH	y3, x2, t3

	MUL	y3, x3, t0
	ADCS	t0, acc6
	UMULH	y3, x3, hlp0
	ADC	$0, ZR, acc7

	ADDS	t1, acc4
	ADCS	t2, acc5
	ADCS	t3, acc6
	ADC	hlp0, acc7
	// Last reduction step
	ADDS	acc3<<32, acc0, acc0
	LSR	$32, acc3, t0
	MUL	acc3, const1, t1
	UMULH	acc3, const1, acc3
	ADCS	t0, acc1
	ADCS	t1, acc2
	ADC	$0, acc3
	// Add bits [511:256] of the mul result
	ADDS	acc4, acc0, acc0
	ADCS	acc5, acc1, acc1
	ADCS	acc6, acc2, acc2
	ADCS	acc7, acc3, acc3
	ADC	$0, ZR, acc4

	SUBS	$-1, acc0, t0
	SBCS	const0, acc1, t1
	SBCS	$0, acc2, t2
	SBCS	const1, acc3, t3
	SBCS	$0, acc4, acc4

	CSEL	CS, t0, acc0, y0
	CSEL	CS, t1, acc1, y1
	CSEL	CS, t2, acc2, y2
	CSEL	CS, t3, acc3, y3
	RET
/* ---------------------------------------*/
#define p256MulBy2Inline       \
	ADDS	y0, y0, x0;    \
	ADCS	y1, y1, x1;    \
	ADCS	y2, y2, x2;    \
	ADCS	y3, y3, x3;    \
	ADC	$0, ZR, hlp0;  \
	SUBS	$-1, x0, t0;   \
	SBCS	const0, x1, t1;\
	SBCS	$0, x2, t2;    \
	SBCS	const1, x3, t3;\
	SBCS	$0, hlp0, hlp0;\
	CSEL	CC, x0, t0, x0;\
	CSEL	CC, x1, t1, x1;\
	CSEL	CC, x2, t2, x2;\
	CSEL	CC, x3, t3, x3;
/* ---------------------------------------*/
#define x1in(off) (off)(a_ptr)
#define y1in(off) (off + 32)(a_ptr)
#define z1in(off) (off + 64)(a_ptr)
#define x2in(off) (off)(b_ptr)
#define z2in(off) (off + 64)(b_ptr)
#define x3out(off) (off)(res_ptr)
#define y3out(off) (off + 32)(res_ptr)
#define z3out(off) (off + 64)(res_ptr)
#define LDx(src) LDP src(0), (x0, x1); LDP src(16), (x2, x3)
#define LDy(src) LDP src(0), (y0, y1); LDP src(16), (y2, y3)
#define STx(src) STP (x0, x1), src(0); STP (x2, x3), src(16)
#define STy(src) STP (y0, y1), src(0); STP (y2, y3), src(16)
/* ---------------------------------------*/
#define y2in(off)  (32*0 + 8 + off)(RSP)
#define s2(off)    (32*1 + 8 + off)(RSP)
#define z1sqr(off) (32*2 + 8 + off)(RSP)
#define h(off)	   (32*3 + 8 + off)(RSP)
#define r(off)	   (32*4 + 8 + off)(RSP)
#define hsqr(off)  (32*5 + 8 + off)(RSP)
#define rsqr(off)  (32*6 + 8 + off)(RSP)
#define hcub(off)  (32*7 + 8 + off)(RSP)

#define z2sqr(off) (32*8 + 8 + off)(RSP)
#define s1(off) (32*9 + 8 + off)(RSP)
#define u1(off) (32*10 + 8 + off)(RSP)
#define u2(off) (32*11 + 8 + off)(RSP)

// func p256PointAddAffineAsm(res, in1, in2 []uint64, sign, sel, zero int)
TEXT ·p256PointAddAffineAsm(SB),0,$264-96
	MOVD	in1+24(FP), a_ptr
	MOVD	in2+48(FP), b_ptr
	MOVD	sign+72(FP), hlp0
	MOVD	sel+80(FP), hlp1
	MOVD	zero+88(FP), t2

	MOVD	$1, t0
	CMP	$0, t2
	CSEL	EQ, ZR, t0, t2
	CMP	$0, hlp1
	CSEL	EQ, ZR, t0, hlp1

	MOVD	p256const0<>(SB), const0
	MOVD	p256const1<>(SB), const1
	EOR	t2<<1, hlp1

	// Negate y2in based on sign
	LDP	2*16(b_ptr), (y0, y1)
	LDP	3*16(b_ptr), (y2, y3)
	MOVD	$-1, acc0

	SUBS	y0, acc0, acc0
	SBCS	y1, const0, acc1
	SBCS	y2, ZR, acc2
	SBCS	y3, const1, acc3
	SBC	$0, ZR, t0

	ADDS	$-1, acc0, acc4
	ADCS	const0, acc1, acc5
	ADCS	$0, acc2, acc6
	ADCS	const1, acc3, acc7
	ADC	$0, t0, t0

	CMP	$0, t0
	CSEL	EQ, acc4, acc0, acc0
	CSEL	EQ, acc5, acc1, acc1
	CSEL	EQ, acc6, acc2, acc2
	CSEL	EQ, acc7, acc3, acc3
	// If condition is 0, keep original value
	CMP	$0, hlp0
	CSEL	EQ, y0, acc0, y0
	CSEL	EQ, y1, acc1, y1
	CSEL	EQ, y2, acc2, y2
	CSEL	EQ, y3, acc3, y3
	// Store result
	STy(y2in)
	// Begin point add
	LDx(z1in)
	CALL	p256SqrInternal<>(SB)    // z1ˆ2
	STy(z1sqr)

	LDx(x2in)
	CALL	p256MulInternal<>(SB)    // x2 * z1ˆ2

	LDx(x1in)
	CALL	p256SubInternal<>(SB)    // h = u2 - u1
	STx(h)

	LDy(z1in)
	CALL	p256MulInternal<>(SB)    // z3 = h * z1

	LDP	4*16(a_ptr), (acc0, acc1)// iff select[0] == 0, z3 = z1
	LDP	5*16(a_ptr), (acc2, acc3)
	ANDS	$1, hlp1, ZR
	CSEL	EQ, acc0, y0, y0
	CSEL	EQ, acc1, y1, y1
	CSEL	EQ, acc2, y2, y2
	CSEL	EQ, acc3, y3, y3
	LDP	p256one<>+0x00(SB), (acc0, acc1)
	LDP	p256one<>+0x10(SB), (acc2, acc3)
	ANDS	$2, hlp1, ZR            // iff select[1] == 0, z3 = 1
	CSEL	EQ, acc0, y0, y0
	CSEL	EQ, acc1, y1, y1
	CSEL	EQ, acc2, y2, y2
	CSEL	EQ, acc3, y3, y3
	LDx(z1in)
	MOVD	res+0(FP), t0
	STP	(y0, y1), 4*16(t0)
	STP	(y2, y3), 5*16(t0)

	LDy(z1sqr)
	CALL	p256MulInternal<>(SB)    // z1 ^ 3

	LDx(y2in)
	CALL	p256MulInternal<>(SB)    // s2 = y2 * z1ˆ3
	STy(s2)

	LDx(y1in)
	CALL	p256SubInternal<>(SB)    // r = s2 - s1
	STx(r)

	CALL	p256SqrInternal<>(SB)    // rsqr = rˆ2
	STy	(rsqr)

	LDx(h)
	CALL	p256SqrInternal<>(SB)    // hsqr = hˆ2
	STy(hsqr)

	CALL	p256MulInternal<>(SB)    // hcub = hˆ3
	STy(hcub)

	LDx(y1in)
	CALL	p256MulInternal<>(SB)    // y1 * hˆ3
	STy(s2)

	LDP	hsqr(0*8), (x0, x1)
	LDP	hsqr(2*8), (x2, x3)
	LDP	0*16(a_ptr), (y0, y1)
	LDP	1*16(a_ptr), (y2, y3)
	CALL	p256MulInternal<>(SB)    // u1 * hˆ2
	STP	(y0, y1), h(0*8)
	STP	(y2, y3), h(2*8)

	p256MulBy2Inline               // u1 * hˆ2 * 2, inline

	LDy(rsqr)
	CALL	p256SubInternal<>(SB)    // rˆ2 - u1 * hˆ2 * 2

	MOVD	x0, y0
	MOVD	x1, y1
	MOVD	x2, y2
	MOVD	x3, y3
	LDx(hcub)
	CALL	p256SubInternal<>(SB)

	LDP	0*16(a_ptr), (acc0, acc1)
	LDP	1*16(a_ptr), (acc2, acc3)
	ANDS	$1, hlp1, ZR           // iff select[0] == 0, x3 = x1
	CSEL	EQ, acc0, x0, x0
	CSEL	EQ, acc1, x1, x1
	CSEL	EQ, acc2, x2, x2
	CSEL	EQ, acc3, x3, x3
	LDP	0*16(b_ptr), (acc0, acc1)
	LDP	1*16(b_ptr), (acc2, acc3)
	ANDS	$2, hlp1, ZR           // iff select[1] == 0, x3 = x2
	CSEL	EQ, acc0, x0, x0
	CSEL	EQ, acc1, x1, x1
	CSEL	EQ, acc2, x2, x2
	CSEL	EQ, acc3, x3, x3
	MOVD	res+0(FP), t0
	STP	(x0, x1), 0*16(t0)
	STP	(x2, x3), 1*16(t0)

	LDP	h(0*8), (y0, y1)
	LDP	h(2*8), (y2, y3)
	CALL	p256SubInternal<>(SB)

	LDP	r(0*8), (y0, y1)
	LDP	r(2*8), (y2, y3)
	CALL	p256MulInternal<>(SB)

	LDP	s2(0*8), (x0, x1)
	LDP	s2(2*8), (x2, x3)
	CALL	p256SubInternal<>(SB)
	LDP	2*16(a_ptr), (acc0, acc1)
	LDP	3*16(a_ptr), (acc2, acc3)
	ANDS	$1, hlp1, ZR           // iff select[0] == 0, y3 = y1
	CSEL	EQ, acc0, x0, x0
	CSEL	EQ, acc1, x1, x1
	CSEL	EQ, acc2, x2, x2
	CSEL	EQ, acc3, x3, x3
	LDP	y2in(0*8), (acc0, acc1)
	LDP	y2in(2*8), (acc2, acc3)
	ANDS	$2, hlp1, ZR            // iff select[1] == 0, y3 = y2
	CSEL	EQ, acc0, x0, x0
	CSEL	EQ, acc1, x1, x1
	CSEL	EQ, acc2, x2, x2
	CSEL	EQ, acc3, x3, x3
	MOVD	res+0(FP), t0
	STP	(x0, x1), 2*16(t0)
	STP	(x2, x3), 3*16(t0)

	RET

#define p256AddInline          \
	ADDS	y0, x0, x0;    \
	ADCS	y1, x1, x1;    \
	ADCS	y2, x2, x2;    \
	ADCS	y3, x3, x3;    \
	ADC	$0, ZR, hlp0;  \
	SUBS	$-1, x0, t0;   \
	SBCS	const0, x1, t1;\
	SBCS	$0, x2, t2;    \
	SBCS	const1, x3, t3;\
	SBCS	$0, hlp0, hlp0;\
	CSEL	CC, x0, t0, x0;\
	CSEL	CC, x1, t1, x1;\
	CSEL	CC, x2, t2, x2;\
	CSEL	CC, x3, t3, x3;

#define s(off)	(32*0 + 8 + off)(RSP)
#define m(off)	(32*1 + 8 + off)(RSP)
#define zsqr(off) (32*2 + 8 + off)(RSP)
#define tmp(off)  (32*3 + 8 + off)(RSP)

//func p256PointDoubleAsm(res, in []uint64)
TEXT ·p256PointDoubleAsm(SB),NOSPLIT,$136-48
	MOVD	res+0(FP), res_ptr
	MOVD	in+24(FP), a_ptr

	MOVD	p256const0<>(SB), const0
	MOVD	p256const1<>(SB), const1

	// Begin point double
	LDP	4*16(a_ptr), (x0, x1)
	LDP	5*16(a_ptr), (x2, x3)
	CALL	p256SqrInternal<>(SB)
	STP	(y0, y1), zsqr(0*8)
	STP	(y2, y3), zsqr(2*8)

	LDP	0*16(a_ptr), (x0, x1)
	LDP	1*16(a_ptr), (x2, x3)
	p256AddInline
	STx(m)

	LDx(z1in)
	LDy(y1in)
	CALL	p256MulInternal<>(SB)
	p256MulBy2Inline
	STx(z3out)

	LDy(x1in)
	LDx(zsqr)
	CALL	p256SubInternal<>(SB)
	LDy(m)
	CALL	p256MulInternal<>(SB)

	// Multiply by 3
	p256MulBy2Inline
	p256AddInline
	STx(m)

	LDy(y1in)
	p256MulBy2Inline
	CALL	p256SqrInternal<>(SB)
	STy(s)
	MOVD	y0, x0
	MOVD	y1, x1
	MOVD	y2, x2
	MOVD	y3, x3
	CALL	p256SqrInternal<>(SB)

	// Divide by 2
	ADDS	$-1, y0, t0
	ADCS	const0, y1, t1
	ADCS	$0, y2, t2
	ADCS	const1, y3, t3
	ADC	$0, ZR, hlp0

	ANDS	$1, y0, ZR
	CSEL	EQ, y0, t0, t0
	CSEL	EQ, y1, t1, t1
	CSEL	EQ, y2, t2, t2
	CSEL	EQ, y3, t3, t3
	AND	y0, hlp0, hlp0

	EXTR	$1, t0, t1, y0
	EXTR	$1, t1, t2, y1
	EXTR	$1, t2, t3, y2
	EXTR	$1, t3, hlp0, y3
	STy(y3out)

	LDx(x1in)
	LDy(s)
	CALL	p256MulInternal<>(SB)
	STy(s)
	p256MulBy2Inline
	STx(tmp)

	LDx(m)
	CALL	p256SqrInternal<>(SB)
	LDx(tmp)
	CALL	p256SubInternal<>(SB)

	STx(x3out)

	LDy(s)
	CALL	p256SubInternal<>(SB)

	LDy(m)
	CALL	p256MulInternal<>(SB)

	LDx(y3out)
	CALL	p256SubInternal<>(SB)
	STx(y3out)
	RET
/* ---------------------------------------*/
#undef y2in
#undef x3out
#undef y3out
#undef z3out
#define y2in(off) (off + 32)(b_ptr)
#define x3out(off) (off)(b_ptr)
#define y3out(off) (off + 32)(b_ptr)
#define z3out(off) (off + 64)(b_ptr)
//func p256PointAddAsm(res, in1, in2 []uint64) int
TEXT ·p256PointAddAsm(SB),0,$392-80
	// See https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-2007-bl
	// Move input to stack in order to free registers
	MOVD	in1+24(FP), a_ptr
	MOVD	in2+48(FP), b_ptr

	MOVD	p256const0<>(SB), const0
	MOVD	p256const1<>(SB), const1

	// Begin point add
	LDx(z2in)
	CALL	p256SqrInternal<>(SB)    // z2^2
	STy(z2sqr)

	CALL	p256MulInternal<>(SB)    // z2^3

	LDx(y1in)
	CALL	p256MulInternal<>(SB)    // s1 = z2ˆ3*y1
	STy(s1)

	LDx(z1in)
	CALL	p256SqrInternal<>(SB)    // z1^2
	STy(z1sqr)

	CALL	p256MulInternal<>(SB)    // z1^3

	LDx(y2in)
	CALL	p256MulInternal<>(SB)    // s2 = z1ˆ3*y2

	LDx(s1)
	CALL	p256SubInternal<>(SB)    // r = s2 - s1
	STx(r)

	MOVD	$1, t2
	ORR	x0, x1, t0             // Check if zero mod p256
	ORR	x2, x3, t1
	ORR	t1, t0, t0
	CMP	$0, t0
	CSEL	EQ, t2, ZR, hlp1

	EOR	$-1, x0, t0
	EOR	const0, x1, t1
	EOR	const1, x3, t3

	ORR	t0, t1, t0
	ORR	x2, t3, t1
	ORR	t1, t0, t0
	CMP	$0, t0
	CSEL	EQ, t2, hlp1, hlp1

	LDx(z2sqr)
	LDy(x1in)
	CALL	p256MulInternal<>(SB)    // u1 = x1 * z2ˆ2
	STy(u1)

	LDx(z1sqr)
	LDy(x2in)
	CALL	p256MulInternal<>(SB)    // u2 = x2 * z1ˆ2
	STy(u2)

	LDx(u1)
	CALL	p256SubInternal<>(SB)    // h = u2 - u1
	STx(h)

	MOVD	$1, t2
	ORR	x0, x1, t0             // Check if zero mod p256
	ORR	x2, x3, t1
	ORR	t1, t0, t0
	CMP	$0, t0
	CSEL	EQ, t2, ZR, hlp0

	EOR	$-1, x0, t0
	EOR	const0, x1, t1
	EOR	const1, x3, t3

	ORR	t0, t1, t0
	ORR	x2, t3, t1
	ORR	t1, t0, t0
	CMP	$0, t0
	CSEL	EQ, t2, hlp0, hlp0

	AND	hlp0, hlp1, hlp1

	LDx(r)
	CALL	p256SqrInternal<>(SB)    // rsqr = rˆ2
	STy(rsqr)

	LDx(h)
	CALL	p256SqrInternal<>(SB)    // hsqr = hˆ2
	STy(hsqr)

	LDx(h)
	CALL	p256MulInternal<>(SB)    // hcub = hˆ3
	STy(hcub)

	LDx(s1)
	CALL	p256MulInternal<>(SB)
	STy(s2)

	LDx(z1in)
	LDy(z2in)
	CALL	p256MulInternal<>(SB)    // z1 * z2
	LDx(h)
	CALL	p256MulInternal<>(SB)    // z1 * z2 * h
	MOVD	res+0(FP), b_ptr
	STy(z3out)

	LDx(hsqr)
	LDy(u1)
	CALL	p256MulInternal<>(SB)    // hˆ2 * u1
	STy(u2)

	p256MulBy2Inline               // u1 * hˆ2 * 2, inline
	LDy(rsqr)
	CALL	p256SubInternal<>(SB)    // rˆ2 - u1 * hˆ2 * 2

	MOVD	x0, y0
	MOVD	x1, y1
	MOVD	x2, y2
	MOVD	x3, y3
	LDx(hcub)
	CALL	p256SubInternal<>(SB)
	STx(x3out)

	LDy(u2)
	CALL	p256SubInternal<>(SB)

	LDy(r)
	CALL	p256MulInternal<>(SB)

	LDx(s2)
	CALL	p256SubInternal<>(SB)
	STx(y3out)

	MOVD	hlp1, R0
	MOVD	R0, ret+72(FP)

	RET
