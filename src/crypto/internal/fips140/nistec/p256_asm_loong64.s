// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

// This file contains constant-time, 64-bit assembly implementation of
// P256. The optimizations performed here are described in detail in:
// S.Gueron and V.Krasnov, "Fast prime field elliptic-curve cryptography with
//                          256-bit primes"
// http://link.springer.com/article/10.1007%2Fs13389-014-0090-x
// https://eprint.iacr.org/2013/816.pdf

#include "textflag.h"

#define res_ptr R4
#define a_ptr   R5
#define b_ptr   R6

#define acc0    R7
#define acc1    R8
#define acc2    R9
#define acc3    R10
#define acc4    R11
#define acc5    R12
#define acc6    R13
#define acc7    R14

#define t0      R15
#define t1      R16
#define t2      R17
#define t3      R18
#define t4      R19
#define t5      R24
#define t6      R25

#define x0      R26
#define x1      R27
#define x2      R28
#define x3      R20

#define y0      R23
#define y1      a_ptr
#define y2      R21
#define y3      R31

#define hlp0    b_ptr
#define hlp1    t2

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
GLOBL p256const0<>(SB), (NOPTR|RODATA), $8
GLOBL p256const1<>(SB), (NOPTR|RODATA), $8
GLOBL p256ordK0<>(SB), (NOPTR|RODATA), $8
GLOBL p256ord<>(SB), (NOPTR|RODATA), $32
GLOBL p256one<>(SB), (NOPTR|RODATA), $32

// func p256MovCond(res, a, b *P256Point, cond int)
// If cond is 0, sets res = b, otherwise sets res = a.
TEXT ·p256MovCond(SB),NOSPLIT,$0
	MOVV	res+0(FP), res_ptr
	MOVV	a+8(FP), a_ptr
	MOVV	b+16(FP), b_ptr
	MOVV	cond+24(FP), t0

	// mask = -(cond != 0)
	SGTU	t0, R0, t1
	SUBV	t1, R0, t1

	// Select each limb as:
	//	b ^ ((a ^ b) & mask)
	//
	// The loop count is fixed at 12 and does not depend on cond.
	MOVV	$0, t2
	MOVV	$12, x0
	MOVV	$0, t3

p256MovCond_loop:
	MOVV	(t3)(a_ptr), t4
	MOVV	(t3)(b_ptr), t5
	XOR	t4, t5, t6
	AND	t1, t6, t6
	XOR	t5, t6, t6
	MOVV	t6, (t3)(res_ptr)

	ADDV	$8, t3, t3
	ADDV	$1, t2, t2
	BNE	t2, x0, p256MovCond_loop
	RET

// func p256NegCond(val *p256Element, cond int)
// If cond is not 0, sets val = -val mod p (aka: val = P - val).
TEXT ·p256NegCond(SB),NOSPLIT,$0
	MOVV	val+0(FP), res_ptr
	MOVV	cond+8(FP), t0

	// mask = -(cond != 0)
	SGTU	t0, R0, acc4
	SUBV	acc4, R0, acc4

	// acc = poly
	MOVV	$-1, acc0
	MOVV	p256const0<>(SB), acc1
	MOVV	$0, acc2
	MOVV	p256const1<>(SB), acc3

	// Load the original value
	MOVV	(0*8)(res_ptr), t0
	MOVV	(1*8)(res_ptr), t1
	MOVV	(2*8)(res_ptr), t2
	MOVV	(3*8)(res_ptr), t3

	// Keep the original value for the final constant-time selection.
	MOVV	t0, x0
	MOVV	t1, x1
	MOVV	t2, x2
	MOVV	t3, x3

	// Speculatively subtract
	SUBV	t0, acc0, t0
	SUBV	t1, acc1, t5	// x1 - y1 = z1', borrow if z1' > x1
	SGTU	t0, acc0, t4
	SUBV	t4, t5, t1	// z1' - c0 = z1, borrow if z1 > z1'
	SGTU	t5, acc1, t6
	SGTU	t1, t5, t4
	OR	t4, t6
	SUBV	t2, acc2, t5
	SUBV	t6, t5, t2
	SGTU	t5, acc2, t4
	SGTU	t2, t5, t5
	OR	t4, t5
	SUBV	t3, acc3, t3
	SUBV	t5, t3

	// Select original value when cond == 0, negated value otherwise.
	// result = original ^ ((original ^ negated) & mask)
	XOR	x0, t0, t4
	AND	acc4, t4, t4
	XOR	x0, t4, t0

	XOR	x1, t1, t4
	AND	acc4, t4, t4
	XOR	x1, t4, t1

	XOR	x2, t2, t4
	AND	acc4, t4, t4
	XOR	x2, t4, t2

	XOR	x3, t3, t4
	AND	acc4, t4, t4
	XOR	x3, t4, t3

	MOVV	t0, (0*8)(res_ptr)
	MOVV	t1, (1*8)(res_ptr)
	MOVV	t2, (2*8)(res_ptr)
	MOVV	t3, (3*8)(res_ptr)
	RET

// p256Select sets res to the point at index idx in the table.
// idx must be in [0, 16]. It executes in constant time.
// idx == 0 returns the point at infinity (all zeroes).
// idx in [1, 16] selects table[idx-1].
//
// func p256Select(res *P256Point, table *p256Table, idx int)
TEXT ·p256Select(SB),NOSPLIT,$0
	MOVV	idx+16(FP), t0
	MOVV	table+8(FP), a_ptr
	MOVV	res+0(FP), res_ptr

	// allones is used to invert the constant-time equality mask.
	MOVV	$-1, x2

	// Process all 12 limbs. For every limb, scan all 16 entries.
	// Both loop counts are fixed and independent of idx.
	MOVV	$0, x0
	MOVV	$12, x1		// limb count
	MOVV	$17, x3		// entry bound: i runs 1..16 (16 entries)

p256Select_limb:
	MOVV	$1, t1
	MOVV	a_ptr, t5
	MOVV	$0, t6

p256Select_entry:
	// i is 1..16. Build:
	//	mask = 0xffff...ffff iff idx == i
	XOR	t0, t1, t2
	SUBV	t2, R0, t3
	OR	t2, t3, t3
	SRAV	$63, t3, t3
	XOR	t3, x2, t3

	MOVV	(t5), t4
	AND	t3, t4, t4
	OR	t6, t4, t6

	ADDV	$96, t5, t5
	ADDV	$1, t1, t1
	BNE	t1, x3, p256Select_entry

	MOVV	t6, (res_ptr)
	ADDV	$8, a_ptr, a_ptr
	ADDV	$8, res_ptr, res_ptr
	ADDV	$1, x0, x0
	BNE	x0, x1, p256Select_limb

	RET

// p256SelectAffine sets res to the point at index idx in the table.
// idx must be in [0, 32]. It executes in constant time.
// idx == 0 returns the point at infinity (all zeroes).
// idx in [1, 32] selects table[idx-1].
//
// func p256SelectAffine(res *p256AffinePoint, table *p256AffineTable, idx int)
TEXT ·p256SelectAffine(SB),NOSPLIT,$0
	MOVV	idx+16(FP), t0
	MOVV	table+8(FP), a_ptr
	MOVV	res+0(FP), res_ptr

	MOVV	$-1, x2

	// Process all 8 limbs (x and y), scanning all 32 entries.
	// The loop counts are fixed and independent of idx.
	MOVV	$0, x0
	MOVV	$8, x1
	MOVV	$33, x3		// entry bound: i runs 1..32 (32 entries)

p256SelectAffine_limb:
	MOVV	$1, t1
	MOVV	a_ptr, t5
	MOVV	$0, t6

p256SelectAffine_entry:
	// i is 1..32. Build:
	//	mask = 0xffff...ffff iff idx == i
	XOR	t0, t1, t2
	SUBV	t2, R0, t3
	OR	t2, t3, t3
	SRAV	$63, t3, t3
	XOR	t3, x2, t3

	MOVV	(t5), t4
	AND	t3, t4, t4
	OR	t6, t4, t6

	ADDV	$64, t5, t5
	ADDV	$1, t1, t1
	BNE	t1, x3, p256SelectAffine_entry

	MOVV	t6, (res_ptr)
	ADDV	$8, a_ptr, a_ptr
	ADDV	$8, res_ptr, res_ptr
	ADDV	$1, x0, x0
	BNE	x0, x1, p256SelectAffine_limb

	RET

// func p256FromMont(res, in *p256Element)
TEXT ·p256FromMont(SB),NOSPLIT,$0-16
	MOVV	res+0(FP), res_ptr
	MOVV	in+8(FP), a_ptr

	// acc4/acc5 hold p256const0/p256const1 (acc4/acc5 are free here).
	MOVV	p256const0<>(SB), acc4
	MOVV	p256const1<>(SB), acc5

	MOVV	0*8(a_ptr), acc0
	MOVV	1*8(a_ptr), acc1
	MOVV	2*8(a_ptr), acc2
	MOVV	3*8(a_ptr), acc3

	// Round 1: eliminate acc0
	SLLV	$32, acc0, t2		// sh = acc0<<32
	SRLV	$32, acc0, t0		// hi32 = acc0>>32
	MULV	acc0, acc5, t1		// lo = low(acc0*const1)
	MULHVU	acc0, acc5, acc0	// acc0 = high(acc0*const1)

	ADDV	t2, acc1, acc1		// acc1 += sh
	SGTU	t2, acc1, t3		// c1 = carry

	ADDV	t0, acc2, t4		// sum = acc2+hi32
	SGTU	t0, t4, t5
	ADDV	t4, t3, acc2		// acc2 = sum + c1
	SGTU	t4, acc2, t3
	ADDV	t5, t3, t3		// c2

	ADDV	t1, acc3, t4
	SGTU	t1, t4, t5
	ADDV	t4, t3, acc3		// acc3 = sum + c2
	SGTU	t4, acc3, t3
	ADDV	t5, t3, t3		// c3

	ADDV	t3, acc0, acc0		// acc0 += c3

	// Round 2: eliminate acc1
	SLLV	$32, acc1, t2
	SRLV	$32, acc1, t0
	MULV	acc1, acc5, t1
	MULHVU	acc1, acc5, acc1

	ADDV	t2, acc2, acc2
	SGTU	t2, acc2, t3

	ADDV	t0, acc3, t4
	SGTU	t0, t4, t5
	ADDV	t4, t3, acc3
	SGTU	t4, acc3, t3
	ADDV	t5, t3, t3

	ADDV	t1, acc0, t4
	SGTU	t1, t4, t5
	ADDV	t4, t3, acc0
	SGTU	t4, acc0, t3
	ADDV	t5, t3, t3

	ADDV	t3, acc1, acc1

	// Round 3: eliminate acc2
	SLLV	$32, acc2, t2
	SRLV	$32, acc2, t0
	MULV	acc2, acc5, t1
	MULHVU	acc2, acc5, acc2

	ADDV	t2, acc3, acc3
	SGTU	t2, acc3, t3

	ADDV	t0, acc0, t4
	SGTU	t0, t4, t5
	ADDV	t4, t3, acc0
	SGTU	t4, acc0, t3
	ADDV	t5, t3, t3

	ADDV	t1, acc1, t4
	SGTU	t1, t4, t5
	ADDV	t4, t3, acc1
	SGTU	t4, acc1, t3
	ADDV	t5, t3, t3

	ADDV	t3, acc2, acc2

	// Round 4: eliminate acc3
	SLLV	$32, acc3, t2
	SRLV	$32, acc3, t0
	MULV	acc3, acc5, t1
	MULHVU	acc3, acc5, acc3

	ADDV	t2, acc0, acc0
	SGTU	t2, acc0, t3

	ADDV	t0, acc1, t4
	SGTU	t0, t4, t5
	ADDV	t4, t3, acc1
	SGTU	t4, acc1, t3
	ADDV	t5, t3, t3

	ADDV	t1, acc2, t4
	SGTU	t1, t4, t5
	ADDV	t4, t3, acc2
	SGTU	t4, acc2, t3
	ADDV	t5, t3, t3

	ADDV	t3, acc3, acc3

	// Final conditional subtraction: if acc >= p, subtract p
	// p256P = {0xFFFFFFFFFFFFFFFF, const0, 0, const1}
	MOVV	$-1, t2			// allones

	SUBV	t2, acc0, t0		// diff0 = acc0 - allones
	SGTU	t2, acc0, t1		// borrow0

	SUBV	acc4, acc1, x0		// tmp = acc1 - const0
	SGTU	acc4, acc1, x1
	SUBV	t1, x0, y1		// diff1 = tmp - borrow0
	SGTU	t1, x0, x2
	ADDV	x1, x2, t1		// borrow1

	SUBV	t1, acc2, x0		// diff2 = acc2 - 0 - borrow1
	SGTU	t1, acc2, t1		// borrow2

	SUBV	acc5, acc3, y2		// tmp = acc3 - const1
	SGTU	acc5, acc3, x3
	SUBV	t1, y2, y3		// diff3 = tmp - borrow2
	SGTU	t1, y2, y0
	ADDV	x3, y0, t1		// borrow3 (final borrow)

	// mask = 0 - borrow3:
	// borrow3 = 0 -> mask = 0 (select diff); borrow3 = 1 -> mask = all ones (keep acc).
	SUBV	t1, R0, t3		// t3 = mask
	MOVV	$-1, t4
	XOR	t3, t4, t4		// notmask = ^mask

	AND	t3, acc0, x1
	AND	t4, t0, x3
	OR	x1, x3, acc0

	AND	t3, acc1, x1
	AND	t4, y1, x3
	OR	x1, x3, acc1

	AND	t3, acc2, x1
	AND	t4, x0, x3
	OR	x1, x3, acc2

	AND	t3, acc3, x1
	AND	t4, y3, x3
	OR	x1, x3, acc3

	MOVV	acc0, 0*8(res_ptr)
	MOVV	acc1, 1*8(res_ptr)
	MOVV	acc2, 2*8(res_ptr)
	MOVV	acc3, 3*8(res_ptr)
	RET

// input:  x0-x3, y0-y3
// output: x0-x3 = (y0-y3) - (x0-x3) mod p256P
// uses:   const0(p256const0<>), const1(p256const1<>)
// clobbers: x0-x3 (overwritten with the result), y0-y3, acc0-acc7,
//           t0-t3
TEXT p256SubInternal<>(SB),NOSPLIT,$0
	// diff = y - x; borrow chain accumulates in t0
	SUBV	x0, y0, acc0
	SGTU	x0, y0, t0		// t0 = borrow0 = (y0 < x0)

	SUBV	x1, y1, t1		// t1 = y1 - x1 (raw, before borrow0)
	SGTU	x1, y1, t2		// t2 = raw borrow
	SUBV	t0, t1, acc1		// acc1 = t1 - borrow0
	SGTU	t0, t1, t3		// t3 = borrow from subtracting borrow0
	OR	t2, t3, t0		// t0 = borrow1

	SUBV	x2, y2, t1
	SGTU	x2, y2, t2
	SUBV	t0, t1, acc2
	SGTU	t0, t1, t3
	OR	t2, t3, t0		// t0 = borrow2

	SUBV	x3, y3, t1
	SGTU	x3, y3, t2
	SUBV	t0, t1, acc3
	SGTU	t0, t1, t3
	OR	t2, t3, t0		// t0 = borrow3 (final borrow: 1 means y < x)

	// acc4..acc7 = diff + p256P (candidate to select on borrow)
	MOVV	$-1, t1			// p256P limb0
	MOVV	p256const0<>(SB), t2	// p256P limb1
	MOVV	p256const1<>(SB), t3	// p256P limb3

	ADDV	t1, acc0, acc4
	SGTU	t1, acc4, x0		// x0 = carry0

	ADDV	t2, acc1, x1		// x1 = acc1 + const0 (raw)
	SGTU	t2, x1, x2		// x2 = raw carry
	ADDV	x0, x1, acc5
	SGTU	x0, acc5, x3		// x3 = carry from adding carry0
	OR	x2, x3, x0		// x0 = carry1

	ADDV	x0, acc2, acc6		// limb2 addend is 0; only carry in
	SGTU	x0, acc6, x1
	MOVV	x1, x0			// x0 = carry2

	ADDV	t3, acc3, x2		// x2 = acc3 + const1 (raw)
	ADDV	x0, x2, acc7		// acc7 = raw + carry2 (top carry discarded)

	// constant-time select: borrow==0 -> diff, borrow==1 -> diff+P
	SUBV	t0, R0, t1		// t1 = mask: 0 (no borrow) or all-1s (borrow)
	MOVV	$-1, t2
	XOR	t1, t2, t2		// t2 = notmask

	AND	t2, acc0, t3
	AND	t1, acc4, y0
	OR	t3, y0, x0

	AND	t2, acc1, t3
	AND	t1, acc5, y0
	OR	t3, y0, x1

	AND	t2, acc2, t3
	AND	t1, acc6, y0
	OR	t3, y0, x2

	AND	t2, acc3, t3
	AND	t1, acc7, y0
	OR	t3, y0, x3

	RET

// p256SqrInternal computes y0..y3 = x0..x3 * x0..x3 mod P (Montgomery domain).
// Result left in y0-y3, matching the arm64 output convention.
// Clobbers t0-t6, acc0-acc7, y0-y3, and hlp0 (b_ptr/R6) -- callers must
// not keep live state in b_ptr across the CALL.
TEXT p256SqrInternal<>(SB),NOSPLIT,$0
	// x[1:] * x[0]
	MULV	x0, x1, acc1
	MULHVU	x0, x1, acc2

	MULV	x0, x2, t0
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t1
	MULHVU	x0, x2, acc3
	ADDV	t1, acc3, acc3

	MULV	x0, x3, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t1
	MULHVU	x0, x3, acc4
	ADDV	t1, acc4, acc4

	// x[2:] * x[1]
	MULV	x1, x2, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t2
	MULHVU	x1, x2, t1
	ADDV	t1, acc4, acc4
	SGTU	t1, acc4, t3
	ADDV	t2, acc4, acc4
	SGTU	t2, acc4, t2
	OR	t3, t2, t2
	ADDV	t2, R0, acc5		// acc5 = carry-out (0 or 1)

	MULV	x1, x3, t0
	ADDV	t0, acc4, acc4
	SGTU	t0, acc4, t1
	MULHVU	x1, x3, t3
	ADDV	t3, acc5, acc5
	SGTU	t3, acc5, t4
	ADDV	t1, acc5, acc5
	SGTU	t1, acc5, t5
	OR	t4, t5, hlp0

	// x[3] * x[2]
	MULV	x2, x3, t0
	ADDV	t0, acc5, acc5
	SGTU	t0, acc5, t1
	MULHVU	x2, x3, acc6
	ADDV	t1, acc6, acc6
	ADDV	hlp0, acc6, acc6

	MOVV	$0, acc7

	// double all cross-products (acc1..acc6)
	SLLV	$1, acc6, t0
	SRLV	$63, acc5, t1
	OR	t1, t0, t0
	SLLV	$1, acc5, t1
	SRLV	$63, acc4, t2
	OR	t2, t1, t1
	SLLV	$1, acc4, t2
	SRLV	$63, acc3, t3
	OR	t3, t2, t2
	SLLV	$1, acc3, t3
	SRLV	$63, acc2, t4
	OR	t4, t3, t3
	SLLV	$1, acc2, t4
	SRLV	$63, acc1, t5
	OR	t5, t4, t4
	SLLV	$1, acc1, t5
	SRLV	$63, acc6, acc7	// top carry-out of the double (from old acc6's high bit)
	MOVV	t5, acc1
	MOVV	t4, acc2
	MOVV	t3, acc3
	MOVV	t2, acc4
	MOVV	t1, acc5
	MOVV	t0, acc6

	// diagonal products: acc0 = low(x0*x0); add high(x0^2), x1^2, x2^2, x3^2
	MULV	x0, x0, acc0
	MULHVU	x0, x0, t0
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t1	// carry into acc2 (propagated below)

	MULV	x1, x1, t0
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t2
	ADDV	t1, acc2, acc2
	SGTU	t1, acc2, t3
	OR	t2, t3, t1
	MULHVU	x1, x1, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t2
	ADDV	t1, acc3, acc3
	SGTU	t1, acc3, t3
	OR	t2, t3, t1

	MULV	x2, x2, t0
	ADDV	t0, acc4, acc4
	SGTU	t0, acc4, t2
	ADDV	t1, acc4, acc4
	SGTU	t1, acc4, t3
	OR	t2, t3, t1
	MULHVU	x2, x2, t0
	ADDV	t0, acc5, acc5
	SGTU	t0, acc5, t2
	ADDV	t1, acc5, acc5
	SGTU	t1, acc5, t3
	OR	t2, t3, t1

	MULV	x3, x3, t0
	ADDV	t0, acc6, acc6
	SGTU	t0, acc6, t2
	ADDV	t1, acc6, acc6
	SGTU	t1, acc6, t3
	OR	t2, t3, t1
	MULHVU	x3, x3, t0
	ADDV	t0, acc7, acc7
	ADDV	t1, acc7, acc7

	// First reduction step
	SLLV	$32, acc0, t2
	SRLV	$32, acc0, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc0, t6, t1
	MULHVU	acc0, t6, acc0
	ADDV	t2, acc1, acc1
	SGTU	t2, acc1, t3
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t4
	ADDV	t3, acc2, acc2
	SGTU	t3, acc2, t5
	OR	t4, t5, t3
	ADDV	t1, acc3, acc3
	SGTU	t1, acc3, t4
	ADDV	t3, acc3, acc3
	SGTU	t3, acc3, t5
	OR	t4, t5, t3
	ADDV	t3, acc0, acc0

	// Second reduction step
	SLLV	$32, acc1, t2
	SRLV	$32, acc1, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc1, t6, t1
	MULHVU	acc1, t6, acc1
	ADDV	t2, acc2, acc2
	SGTU	t2, acc2, t3
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t4
	ADDV	t3, acc3, acc3
	SGTU	t3, acc3, t5
	OR	t4, t5, t3
	ADDV	t1, acc0, acc0
	SGTU	t1, acc0, t4
	ADDV	t3, acc0, acc0
	SGTU	t3, acc0, t5
	OR	t4, t5, t3
	ADDV	t3, acc1, acc1

	// Third reduction step
	SLLV	$32, acc2, t2
	SRLV	$32, acc2, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc2, t6, t1
	MULHVU	acc2, t6, acc2
	ADDV	t2, acc3, acc3
	SGTU	t2, acc3, t3
	ADDV	t0, acc0, acc0
	SGTU	t0, acc0, t4
	ADDV	t3, acc0, acc0
	SGTU	t3, acc0, t5
	OR	t4, t5, t3
	ADDV	t1, acc1, acc1
	SGTU	t1, acc1, t4
	ADDV	t3, acc1, acc1
	SGTU	t3, acc1, t5
	OR	t4, t5, t3
	ADDV	t3, acc2, acc2

	// Last reduction step
	SLLV	$32, acc3, t2
	SRLV	$32, acc3, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc3, t6, t1
	MULHVU	acc3, t6, acc3
	ADDV	t2, acc0, acc0
	SGTU	t2, acc0, t3
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t4
	ADDV	t3, acc1, acc1
	SGTU	t3, acc1, t5
	OR	t4, t5, t3
	ADDV	t1, acc2, acc2
	SGTU	t1, acc2, t4
	ADDV	t3, acc2, acc2
	SGTU	t3, acc2, t5
	OR	t4, t5, t3
	ADDV	t3, acc3, acc3

	// Add bits [511:256] of the square result (acc4-acc7)
	ADDV	acc4, acc0, acc0
	SGTU	acc4, acc0, t0
	ADDV	acc5, acc1, acc1
	SGTU	acc5, acc1, t1
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t2
	OR	t1, t2, t0
	ADDV	acc6, acc2, acc2
	SGTU	acc6, acc2, t1
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t2
	OR	t1, t2, t0
	ADDV	acc7, acc3, acc3
	SGTU	acc7, acc3, t1
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t2
	OR	t1, t2, acc4	// final overflow bit (0 or 1)

	// Final conditional subtraction of P
	SUBV	$-1, acc0, t0
	SGTU	t0, acc0, t3
	MOVV	p256const0<>(SB), t5
	SUBV	t5, acc1, t1
	SGTU	t5, acc1, t4
	SGTU	t3, t1, t5
	SUBV	t3, t1, t1
	OR	t4, t5, t3
	SUBV	$0, acc2, t2
	SGTU	t3, t2, t4
	SUBV	t3, t2, t2
	MOVV	t4, t3
	MOVV	p256const1<>(SB), t6
	SUBV	t6, acc3, y0
	SGTU	y0, acc3, t4
	SGTU	t3, y0, t5
	SUBV	t3, y0, y0
	OR	t4, t5, t3
	SUBV	t3, acc4, acc4

	// acc4 is -1 on borrow (result < P) and 0 otherwise; SGTU against
	// the always-zero R0 turns it into a 0/1 flag, then into a mask.
	SGTU	acc4, R0, t3
	SUBV	t3, R0, t3	// t3 = borrow mask (0 or all-1s)
	MOVV	$-1, t4
	XOR	t3, t4, t4

	AND	t4, t0, t5
	AND	t3, acc0, t6
	OR	t5, t6, acc0

	AND	t4, t1, t5
	AND	t3, acc1, t6
	OR	t5, t6, acc1

	AND	t4, t2, t5
	AND	t3, acc2, t6
	OR	t5, t6, acc2

	AND	t4, y0, t5
	AND	t3, acc3, t6
	OR	t5, t6, acc3

	MOVV	acc0, y0
	MOVV	acc1, y1
	MOVV	acc2, y2
	MOVV	acc3, y3

	RET

// Non-leaf: saves R1 and keeps n on the stack, because
// p256SqrInternal clobbers both R1 (CALL) and b_ptr (hlp0).
TEXT ·p256Sqr(SB),NOSPLIT,$16-24
	MOVV	R1, 8(R3)		// save return address
	MOVV	res+0(FP), res_ptr
	MOVV	in+8(FP), a_ptr
	MOVV	n+16(FP), t0
	MOVV	t0, 16(R3)		// spill n (b_ptr is clobbered by callee)

	MOVV	0*8(a_ptr), x0
	MOVV	1*8(a_ptr), x1
	MOVV	2*8(a_ptr), x2
	MOVV	3*8(a_ptr), x3

sqrLoop:
	MOVV	16(R3), t0
	SUBV	$1, t0, t0
	MOVV	t0, 16(R3)
	CALL	p256SqrInternal<>(SB)
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	MOVV	16(R3), t0
	BNE	t0, R0, sqrLoop

	MOVV	y0, 0*8(res_ptr)
	MOVV	y1, 1*8(res_ptr)
	MOVV	y2, 2*8(res_ptr)
	MOVV	y3, 3*8(res_ptr)
	MOVV	8(R3), R1		// restore return address
	RET

TEXT p256MulInternal<>(SB),NOSPLIT,$0
	// Initialize the upper half of the 512-bit accumulator.
	// The first y0*x row only produces limbs acc0..acc4;
	// acc5..acc7 must not contain stale register values.
	MOVV	$0, acc4
	MOVV	$0, acc5
	MOVV	$0, acc6
	MOVV	$0, acc7

	// y0 * x
	MULV	y0, x0, acc0
	MULHVU	y0, x0, acc1

	MULV	y0, x1, t0
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t1
	MULHVU	y0, x1, acc2
	ADDV	t1, acc2, acc2

	MULV	y0, x2, t0
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t1
	MULHVU	y0, x2, acc3
	ADDV	t1, acc3, acc3

	MULV	y0, x3, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t1
	MULHVU	y0, x3, acc4
	ADDV	t1, acc4, acc4

	// first reduction step
	SLLV	$32, acc0, t2
	SRLV	$32, acc0, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc0, t6, t1
	MULHVU	acc0, t6, acc0

	ADDV	t2, acc1, acc1
	SGTU	t2, acc1, t3
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t4
	ADDV	t3, acc2, acc2
	SGTU	t3, acc2, t3
	OR	t4, t3, t3
	ADDV	t1, acc3, acc3
	SGTU	t1, acc3, t4
	ADDV	t3, acc3, acc3
	SGTU	t3, acc3, t3
	OR	t4, t3, t3
	ADDV	t3, acc0, acc0		// acc0 = hi + carry: new pos-4 window limb

	// y1 * x
	MULV	y1, x0, t0
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t1
	MULHVU	y1, x0, t2

	MULV	y1, x1, t0
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t3
	ADDV	t1, acc2, acc2
	SGTU	t1, acc2, t1
	OR	t3, t1, t1
	MULHVU	y1, x1, t4

	MULV	y1, x2, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t5
	ADDV	t1, acc3, acc3
	SGTU	t1, acc3, t1
	OR	t5, t1, t1
	MULHVU	y1, x2, t6

	MULV	y1, x3, t0
	ADDV	t0, acc4, acc4
	SGTU	t0, acc4, t5
	ADDV	t1, acc4, acc4
	SGTU	t1, acc4, t1
	OR	t5, t1, t1

	// acc5 starts with the carry from the low-half chain.
	// The high-half chain has its own carry, initially zero.
	MOVV	t1, acc5
	MULHVU	y1, x3, t0

	// acc2 += high(y1*x0)
	ADDV	t2, acc2, acc2
	SGTU	t2, acc2, t3

	// acc3 += high(y1*x1) + high-half carry
	ADDV	t4, acc3, acc3
	SGTU	t4, acc3, t5
	ADDV	t3, acc3, acc3
	SGTU	t3, acc3, t1
	OR	t5, t1, t3

	// acc4 += high(y1*x2) + high-half carry
	ADDV	t6, acc4, acc4
	SGTU	t6, acc4, t5
	ADDV	t3, acc4, acc4
	SGTU	t3, acc4, t1
	OR	t5, t1, t3

	// acc5 = low-half carry + high(y1*x3) + high-half carry
	ADDV	t0, acc5, acc5
	SGTU	t0, acc5, t5
	ADDV	t3, acc5, acc5
	SGTU	t3, acc5, t1
	OR	t5, t1, t3

	// second reduction step
	SLLV	$32, acc1, t2
	SRLV	$32, acc1, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc1, t6, t1
	MULHVU	acc1, t6, acc1

	ADDV	t2, acc2, acc2
	SGTU	t2, acc2, t3
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t4
	ADDV	t3, acc3, acc3
	SGTU	t3, acc3, t3
	OR	t4, t3, t3
	ADDV	t1, acc0, acc0
	SGTU	t1, acc0, t4
	ADDV	t3, acc0, acc0
	SGTU	t3, acc0, t3
	OR	t4, t3, t3
	ADDV	t3, acc1, acc1

	// y2 * x
	MULV	y2, x0, t0
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t1
	MULHVU	y2, x0, t2

	MULV	y2, x1, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t3
	MULHVU	y2, x1, t4

	ADDV	t1, acc3, acc3
	SGTU	t1, acc3, t1
	OR	t3, t1, t1

	MULV	y2, x2, t0
	ADDV	t0, acc4, acc4
	SGTU	t0, acc4, t5
	ADDV	t1, acc4, acc4
	SGTU	t1, acc4, t1
	OR	t5, t1, t1
	MULHVU	y2, x2, t6

	MULV	y2, x3, t0
	ADDV	t0, acc5, acc5
	SGTU	t0, acc5, t5

	ADDV	t1, acc5, acc5
	SGTU	t1, acc5, t1
	OR	t5, t1, t1

	// acc6 starts with the carry from the low-half chain.
	// The high-half chain has its own carry, initially zero.
	MOVV	t1, acc6
	MULHVU	y2, x3, t0

	// acc3 += high(y2*x0)
	ADDV	t2, acc3, acc3
	SGTU	t2, acc3, t3

	// acc4 += high(y2*x1) + high-half carry
	ADDV	t4, acc4, acc4
	SGTU	t4, acc4, t5
	ADDV	t3, acc4, acc4
	SGTU	t3, acc4, t1
	OR	t5, t1, t3

	// acc5 += high(y2*x2) + high-half carry
	ADDV	t6, acc5, acc5
	SGTU	t6, acc5, t5
	ADDV	t3, acc5, acc5
	SGTU	t3, acc5, t1
	OR	t5, t1, t3

	// acc6 = low-half carry + high(y2*x3) + high-half carry
	ADDV	t0, acc6, acc6
	SGTU	t0, acc6, t5
	ADDV	t3, acc6, acc6
	SGTU	t3, acc6, t1
	OR	t5, t1, t3

	// third reduction step
	SLLV	$32, acc2, t2
	SRLV	$32, acc2, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc2, t6, t1
	MULHVU	acc2, t6, acc2

	ADDV	t2, acc3, acc3
	SGTU	t2, acc3, t3
	ADDV	t0, acc0, acc0
	SGTU	t0, acc0, t4
	ADDV	t3, acc0, acc0
	SGTU	t3, acc0, t3
	OR	t4, t3, t3
	ADDV	t1, acc1, acc1
	SGTU	t1, acc1, t4
	ADDV	t3, acc1, acc1
	SGTU	t3, acc1, t3
	OR	t4, t3, t3
	ADDV	t3, acc2, acc2

	// y3 * x
	MULV	y3, x0, t0
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t1
	MULHVU	y3, x0, t2

	MULV	y3, x1, t0
	ADDV	t0, acc4, acc4
	SGTU	t0, acc4, t3
	MULHVU	y3, x1, t4

	ADDV	t1, acc4, acc4
	SGTU	t1, acc4, t1
	OR	t3, t1, t1

	MULV	y3, x2, t0
	ADDV	t0, acc5, acc5
	SGTU	t0, acc5, t5
	ADDV	t1, acc5, acc5
	SGTU	t1, acc5, t1
	OR	t5, t1, t1
	MULHVU	y3, x2, t6

	MULV	y3, x3, t0
	ADDV	t0, acc6, acc6
	SGTU	t0, acc6, t5

	ADDV	t1, acc6, acc6
	SGTU	t1, acc6, t1
	OR	t5, t1, t1

	// acc7 starts with the carry from the low-half chain.
	// The high-half chain has its own carry, initially zero.
	MOVV	t1, acc7
	MULHVU	y3, x3, t0

	// acc4 += high(y3*x0)
	ADDV	t2, acc4, acc4
	SGTU	t2, acc4, t3

	// acc5 += high(y3*x1) + high-half carry
	ADDV	t4, acc5, acc5
	SGTU	t4, acc5, t5
	ADDV	t3, acc5, acc5
	SGTU	t3, acc5, t1
	OR	t5, t1, t3

	// acc6 += high(y3*x2) + high-half carry
	ADDV	t6, acc6, acc6
	SGTU	t6, acc6, t5
	ADDV	t3, acc6, acc6
	SGTU	t3, acc6, t1
	OR	t5, t1, t3

	// acc7 = low-half carry + high(y3*x3) + high-half carry
	ADDV	t0, acc7, acc7
	SGTU	t0, acc7, t5
	ADDV	t3, acc7, acc7
	SGTU	t3, acc7, t1
	OR	t5, t1, t3
	// carry beyond acc7 is outside the 512-bit product and is discarded.

	// fourth reduction step
	SLLV	$32, acc3, t2
	SRLV	$32, acc3, t0
	MOVV	p256const1<>(SB), t6
	MULV	acc3, t6, t1
	MULHVU	acc3, t6, acc3

	ADDV	t2, acc0, acc0
	SGTU	t2, acc0, t3
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t4
	ADDV	t3, acc1, acc1
	SGTU	t3, acc1, t3
	OR	t4, t3, t3
	ADDV	t1, acc2, acc2
	SGTU	t1, acc2, t4
	ADDV	t3, acc2, acc2
	SGTU	t3, acc2, t3
	OR	t4, t3, t3
	ADDV	t3, acc3, acc3

	// fold high half [511:256]
	ADDV	acc4, acc0, acc0
	SGTU	acc4, acc0, t0
	ADDV	acc5, acc1, acc1
	SGTU	acc5, acc1, t1
	ADDV	t0, acc1, acc1
	SGTU	t0, acc1, t0
	OR	t1, t0, t0

	ADDV	acc6, acc2, acc2
	SGTU	acc6, acc2, t1
	ADDV	t0, acc2, acc2
	SGTU	t0, acc2, t0
	OR	t1, t0, t0

	ADDV	acc7, acc3, acc3
	SGTU	acc7, acc3, t1
	ADDV	t0, acc3, acc3
	SGTU	t0, acc3, t0
	OR	t1, t0, t0

	MOVV	t0, acc4		// final overflow flag (0 or 1)

	// conditional subtraction of P (mask select)
	// compute borrows
	MOVV	$-1, t6
	SGTU	t6, acc0, t5		// borrow0 = 1 if acc0 < 0xFFFFFFFFFFFFFFFF
	ADDV	$1, acc0, t0		// d0 = acc0 + 1

	MOVV	p256const0<>(SB), t6
	SUBV	t6, acc1, t1
	SGTU	t6, acc1, t4
	MOVV	t1, t6
	SUBV	t5, t1, t1
	SGTU	t1, t6, t6
	OR	t4, t6, t5		// borrow1

	SUBV	$0, acc2, t2
	SGTU	$0, acc2, t4		// always 0
	MOVV	t2, t6
	SUBV	t5, t2, t2
	SGTU	t2, t6, t6
	OR	t4, t6, t5

	MOVV	p256const1<>(SB), t6
	SUBV	t6, acc3, t3
	SGTU	t6, acc3, t4
	MOVV	t3, t6
	SUBV	t5, t3, t3
	SGTU	t3, t6, t6
	OR	t4, t6, t5		// final borrow in t5 (0 or 1)

	// Fold the extra overflow limb into the borrow chain:
	// acc4 -= borrow(acc0..acc3).
	//   acc4=1, no borrow  -> acc4 stays 1, still >= P, subtract P
	//   acc4=0, borrow     -> acc4 becomes -1, value < P, keep original
	//   acc4=1, borrow     -> acc4 becomes 0, exactly cancels, subtract P
	//   acc4=0, no borrow  -> acc4 stays 0, subtract P
	SUBV	t5, acc4, acc4

	// acc4 != 0 means borrow occurred (value < P): keep acc0..acc3.
	// acc4 == 0 means no borrow (value >= P): select the subtraction t0..t3.
	SGTU	acc4, R0, t5
	SUBV	t5, R0, t5		// t5 = borrow mask: all-1s keeps original
	MOVV	$-1, t6
	XOR	t5, t6, t6		// t6 = no-borrow mask: all-1s selects diff

	// borrow==0 (acc >= P) -> take t0..t3; borrow==1 (acc < P) -> keep acc0..acc3
	AND	t6, t0, t4
	AND	t5, acc0, acc4
	OR	t4, acc4, y0

	AND	t6, t1, t4
	AND	t5, acc1, acc4
	OR	t4, acc4, y1

	AND	t6, t2, t4
	AND	t5, acc2, acc4
	OR	t4, acc4, y2

	AND	t6, t3, t4
	AND	t5, acc3, acc4
	OR	t4, acc4, y3

	RET

// func p256Mul(res, in1, in2 *p256Element)
TEXT ·p256Mul(SB),NOSPLIT,$16-24
	MOVV	R1, 8(R3)		// save return address
	MOVV	res+0(FP), res_ptr
	MOVV	in1+8(FP), a_ptr
	MOVV	in2+16(FP), b_ptr

	MOVV	0*8(a_ptr), x0
	MOVV	1*8(a_ptr), x1
	MOVV	2*8(a_ptr), x2
	MOVV	3*8(a_ptr), x3

	MOVV	0*8(b_ptr), y0
	MOVV	1*8(b_ptr), y1
	MOVV	2*8(b_ptr), y2
	MOVV	3*8(b_ptr), y3

	CALL	p256MulInternal<>(SB)

	MOVV	y0, 0*8(res_ptr)
	MOVV	y1, 1*8(res_ptr)
	MOVV	y2, 2*8(res_ptr)
	MOVV	y3, 3*8(res_ptr)
	MOVV	8(R3), R1		// restore return address
	RET

#define p256MulBy2Inline \
	/* x0..x3 = y0..y3 + y0..y3 (double), carry out -> hlp0 */ \
	ADDV	y0, y0, x0		;\
	SGTU	y0, x0, t0		;\
	\
	ADDV	y1, y1, t1		;\
	SGTU	y1, t1, t2		;\
	ADDV	t0, t1, x1		;\
	SGTU	t0, x1, t3		;\
	OR	t2, t3, t0		;\
	\
	ADDV	y2, y2, t1		;\
	SGTU	y2, t1, t2		;\
	ADDV	t0, t1, x2		;\
	SGTU	t0, x2, t3		;\
	OR	t2, t3, t0		;\
	\
	ADDV	y3, y3, t1		;\
	SGTU	y3, t1, t2		;\
	ADDV	t0, t1, x3		;\
	SGTU	t0, x3, t3		;\
	OR	t2, t3, hlp0		;\
	\
	/* t0..t3 = x0..x3 - P (4-limb borrow chain, same pattern as in p256MulInternal) */ \
	MOVV	$-1, t6			;\
	SGTU	t6, x0, t5		;\
	ADDV	$1, x0, t0		;\
	\
	MOVV	p256const0<>(SB), t6	;\
	SUBV	t6, x1, t1		;\
	SGTU	t6, x1, t4		;\
	MOVV	t1, t6			;\
	SUBV	t5, t1, t1		;\
	SGTU	t5, t6, t6		;\	/* borrow from subtracting borrow_in */
	OR	t4, t6, t5		;\
	\
	SUBV	$0, x2, t2		;\
	SGTU	$0, x2, t4		;\
	MOVV	t2, t6			;\
	SUBV	t5, t2, t2		;\
	SGTU	t5, t6, t6		;\
	OR	t4, t6, t5		;\
	\
	MOVV	p256const1<>(SB), t6	;\
	SUBV	t6, x3, t3		;\
	SGTU	t6, x3, t4		;\
	MOVV	t3, t6			;\
	SUBV	t5, t3, t3		;\
	SGTU	t5, t6, t6		;\
	OR	t4, t6, t5		;\
	\
	/* fold the doubling carry bit into the borrow chain */ \
	SUBV	t5, hlp0, hlp0		;\	/* hlp0 = ovf - borrow */
	SRAV	$63, hlp0, t5		;\
	MOVV	$-1, t6			;\
	XOR	t5, t6, t6		;\
	\
	/* select: no borrow (>= P) -> t0..t3; borrow (< P) -> original x0..x3 */ \
	AND	t6, t0, t4		;\
	AND	t5, x0, hlp0		;\
	OR	t4, hlp0, x0		;\
	\
	AND	t6, t1, t4		;\
	AND	t5, x1, hlp0		;\
	OR	t4, hlp0, x1		;\
	\
	AND	t6, t2, t4		;\
	AND	t5, x2, hlp0		;\
	OR	t4, hlp0, x2		;\
	\
	AND	t6, t3, t4		;\
	AND	t5, x3, hlp0		;\
	OR	t4, hlp0, x3		;

#define y2in(off)  (8 + 32*0 + off)(R3)
#define s2v(off)   (8 + 32*1 + off)(R3)
#define z1sqr(off) (8 + 32*2 + off)(R3)
#define hv(off)    (8 + 32*3 + off)(R3)
#define rv(off)    (8 + 32*4 + off)(R3)
#define hsqr(off)  (8 + 32*5 + off)(R3)
#define rsqr(off)  (8 + 32*6 + off)(R3)
#define hcub(off)  (8 + 32*7 + off)(R3)

/*
 * slot 8:
 *  +0 : sel | (zero << 1)
 *  +8 : normalized sign
 *
 * Both values need to survive CALLs, so keep them on the stack.
 */
#define flagbase(off)	(8 + 32*8 + off)(R3)
#define selflag(off)	flagbase(off)
#define signflag(off)	flagbase(8 + off)

// z3 staging slot: deferred so that res==in1 aliasing cannot clobber z1in
// before its last read.
#define z3save(off)	(8 + 32*12 + off)(R3)

#define x1in(off)	(off)(a_ptr)
#define y1in(off)	(off+32)(a_ptr)
#define z1in(off)	(off+64)(a_ptr)
#define x2in(off)	(off)(b_ptr)
#define z2in(off)	(off + 64)(b_ptr)
#define x3out(off)	(off)(res_ptr)
#define y3out(off)	(off + 32)(res_ptr)
#define z3out(off)	(off + 64)(res_ptr)
#define y2inptr(off)	(off+32)(b_ptr)

#define LDx(src) \
	MOVV src(0*8),x0; \
	MOVV src(1*8),x1; \
	MOVV src(2*8),x2; \
	MOVV src(3*8),x3

// LDy deliberately loads limb 1 into t4 first because y1 must not alias
// either a_ptr or b_ptr. All source limbs are loaded before y1 is written.
#define LDy(src) \
	MOVV src(0*8),y0; \
	MOVV src(1*8),t4; \
	MOVV src(2*8),y2; \
	MOVV src(3*8),y3; \
	MOVV t4, y1

#define STx(dst) MOVV x0,dst(0*8); MOVV x1,dst(1*8); MOVV x2,dst(2*8); MOVV x3,dst(3*8)
#define STy(dst) MOVV y0,dst(0*8); MOVV y1,dst(1*8); MOVV y2,dst(2*8); MOVV y3,dst(3*8)

#define RELOAD_PTRS \
	MOVV	in1+8(FP), a_ptr	;\
	MOVV	in2+16(FP), b_ptr

// func p256PointAddAffineAsm(res, in1 *P256Point, in2 *p256AffinePoint, sign, sel, zero int)
TEXT ·p256PointAddAffineAsm(SB),NOSPLIT,$432-48
	MOVV	R1, (8+32*9)(R3)	// save return address across CALLs
	// sign/sel/zero cannot live in hlp0(=b_ptr)/y1(=a_ptr); use t5/t6 first.
	MOVV	sign+24(FP), t5		// t5 = sign
	MOVV	sel+32(FP), t6		// t6 = sel
	MOVV	zero+40(FP), t0		// t0 = zero

	// Normalize sel/zero to 0/1 and combine them into a bitmask saved in t1.
	SGTU	t6, R0, t2		// t2 = (sel != 0) ? 1 : 0
	SGTU	t0, R0, t3		// t3 = (zero != 0) ? 1 : 0
	SLLV	$1, t3, t3
	OR	t2, t3, t1		// t1 = sel_norm | (zero_norm << 1)
	MOVV	t1, selflag(0)		// spill to stack; must survive CALLs below

	// Normalize sign to 0/1 and spill it for the conditional select below.
	SGTU	t5, R0, t4
	MOVV	t4, signflag(0)		// sign must survive CALLs; store on stack

	RELOAD_PTRS			// ensure a_ptr/b_ptr are clean

	// Negate y2in based on sign: compute P - y2in via p256SubInternal
	// p256SubInternal computes diff = y - x (result in x0..x3), so:
	//   x0..x3 = y2in (value to subtract)
	//   y0..y3 = P
	MOVV	y2inptr(0*8), x0
	MOVV	y2inptr(1*8), x1
	MOVV	y2inptr(2*8), x2
	MOVV	y2inptr(3*8), x3

	MOVV	$-1, y0
	MOVV	p256const0<>(SB), y1
	MOVV	$0, y2
	MOVV	p256const1<>(SB), y3

	CALL	p256SubInternal<>(SB)	// x0..x3 = P - y2in
	RELOAD_PTRS			// a_ptr/b_ptr(=hlp0)/y1 clobbered by the call

	// reload original y2in (y0..y3 were consumed as the P operand above)
	MOVV	y2inptr(0*8), y0
	MOVV	y2inptr(1*8), y1
	MOVV	y2inptr(2*8), y2
	MOVV	y2inptr(3*8), y3

	// Restore normalized sign from the stack.
	// mask = 0 (sign==0, keep original) or all-ones (sign!=0, use negated)
	MOVV	signflag(0), t4		// restore normalized sign
	SUBV	t4, R0, t2		// t2 = mask
	MOVV	$-1, t3
	XOR	t2, t3, t3		// t3 = notmask

	AND	t3, y0, acc0
	AND	t2, x0, acc4
	OR	acc0, acc4, y0

	AND	t3, y1, acc0
	AND	t2, x1, acc4
	OR	acc0, acc4, y1

	AND	t3, y2, acc0
	AND	t2, x2, acc4
	OR	acc0, acc4, y2

	AND	t3, y3, acc0
	AND	t2, x3, acc4
	OR	acc0, acc4, y3

	STy(y2in)

	// Begin point add
	RELOAD_PTRS

	LDx(z1in)
	CALL	p256SqrInternal<>(SB)	// z1^2 -- a_ptr/b_ptr(=hlp0)/y1 clobbered
	STy(z1sqr)
	RELOAD_PTRS

	LDx(x2in)
	LDy(z1sqr)			// y = z1^2 (RELOAD_PTRS clobbered y1)
	CALL	p256MulInternal<>(SB)	// u2 = x2 * z1^2
	MOVV	y0, rsqr(0*8)
	MOVV	y1, rsqr(1*8)
	MOVV	y2, rsqr(2*8)
	MOVV	y3, rsqr(3*8)
	RELOAD_PTRS

	LDx(x1in)
	LDy(rsqr)
	CALL	p256SubInternal<>(SB)	// h = u2 - x1
	STx(hv)
	RELOAD_PTRS

	LDy(z1in)
	CALL	p256MulInternal<>(SB)	// z3 = h * z1
	STy(s2v)
	RELOAD_PTRS

	// Conditional z3 select: bit0 clear -> z1; bit1 clear -> 1.
	// Read z1in first (needs a_ptr; restored by RELOAD_PTRS above).
	MOVV	z1in(0*8), acc0
	MOVV	z1in(1*8), acc1
	MOVV	z1in(2*8), acc2
	MOVV	z1in(3*8), acc3

	// Reload y = z3 (s2v is an R3-relative slot, independent of a_ptr;
	// y1 is written last, and no a_ptr dereference follows).
	MOVV	s2v(0*8), y0
	MOVV	s2v(1*8), t4
	MOVV	s2v(2*8), y2
	MOVV	s2v(3*8), y3
	MOVV	t4, y1

	MOVV	selflag(0), t1		// reload bitmask; t1 may have been clobbered
	AND	$1, t1, t2
	NEGV	t2, t2			// 0/1 -> 0/all-ones mask
	MOVV	$-1, t3
	XOR	t2, t3, t3		// t3 = ~mask
	AND	t2, y0, y0; AND t3, acc0, acc0; OR y0, acc0, y0
	AND	t2, y1, y1; AND t3, acc1, acc1; OR y1, acc1, y1
	AND	t2, y2, y2; AND t3, acc2, acc2; OR y2, acc2, y2
	AND	t2, y3, y3; AND t3, acc3, acc3; OR y3, acc3, y3

	MOVV	p256one<>+0x00(SB), acc0
	MOVV	p256one<>+0x08(SB), acc1
	MOVV	p256one<>+0x10(SB), acc2
	MOVV	p256one<>+0x18(SB), acc3
	MOVV	selflag(0), t1
	AND	$2, t1, t2
	SRLV	$1, t2, t2
	NEGV	t2, t2
	MOVV	$-1, t3
	XOR	t2, t3, t3
	AND	t2, y0, y0; AND t3, acc0, acc0; OR y0, acc0, y0
	AND	t2, y1, y1; AND t3, acc1, acc1; OR y1, acc1, y1
	AND	t2, y2, y2; AND t3, acc2, acc2; OR y2, acc2, y2
	AND	t2, y3, y3; AND t3, acc3, acc3; OR y3, acc3, y3

	// Save the selected z3 before loading the next temporary.
	// It must not be written to res.z yet: when res==in1, the later
	// LDx(z1in) below still needs the old z1 value.
	STy(z3save)			// y0..y3 -> z3save

	MOVV	in1+8(FP), a_ptr	// y is spilled; restore a_ptr to read in1
	LDx(z1in)			// x = z1 (x previously held h)
	LDy(z1sqr)
	CALL	p256MulInternal<>(SB)	// z1^3
	MOVV	y0, rsqr(0*8)
	MOVV	y1, rsqr(1*8)
	MOVV	y2, rsqr(2*8)
	MOVV	y3, rsqr(3*8)
	RELOAD_PTRS

	LDx(y2in)
	LDy(rsqr)
	CALL	p256MulInternal<>(SB)	// s2 = y2in * z1^3
	STy(s2v)
	RELOAD_PTRS

	LDx(y1in)
	LDy(s2v)			// y = s2
	CALL	p256SubInternal<>(SB)	// r = s2 - y1
	STx(rv)
	RELOAD_PTRS

	CALL	p256SqrInternal<>(SB)	// rsqr = r^2
	STy(rsqr)
	RELOAD_PTRS

	LDx(hv)
	CALL	p256SqrInternal<>(SB)	// hsqr = h^2
	STy(hsqr)
	RELOAD_PTRS

	LDy(hsqr)			// y = h^2
	CALL	p256MulInternal<>(SB)	// hcub = h^3
	STy(hcub)
	RELOAD_PTRS

	LDx(y1in)
	LDy(hcub)			// y = hcub
	CALL	p256MulInternal<>(SB)	// s2' = y1 * hcub
	STy(s2v)
	RELOAD_PTRS

	MOVV	hsqr(0*8), x0
	MOVV	hsqr(1*8), x1
	MOVV	hsqr(2*8), x2
	MOVV	hsqr(3*8), x3
	MOVV	x1in(0*8), y0
	MOVV	x1in(1*8), t4
	MOVV	x1in(2*8), y2
	MOVV	x1in(3*8), y3
	MOVV	t4, y1
	CALL	p256MulInternal<>(SB)	// u1' = x1 * hsqr
	STy(hv)				// overwrite spilled h
	// NOTE: MulBy2Inline reads y1 which aliases a_ptr; it must run
	// BEFORE RELOAD_PTRS. LDy(rsqr) only needs the stack pointer.
	p256MulBy2Inline		// u1'2 = 2*u1' (y regs still hold u1')

	LDy(rsqr)
	CALL	p256SubInternal<>(SB)	// rsqr - u1'2
	RELOAD_PTRS

	MOVV	x0, y0
	MOVV	x1, y1
	MOVV	x2, y2
	MOVV	x3, y3

	LDx(hcub)
	CALL	p256SubInternal<>(SB)	// x3 = previous result - hcub
	RELOAD_PTRS

	// Conditional x3 select.
	MOVV	x1in(0*8), acc0
	MOVV	x1in(1*8), acc1
	MOVV	x1in(2*8), acc2
	MOVV	x1in(3*8), acc3
	MOVV	selflag(0), t1
	AND	$1, t1, t2
	NEGV	t2, t2
	MOVV	$-1, t3
	XOR	t2, t3, t3
	AND	t2, x0, x0; AND t3, acc0, acc0; OR x0, acc0, x0
	AND	t2, x1, x1; AND t3, acc1, acc1; OR x1, acc1, x1
	AND	t2, x2, x2; AND t3, acc2, acc2; OR x2, acc2, x2
	AND	t2, x3, x3; AND t3, acc3, acc3; OR x3, acc3, x3

	RELOAD_PTRS
	MOVV	x2in(0*8), acc0
	MOVV	x2in(1*8), acc1
	MOVV	x2in(2*8), acc2
	MOVV	x2in(3*8), acc3
	MOVV	selflag(0), t1
	AND	$2, t1, t2
	SRLV	$1, t2, t2
	NEGV	t2, t2
	MOVV	$-1, t3
	XOR	t2, t3, t3
	AND	t2, x0, x0; AND t3, acc0, acc0; OR x0, acc0, x0
	AND	t2, x1, x1; AND t3, acc1, acc1; OR x1, acc1, x1
	AND	t2, x2, x2; AND t3, acc2, acc2; OR x2, acc2, x2
	AND	t2, x3, x3; AND t3, acc3, acc3; OR x3, acc3, x3

	RELOAD_PTRS
	MOVV	res+0(FP), t0
	MOVV	x0, 0*8(t0)
	MOVV	x1, 1*8(t0)
	MOVV	x2, 2*8(t0)
	MOVV	x3, 3*8(t0)

	MOVV	0*8(t0), x0
	MOVV	1*8(t0), x1
	MOVV	2*8(t0), x2
	MOVV	3*8(t0), x3
	LDy(hv)				// u1'
	CALL	p256SubInternal<>(SB)	// tmp = u1' - x3
	RELOAD_PTRS

	MOVV	rv(0*8), y0
	MOVV	rv(1*8), y1
	MOVV	rv(2*8), y2
	MOVV	rv(3*8), y3
	CALL	p256MulInternal<>(SB)	// tmp2 = r * tmp
	MOVV	y0, rv(0*8)
	MOVV	y1, rv(1*8)
	MOVV	y2, rv(2*8)
	MOVV	y3, rv(3*8)
	RELOAD_PTRS

	LDx(s2v)
	LDy(rv)
	CALL	p256SubInternal<>(SB)	// y3 = tmp2 - s2'
	RELOAD_PTRS

	// Conditional y3 select.
	MOVV	y1in(0*8), acc0
	MOVV	y1in(1*8), acc1
	MOVV	y1in(2*8), acc2
	MOVV	y1in(3*8), acc3
	MOVV	selflag(0), t1
	AND	$1, t1, t2
	NEGV	t2, t2
	MOVV	$-1, t3
	XOR	t2, t3, t3
	AND	t2, x0, x0; AND t3, acc0, acc0; OR x0, acc0, x0
	AND	t2, x1, x1; AND t3, acc1, acc1; OR x1, acc1, x1
	AND	t2, x2, x2; AND t3, acc2, acc2; OR x2, acc2, x2
	AND	t2, x3, x3; AND t3, acc3, acc3; OR x3, acc3, x3

	MOVV	y2in(0*8), acc0
	MOVV	y2in(1*8), acc1
	MOVV	y2in(2*8), acc2
	MOVV	y2in(3*8), acc3
	MOVV	selflag(0), t1
	AND	$2, t1, t2
	SRLV	$1, t2, t2
	NEGV	t2, t2
	MOVV	$-1, t3
	XOR	t2, t3, t3
	AND	t2, x0, x0; AND t3, acc0, acc0; OR x0, acc0, x0
	AND	t2, x1, x1; AND t3, acc1, acc1; OR x1, acc1, x1
	AND	t2, x2, x2; AND t3, acc2, acc2; OR x2, acc2, x2
	AND	t2, x3, x3; AND t3, acc3, acc3; OR x3, acc3, x3

	MOVV	res+0(FP), t0
	MOVV	x0, 4*8(t0)
	MOVV	x1, 5*8(t0)
	MOVV	x2, 6*8(t0)
	MOVV	x3, 7*8(t0)

	// Deferred res.z write: all z1in reads are complete at this point.
	LDx(z3save)
	MOVV	x0, 8*8(t0)
	MOVV	x1, 9*8(t0)
	MOVV	x2, 10*8(t0)
	MOVV	x3, 11*8(t0)

	MOVV	(8+32*9)(R3), R1	// restore return address
	RET

// Stack temporaries for PointDouble (R3-relative, avoids the saved LR).
#define zsqrv(off)	(8 + 32*0 + off)(R3)
#define mv(off)		(8 + 32*1 + off)(R3)
#define h2v(off)	(8 + 32*2 + off)(R3)
#define sv(off)		(8 + 32*3 + off)(R3)
#define tmpv(off)	(8 + 32*4 + off)(R3)
#define y3tmp(off)	(8 + 32*5 + off)(R3)

// mod-P halving: y0..y3 = (y0..y3) / 2 mod P
// If even, shift right by 1; if odd, add P first, then shift right by 1.
#define p256HalveInline \
	ADDV	$-1, y0, t0		;\
	SGTU	$-1, t0, t4		;\
	MOVV	p256const0<>(SB), t5	;\
	ADDV	t5, y1, t1		;\
	SGTU	t5, t1, t5		;\
	ADDV	t4, t1, t1		;\
	SGTU	t4, t1, t6		;\
	OR	t5, t6, t4		;\
	ADDV	$0, y2, t2		;\
	ADDV	t4, t2, t2		;\
	SGTU	t4, t2, t4		;\
	MOVV	p256const1<>(SB), t5	;\
	ADDV	t5, y3, t3		;\
	SGTU	t5, t3, t5		;\
	ADDV	t4, t3, t3		;\
	SGTU	t4, t3, t6		;\
	OR	t5, t6, hlp0		;\
	\
	AND	$1, y0, t4		;\
	NEGV	t4, t4			;\	/* t4 = 0 (even) / all-ones (odd) */
	MOVV	$-1, t5			;\
	XOR	t4, t5, t5		;\	/* t5 = notmask */
	AND	t5, y0, acc0; AND t4, t0, acc4; OR acc0, acc4, t0 ;\
	AND	t5, y1, acc0; AND t4, t1, acc4; OR acc0, acc4, t1 ;\
	AND	t5, y2, acc0; AND t4, t2, acc4; OR acc0, acc4, t2 ;\
	AND	t5, y3, acc0; AND t4, t3, acc4; OR acc0, acc4, t3 ;\
	AND	t4, hlp0, hlp0		;\
	\
	SRLV	$1, t0, y0		;\
	SLLV	$63, t1, acc0		;\
	OR	acc0, y0, y0		;\
	SRLV	$1, t1, y1		;\
	SLLV	$63, t2, acc0		;\
	OR	acc0, y1, y1		;\
	SRLV	$1, t2, y2		;\
	SLLV	$63, t3, acc0		;\
	OR	acc0, y2, y2		;\
	SRLV	$1, t3, y3		;\
	SLLV	$63, hlp0, acc0		;\
	OR	acc0, y3, y3

// x = (x + y) mod P; result written back in place to x0..x3
// y0..y3 are read but not written; their contents are preserved
#define p256AddInline				\
	ADDV	y0, x0, x0			;\
	SGTU	y0, x0, hlp0			;\ /* hlp0 = limb0 carry */
	ADDV	y1, x1, x1			;\
	SGTU	y1, x1, t4			;\
	ADDV	hlp0, x1, x1			;\
	SGTU	hlp0, x1, t5			;\
	OR	t4, t5, hlp0			;\
	ADDV	y2, x2, x2			;\
	SGTU	y2, x2, t4			;\
	ADDV	hlp0, x2, x2			;\
	SGTU	hlp0, x2, t5			;\
	OR	t4, t5, hlp0			;\
	ADDV	y3, x3, x3			;\
	SGTU	y3, x3, t4			;\
	ADDV	hlp0, x3, x3			;\
	SGTU	hlp0, x3, t5			;\
	OR	t4, t5, hlp0			;\ /* hlp0 = final overflow (0/1); 1 means x+y >= 2^256 */
	\
	SUBV	$-1, x0, t0			;\ /* t0 = x0 - P0 (P0 = -1, i.e. 0xFFFF...FFFF) */
	SGTU	$-1, x0, t4			;\ /* t4 = borrow0 = (x0 < P0) */
	MOVV	p256const0<>(SB), t5		;\
	SUBV	t5, x1, t1			;\
	SGTU	t5, x1, t5			;\
	SUBV	t4, t1, t1			;\
	SGTU	t4, t1, t6			;\
	OR	t5, t6, t4			;\
	SUBV	$0, x2, t2			;\
	SGTU	$0, x2, t5			;\
	SUBV	t4, t2, t2			;\
	SGTU	t4, t2, t6			;\
	OR	t5, t6, t4			;\
	MOVV	p256const1<>(SB), t5		;\
	SUBV	t5, x3, t3			;\
	SGTU	t5, x3, t5			;\
	SUBV	t4, t3, t3			;\
	SGTU	t4, t3, t6			;\
	OR	t5, t6, t4			;\ /* t4 = 4-limb borrow (1 means x < P, no subtraction needed) */
	\
	SUBV	t4, hlp0, hlp0			;\ /* hlp0 - t4: 1/0 (subtract P) or -1 (no subtract) */
	SRAV	$63, hlp0, t5			;\ /* t5 = all-1s (no subtract) or all-0s (subtract) */
	MOVV	$-1, t6				;\
	XOR	t5, t6, t6			;\ /* t6 = notmask: all-1s (subtract) or all-0s (no subtract) */
	\
	AND	t6, t0, hlp0			;\
	AND	t5, x0, x0			;\
	OR	hlp0, x0, x0			;\
	AND	t6, t1, hlp0			;\
	AND	t5, x1, x1			;\
	OR	hlp0, x1, x1			;\
	AND	t6, t2, hlp0			;\
	AND	t5, x2, x2			;\
	OR	hlp0, x2, x2			;\
	AND	t6, t3, hlp0			;\
	AND	t5, x3, x3			;\
	OR	hlp0, x3, x3

// func p256PointDoubleAsm(res, in *P256Point)
TEXT ·p256PointDoubleAsm(SB),NOSPLIT,$216-16
	MOVV	R1, (8+32*6)(R3)		// save return address; NOSPLIT => no auto-save
	MOVV	res+0(FP), res_ptr
	MOVV	in+8(FP), a_ptr

	// zsqr = Z1^2
	LDx(z1in)
	CALL	p256SqrInternal<>(SB)
	STy(zsqrv)
	MOVV	in+8(FP), a_ptr

	// m_pre = X1 + zsqr  (x = X1, y = zsqr)
	LDx(x1in)
	LDy(zsqrv)
	p256AddInline				// x0..x3 = x + y = X1 + zsqr
	STx(mv)					// stash m_pre; x1in/zsqr are reused below

	// a_ptr is caller-clobbered by previous CALLs.
	MOVV	in+8(FP), a_ptr

	// z3 = 2*(Y1*Z1) -> res.z
	LDx(z1in)
	LDy(y1in)
	CALL	p256MulInternal<>(SB)
	p256MulBy2Inline
	STx(z3out)
	MOVV	in+8(FP), a_ptr			// reload AFTER y is consumed, BEFORE next in1 access

	// h = X1 - zsqr  (p256SubInternal: diff = y - x, so y=X1, x=zsqr)
	LDy(x1in)
	LDx(zsqrv)
	CALL	p256SubInternal<>(SB)
	MOVV	in+8(FP), a_ptr

	// h2 = h * m_pre
	MOVV	x0, y0
	MOVV	x1, y1
	MOVV	x2, y2
	MOVV	x3, y3
	LDx(mv)
	CALL	p256MulInternal<>(SB)
	STy(h2v)				// save h2 before MulBy2 consumes y

	// M = 3*h2 = 2*h2 + h2
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	p256MulBy2Inline			// x0..x3 = 2*h2
	LDy(h2v)				// y0..y3 = original h2 (not overwritten)
	p256AddInline				// x0..x3 = 2*h2 + h2 = 3*h2 = M
	STx(mv)					// store M into mv
	MOVV	in+8(FP), a_ptr			// reload BEFORE next in1 dereference;
						// LDy(h2v) above re-clobbered y1==a_ptr

	// s = (2*Y1)^2 = 4*Y1^2
	LDx(y1in)
	MOVV	x0, y0
	MOVV	x1, y1
	MOVV	x2, y2
	MOVV	x3, y3
	p256MulBy2Inline			// x0..x3 = 2*Y1
	MOVV	x0, y0
	MOVV	x1, y1
	MOVV	x2, y2
	MOVV	x3, y3
	CALL	p256SqrInternal<>(SB)		// y0..y3 = (2*Y1)^2 = s
	STy(sv)

	// yyyy = s^2 = 16*Y1^4
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	MOVV	in+8(FP), a_ptr
	CALL	p256SqrInternal<>(SB)		// y0..y3 = s^2 = 16*Y1^4

	// 8*Y1^4 = yyyy / 2 (mod P halving)
	p256HalveInline				// y0..y3 = yyyy/2
	STy(y3tmp)				// stash; needed later for Y3

	// S = X1 * s = 4*X1*Y1^2
	MOVV	in+8(FP), a_ptr
	LDx(x1in)
	LDy(sv)
	CALL	p256MulInternal<>(SB)
	STy(sv)					// overwrite sv with S

	// 2S = 8*X1*Y1^2
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	p256MulBy2Inline
	STx(tmpv)				// tmp = 2S

	// Msqr = M^2
	LDx(mv)
	CALL	p256SqrInternal<>(SB)		// y0..y3 = M^2

	// X3 = Msqr - 2S  (diff = y - x, y = Msqr, x = 2S)
	LDx(tmpv)
	CALL	p256SubInternal<>(SB)
	MOVV	in+8(FP), a_ptr
	STx(x3out)				// write res.x

	// S - X3  (diff = y - x, y = S, x = X3)
	LDy(sv)
	CALL	p256SubInternal<>(SB)
	MOVV	in+8(FP), a_ptr

	// M * (S - X3)
	MOVV	x0, y0
	MOVV	x1, y1
	MOVV	x2, y2
	MOVV	x3, y3
	LDx(mv)
	CALL	p256MulInternal<>(SB)

	// Y3 = M*(S-X3) - 8*Y1^4  (diff = y - x, y = previous result, x = y3tmp)
	LDx(y3tmp)
	CALL	p256SubInternal<>(SB)
	MOVV	in+8(FP), a_ptr
	STx(y3out)				// write res.y

	MOVV	(8+32*6)(R3), R1		// restore return address
	RET

#undef y2in
#undef x3out
#undef y3out
#undef z3out
#define y2in(off) (off + 32)(b_ptr)
#define x3out(off) (off)(b_ptr)
#define y3out(off) (off + 32)(b_ptr)
#define z3out(off) (off + 64)(b_ptr)

// p256PointAddAsm stack slots (addressed via the SP pseudo-register)
#define z2sqrv(off)	(8 + 32*0  + off)(R3)
#define s1v(off)	(8 + 32*1  + off)(R3)
#define z1sqrv(off)	(8 + 32*2  + off)(R3)
#define rv2(off)	(8 + 32*3  + off)(R3)	// r
#define u1v(off)	(8 + 32*4  + off)(R3)
#define u2v(off)	(8 + 32*5  + off)(R3)
#define hsqrv(off)	(8 + 32*6  + off)(R3)
#define rsqrv(off)	(8 + 32*7  + off)(R3)
#define hcubv(off)	(8 + 32*8  + off)(R3)
#define s2v2(off)	(8 + 32*9  + off)(R3)	// s2
#define hv2(off)	(8 + 32*10 + off)(R3)	// h
#define degflag(off)	(8 + 32*11 + off)(R3)	// degenerate-point flag (8 bytes used)

// y2inp is only valid while b_ptr still points at in2
#define y2inp(off) (off+32)(b_ptr)

// func p256PointAddAsm(res, in1, in2 *P256Point) int
TEXT ·p256PointAddAsm(SB),0,$392-32
	MOVV	in1+8(FP), a_ptr
	MOVV	in2+16(FP), b_ptr

	// z2sqr = Z2^2
	LDx(z2in)
	CALL	p256SqrInternal<>(SB)
	STy(z2sqrv)

	// z2cub = z2sqr * Z2
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	RELOAD_PTRS
	LDy(z2in)
	CALL	p256MulInternal<>(SB)

	// s1 = z2cub * Y1
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	RELOAD_PTRS
	LDy(y1in)
	CALL	p256MulInternal<>(SB)
	STy(s1v)
	RELOAD_PTRS

	// z1sqr = Z1^2
	LDx(z1in)
	CALL	p256SqrInternal<>(SB)
	STy(z1sqrv)
	RELOAD_PTRS

	// z1cub = z1sqr * Z1
	LDx(z1sqrv)
	LDy(z1in)
	CALL	p256MulInternal<>(SB)

	// s2 = z1cub * Y2
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	RELOAD_PTRS
	LDy(y2in)
	CALL	p256MulInternal<>(SB)

	// r = s2 - s1  (diff = y - x, y = s2, x = s1)
	LDx(s1v)
	CALL	p256SubInternal<>(SB)
	STx(rv2)

	// detect r == 0 mod P (r == 0 or r == P)
	MOVV	$1, t2
	OR	x0, x1, t0
	OR	x2, x3, t1
	OR	t1, t0, t0
	SGTU	t0, R0, t1			// t1 = (t0 != 0) => 1 means nonzero
	XOR	$1, t1, t1			// t1 = (t0 == 0) => 1 means r == 0
	MOVV	t1, hlp1

	XOR	$-1, x0, t0
	MOVV	p256const0<>(SB), t5
	XOR	t5, x1, t1
	MOVV	p256const1<>(SB), t6
	XOR	t6, x3, t3
	OR	t0, t1, t0
	OR	x2, t3, t1
	OR	t1, t0, t0
	SGTU	t0, R0, t1
	XOR	$1, t1, t1			// t1 = (r == P) => 1
	OR	t1, hlp1, hlp1			// hlp1 = (r==0) OR (r==P)

	MOVV	hlp1, degflag(0)		// spill immediately so later CALLs can't clobber it

	// u1 = X1 * z2sqr
	RELOAD_PTRS				// a_ptr/b_ptr were clobbered by previous CALL
	LDx(x1in)
	LDy(z2sqrv)
	CALL	p256MulInternal<>(SB)
	STy(u1v)
	RELOAD_PTRS

	// u2 = X2 * z1sqr
	LDx(x2in)
	LDy(z1sqrv)
	CALL	p256MulInternal<>(SB)

	// h = u2 - u1  (diff = y - x, y = u2, x = u1)
	LDx(u1v)
	CALL	p256SubInternal<>(SB)
	STx(hv2)

	// detect h == 0 mod P; AND into degflag (degenerate only if both are)
	MOVV	$1, t2
	OR	x0, x1, t0
	OR	x2, x3, t1
	OR	t1, t0, t0
	SGTU	t0, R0, t1
	XOR	$1, t1, t1
	AND	t1, t2, hlp0

	XOR	$-1, x0, t0
	MOVV	p256const0<>(SB), t5
	XOR	t5, x1, t1
	MOVV	p256const1<>(SB), t6
	XOR	t6, x3, t3
	OR	t0, t1, t0
	OR	x2, t3, t1
	OR	t1, t0, t0
	SGTU	t0, R0, t1
	XOR	$1, t1, t1
	OR	t1, hlp0, hlp0			// hlp0 = (h==0) OR (h==P)

	MOVV	degflag(0), t4
	AND	hlp0, t4, t4
	MOVV	t4, degflag(0)			// degflag = r-degenerate AND h-degenerate

	// rsqr = r^2
	LDx(rv2)
	CALL	p256SqrInternal<>(SB)
	STy(rsqrv)
	RELOAD_PTRS

	// hsqr = h^2
	LDx(hv2)
	CALL	p256SqrInternal<>(SB)
	STy(hsqrv)
	RELOAD_PTRS

	// hcub = h * hsqr
	LDx(hsqrv)
	LDy(hv2)
	CALL	p256MulInternal<>(SB)
	STy(hcubv)
	RELOAD_PTRS

	// s2' = s1 * hcub
	LDx(s1v)
	LDy(hcubv)
	CALL	p256MulInternal<>(SB)
	STy(s2v2)
	RELOAD_PTRS

	// z3 = z1 * z2 * h
	LDx(z1in)
	LDy(z2in)
	CALL	p256MulInternal<>(SB)
	MOVV	y0, x0
	MOVV	y1, x1
	MOVV	y2, x2
	MOVV	y3, x3
	RELOAD_PTRS
	LDy(hv2)
	CALL	p256MulInternal<>(SB)
	MOVV	res+0(FP), b_ptr		// b_ptr now points at res; all reads
						// via the old b_ptr (in2) must be done
	STy(z3out)

	// u2' = hsqr * u1
	LDx(hsqrv)
	LDy(u1v)
	CALL	p256MulInternal<>(SB)
	STy(u2v)

	// tmp = 2 * u2'
	p256MulBy2Inline
	RELOAD_PTRS

	// x3_0 = rsqr - tmp  (diff = y - x, y = rsqr, x = tmp)
	LDy(rsqrv)
	CALL	p256SubInternal<>(SB)
	RELOAD_PTRS
	MOVV	res+0(FP), b_ptr

	// x3 = x3_0 - hcub
	MOVV	x0, y0
	MOVV	x1, y1
	MOVV	x2, y2
	MOVV	x3, y3
	LDx(hcubv)
	CALL	p256SubInternal<>(SB)
	RELOAD_PTRS
	MOVV	res+0(FP), b_ptr
	STx(x3out)

	// tmp2 = u2' - x3
	LDy(u2v)
	CALL	p256SubInternal<>(SB)
	RELOAD_PTRS
	MOVV	res+0(FP), b_ptr

	// tmp3 = r * tmp2
	LDy(rv2)
	CALL	p256MulInternal<>(SB)

	// y3 = tmp3 - s2'  (diff = y - x, y = tmp3, x = s2')
	LDx(s2v2)
	CALL	p256SubInternal<>(SB)
	RELOAD_PTRS
	MOVV	res+0(FP), b_ptr
	STx(y3out)

	MOVV	degflag(0), t0
	MOVV	t0, ret+24(FP)

	RET
