// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

#include "textflag.h"

// P256 constants
DATA p256const0<>+0x00(SB)/8, $0x00000000ffffffff
DATA p256const1<>+0x00(SB)/8, $0xffffffff00000001
GLOBL p256const0<>(SB), RODATA, $8
GLOBL p256const1<>(SB), RODATA, $8

// func p256Mul(out, a, b *p256MontgomeryDomainFieldElement)
// RISC-V assembly implementation of P256 Montgomery multiplication
// Register allocation:
//   X10 = out pointer
//   X11 = a pointer (temporary), later used as temp variable
//   X12 = b pointer (temporary)
//   X13-X16 = x0-x3 (input a)
//   X17-X20 = y0-y3 (input b)
//   X21-X24 = acc0-acc3 (accumulator low bits)
//   X25-X26, X28-X29 = acc4-acc7 (accumulator high bits)
//   X30 = const1 = 0xFFFFFFFF00000001
//   X5-X9, X12 = temporary variables
TEXT ·p256Mul(SB),NOSPLIT,$0-24
	MOV	out+0(FP), X10
	MOV	a+8(FP), X11
	MOV	b+16(FP), X12

	// Load const1
	MOV	$p256const1<>(SB), X30
	MOV	(X30), X30

	// Load input x (a): X13-X16 = x0-x3
	MOV	0(X11), X13
	MOV	8(X11), X14
	MOV	16(X11), X15
	MOV	24(X11), X16

	// Load input y (b): X17-X20 = y0-y3
	MOV	0(X12), X17
	MOV	8(X12), X18
	MOV	16(X12), X19
	MOV	24(X12), X20

	// =====================================
	// Round 0: y[0] * x
	// =====================================
	MUL	X17, X13, X21		// acc0
	MULHU	X17, X13, X22		// acc1

	MUL	X17, X14, X5
	MULHU	X17, X14, X23
	ADD	X22, X5, X22
	SLTU	X5, X22, X6
	ADD	X23, X6, X23

	MUL	X17, X15, X5
	MULHU	X17, X15, X24
	ADD	X23, X5, X23
	SLTU	X5, X23, X6
	ADD	X24, X6, X24

	MUL	X17, X16, X5
	MULHU	X17, X16, X25
	ADD	X24, X5, X24
	SLTU	X5, X24, X6
	ADD	X25, X6, X25

	// {x25 x24 x23 x22 x21}

	// =====================================
	// First reduction step
	// =====================================
	SLL	$32, X21, X5		// t0 = acc0 << 32
	SRL	$32, X21, X7		// t1 = acc0 >> 32
	MUL	X21, X30, X8		// t2 = lo(acc0 * const1)
	MULHU	X21, X30, X21		// acc0 = hi(acc0 * const1)

	ADD	X22, X5, X22
	SLTU	X5, X22, X6

	ADD	X23, X7, X23
	SLTU	X7, X23, X9
	ADD	X23, X6, X23
	SLTU	X6, X23, X5
	ADD	X9, X5, X6

	ADD	X24, X8, X24
	SLTU	X8, X24, X9
	ADD	X24, X6, X24
	SLTU	X6, X24, X5
	ADD	X9, X5, X6

	ADD	X21, X6, X21

	// =====================================
	// Round 1: y[1] * x
	// =====================================
	MULHU	X18, X13, X7
	MULHU	X18, X14, X8
	MULHU	X18, X15, X9
	MULHU	X18, X16, X26

	MUL	X18, X13, X5
	ADD	X22, X5, X22
	SLTU	X5, X22, X6

	MUL	X18, X14, X5
	ADD	X23, X5, X23
	SLTU	X5, X23, X12
	ADD	X23, X6, X23
	SLTU	X6, X23, X5
	ADD	X12, X5, X6

	MUL	X18, X15, X5
	ADD	X24, X5, X24
	SLTU	X5, X24, X12
	ADD	X24, X6, X24
	SLTU	X6, X24, X5
	ADD	X12, X5, X6

	MUL	X18, X16, X5
	ADD	X25, X5, X25
	SLTU	X5, X25, X12
	ADD	X25, X6, X25
	SLTU	X6, X25, X5
	ADD	X12, X5, X6

	MOV	X6, X29			// Temporarily store carry_lo

	ADD	X23, X7, X23
	SLTU	X7, X23, X6

	ADD	X24, X8, X24
	SLTU	X8, X24, X5
	ADD	X24, X6, X24
	SLTU	X6, X24, X12
	ADD	X5, X12, X6

	ADD	X25, X9, X25
	SLTU	X9, X25, X5
	ADD	X25, X6, X25
	SLTU	X6, X25, X12
	ADD	X5, X12, X6

	ADD	X26, X29, X26		// acc5 = hi(y1*x3) + carry_lo
	ADD	X26, X6, X26		// acc5 += carry_hi

	// =====================================
	// Second reduction step
	// =====================================
	SLL	$32, X22, X5
	SRL	$32, X22, X7
	MUL	X22, X30, X8
	MULHU	X22, X30, X22

	ADD	X23, X5, X23
	SLTU	X5, X23, X6

	ADD	X24, X7, X24
	SLTU	X7, X24, X9
	ADD	X24, X6, X24
	SLTU	X6, X24, X5
	ADD	X9, X5, X6

	ADD	X21, X8, X21
	SLTU	X8, X21, X9
	ADD	X21, X6, X21
	SLTU	X6, X21, X5
	ADD	X9, X5, X6

	ADD	X22, X6, X22

	// =====================================
	// Round 2: y[2] * x
	// =====================================
	MULHU	X19, X13, X7
	MULHU	X19, X14, X8
	MULHU	X19, X15, X9
	MULHU	X19, X16, X28

	MUL	X19, X13, X5
	ADD	X23, X5, X23
	SLTU	X5, X23, X6

	MUL	X19, X14, X5
	ADD	X24, X5, X24
	SLTU	X5, X24, X12
	ADD	X24, X6, X24
	SLTU	X6, X24, X5
	ADD	X12, X5, X6

	MUL	X19, X15, X5
	ADD	X25, X5, X25
	SLTU	X5, X25, X12
	ADD	X25, X6, X25
	SLTU	X6, X25, X5
	ADD	X12, X5, X6

	MUL	X19, X16, X5
	ADD	X26, X5, X26
	SLTU	X5, X26, X12
	ADD	X26, X6, X26
	SLTU	X6, X26, X5
	ADD	X12, X5, X6

	MOV	X6, X29

	ADD	X24, X7, X24
	SLTU	X7, X24, X6

	ADD	X25, X8, X25
	SLTU	X8, X25, X5
	ADD	X25, X6, X25
	SLTU	X6, X25, X12
	ADD	X5, X12, X6

	ADD	X26, X9, X26
	SLTU	X9, X26, X5
	ADD	X26, X6, X26
	SLTU	X6, X26, X12
	ADD	X5, X12, X6

	ADD	X28, X29, X28
	ADD	X28, X6, X28

	MOV	X0, X29

	// =====================================
	// Third reduction step
	// =====================================
	SLL	$32, X23, X5
	SRL	$32, X23, X7
	MUL	X23, X30, X8
	MULHU	X23, X30, X23

	ADD	X24, X5, X24
	SLTU	X5, X24, X6

	ADD	X21, X7, X21
	SLTU	X7, X21, X9
	ADD	X21, X6, X21
	SLTU	X6, X21, X5
	ADD	X9, X5, X6

	ADD	X22, X8, X22
	SLTU	X8, X22, X9
	ADD	X22, X6, X22
	SLTU	X6, X22, X5
	ADD	X9, X5, X6

	ADD	X23, X6, X23

	// =====================================
	// Round 3: y[3] * x
	// =====================================
	MULHU	X20, X13, X7
	MULHU	X20, X14, X8
	MULHU	X20, X15, X9
	MULHU	X20, X16, X29

	MUL	X20, X13, X5
	ADD	X24, X5, X24
	SLTU	X5, X24, X6

	MUL	X20, X14, X5
	ADD	X25, X5, X25
	SLTU	X5, X25, X12
	ADD	X25, X6, X25
	SLTU	X6, X25, X5
	ADD	X12, X5, X6

	MUL	X20, X15, X5
	ADD	X26, X5, X26
	SLTU	X5, X26, X12
	ADD	X26, X6, X26
	SLTU	X6, X26, X5
	ADD	X12, X5, X6

	MUL	X20, X16, X5
	ADD	X28, X5, X28
	SLTU	X5, X28, X12
	ADD	X28, X6, X28
	SLTU	X6, X28, X5
	ADD	X12, X5, X6

	MOV	X6, X12			// Temporarily store carry_lo

	ADD	X25, X7, X25
	SLTU	X7, X25, X6

	ADD	X26, X8, X26
	SLTU	X8, X26, X5
	ADD	X26, X6, X26
	SLTU	X6, X26, X7
	ADD	X5, X7, X6

	ADD	X28, X9, X28
	SLTU	X9, X28, X5
	ADD	X28, X6, X28
	SLTU	X6, X28, X7
	ADD	X5, X7, X6

	ADD	X29, X12, X29
	ADD	X29, X6, X29

	// =====================================
	// Last reduction step
	// =====================================
	SLL	$32, X24, X5
	SRL	$32, X24, X7
	MUL	X24, X30, X8
	MULHU	X24, X30, X24

	ADD	X21, X5, X21
	SLTU	X5, X21, X6

	ADD	X22, X7, X22
	SLTU	X7, X22, X9
	ADD	X22, X6, X22
	SLTU	X6, X22, X5
	ADD	X9, X5, X6

	ADD	X23, X8, X23
	SLTU	X8, X23, X9
	ADD	X23, X6, X23
	SLTU	X6, X23, X5
	ADD	X9, X5, X6

	ADD	X24, X6, X24

	// =====================================
	// Add bits [511:256]
	// =====================================
	ADD	X21, X25, X21
	SLTU	X25, X21, X6

	ADD	X22, X26, X22
	SLTU	X26, X22, X9
	ADD	X22, X6, X22
	SLTU	X6, X22, X5
	ADD	X9, X5, X6

	ADD	X23, X28, X23
	SLTU	X28, X23, X9
	ADD	X23, X6, X23
	SLTU	X6, X23, X5
	ADD	X9, X5, X6

	ADD	X24, X29, X24
	SLTU	X29, X24, X9
	ADD	X24, X6, X24
	SLTU	X6, X24, X5
	ADD	X9, X5, X25

	// =====================================
	// Conditional subtraction
	// =====================================
	MOV	$p256const0<>(SB), X5
	MOV	(X5), X5		// X5 = const0 = 0x00000000FFFFFFFF
	MOV	$-1, X6			// X6 = p0 = 0xFFFFFFFFFFFFFFFF

	// Step 1: t0, b0 = Sub64(acc0, p0, 0)
	ADD	$1, X21, X13		// X13 = t0 = acc0 + 1
	SLTU	X6, X21, X7		// X7 = b0 = (acc0 < p0)

	// Step 2: t1, b1 = Sub64(acc1, p1, b0)
	SUB	X5, X22, X8		// X8 = tmp = acc1 - const0
	SLTU	X5, X22, X9		// X9 = (acc1 < const0)
	SUB	X7, X8, X14		// X14 = t1 = tmp - b0
	SLTU	X7, X8, X12		// X12 = (tmp < b0)
	OR	X9, X12, X7		// X7 = b1

	// Step 3: t2, b2 = Sub64(acc2, 0, b1)
	SUB	X7, X23, X15		// X15 = t2 = acc2 - b1
	SLTU	X7, X23, X7		// X7 = b2 = (acc2 < b1)

	// Step 4: t3, b3 = Sub64(acc3, p3, b2)
	SUB	X30, X24, X8		// X8 = tmp = acc3 - const1
	SLTU	X30, X24, X9		// X9 = (acc3 < const1)
	SUB	X7, X8, X16		// X16 = t3 = tmp - b2
	SLTU	X7, X8, X12		// X12 = (tmp < b2)
	OR	X9, X12, X7		// X7 = b3

	// Step 5: final_borrow = (carry < b3)
	SLTU	X7, X25, X8		// X8 = (carry < b3) = final_borrow

	// Conditional select: mask = final_borrow ? -1 : 0
	NEG	X8, X8			// X8 = mask

	// Select result
	XOR	X21, X13, X9
	AND	X8, X9, X9
	XOR	X9, X13, X13

	XOR	X22, X14, X9
	AND	X8, X9, X9
	XOR	X9, X14, X14

	XOR	X23, X15, X9
	AND	X8, X9, X9
	XOR	X9, X15, X15

	XOR	X24, X16, X9
	AND	X8, X9, X9
	XOR	X9, X16, X16

	// Write result
	MOV	X13, 0(X10)
	MOV	X14, 8(X10)
	MOV	X15, 16(X10)
	MOV	X16, 24(X10)

	RET
	