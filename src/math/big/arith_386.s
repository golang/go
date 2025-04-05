// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),NOSPLIT,$0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL y+24(FP), CX
	MOVL z_len+4(FP), BP
	MOVL $0, BX		// i = 0
	MOVL $0, DX		// c = 0
	JMP E1

L1:	MOVL (SI)(BX*4), AX
	ADDL DX, DX		// restore CF
	ADCL (CX)(BX*4), AX
	SBBL DX, DX		// save CF
	MOVL AX, (DI)(BX*4)
	ADDL $1, BX		// i++

E1:	CMPL BX, BP		// i < n
	JL L1

	NEGL DX
	MOVL DX, c+36(FP)
	RET


// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SBBL instead of ADCL and label names)
TEXT ·subVV(SB),NOSPLIT,$0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL y+24(FP), CX
	MOVL z_len+4(FP), BP
	MOVL $0, BX		// i = 0
	MOVL $0, DX		// c = 0
	JMP E2

L2:	MOVL (SI)(BX*4), AX
	ADDL DX, DX		// restore CF
	SBBL (CX)(BX*4), AX
	SBBL DX, DX		// save CF
	MOVL AX, (DI)(BX*4)
	ADDL $1, BX		// i++

E2:	CMPL BX, BP		// i < n
	JL L2

	NEGL DX
	MOVL DX, c+36(FP)
	RET


// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),NOSPLIT,$0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL y+24(FP), AX	// c = y
	MOVL z_len+4(FP), BP
	MOVL $0, BX		// i = 0
	JMP E3

L3:	ADDL (SI)(BX*4), AX
	MOVL AX, (DI)(BX*4)
	SBBL AX, AX		// save CF
	NEGL AX
	ADDL $1, BX		// i++

E3:	CMPL BX, BP		// i < n
	JL L3

	MOVL AX, c+28(FP)
	RET


// func subVW(z, x []Word, y Word) (c Word)
TEXT ·subVW(SB),NOSPLIT,$0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL y+24(FP), AX	// c = y
	MOVL z_len+4(FP), BP
	MOVL $0, BX		// i = 0
	JMP E4

L4:	MOVL (SI)(BX*4), DX
	SUBL AX, DX
	MOVL DX, (DI)(BX*4)
	SBBL AX, AX		// save CF
	NEGL AX
	ADDL $1, BX		// i++

E4:	CMPL BX, BP		// i < n
	JL L4

	MOVL AX, c+28(FP)
	RET


// func shlVU(z, x []Word, s uint) (c Word)
TEXT ·shlVU(SB),NOSPLIT,$0
	MOVL z_len+4(FP), BX	// i = z
	SUBL $1, BX		// i--
	JL X8b			// i < 0	(n <= 0)

	// n > 0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL s+24(FP), CX
	MOVL (SI)(BX*4), AX	// w1 = x[n-1]
	MOVL $0, DX
	SHLL CX, AX, DX		// w1>>ŝ
	MOVL DX, c+28(FP)

	CMPL BX, $0
	JLE X8a			// i <= 0

	// i > 0
L8:	MOVL AX, DX		// w = w1
	MOVL -4(SI)(BX*4), AX	// w1 = x[i-1]
	SHLL CX, AX, DX		// w<<s | w1>>ŝ
	MOVL DX, (DI)(BX*4)	// z[i] = w<<s | w1>>ŝ
	SUBL $1, BX		// i--
	JG L8			// i > 0

	// i <= 0
X8a:	SHLL CX, AX		// w1<<s
	MOVL AX, (DI)		// z[0] = w1<<s
	RET

X8b:	MOVL $0, c+28(FP)
	RET


// func shrVU(z, x []Word, s uint) (c Word)
TEXT ·shrVU(SB),NOSPLIT,$0
	MOVL z_len+4(FP), BP
	SUBL $1, BP		// n--
	JL X9b			// n < 0	(n <= 0)

	// n > 0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL s+24(FP), CX
	MOVL (SI), AX		// w1 = x[0]
	MOVL $0, DX
	SHRL CX, AX, DX		// w1<<ŝ
	MOVL DX, c+28(FP)

	MOVL $0, BX		// i = 0
	JMP E9

	// i < n-1
L9:	MOVL AX, DX		// w = w1
	MOVL 4(SI)(BX*4), AX	// w1 = x[i+1]
	SHRL CX, AX, DX		// w>>s | w1<<ŝ
	MOVL DX, (DI)(BX*4)	// z[i] = w>>s | w1<<ŝ
	ADDL $1, BX		// i++

E9:	CMPL BX, BP
	JL L9			// i < n-1

	// i >= n-1
X9a:	SHRL CX, AX		// w1>>s
	MOVL AX, (DI)(BP*4)	// z[n-1] = w1>>s
	RET

X9b:	MOVL $0, c+28(FP)
	RET


// func mulAddVWW(z, x []Word, m, a Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVL z+0(FP), DI
	MOVL x+12(FP), SI
	MOVL m+24(FP), BP
	MOVL a+28(FP), CX	// c = a
	MOVL z_len+4(FP), BX
	LEAL (DI)(BX*4), DI
	LEAL (SI)(BX*4), SI
	NEGL BX			// i = -n
	JMP E5

L5:	MOVL (SI)(BX*4), AX
	MULL BP
	ADDL CX, AX
	ADCL $0, DX
	MOVL AX, (DI)(BX*4)
	MOVL DX, CX
	ADDL $1, BX		// i++

E5:	CMPL BX, $0		// i < 0
	JL L5

	MOVL CX, c+32(FP)
	RET


// func addMulVVWW(z, x, y []Word, m, a Word) (c Word)
TEXT ·addMulVVWW(SB),NOSPLIT,$0
	MOVL z+0(FP), BP
	MOVL x+12(FP), DI
	MOVL y+24(FP), SI
	MOVL a+40(FP), CX
	MOVL z_len+4(FP), BX
	LEAL (DI)(BX*4), DI
	LEAL (SI)(BX*4), SI
	LEAL (BP)(BX*4), BP
	NEGL BX			// i = -n
	JMP E6

L6:	MOVL (SI)(BX*4), AX
	MULL m+36(FP)
	ADDL CX, AX
	ADCL $0, DX
	ADDL (DI)(BX*4), AX
	MOVL AX, (BP)(BX*4)
	ADCL $0, DX
	MOVL DX, CX
	ADDL $1, BX		// i++

E6:	CMPL BX, $0		// i < 0
	JL L6

	MOVL CX, c+44(FP)
	RET



