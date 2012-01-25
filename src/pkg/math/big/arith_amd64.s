// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// TODO(gri) - experiment with unrolled loops for faster execution

// func mulWW(x, y Word) (z1, z0 Word)
TEXT ·mulWW(SB),7,$0
	MOVQ x+0(FP), AX
	MULQ y+8(FP)
	MOVQ DX, z1+16(FP)
	MOVQ AX, z0+24(FP)
	RET


// func divWW(x1, x0, y Word) (q, r Word)
TEXT ·divWW(SB),7,$0
	MOVQ x1+0(FP), DX
	MOVQ x0+8(FP), AX
	DIVQ y+16(FP)
	MOVQ AX, q+24(FP)
	MOVQ DX, r+32(FP)
	RET


// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVQ y+32(FP), R9
	MOVL n+8(FP), R11
	MOVQ $0, BX		// i = 0
	MOVQ $0, DX		// c = 0
	JMP E1

L1:	MOVQ (R8)(BX*8), AX
	RCRQ $1, DX
	ADCQ (R9)(BX*8), AX
	RCLQ $1, DX
	MOVQ AX, (R10)(BX*8)
	ADDL $1, BX		// i++

E1:	CMPQ BX, R11		// i < n
	JL L1

	MOVQ DX, c+48(FP)
	RET


// func subVV(z, x, y []Word) (c Word)
// (same as addVV_s except for SBBQ instead of ADCQ and label names)
TEXT ·subVV(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVQ y+32(FP), R9
	MOVL n+8(FP), R11
	MOVQ $0, BX		// i = 0
	MOVQ $0, DX		// c = 0
	JMP E2

L2:	MOVQ (R8)(BX*8), AX
	RCRQ $1, DX
	SBBQ (R9)(BX*8), AX
	RCLQ $1, DX
	MOVQ AX, (R10)(BX*8)
	ADDL $1, BX		// i++

E2:	CMPQ BX, R11		// i < n
	JL L2

	MOVQ DX, c+48(FP)
	RET


// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVQ y+32(FP), AX	// c = y
	MOVL n+8(FP), R11
	MOVQ $0, BX		// i = 0
	JMP E3

L3:	ADDQ (R8)(BX*8), AX
	MOVQ AX, (R10)(BX*8)
	RCLQ $1, AX
	ANDQ $1, AX
	ADDL $1, BX		// i++

E3:	CMPQ BX, R11		// i < n
	JL L3

	MOVQ AX, c+40(FP)
	RET


// func subVW(z, x []Word, y Word) (c Word)
TEXT ·subVW(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVQ y+32(FP), AX	// c = y
	MOVL n+8(FP), R11
	MOVQ $0, BX		// i = 0
	JMP E4

L4:	MOVQ (R8)(BX*8), DX	// TODO(gri) is there a reverse SUBQ?
	SUBQ AX, DX
	MOVQ DX, (R10)(BX*8)
	RCLQ $1, AX
	ANDQ $1, AX
	ADDL $1, BX		// i++

E4:	CMPQ BX, R11		// i < n
	JL L4

	MOVQ AX, c+40(FP)
	RET


// func shlVU(z, x []Word, s uint) (c Word)
TEXT ·shlVU(SB),7,$0
	MOVL n+8(FP), BX	// i = n
	SUBL $1, BX		// i--
	JL X8b			// i < 0	(n <= 0)

	// n > 0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVL s+32(FP), CX
	MOVQ (R8)(BX*8), AX	// w1 = x[n-1]
	MOVQ $0, DX
	SHLQ CX, DX:AX		// w1>>ŝ
	MOVQ DX, c+40(FP)

	CMPL BX, $0
	JLE X8a			// i <= 0

	// i > 0
L8:	MOVQ AX, DX		// w = w1
	MOVQ -8(R8)(BX*8), AX	// w1 = x[i-1]
	SHLQ CX, DX:AX		// w<<s | w1>>ŝ
	MOVQ DX, (R10)(BX*8)	// z[i] = w<<s | w1>>ŝ
	SUBL $1, BX		// i--
	JG L8			// i > 0

	// i <= 0
X8a:	SHLQ CX, AX		// w1<<s
	MOVQ AX, (R10)		// z[0] = w1<<s
	RET

X8b:	MOVQ $0, c+40(FP)
	RET


// func shrVU(z, x []Word, s uint) (c Word)
TEXT ·shrVU(SB),7,$0
	MOVL n+8(FP), R11
	SUBL $1, R11		// n--
	JL X9b			// n < 0	(n <= 0)

	// n > 0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVL s+32(FP), CX
	MOVQ (R8), AX		// w1 = x[0]
	MOVQ $0, DX
	SHRQ CX, DX:AX		// w1<<ŝ
	MOVQ DX, c+40(FP)

	MOVQ $0, BX		// i = 0
	JMP E9

	// i < n-1
L9:	MOVQ AX, DX		// w = w1
	MOVQ 8(R8)(BX*8), AX	// w1 = x[i+1]
	SHRQ CX, DX:AX		// w>>s | w1<<ŝ
	MOVQ DX, (R10)(BX*8)	// z[i] = w>>s | w1<<ŝ
	ADDL $1, BX		// i++
	
E9:	CMPQ BX, R11
	JL L9			// i < n-1

	// i >= n-1
X9a:	SHRQ CX, AX		// w1>>s
	MOVQ AX, (R10)(R11*8)	// z[n-1] = w1>>s
	RET

X9b:	MOVQ $0, c+40(FP)
	RET


// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVQ y+32(FP), R9
	MOVQ r+40(FP), CX	// c = r
	MOVL n+8(FP), R11
	MOVQ $0, BX		// i = 0
	JMP E5

L5:	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (R10)(BX*8)
	MOVQ DX, CX
	ADDL $1, BX		// i++

E5:	CMPQ BX, R11		// i < n
	JL L5

	MOVQ CX, c+48(FP)
	RET


// func addMulVVW(z, x []Word, y Word) (c Word)
TEXT ·addMulVVW(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+16(FP), R8
	MOVQ y+32(FP), R9
	MOVL n+8(FP), R11
	MOVQ $0, BX		// i = 0
	MOVQ $0, CX		// c = 0
	JMP E6

L6:	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	ADDQ AX, (R10)(BX*8)
	ADCQ $0, DX
	MOVQ DX, CX
	ADDL $1, BX		// i++

E6:	CMPQ BX, R11		// i < n
	JL L6

	MOVQ CX, c+40(FP)
	RET


// func divWVW(z []Word, xn Word, x []Word, y Word) (r Word)
TEXT ·divWVW(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ xn+16(FP), DX	// r = xn
	MOVQ x+24(FP), R8
	MOVQ y+40(FP), R9
	MOVL n+8(FP), BX	// i = n
	JMP E7

L7:	MOVQ (R8)(BX*8), AX
	DIVQ R9
	MOVQ AX, (R10)(BX*8)

E7:	SUBL $1, BX		// i--
	JGE L7			// i >= 0

	MOVQ DX, r+48(FP)
	RET

// func bitLen(x Word) (n int)
TEXT ·bitLen(SB),7,$0
	BSRQ x+0(FP), AX
	JZ Z1
	INCQ AX
	MOVQ AX, n+8(FP)
	RET

Z1:	MOVQ $0, n+8(FP)
	RET
