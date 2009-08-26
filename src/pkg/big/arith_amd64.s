// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT big·useAsm(SB),7,$0
	MOVB $1, 8(SP)  // assembly routines enabled
	RET


// TODO(gri) - experiment with unrolled loops for faster execution

// func addVV_s(z, x, y *Word, n int) (c Word)
TEXT big·addVV_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+8(FP), R8
	MOVQ y+16(FP), R9
	MOVL n+24(FP), R11
	MOVQ $0, BX         // i = 0
	MOVQ $0, DX         // c = 0
	JMP E1

L1:	MOVQ (R8)(BX*8), AX
	RCRQ $1, DX
	ADCQ (R9)(BX*8), AX
	RCLQ $1, DX
	MOVQ AX, (R10)(BX*8)
	ADDL $1, BX			// i++

E1:	CMPQ BX, R11		// i < n
	JL L1

	MOVQ DX, c+32(FP)
	RET


// func subVV_s(z, x, y *Word, n int) (c Word)
// (same as addVV_s except for SBBQ instead of ADCQ and label names)
TEXT big·subVV_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+8(FP), R8
	MOVQ y+16(FP), R9
	MOVL n+24(FP), R11
	MOVQ $0, BX         // i = 0
	MOVQ $0, DX         // c = 0
	JMP E2

L2:	MOVQ (R8)(BX*8), AX
	RCRQ $1, DX
	SBBQ (R9)(BX*8), AX
	RCLQ $1, DX
	MOVQ AX, (R10)(BX*8)
	ADDL $1, BX         // i++

E2:	CMPQ BX, R11        // i < n
	JL L2

	MOVQ DX, c+32(FP)
	RET


// func addVW_s(z, x *Word, y Word, n int) (c Word)
TEXT big·addVW_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+8(FP), R8
	MOVQ y+16(FP), AX   // c = y
	MOVL n+24(FP), R11
	MOVQ $0, BX         // i = 0
	JMP E3

L3:	ADDQ (R8)(BX*8), AX
	MOVQ AX, (R10)(BX*8)
	RCLQ $1, AX
	ANDQ $1, AX
	ADDL $1, BX         // i++

E3:	CMPQ BX, R11        // i < n
	JL L3

	MOVQ AX, c+32(FP)
	RET


// func subVW_s(z, x *Word, y Word, n int) (c Word)
TEXT big·subVW_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+8(FP), R8
	MOVQ y+16(FP), AX   // c = y
	MOVL n+24(FP), R11
	MOVQ $0, BX         // i = 0
	JMP E4

L4:	MOVQ (R8)(BX*8), DX	// TODO(gri) is there a reverse SUBQ?
	SUBQ AX, DX
	MOVQ DX, (R10)(BX*8)
	RCLQ $1, AX
	ANDQ $1, AX
	ADDL $1, BX          // i++

E4:	CMPQ BX, R11         // i < n
	JL L4

	MOVQ AX, c+32(FP)
	RET


// func mulAddVWW_s(z, x *Word, y, r Word, n int) (c Word)
TEXT big·mulAddVWW_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+8(FP), R8
	MOVQ y+16(FP), R9
	MOVQ r+24(FP), CX   // c = r
	MOVL n+32(FP), R11
	MOVQ $0, BX         // i = 0
	JMP E5

L5:	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (R10)(BX*8)
	MOVQ DX, CX
	ADDL $1, BX         // i++

E5:	CMPQ BX, R11        // i < n
	JL L5

	MOVQ CX, c+40(FP)
	RET


// func addMulVVW_s(z, x *Word, y Word, n int) (c Word)
TEXT big·addMulVVW_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ x+8(FP), R8
	MOVQ y+16(FP), R9
	MOVL n+24(FP), R11
	MOVQ $0, BX         // i = 0
	MOVQ $0, CX         // c = 0
	JMP E6

L6:	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ (R10)(BX*8), AX
	ADCQ $0, DX
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (R10)(BX*8)
	MOVQ DX, CX
	ADDL $1, BX         // i++

E6:	CMPQ BX, R11        // i < n
	JL L6

	MOVQ CX, c+32(FP)
	RET


// divWVW_s(z* Word, xn Word, x *Word, y Word, n int) (r Word)
TEXT big·divWVW_s(SB),7,$0
	MOVQ z+0(FP), R10
	MOVQ xn+8(FP), DX   // r = xn
	MOVQ x+16(FP), R8
	MOVQ y+24(FP), R9
	MOVL n+32(FP), BX   // i = n
	JMP E7

L7:	MOVQ (R8)(BX*8), AX
	DIVQ R9
	MOVQ AX, (R10)(BX*8)

E7:	SUBL $1, BX         // i--
	JGE L7              // i >= 0

	MOVQ DX, r+40(FP)
	RET


// TODO(gri) Implement this routine completely in Go.
//           At the moment we need this assembly version.
TEXT big·divWWW_s(SB),7,$0
	MOVQ x1+0(FP), DX
	MOVQ x0+8(FP), AX
	DIVQ y+16(FP)
	MOVQ AX, q+24(FP)
	MOVQ DX, r+32(FP)
	RET
