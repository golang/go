// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// The carry bit is saved with SBBQ Rx, Rx: if the carry was set, Rx is -1, otherwise it is 0.
// It is restored with ADDQ Rx, Rx: if Rx was -1 the carry is set, otherwise it is cleared.
// This is faster than using rotate instructions.

// func addVV(z, x, y []Word) (c Word)
TEXT ·addVV(SB),NOSPLIT,$0
	MOVQ z_len+8(FP), DI
	MOVQ x+24(FP), R8
	MOVQ y+48(FP), R9
	MOVQ z+0(FP), R10

	MOVQ $0, CX		// c = 0
	MOVQ $0, SI		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUBQ $4, DI		// n -= 4
	JL V1			// if n < 0 goto V1

U1:	// n >= 0
	// regular loop body unrolled 4x
	ADDQ CX, CX		// restore CF
	MOVQ 0(R8)(SI*8), R11
	MOVQ 8(R8)(SI*8), R12
	MOVQ 16(R8)(SI*8), R13
	MOVQ 24(R8)(SI*8), R14
	ADCQ 0(R9)(SI*8), R11
	ADCQ 8(R9)(SI*8), R12
	ADCQ 16(R9)(SI*8), R13
	ADCQ 24(R9)(SI*8), R14
	MOVQ R11, 0(R10)(SI*8)
	MOVQ R12, 8(R10)(SI*8)
	MOVQ R13, 16(R10)(SI*8)
	MOVQ R14, 24(R10)(SI*8)
	SBBQ CX, CX		// save CF

	ADDQ $4, SI		// i += 4
	SUBQ $4, DI		// n -= 4
	JGE U1			// if n >= 0 goto U1

V1:	ADDQ $4, DI		// n += 4
	JLE E1			// if n <= 0 goto E1

L1:	// n > 0
	ADDQ CX, CX		// restore CF
	MOVQ 0(R8)(SI*8), R11
	ADCQ 0(R9)(SI*8), R11
	MOVQ R11, 0(R10)(SI*8)
	SBBQ CX, CX		// save CF

	ADDQ $1, SI		// i++
	SUBQ $1, DI		// n--
	JG L1			// if n > 0 goto L1

E1:	NEGQ CX
	MOVQ CX, c+72(FP)	// return c
	RET


// func subVV(z, x, y []Word) (c Word)
// (same as addVV except for SBBQ instead of ADCQ and label names)
TEXT ·subVV(SB),NOSPLIT,$0
	MOVQ z_len+8(FP), DI
	MOVQ x+24(FP), R8
	MOVQ y+48(FP), R9
	MOVQ z+0(FP), R10

	MOVQ $0, CX		// c = 0
	MOVQ $0, SI		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUBQ $4, DI		// n -= 4
	JL V2			// if n < 0 goto V2

U2:	// n >= 0
	// regular loop body unrolled 4x
	ADDQ CX, CX		// restore CF
	MOVQ 0(R8)(SI*8), R11
	MOVQ 8(R8)(SI*8), R12
	MOVQ 16(R8)(SI*8), R13
	MOVQ 24(R8)(SI*8), R14
	SBBQ 0(R9)(SI*8), R11
	SBBQ 8(R9)(SI*8), R12
	SBBQ 16(R9)(SI*8), R13
	SBBQ 24(R9)(SI*8), R14
	MOVQ R11, 0(R10)(SI*8)
	MOVQ R12, 8(R10)(SI*8)
	MOVQ R13, 16(R10)(SI*8)
	MOVQ R14, 24(R10)(SI*8)
	SBBQ CX, CX		// save CF

	ADDQ $4, SI		// i += 4
	SUBQ $4, DI		// n -= 4
	JGE U2			// if n >= 0 goto U2

V2:	ADDQ $4, DI		// n += 4
	JLE E2			// if n <= 0 goto E2

L2:	// n > 0
	ADDQ CX, CX		// restore CF
	MOVQ 0(R8)(SI*8), R11
	SBBQ 0(R9)(SI*8), R11
	MOVQ R11, 0(R10)(SI*8)
	SBBQ CX, CX		// save CF

	ADDQ $1, SI		// i++
	SUBQ $1, DI		// n--
	JG L2			// if n > 0 goto L2

E2:	NEGQ CX
	MOVQ CX, c+72(FP)	// return c
	RET


// func addVW(z, x []Word, y Word) (c Word)
TEXT ·addVW(SB),NOSPLIT,$0
	MOVQ z_len+8(FP), DI
	CMPQ DI, $32
	JG large
	MOVQ x+24(FP), R8
	MOVQ y+48(FP), CX	// c = y
	MOVQ z+0(FP), R10

	MOVQ $0, SI		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUBQ $4, DI		// n -= 4
	JL V3			// if n < 4 goto V3

U3:	// n >= 0
	// regular loop body unrolled 4x
	MOVQ 0(R8)(SI*8), R11
	MOVQ 8(R8)(SI*8), R12
	MOVQ 16(R8)(SI*8), R13
	MOVQ 24(R8)(SI*8), R14
	ADDQ CX, R11
	ADCQ $0, R12
	ADCQ $0, R13
	ADCQ $0, R14
	SBBQ CX, CX		// save CF
	NEGQ CX
	MOVQ R11, 0(R10)(SI*8)
	MOVQ R12, 8(R10)(SI*8)
	MOVQ R13, 16(R10)(SI*8)
	MOVQ R14, 24(R10)(SI*8)

	ADDQ $4, SI		// i += 4
	SUBQ $4, DI		// n -= 4
	JGE U3			// if n >= 0 goto U3

V3:	ADDQ $4, DI		// n += 4
	JLE E3			// if n <= 0 goto E3

L3:	// n > 0
	ADDQ 0(R8)(SI*8), CX
	MOVQ CX, 0(R10)(SI*8)
	SBBQ CX, CX		// save CF
	NEGQ CX

	ADDQ $1, SI		// i++
	SUBQ $1, DI		// n--
	JG L3			// if n > 0 goto L3

E3:	MOVQ CX, c+56(FP)	// return c
	RET
large:
	JMP ·addVWlarge(SB)


// func subVW(z, x []Word, y Word) (c Word)
// (same as addVW except for SUBQ/SBBQ instead of ADDQ/ADCQ and label names)
TEXT ·subVW(SB),NOSPLIT,$0
	MOVQ z_len+8(FP), DI
	CMPQ DI, $32
	JG large
	MOVQ x+24(FP), R8
	MOVQ y+48(FP), CX	// c = y
	MOVQ z+0(FP), R10

	MOVQ $0, SI		// i = 0

	// s/JL/JMP/ below to disable the unrolled loop
	SUBQ $4, DI		// n -= 4
	JL V4			// if n < 4 goto V4

U4:	// n >= 0
	// regular loop body unrolled 4x
	MOVQ 0(R8)(SI*8), R11
	MOVQ 8(R8)(SI*8), R12
	MOVQ 16(R8)(SI*8), R13
	MOVQ 24(R8)(SI*8), R14
	SUBQ CX, R11
	SBBQ $0, R12
	SBBQ $0, R13
	SBBQ $0, R14
	SBBQ CX, CX		// save CF
	NEGQ CX
	MOVQ R11, 0(R10)(SI*8)
	MOVQ R12, 8(R10)(SI*8)
	MOVQ R13, 16(R10)(SI*8)
	MOVQ R14, 24(R10)(SI*8)

	ADDQ $4, SI		// i += 4
	SUBQ $4, DI		// n -= 4
	JGE U4			// if n >= 0 goto U4

V4:	ADDQ $4, DI		// n += 4
	JLE E4			// if n <= 0 goto E4

L4:	// n > 0
	MOVQ 0(R8)(SI*8), R11
	SUBQ CX, R11
	MOVQ R11, 0(R10)(SI*8)
	SBBQ CX, CX		// save CF
	NEGQ CX

	ADDQ $1, SI		// i++
	SUBQ $1, DI		// n--
	JG L4			// if n > 0 goto L4

E4:	MOVQ CX, c+56(FP)	// return c
	RET
large:
	JMP ·subVWlarge(SB)


// func lshVU(z, x []Word, s uint) (c Word)
TEXT ·lshVU(SB),NOSPLIT,$0
	MOVQ z_len+8(FP), BX	// i = z
	SUBQ $1, BX		// i--
	JL X8b			// i < 0	(n <= 0)

	// n > 0
	MOVQ z+0(FP), R10
	MOVQ x+24(FP), R8
	MOVQ s+48(FP), CX
	MOVQ (R8)(BX*8), AX	// w1 = x[n-1]
	MOVQ $0, DX
	SHLQ CX, AX, DX		// w1>>ŝ
	MOVQ DX, c+56(FP)

	CMPQ BX, $0
	JLE X8a			// i <= 0

	// i > 0
L8:	MOVQ AX, DX		// w = w1
	MOVQ -8(R8)(BX*8), AX	// w1 = x[i-1]
	SHLQ CX, AX, DX		// w<<s | w1>>ŝ
	MOVQ DX, (R10)(BX*8)	// z[i] = w<<s | w1>>ŝ
	SUBQ $1, BX		// i--
	JG L8			// i > 0

	// i <= 0
X8a:	SHLQ CX, AX		// w1<<s
	MOVQ AX, (R10)		// z[0] = w1<<s
	RET

X8b:	MOVQ $0, c+56(FP)
	RET


// func rshVU(z, x []Word, s uint) (c Word)
TEXT ·rshVU(SB),NOSPLIT,$0
	MOVQ z_len+8(FP), R11
	SUBQ $1, R11		// n--
	JL X9b			// n < 0	(n <= 0)

	// n > 0
	MOVQ z+0(FP), R10
	MOVQ x+24(FP), R8
	MOVQ s+48(FP), CX
	MOVQ (R8), AX		// w1 = x[0]
	MOVQ $0, DX
	SHRQ CX, AX, DX		// w1<<ŝ
	MOVQ DX, c+56(FP)

	MOVQ $0, BX		// i = 0
	JMP E9

	// i < n-1
L9:	MOVQ AX, DX		// w = w1
	MOVQ 8(R8)(BX*8), AX	// w1 = x[i+1]
	SHRQ CX, AX, DX		// w>>s | w1<<ŝ
	MOVQ DX, (R10)(BX*8)	// z[i] = w>>s | w1<<ŝ
	ADDQ $1, BX		// i++

E9:	CMPQ BX, R11
	JL L9			// i < n-1

	// i >= n-1
X9a:	SHRQ CX, AX		// w1>>s
	MOVQ AX, (R10)(R11*8)	// z[n-1] = w1>>s
	RET

X9b:	MOVQ $0, c+56(FP)
	RET


// func mulAddVWW(z, x []Word, m, a Word) (c Word)
TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOVQ z+0(FP), R10
	MOVQ x+24(FP), R8
	MOVQ m+48(FP), R9
	MOVQ a+56(FP), CX	// c = a
	MOVQ z_len+8(FP), R11
	MOVQ $0, BX		// i = 0

	CMPQ R11, $4
	JL E5

U5:	// i+4 <= n
	// regular loop body unrolled 4x
	MOVQ (0*8)(R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (0*8)(R10)(BX*8)
	MOVQ DX, CX
	MOVQ (1*8)(R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (1*8)(R10)(BX*8)
	MOVQ DX, CX
	MOVQ (2*8)(R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (2*8)(R10)(BX*8)
	MOVQ DX, CX
	MOVQ (3*8)(R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (3*8)(R10)(BX*8)
	MOVQ DX, CX
	ADDQ $4, BX		// i += 4

	LEAQ 4(BX), DX
	CMPQ DX, R11
	JLE U5
	JMP E5

L5:	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ AX, (R10)(BX*8)
	MOVQ DX, CX
	ADDQ $1, BX		// i++

E5:	CMPQ BX, R11		// i < n
	JL L5

	MOVQ CX, c+64(FP)
	RET


// func addMulVVWW(z, x, y []Word, m, a Word) (c Word)
TEXT ·addMulVVWW(SB),NOSPLIT,$0
	CMPB ·support_adx(SB), $1
	JEQ adx
	MOVQ z+0(FP), R14
	MOVQ x+24(FP), R10
	MOVQ y+48(FP), R8
	MOVQ m+72(FP), R9
	MOVQ z_len+8(FP), R11
	MOVQ $0, BX		// i = 0
	MOVQ a+80(FP), CX		// c = 0
	MOVQ R11, R12
	ANDQ $-2, R12
	CMPQ R11, $2
	JAE A6
	JMP E6

A6:
	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ (R10)(BX*8), AX
	ADCQ $0, DX
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ DX, CX
	MOVQ AX, (R14)(BX*8)

	MOVQ (8)(R8)(BX*8), AX
	MULQ R9
	ADDQ (8)(R10)(BX*8), AX
	ADCQ $0, DX
	ADDQ CX, AX
	ADCQ $0, DX
	MOVQ DX, CX
	MOVQ AX, (8)(R14)(BX*8)

	ADDQ $2, BX
	CMPQ BX, R12
	JL A6
	JMP E6

L6:	MOVQ (R8)(BX*8), AX
	MULQ R9
	ADDQ CX, AX
	ADCQ $0, DX
	ADDQ (R10)(BX*8), AX
	MOVQ AX, (R14)(BX*8)
	ADCQ $0, DX
	MOVQ DX, CX
	ADDQ $1, BX		// i++

E6:	CMPQ BX, R11		// i < n
	JL L6

	MOVQ CX, c+88(FP)
	RET

adx:
	MOVQ z_len+8(FP), R11
	MOVQ z+0(FP), R14
	MOVQ x+24(FP), R10
	MOVQ y+48(FP), R8
	MOVQ m+72(FP), DX
	MOVQ $0, BX   // i = 0
	MOVQ $0, CX   // carry
	CMPQ R11, $8
	JAE  adx_loop_header
	CMPQ BX, R11
	JL adx_short
	MOVQ CX, c+88(FP)
	RET

adx_loop_header:
	MOVQ  R11, R13
	ANDQ  $-8, R13
adx_loop:
	XORQ  R9, R9  // unset flags
	MULXQ (R8), SI, DI
	ADCXQ CX,SI
	ADOXQ (R10), SI
	MOVQ  SI,(R10)

	MULXQ 8(R8), AX, CX
	ADCXQ DI, AX
	ADOXQ 8(R10), AX
	MOVQ  AX, 8(R14)

	MULXQ 16(R8), SI, DI
	ADCXQ CX, SI
	ADOXQ 16(R10), SI
	MOVQ  SI, 16(R14)

	MULXQ 24(R8), AX, CX
	ADCXQ DI, AX
	ADOXQ 24(R10), AX
	MOVQ  AX, 24(R14)

	MULXQ 32(R8), SI, DI
	ADCXQ CX, SI
	ADOXQ 32(R10), SI
	MOVQ  SI, 32(R14)

	MULXQ 40(R8), AX, CX
	ADCXQ DI, AX
	ADOXQ 40(R10), AX
	MOVQ  AX, 40(R14)

	MULXQ 48(R8), SI, DI
	ADCXQ CX, SI
	ADOXQ 48(R10), SI
	MOVQ  SI, 48(R14)

	MULXQ 56(R8), AX, CX
	ADCXQ DI, AX
	ADOXQ 56(R10), AX
	MOVQ  AX, 56(R14)

	ADCXQ R9, CX
	ADOXQ R9, CX

	ADDQ $64, R8
	ADDQ $64, R10
	ADDQ $64, R14
	ADDQ $8, BX

	CMPQ BX, R13
	JL adx_loop
	MOVQ z+0(FP), R14
	MOVQ x+24(FP), R10
	MOVQ y+48(FP), R8
	CMPQ BX, R11
	JL adx_short
	MOVQ CX, c+88(FP)
	RET

adx_short:
	MULXQ (R8)(BX*8), SI, DI
	ADDQ CX, SI
	ADCQ $0, DI
	ADDQ SI, (R10)(BX*8)
	ADCQ $0, DI
	MOVQ DI, CX
	ADDQ $1, BX		// i++

	CMPQ BX, R11
	JL adx_short

	MOVQ CX, c+88(FP)
	RET



