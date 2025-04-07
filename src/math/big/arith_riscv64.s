// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go && riscv64

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

TEXT ·addVV(SB),NOSPLIT,$0
	MOV	x+24(FP), X5
	MOV	y+48(FP), X6
	MOV	z+0(FP), X7
	MOV	z_len+8(FP), X30

	MOV	$4, X28
	MOV	$0, X29		// c = 0

	BEQZ	X30, done
	BLTU	X30, X28, loop1

loop4:
	MOV	0(X5), X8	// x[0]
	MOV	0(X6), X9	// y[0]
	MOV	8(X5), X11	// x[1]
	MOV	8(X6), X12	// y[1]
	MOV	16(X5), X14	// x[2]
	MOV	16(X6), X15	// y[2]
	MOV	24(X5), X17	// x[3]
	MOV	24(X6), X18	// y[3]

	ADD	X8, X9, X21	// z[0] = x[0] + y[0]
	SLTU	X8, X21, X22
	ADD	X21, X29, X10	// z[0] = x[0] + y[0] + c
	SLTU	X21, X10, X23
	ADD	X22, X23, X29	// next c

	ADD	X11, X12, X24	// z[1] = x[1] + y[1]
	SLTU	X11, X24, X25
	ADD	X24, X29, X13	// z[1] = x[1] + y[1] + c
	SLTU	X24, X13, X26
	ADD	X25, X26, X29	// next c

	ADD	X14, X15, X21	// z[2] = x[2] + y[2]
	SLTU	X14, X21, X22
	ADD	X21, X29, X16	// z[2] = x[2] + y[2] + c
	SLTU	X21, X16, X23
	ADD	X22, X23, X29	// next c

	ADD	X17, X18, X21	// z[3] = x[3] + y[3]
	SLTU	X17, X21, X22
	ADD	X21, X29, X19	// z[3] = x[3] + y[3] + c
	SLTU	X21, X19, X23
	ADD	X22, X23, X29	// next c

	MOV	X10, 0(X7)	// z[0]
	MOV	X13, 8(X7)	// z[1]
	MOV	X16, 16(X7)	// z[2]
	MOV	X19, 24(X7)	// z[3]

	ADD	$32, X5
	ADD	$32, X6
	ADD	$32, X7
	SUB	$4, X30

	BGEU	X30, X28, loop4
	BEQZ	X30, done

loop1:
	MOV	0(X5), X10	// x
	MOV	0(X6), X11	// y

	ADD	X10, X11, X12	// z = x + y
	SLTU	X10, X12, X14
	ADD	X12, X29, X13	// z = x + y + c
	SLTU	X12, X13, X15
	ADD	X14, X15, X29	// next c

	MOV	X13, 0(X7)	// z

	ADD	$8, X5
	ADD	$8, X6
	ADD	$8, X7
	SUB	$1, X30

	BNEZ	X30, loop1

done:
	MOV	X29, c+72(FP)	// return c
	RET

TEXT ·subVV(SB),NOSPLIT,$0
	MOV	x+24(FP), X5
	MOV	y+48(FP), X6
	MOV	z+0(FP), X7
	MOV	z_len+8(FP), X30

	MOV	$4, X28
	MOV	$0, X29		// b = 0

	BEQZ	X30, done
	BLTU	X30, X28, loop1

loop4:
	MOV	0(X5), X8	// x[0]
	MOV	0(X6), X9	// y[0]
	MOV	8(X5), X11	// x[1]
	MOV	8(X6), X12	// y[1]
	MOV	16(X5), X14	// x[2]
	MOV	16(X6), X15	// y[2]
	MOV	24(X5), X17	// x[3]
	MOV	24(X6), X18	// y[3]

	SUB	X9, X8, X21	// z[0] = x[0] - y[0]
	SLTU	X21, X8, X22
	SUB	X29, X21, X10	// z[0] = x[0] - y[0] - b
	SLTU	X10, X21, X23
	ADD	X22, X23, X29	// next b

	SUB	X12, X11, X24	// z[1] = x[1] - y[1]
	SLTU	X24, X11, X25
	SUB	X29, X24, X13	// z[1] = x[1] - y[1] - b
	SLTU	X13, X24, X26
	ADD	X25, X26, X29	// next b

	SUB	X15, X14, X21	// z[2] = x[2] - y[2]
	SLTU	X21, X14, X22
	SUB	X29, X21, X16	// z[2] = x[2] - y[2] - b
	SLTU	X16, X21, X23
	ADD	X22, X23, X29	// next b

	SUB	X18, X17, X21	// z[3] = x[3] - y[3]
	SLTU	X21, X17, X22
	SUB	X29, X21, X19	// z[3] = x[3] - y[3] - b
	SLTU	X19, X21, X23
	ADD	X22, X23, X29	// next b

	MOV	X10, 0(X7)	// z[0]
	MOV	X13, 8(X7)	// z[1]
	MOV	X16, 16(X7)	// z[2]
	MOV	X19, 24(X7)	// z[3]

	ADD	$32, X5
	ADD	$32, X6
	ADD	$32, X7
	SUB	$4, X30

	BGEU	X30, X28, loop4
	BEQZ	X30, done

loop1:
	MOV	0(X5), X10	// x
	MOV	0(X6), X11	// y

	SUB	X11, X10, X12	// z = x - y
	SLTU	X12, X10, X14
	SUB	X29, X12, X13	// z = x - y - b
	SLTU	X13, X12, X15
	ADD	X14, X15, X29	// next b

	MOV	X13, 0(X7)	// z

	ADD	$8, X5
	ADD	$8, X6
	ADD	$8, X7
	SUB	$1, X30

	BNEZ	X30, loop1

done:
	MOV	X29, c+72(FP)	// return b
	RET

TEXT ·lshVU(SB),NOSPLIT,$0
	JMP ·lshVU_g(SB)

TEXT ·rshVU(SB),NOSPLIT,$0
	JMP ·rshVU_g(SB)

TEXT ·mulAddVWW(SB),NOSPLIT,$0
	MOV	x+24(FP), X5
	MOV	m+48(FP), X6
	MOV	z+0(FP), X7
	MOV	z_len+8(FP), X30
	MOV	a+56(FP), X29

	MOV	$4, X28

	BEQ	ZERO, X30, done
	BLTU	X30, X28, loop1

loop4:
	MOV	0(X5), X8	// x[0]
	MOV	8(X5), X11	// x[1]
	MOV	16(X5), X14	// x[2]
	MOV	24(X5), X17	// x[3]

	MULHU	X8, X6, X9	// z_hi[0] = x[0] * m
	MUL	X8, X6, X8	// z_lo[0] = x[0] * m
	ADD	X8, X29, X10	// z[0] = z_lo[0] + c
	SLTU	X8, X10, X23
	ADD	X23, X9, X29	// next c

	MULHU	X11, X6, X12	// z_hi[1] = x[1] * m
	MUL	X11, X6, X11	// z_lo[1] = x[1] * m
	ADD	X11, X29, X13	// z[1] = z_lo[1] + c
	SLTU	X11, X13, X23
	ADD	X23, X12, X29	// next c

	MULHU	X14, X6, X15	// z_hi[2] = x[2] * m
	MUL	X14, X6, X14	// z_lo[2] = x[2] * m
	ADD	X14, X29, X16	// z[2] = z_lo[2] + c
	SLTU	X14, X16, X23
	ADD	X23, X15, X29	// next c

	MULHU	X17, X6, X18	// z_hi[3] = x[3] * m
	MUL	X17, X6, X17	// z_lo[3] = x[3] * m
	ADD	X17, X29, X19	// z[3] = z_lo[3] + c
	SLTU	X17, X19, X23
	ADD	X23, X18, X29	// next c

	MOV	X10, 0(X7)	// z[0]
	MOV	X13, 8(X7)	// z[1]
	MOV	X16, 16(X7)	// z[2]
	MOV	X19, 24(X7)	// z[3]

	ADD	$32, X5
	ADD	$32, X7
	SUB	$4, X30

	BGEU	X30, X28, loop4
	BEQZ	X30, done

loop1:
	MOV	0(X5), X10	// x

	MULHU	X10, X6, X12	// z_hi = x * m
	MUL	X10, X6, X10	// z_lo = x * m
	ADD	X10, X29, X13	// z_lo + c
	SLTU	X10, X13, X15
	ADD	X12, X15, X29	// next c

	MOV	X13, 0(X7)	// z

	ADD	$8, X5
	ADD	$8, X7
	SUB	$1, X30

	BNEZ	X30, loop1

done:
	MOV	X29, c+64(FP)	// return c
	RET

TEXT ·addMulVVWW(SB),NOSPLIT,$0
	MOV	y+48(FP), X5
	MOV	m+72(FP), X6
	MOV	x+24(FP), X7
	MOV z+0(FP), X20
	MOV	z_len+8(FP), X30

	MOV	$4, X28
	MOV	a+80(FP), X29		// c = a

	BEQZ	X30, done
	BLTU	X30, X28, loop1

loop4:
	MOV	0(X5), X8	// y[0]
	MOV	0(X7), X10	// x[0]
	MOV	8(X5), X11	// y[1]
	MOV	8(X7), X13	// x[1]
	MOV	16(X5), X14	// y[2]
	MOV	16(X7), X16	// x[2]
	MOV	24(X5), X17	// y[3]
	MOV	24(X7), X19	// x[3]

	MULHU	X8, X6, X9	// x_hi[0] = y[0] * m
	MUL	X8, X6, X8	// x_lo[0] = y[0] * m
	ADD	X8, X10, X21	// x_lo[0] = y[0] * m + x[0]
	SLTU	X8, X21, X22
	ADD	X9, X22, X9	// x_hi[0] = y[0] * m + x[0]
	ADD	X21, X29, X10	// x[0] = y[0] * m + x[0] + c
	SLTU	X21, X10, X22
	ADD	X9, X22, X29	// next c

	MULHU	X11, X6, X12	// x_hi[1] = y[1] * m
	MUL	X11, X6, X11	// x_lo[1] = y[1] * m
	ADD	X11, X13, X21	// x_lo[1] = y[1] * m + x[1]
	SLTU	X11, X21, X22
	ADD	X12, X22, X12	// x_hi[1] = y[1] * m + x[1]
	ADD	X21, X29, X13	// x[1] = y[1] * m + x[1] + c
	SLTU	X21, X13, X22
	ADD	X12, X22, X29	// next c

	MULHU	X14, X6, X15	// x_hi[2] = y[2] * m
	MUL	X14, X6, X14	// x_lo[2] = y[2] * m
	ADD	X14, X16, X21	// x_lo[2] = y[2] * m + x[2]
	SLTU	X14, X21, X22
	ADD	X15, X22, X15	// x_hi[2] = y[2] * m + x[2]
	ADD	X21, X29, X16	// x[2] = y[2] * m + x[2] + c
	SLTU	X21, X16, X22
	ADD	X15, X22, X29	// next c

	MULHU	X17, X6, X18	// x_hi[3] = y[3] * m
	MUL	X17, X6, X17	// x_lo[3] = y[3] * m
	ADD	X17, X19, X21	// x_lo[3] = y[3] * m + x[3]
	SLTU	X17, X21, X22
	ADD	X18, X22, X18	// x_hi[3] = y[3] * m + x[3]
	ADD	X21, X29, X19	// x[3] = y[3] * m + x[3] + c
	SLTU	X21, X19, X22
	ADD	X18, X22, X29	// next c

	MOV	X10, 0(X20)	// z[0]
	MOV	X13, 8(X20)	// z[1]
	MOV	X16, 16(X20)	// z[2]
	MOV	X19, 24(X20)	// z[3]

	ADD	$32, X5
	ADD	$32, X7
	ADD	$32, X20
	SUB	$4, X30

	BGEU	X30, X28, loop4
	BEQZ	X30, done

loop1:
	MOV	0(X5), X10	// y
	MOV	0(X7), X11	// x

	MULHU	X10, X6, X12	// z_hi = y * m
	MUL	X10, X6, X10	// z_lo = y * m
	ADD	X10, X11, X13	// z_lo = y * m + x
	SLTU	X10, X13, X15
	ADD	X12, X15, X12	// z_hi = y * m + x
	ADD	X13, X29, X10	// z = y * m + x + c
	SLTU	X13, X10, X15
	ADD	X12, X15, X29	// next c

	MOV	X10, 0(X20)	// z

	ADD	$8, X5
	ADD	$8, X7
	ADD	$8, X20
	SUB	$1, X30

	BNEZ	X30, loop1

done:
	MOV	X29, c+88(FP)	// return c
	RET
