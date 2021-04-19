// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !math_big_pure_go,ppc64 !math_big_pure_go,ppc64le

#include "textflag.h"

// This file provides fast assembly versions for the elementary
// arithmetic operations on vectors implemented in arith.go.

// func mulWW(x, y Word) (z1, z0 Word)
TEXT ·mulWW(SB), NOSPLIT, $0
	MOVD   x+0(FP), R4
	MOVD   y+8(FP), R5
	MULHDU R4, R5, R6
	MULLD  R4, R5, R7
	MOVD   R6, z1+16(FP)
	MOVD   R7, z0+24(FP)
	RET

TEXT ·addVV(SB), NOSPLIT, $0
	BR ·addVV_g(SB)

// func subVV(z, x, y []Word) (c Word)
// z[i] = x[i] - y[i] for all i, carrying
TEXT ·subVV(SB), NOSPLIT, $0
	MOVD z_len+8(FP), R7
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z+0(FP), R10

	MOVD $0, R4  // c = 0
	MOVD $0, R5  // i = 0
	MOVD $1, R29 // work around lack of ADDI
	MOVD $8, R28 // work around lack of scaled addressing

	SUBC R0, R0  // clear CA
	JMP  sublend

// amd64 saves and restores CF, but I believe they only have to do that because all of
// their math operations clobber it - we should just be able to recover it at the end.
subloop:
	MULLD R5, R28, R6
	MOVD  (R8)(R6), R11 // x[i]
	MOVD  (R9)(R6), R12 // y[i]

	SUBE R12, R11, R15
	MOVD R15, (R10)(R6)

	ADD R29, R5 // i++

sublend:
	CMP R5, R7
	BLT subloop

	ADDZE R4
	XOR   R29, R4
	MOVD  R4, c+72(FP)
	RET

TEXT ·addVW(SB), NOSPLIT, $0
	BR ·addVW_g(SB)

TEXT ·subVW(SB), NOSPLIT, $0
	BR ·subVW_g(SB)

TEXT ·shlVU(SB), NOSPLIT, $0
	BR ·shlVU_g(SB)

TEXT ·shrVU(SB), NOSPLIT, $0
	BR ·shrVU_g(SB)

// func mulAddVWW(z, x []Word, y, r Word) (c Word)
TEXT ·mulAddVWW(SB), NOSPLIT, $0
	MOVD z+0(FP), R10
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD r+56(FP), R4     // c = r
	MOVD z_len+8(FP), R11
	MOVD $0, R3           // i = 0
	MOVD $8, R18
	MOVD $1, R19

	JMP e5

l5:
	MULLD  R18, R3, R5
	MOVD   (R8)(R5), R20
	MULLD  R9, R20, R6
	MULHDU R9, R20, R7
	ADDC   R4, R6
	ADDZE  R7
	MOVD   R6, (R10)(R5)
	MOVD   R7, R4
	ADD    R19, R3

e5:
	CMP R3, R11
	BLT l5

	MOVD R4, c+64(FP)
	RET

// func addMulVVW(z, x []Word, y Word) (c Word)
TEXT ·addMulVVW(SB), NOSPLIT, $0
	MOVD z+0(FP), R10
	MOVD x+24(FP), R8
	MOVD y+48(FP), R9
	MOVD z_len+8(FP), R22

	MOVD $0, R5   // i = 0
	MOVD $0, R4   // c = 0
	MOVD $8, R28
	MOVD $-2, R23
	AND  R22, R23 // mask the last bit of z.len
	MOVD $2, R24
	CMP  R23, R24
	BGE  unrolled
	JMP  end

unrolled:
	MOVD  $8, R19         // no (RA)(RB*8) on power
	MULLD R5, R19
	MOVD  (R10)(R19), R11 // R11 = z[i]
	MOVD  (R8)(R19), R16  // R16 = x[i]
	ADD   R28, R19, R25
	MOVD  (R10)(R25), R17
	MOVD  (R8)(R25), R18

	MULLD  R9, R16, R12
	MULHDU R9, R16, R14
	MULLD  R9, R18, R6
	MULHDU R9, R18, R7
	ADDC   R4, R12
	ADDZE  R14
	ADDC   R11, R12        // z[i] = (x[i]*y) + z[i] + carry
	ADDZE  R14             // carry = high order bits + add carry
	MOVD   R12, (R10)(R19)
	ADDC   R14, R6
	ADDZE  R7
	ADDC   R17, R6
	ADDZE  R7
	MOVD   R6, (R10)(R25)
	MOVD   R7, R4

	ADD R24, R5
	CMP R5, R23
	BLT unrolled
	JMP end

loop:
	MOVD   $8, R19
	MULLD  R5, R19
	MOVD   (R10)(R19), R11
	MOVD   (R8)(R19), R16
	MULLD  R9, R16, R12
	MULHDU R9, R16, R14
	ADDC   R4, R12
	ADDZE  R14
	ADDC   R11, R12
	ADDZE  R14
	MOVD   R12, (R10)(R19)
	MOVD   R14, R4

	MOVD $1, R15
	ADD  R15, R5

end:
	CMP R5, R22
	BLT loop

	MOVD R4, c+56(FP)
	RET

TEXT ·divWVW(SB), NOSPLIT, $0
	BR ·divWVW_g(SB)

// func bitLen(x Word) int
TEXT ·bitLen(SB), NOSPLIT, $0
	MOVD   x+0(FP), R4
	CNTLZD R4, R4
	MOVD   $64, R5
	SUB    R4, R5
	MOVD   R5, n+8(FP)
	RET
