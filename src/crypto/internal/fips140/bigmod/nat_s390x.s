// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

// Register usage (z13 convention):
// R2 = rp (result pointer)
// R3 = ap (source pointer)
// R4 = an / idx (loop counter)
// R5 = b0 (multiplier limb)
// R6 = cy (carry)

#include "textflag.h"

// func addMulVVW1024(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1024(SB), $0-32
	MOVD	$16, R4
	JMP	addMulVVWx(SB)

// func addMulVVW1536(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW1536(SB), $0-32
	MOVD	$24, R4
	JMP	addMulVVWx(SB)

// func addMulVVW2048(z, x *uint, y uint) (c uint)
TEXT ·addMulVVW2048(SB), $0-32
	MOVD	$32, R4
	JMP	addMulVVWx(SB)

TEXT addMulVVWx(SB), NOFRAME|NOSPLIT, $0
	MOVD z+0(FP), R2
	MOVD x+8(FP), R3
	MOVD y+16(FP), R5

	MOVD	$0, R6

L_ent:
	VZERO	V0
	VZERO	V2
	SRD	$2, R4, R10
	TMLL	R4, $1
	BRC     $8, L_bx0

L_bx1:
	VLEG	$1, 0(R2), V2
	VZERO	V4
	TMLL	R4, $2
	BRC     $7, L_b11

L_b01:
	MOVD	$-24, R4
	MOVD	R6, R0
	MOVD	0(R3), R7
	MLGR	R5, R6
	ADDC	R0, R7
	MOVD	$0, R0
	ADDE	R0, R6
	VLVGG	$1, R7, V4
	VAQ	V2, V4, V2
	VSTEG	$1, V2, 0(R2)
	VMRHG	V2, V2, V2
	CMPBEQ	R10, $0, L_1
	BR	L_cj0

L_b11:
	MOVD	$-8, R4
	MOVD	0(R3), R9
	MLGR	R5, R8
	ADDC	R6, R9
	MOVD	$0, R6
	ADDE	R6, R8
	VLVGG	$1, R9, V4
	VAQ	V2, V4, V2
	VSTEG	$1, V2, 0(R2)
	VMRHG	V2, V2, V2
	BR	L_cj1

L_bx0:
	TMLL	R4, $2
	BRC	$7, L_b10

L_b00:
	MOVD	$-32, R4

L_cj0:
	MOVD	32(R3)(R4), R1
	MOVD	40(R3)(R4), R9
	MLGR	R5, R0
	MLGR	R5, R8
	VL	32(R4)(R2), V1
	VPDI	$4, V1, V1, V1
	VLVGP	R0, R1, V6
	VLVGP	R9, R6, V7
	BR	L_mid

L_b10:
	MOVD	$-16, R4
	MOVD	R6, R8

L_cj1:
	MOVD	16(R4)(R3), R1
	MOVD	24(R4)(R3), R7
	MLGR	R5, R0
	MLGR	R5, R6
	VL	16(R4)(R2), V1
	VPDI	$4, V1, V1, V1
	VLVGP	R0, R1, V6
	VLVGP	R7, R8, V7
	CMPBEQ	R10, $0, L_end

L_top:
	MOVD	32(R4)(R3), R1
	MOVD	40(R4)(R3), R9
	MLGR	R5, R0
	MLGR R5, R8
	VACQ	V6, V1, V0, V5
	VACCCQ	V6, V1, V0, V0
	VACQ	V5, V7, V2, V3
	VACCCQ	V5, V7, V2, V2
	VPDI	$4, V3, V3, V3
	VL	32(R4)(R2), V1
	VPDI	$4, V1, V1, V1
	VST	V3, 16(R4)(R2)
	VLVGP	R0, R1, V6
	VLVGP	R9, R6, V7

L_mid:
	MOVD	48(R4)(R3), R1
	MOVD	56(R4)(R3), R7
	MLGR	R5, R0
	MLGR	R5, R6
	VACQ	V6, V1, V0, V5
	VACCCQ	V6, V1, V0, V0
	VACQ	V5, V7, V2, V3
	VACCCQ	V5, V7, V2, V2
	VPDI	$4, V3, V3, V3
	VL	48(R4)(R2), V1
	VPDI	$4, V1, V1, V1
	VST	V3, 32(R4)(R2)
	VLVGP	R0, R1, V6
	VLVGP	R7, R8, V7
	MOVD	$32(R4), R4
	BRCTG	R10, L_top

L_end:
	VACQ	V6, V1, V0, V5
	VACCCQ	V6, V1, V0, V0
	VACQ	V5, V7, V2, V3
	VACCCQ	V5, V7, V2, V2
	VPDI	$4, V3, V3, V3
	VST	V3, 16(R2)(R4)
	VAG	V0, V2, V2

L_1:
	VLGVG	$1, V2, R2
	ADDC	R6, R2
	MOVD	R2, c+24(FP)
	RET

