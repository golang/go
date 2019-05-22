// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#define PosInf		0x7FF0000000000000
#define NegInf		0xFFF0000000000000
#define NegZero		0x8000000000000000
#define Pi		0x400921FB54442D18
#define NegPi		0xC00921FB54442D18
#define Pi3Div4		0x4002D97C7F3321D2	// 3Pi/4
#define NegPi3Div4	0xC002D97C7F3321D2	// -3Pi/4
#define PiDiv4		0x3FE921FB54442D18	// Pi/4
#define NegPiDiv4	0xBFE921FB54442D18	// -Pi/4

// Minimax polynomial coefficients and other constants
DATA ·atan2rodataL25<> + 0(SB)/8, $0.199999999999554423E+00
DATA ·atan2rodataL25<> + 8(SB)/8, $-.333333333333330928E+00
DATA ·atan2rodataL25<> + 16(SB)/8, $0.111111110136634272E+00
DATA ·atan2rodataL25<> + 24(SB)/8, $-.142857142828026806E+00
DATA ·atan2rodataL25<> + 32(SB)/8, $0.769228118888682505E-01
DATA ·atan2rodataL25<> + 40(SB)/8, $0.588059263575587687E-01
DATA ·atan2rodataL25<> + 48(SB)/8, $-.909090711945939878E-01
DATA ·atan2rodataL25<> + 56(SB)/8, $-.666641501287528609E-01
DATA ·atan2rodataL25<> + 64(SB)/8, $0.472329433805024762E-01
DATA ·atan2rodataL25<> + 72(SB)/8, $-.525380587584426406E-01
DATA ·atan2rodataL25<> + 80(SB)/8, $-.422172007412067035E-01
DATA ·atan2rodataL25<> + 88(SB)/8, $0.366935664549587481E-01
DATA ·atan2rodataL25<> + 96(SB)/8, $0.220852012160300086E-01
DATA ·atan2rodataL25<> + 104(SB)/8, $-.299856214685512712E-01
DATA ·atan2rodataL25<> + 112(SB)/8, $0.726338160757602439E-02
DATA ·atan2rodataL25<> + 120(SB)/8, $0.134893651284712515E-04
DATA ·atan2rodataL25<> + 128(SB)/8, $-.291935324869629616E-02
DATA ·atan2rodataL25<> + 136(SB)/8, $-.154797890856877418E-03
DATA ·atan2rodataL25<> + 144(SB)/8, $0.843488472994227321E-03
DATA ·atan2rodataL25<> + 152(SB)/8, $-.139950258898989925E-01
GLOBL ·atan2rodataL25<> + 0(SB), RODATA, $160

DATA ·atan2xpi2h<> + 0(SB)/8, $0x3ff330e4e4fa7b1b
DATA ·atan2xpi2h<> + 8(SB)/8, $0xbff330e4e4fa7b1b
DATA ·atan2xpi2h<> + 16(SB)/8, $0x400330e4e4fa7b1b
DATA ·atan2xpi2h<> + 24(SB)/8, $0xc00330e4e4fa7b1b
GLOBL ·atan2xpi2h<> + 0(SB), RODATA, $32
DATA ·atan2xpim<> + 0(SB)/8, $0x3ff4f42b00000000
GLOBL ·atan2xpim<> + 0(SB), RODATA, $8

// Atan2 returns the arc tangent of y/x, using
// the signs of the two to determine the quadrant
// of the return value.
//
// Special cases are (in order):
//      Atan2(y, NaN) = NaN
//      Atan2(NaN, x) = NaN
//      Atan2(+0, x>=0) = +0
//      Atan2(-0, x>=0) = -0
//      Atan2(+0, x<=-0) = +Pi
//      Atan2(-0, x<=-0) = -Pi
//      Atan2(y>0, 0) = +Pi/2
//      Atan2(y<0, 0) = -Pi/2
//      Atan2(+Inf, +Inf) = +Pi/4
//      Atan2(-Inf, +Inf) = -Pi/4
//      Atan2(+Inf, -Inf) = 3Pi/4
//      Atan2(-Inf, -Inf) = -3Pi/4
//      Atan2(y, +Inf) = 0
//      Atan2(y>0, -Inf) = +Pi
//      Atan2(y<0, -Inf) = -Pi
//      Atan2(+Inf, x) = +Pi/2
//      Atan2(-Inf, x) = -Pi/2
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT	·atan2Asm(SB), NOSPLIT, $0-24
	// special case
	MOVD	x+0(FP), R1
	MOVD	y+8(FP), R2

	// special case Atan2(NaN, y) = NaN
	MOVD	$~(1<<63), R5
	AND	R1, R5		// x = |x|
	MOVD	$PosInf, R3
	CMPUBLT	R3, R5, returnX

	// special case Atan2(x, NaN) = NaN
	MOVD	$~(1<<63), R5
	AND	R2, R5
	CMPUBLT R3, R5, returnY

	MOVD	$NegZero, R3
	CMPUBEQ	R3, R1, xIsNegZero

	MOVD	$0, R3
	CMPUBEQ	R3, R1, xIsPosZero

	MOVD	$PosInf, R4
	CMPUBEQ	R4, R2, yIsPosInf

	MOVD	$NegInf, R4
	CMPUBEQ	R4, R2, yIsNegInf
	BR	Normal
xIsNegZero:
	// special case Atan(-0, y>=0) = -0
	MOVD	$0, R4
	CMPBLE	R4, R2, returnX

	//special case Atan2(-0, y<=-0) = -Pi
	MOVD	$NegZero, R4
	CMPBGE	R4, R2, returnNegPi
	BR	Normal
xIsPosZero:
	//special case Atan2(0, 0) = 0
	MOVD	$0, R4
	CMPUBEQ	R4, R2, returnX

	//special case Atan2(0, y<=-0) = Pi
	MOVD	$NegZero, R4
	CMPBGE	R4, R2, returnPi
	BR Normal
yIsNegInf:
	//special case Atan2(+Inf, -Inf) = 3Pi/4
	MOVD	$PosInf, R3
	CMPUBEQ	R3, R1, posInfNegInf

	//special case Atan2(-Inf, -Inf) = -3Pi/4
	MOVD	$NegInf, R3
	CMPUBEQ	R3, R1, negInfNegInf
	BR Normal
yIsPosInf:
	//special case Atan2(+Inf, +Inf) = Pi/4
	MOVD	$PosInf, R3
	CMPUBEQ	R3, R1, posInfPosInf

	//special case Atan2(-Inf, +Inf) = -Pi/4
	MOVD	$NegInf, R3
	CMPUBEQ	R3, R1, negInfPosInf

	//special case Atan2(-Pi, +Inf) = Pi
	MOVD	$NegPi, R3
	CMPUBEQ	R3, R1, negPiPosInf

Normal:
	FMOVD	x+0(FP), F0
	FMOVD	y+8(FP), F2
	MOVD	$·atan2rodataL25<>+0(SB), R9
	LGDR	F0, R2
	LGDR	F2, R1
	RISBGNZ	$32, $63, $32, R2, R2
	RISBGNZ	$32, $63, $32, R1, R1
	WORD	$0xB9170032	//llgtr	%r3,%r2
	RISBGZ	$63, $63, $33, R2, R5
	WORD	$0xB9170041	//llgtr	%r4,%r1
	WFLCDB	V0, V20
	MOVW	R4, R6
	MOVW	R3, R7
	CMPUBLT	R6, R7, L17
	WFDDB	V2, V0, V3
	ADDW	$2, R5, R2
	MOVW	R4, R6
	MOVW	R3, R7
	CMPUBLE	R6, R7, L20
L3:
	WFMDB	V3, V3, V4
	VLEG	$0, 152(R9), V18
	VLEG	$0, 144(R9), V16
	FMOVD	136(R9), F1
	FMOVD	128(R9), F5
	FMOVD	120(R9), F6
	WFMADB	V4, V16, V5, V16
	WFMADB	V4, V6, V1, V6
	FMOVD	112(R9), F7
	WFMDB	V4, V4, V1
	WFMADB	V4, V7, V18, V7
	VLEG	$0, 104(R9), V18
	WFMADB	V1, V6, V16, V6
	CMPWU	R4, R3
	FMOVD	96(R9), F5
	VLEG	$0, 88(R9), V16
	WFMADB	V4, V5, V18, V5
	VLEG	$0, 80(R9), V18
	VLEG	$0, 72(R9), V22
	WFMADB	V4, V16, V18, V16
	VLEG	$0, 64(R9), V18
	WFMADB	V1, V7, V5, V7
	WFMADB	V4, V18, V22, V18
	WFMDB	V1, V1, V5
	WFMADB	V1, V16, V18, V16
	VLEG	$0, 56(R9), V18
	WFMADB	V5, V6, V7, V6
	VLEG	$0, 48(R9), V22
	FMOVD	40(R9), F7
	WFMADB	V4, V7, V18, V7
	VLEG	$0, 32(R9), V18
	WFMADB	V5, V6, V16, V6
	WFMADB	V4, V18, V22, V18
	VLEG	$0, 24(R9), V16
	WFMADB	V1, V7, V18, V7
	VLEG	$0, 16(R9), V18
	VLEG	$0, 8(R9), V22
	WFMADB	V4, V18, V16, V18
	VLEG	$0, 0(R9), V16
	WFMADB	V5, V6, V7, V6
	WFMADB	V4, V16, V22, V16
	FMUL	F3, F4
	WFMADB	V1, V18, V16, V1
	FMADD	F6, F5, F1
	WFMADB	V4, V1, V3, V4
	BLT	L18
	BGT	L7
	LTDBR	F2, F2
	BLTU	L21
L8:
	LTDBR	F0, F0
	BLTU	L22
L9:
	WFCHDBS	V2, V0, V0
	BNE	L18
L7:
	MOVW	R1, R6
	CMPBGE	R6, $0, L1
L18:
	RISBGZ	$58, $60, $3, R2, R2
	MOVD	$·atan2xpi2h<>+0(SB), R1
	MOVD	·atan2xpim<>+0(SB), R3
	LDGR	R3, F0
	WORD	$0xED021000	//madb	%f4,%f0,0(%r2,%r1)
	BYTE	$0x40
	BYTE	$0x1E
L1:
	FMOVD	F4, ret+16(FP)
	RET

L20:
	LTDBR	F2, F2
	BLTU	L23
	FMOVD	F2, F6
L4:
	LTDBR	F0, F0
	BLTU	L24
	FMOVD	F0, F4
L5:
	WFCHDBS	V6, V4, V4
	BEQ	L3
L17:
	WFDDB	V0, V2, V4
	BYTE	$0x18	//lr	%r2,%r5
	BYTE	$0x25
	WORD	$0xB3130034	//lcdbr	%f3,%f4
	BR	L3
L23:
	WORD	$0xB3130062	//lcdbr	%f6,%f2
	BR	L4
L22:
	VLR	V20, V0
	BR	L9
L21:
	WORD	$0xB3130022	//lcdbr	%f2,%f2
	BR	L8
L24:
	VLR	V20, V4
	BR	L5
returnX:	//the result is same as the first argument
	MOVD	R1, ret+16(FP)
	RET
returnY:	//the result is same as the second argument
	MOVD	R2, ret+16(FP)
	RET
returnPi:
	MOVD	$Pi, R1
	MOVD	R1, ret+16(FP)
	RET
returnNegPi:
	MOVD	$NegPi, R1
	MOVD	R1, ret+16(FP)
	RET
posInfNegInf:
	MOVD	$Pi3Div4, R1
	MOVD	R1, ret+16(FP)
	RET
negInfNegInf:
	MOVD	$NegPi3Div4, R1
	MOVD	R1, ret+16(FP)
	RET
posInfPosInf:
	MOVD	$PiDiv4, R1
	MOVD	R1, ret+16(FP)
	RET
negInfPosInf:
	MOVD	$NegPiDiv4, R1
	MOVD	R1, ret+16(FP)
	RET
negPiPosInf:
	MOVD	$NegZero, R1
	MOVD	R1, ret+16(FP)
	RET
