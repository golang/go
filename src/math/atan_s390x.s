// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial coefficients and other constants
DATA ·atanrodataL8<> + 0(SB)/8, $0.199999999999554423E+00
DATA ·atanrodataL8<> + 8(SB)/8, $0.111111110136634272E+00
DATA ·atanrodataL8<> + 16(SB)/8, $-.142857142828026806E+00
DATA ·atanrodataL8<> + 24(SB)/8, $-.333333333333330928E+00
DATA ·atanrodataL8<> + 32(SB)/8, $0.769228118888682505E-01
DATA ·atanrodataL8<> + 40(SB)/8, $0.588059263575587687E-01
DATA ·atanrodataL8<> + 48(SB)/8, $-.666641501287528609E-01
DATA ·atanrodataL8<> + 56(SB)/8, $-.909090711945939878E-01
DATA ·atanrodataL8<> + 64(SB)/8, $0.472329433805024762E-01
DATA ·atanrodataL8<> + 72(SB)/8, $0.366935664549587481E-01
DATA ·atanrodataL8<> + 80(SB)/8, $-.422172007412067035E-01
DATA ·atanrodataL8<> + 88(SB)/8, $-.299856214685512712E-01
DATA ·atanrodataL8<> + 96(SB)/8, $0.220852012160300086E-01
DATA ·atanrodataL8<> + 104(SB)/8, $0.726338160757602439E-02
DATA ·atanrodataL8<> + 112(SB)/8, $0.843488472994227321E-03
DATA ·atanrodataL8<> + 120(SB)/8, $0.134893651284712515E-04
DATA ·atanrodataL8<> + 128(SB)/8, $-.525380587584426406E-01
DATA ·atanrodataL8<> + 136(SB)/8, $-.139950258898989925E-01
DATA ·atanrodataL8<> + 144(SB)/8, $-.291935324869629616E-02
DATA ·atanrodataL8<> + 152(SB)/8, $-.154797890856877418E-03
GLOBL ·atanrodataL8<> + 0(SB), RODATA, $160

DATA ·atanxpi2h<> + 0(SB)/8, $0x3ff330e4e4fa7b1b
DATA ·atanxpi2h<> + 8(SB)/8, $0xbff330e4e4fa7b1b
DATA ·atanxpi2h<> + 16(SB)/8, $0x400330e4e4fa7b1b
DATA ·atanxpi2h<> + 24(SB)/4, $0xc00330e4e4fa7b1b
GLOBL ·atanxpi2h<> + 0(SB), RODATA, $32
DATA ·atanxpim<> + 0(SB)/8, $0x3ff4f42b00000000
GLOBL ·atanxpim<> + 0(SB), RODATA, $8
DATA ·atanxmone<> + 0(SB)/8, $-1.0
GLOBL ·atanxmone<> + 0(SB), RODATA, $8

// Atan returns the arctangent, in radians, of the argument.
//
// Special cases are:
//      Atan(±0) = ±0
//      Atan(±Inf) = ±Pi/2Pi
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT	·atanAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	//special case Atan(±0) = ±0
	FMOVD   $(0.0), F1
	FCMPU   F0, F1
	BEQ     atanIsZero

	MOVD	$·atanrodataL8<>+0(SB), R5
	MOVH	$0x3FE0, R3
	WORD	$0xB3CD0010	//lgdr	%r1,%f0
	WORD	$0xEC1120BF	//risbgn	%r1,%r1,64-32,128+63,64+0+32
	BYTE	$0x60
	BYTE	$0x59
	RLL	$16, R1, R2
	ANDW	$0x7FF0, R2
	MOVW	R2, R6
	MOVW	R3, R7
	CMPUBLE	R6, R7, L6
	MOVD	$·atanxmone<>+0(SB), R3
	FMOVD	0(R3), F2
	WFDDB	V0, V2, V0
	WORD	$0xEC113FBF	//risbg	%r1,%r1,64-1,128+63,64+32+1
	BYTE	$0x61
	BYTE	$0x55
	MOVD	$·atanxpi2h<>+0(SB), R3
	MOVWZ	R1, R1
	SLD	$3, R1, R1
	WORD	$0x68813000	//ld	%f8,0(%r1,%r3)
L6:
	WFMDB	V0, V0, V2
	FMOVD	152(R5), F6
	FMOVD	144(R5), F1
	FMOVD	136(R5), F7
	VLEG	$0, 128(R5), V16
	FMOVD	120(R5), F4
	FMOVD	112(R5), F5
	WFMADB	V2, V4, V6, V4
	WFMADB	V2, V5, V1, V5
	WFMDB	V2, V2, V6
	FMOVD	104(R5), F3
	FMOVD	96(R5), F1
	WFMADB	V2, V3, V7, V3
	MOVH	$0x3FE0, R1
	FMOVD	88(R5), F7
	WFMADB	V2, V1, V7, V1
	FMOVD	80(R5), F7
	WFMADB	V6, V3, V1, V3
	WFMADB	V6, V4, V5, V4
	WFMDB	V6, V6, V1
	FMOVD	72(R5), F5
	WFMADB	V2, V5, V7, V5
	FMOVD	64(R5), F7
	WFMADB	V2, V7, V16, V7
	VLEG	$0, 56(R5), V16
	WFMADB	V6, V5, V7, V5
	WFMADB	V1, V4, V3, V4
	FMOVD	48(R5), F7
	FMOVD	40(R5), F3
	WFMADB	V2, V3, V7, V3
	FMOVD	32(R5), F7
	WFMADB	V2, V7, V16, V7
	VLEG	$0, 24(R5), V16
	WFMADB	V1, V4, V5, V4
	FMOVD	16(R5), F5
	WFMADB	V6, V3, V7, V3
	FMOVD	8(R5), F7
	WFMADB	V2, V7, V5, V7
	FMOVD	0(R5), F5
	WFMADB	V2, V5, V16, V5
	WFMADB	V1, V4, V3, V4
	WFMADB	V6, V7, V5, V6
	FMUL	F0, F2
	FMADD	F4, F1, F6
	FMADD	F6, F2, F0
	MOVW	R2, R6
	MOVW	R1, R7
	CMPUBLE	R6, R7, L1
	MOVD	$·atanxpim<>+0(SB), R1
	WORD	$0xED801000	//madb	%f0,%f8,0(%r1)
	BYTE	$0x00
	BYTE	$0x1E
L1:
atanIsZero:
	FMOVD	F0, ret+8(FP)
	RET
