// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial approximations
DATA ·tanrodataL13<> + 0(SB)/8, $0.181017336383229927e-07
DATA ·tanrodataL13<> + 8(SB)/8, $-.256590857271311164e-03
DATA ·tanrodataL13<> + 16(SB)/8, $-.464359274328689195e+00
DATA ·tanrodataL13<> + 24(SB)/8, $1.0
DATA ·tanrodataL13<> + 32(SB)/8, $-.333333333333333464e+00
DATA ·tanrodataL13<> + 40(SB)/8, $0.245751217306830032e-01
DATA ·tanrodataL13<> + 48(SB)/8, $-.245391301343844510e-03
DATA ·tanrodataL13<> + 56(SB)/8, $0.214530914428992319e-01
DATA ·tanrodataL13<> + 64(SB)/8, $0.108285667160535624e-31
DATA ·tanrodataL13<> + 72(SB)/8, $0.612323399573676480e-16
DATA ·tanrodataL13<> + 80(SB)/8, $0.157079632679489656e+01
DATA ·tanrodataL13<> + 88(SB)/8, $0.636619772367581341e+00
GLOBL ·tanrodataL13<> + 0(SB), RODATA, $96

// Constants
DATA ·tanxnan<> + 0(SB)/8, $0x7ff8000000000000
GLOBL ·tanxnan<> + 0(SB), RODATA, $8
DATA ·tanxlim<> + 0(SB)/8, $0x432921fb54442d19
GLOBL ·tanxlim<> + 0(SB), RODATA, $8
DATA ·tanxadd<> + 0(SB)/8, $0xc338000000000000
GLOBL ·tanxadd<> + 0(SB), RODATA, $8

// Tan returns the tangent of the radian argument.
//
// Special cases are:
//      Tan(±0) = ±0
//      Tan(±Inf) = NaN
//      Tan(NaN) = NaN
// The algorithm used is minimax polynomial approximation using a table of
// polynomial coefficients determined with a Remez exchange algorithm.

TEXT	·tanAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	//specail case Tan(±0) = ±0
	FMOVD   $(0.0), F1
	FCMPU   F0, F1
	BEQ     atanIsZero

	MOVD	$·tanrodataL13<>+0(SB), R5
	LTDBR	F0, F0
	BLTU	L10
	FMOVD	F0, F2
L2:
	MOVD	$·tanxlim<>+0(SB), R1
	WORD	$0xED201000	//cdb	%f2,0(%r1)
	BYTE	$0x00
	BYTE	$0x19
	BGE	L11
	BVS	L11
	MOVD	$·tanxadd<>+0(SB), R1
	FMOVD	88(R5), F6
	FMOVD	0(R1), F4
	WFMSDB	V0, V6, V4, V6
	FMOVD	80(R5), F1
	FADD	F6, F4
	FMOVD	72(R5), F2
	FMSUB	F1, F4, F0
	FMOVD	64(R5), F3
	WFMADB	V4, V2, V0, V2
	FMOVD	56(R5), F1
	WFMADB	V4, V3, V2, V4
	FMUL	F2, F2
	VLEG	$0, 48(R5), V18
	WORD	$0xB3CD0016	//lgdr	%r1,%f6
	FMOVD	40(R5), F5
	FMOVD	32(R5), F3
	FMADD	F1, F2, F3
	FMOVD	24(R5), F1
	FMOVD	16(R5), F7
	FMOVD	8(R5), F0
	WFMADB	V2, V7, V1, V7
	WFMADB	V2, V0, V5, V0
	WFMDB	V2, V2, V1
	FMOVD	0(R5), F5
	WFLCDB	V4, V16
	WFMADB	V2, V5, V18, V5
	WFMADB	V1, V0, V7, V0
	WORD	$0xA7110001	//tmll	%r1,1
	WFMADB	V1, V5, V3, V1
	BNE	L12
	WFDDB	V0, V1, V0
	WFMDB	V2, V16, V2
	WFMADB	V2, V0, V4, V0
	WORD	$0xB3130000	//lcdbr	%f0,%f0
	FMOVD	F0, ret+8(FP)
	RET
L12:
	WFMSDB	V2, V1, V0, V2
	WFMDB	V16, V2, V2
	FDIV	F2, F0
	FMOVD	F0, ret+8(FP)
	RET
L11:
	MOVD	$·tanxnan<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET
L10:
	WORD	$0xB3130020	//lcdbr	%f2,%f0
	BR	L2
atanIsZero:
	FMOVD	F0, ret+8(FP)
	RET
