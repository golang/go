// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial approximations
DATA ·logrodataL21<> + 0(SB)/8, $-.499999999999999778E+00
DATA ·logrodataL21<> + 8(SB)/8, $0.333333333333343751E+00
DATA ·logrodataL21<> + 16(SB)/8, $-.250000000001606881E+00
DATA ·logrodataL21<> + 24(SB)/8, $0.199999999971603032E+00
DATA ·logrodataL21<> + 32(SB)/8, $-.166666663114122038E+00
DATA ·logrodataL21<> + 40(SB)/8, $-.125002923782692399E+00
DATA ·logrodataL21<> + 48(SB)/8, $0.111142014580396256E+00
DATA ·logrodataL21<> + 56(SB)/8, $0.759438932618934220E-01
DATA ·logrodataL21<> + 64(SB)/8, $0.142857144267212549E+00
DATA ·logrodataL21<> + 72(SB)/8, $-.993038938793590759E-01
DATA ·logrodataL21<> + 80(SB)/8, $-1.0
GLOBL ·logrodataL21<> + 0(SB), RODATA, $88

// Constants
DATA ·logxminf<> + 0(SB)/8, $0xfff0000000000000
GLOBL ·logxminf<> + 0(SB), RODATA, $8
DATA ·logxnan<> + 0(SB)/8, $0x7ff8000000000000
GLOBL ·logxnan<> + 0(SB), RODATA, $8
DATA ·logx43f<> + 0(SB)/8, $0x43f0000000000000
GLOBL ·logx43f<> + 0(SB), RODATA, $8
DATA ·logxl2<> + 0(SB)/8, $0x3fda7aecbeba4e46
GLOBL ·logxl2<> + 0(SB), RODATA, $8
DATA ·logxl1<> + 0(SB)/8, $0x3ffacde700000000
GLOBL ·logxl1<> + 0(SB), RODATA, $8

/* Input transform scale and add constants */
DATA ·logxm<> + 0(SB)/8, $0x3fc77604e63c84b1
DATA ·logxm<> + 8(SB)/8, $0x40fb39456ab53250
DATA ·logxm<> + 16(SB)/8, $0x3fc9ee358b945f3f
DATA ·logxm<> + 24(SB)/8, $0x40fb39418bf3b137
DATA ·logxm<> + 32(SB)/8, $0x3fccfb2e1304f4b6
DATA ·logxm<> + 40(SB)/8, $0x40fb393d3eda3022
DATA ·logxm<> + 48(SB)/8, $0x3fd0000000000000
DATA ·logxm<> + 56(SB)/8, $0x40fb393969e70000
DATA ·logxm<> + 64(SB)/8, $0x3fd11117aafbfe04
DATA ·logxm<> + 72(SB)/8, $0x40fb3936eaefafcf
DATA ·logxm<> + 80(SB)/8, $0x3fd2492af5e658b2
DATA ·logxm<> + 88(SB)/8, $0x40fb39343ff01715
DATA ·logxm<> + 96(SB)/8, $0x3fd3b50c622a43dd
DATA ·logxm<> + 104(SB)/8, $0x40fb39315adae2f3
DATA ·logxm<> + 112(SB)/8, $0x3fd56bbeea918777
DATA ·logxm<> + 120(SB)/8, $0x40fb392e21698552
GLOBL ·logxm<> + 0(SB), RODATA, $128

// Log returns the natural logarithm of the argument.
//
// Special cases are:
//      Log(+Inf) = +Inf
//      Log(0) = -Inf
//      Log(x < 0) = NaN
//      Log(NaN) = NaN
// The algorithm used is minimax polynomial approximation using a table of
// polynomial coefficients determined with a Remez exchange algorithm.

TEXT	·logAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·logrodataL21<>+0(SB), R9
	MOVH	$0x8006, R4
	WORD	$0xB3CD0010	//lgdr	%r1,%f0
	MOVD	$0x3FF0000000000000, R6
	SRAD	$48, R1, R1
	MOVD	$0x40F03E8000000000, R8
	SUBW	R1, R4
	WORD	$0xEC2420BB	//risbg	%r2,%r4,32,128+59,0
	BYTE	$0x00
	BYTE	$0x55
	WORD	$0xEC62000F	//risbgn	%r6,%r2,64-64+0,64-64+0+16-1,64-0-16
	BYTE	$0x30
	BYTE	$0x59
	WORD	$0xEC82101F	//risbgn	%r8,%r2,64-64+16,64-64+16+16-1,64-16-16
	BYTE	$0x20
	BYTE	$0x59
	MOVW	R1, R7
	CMPBGT	R7, $22, L17
	WORD	$0xB3120000	//ltdbr	%f0,%f0
	MOVD	$·logx43f<>+0(SB), R1
	FMOVD	0(R1), F2
	BLEU	L3
	MOVH	$0x8005, R12
	MOVH	$0x8405, R0
	BR	L15
L7:
	WORD	$0xB3120000	//ltdbr	%f0,%f0
	BLEU	L3
L15:
	FMUL	F2, F0
	WORD	$0xB3CD0010	//lgdr	%r1,%f0
	SRAD	$48, R1, R1
	SUBW	R1, R0, R2
	SUBW	R1, R12, R3
	BYTE	$0x18	//lr	%r4,%r2
	BYTE	$0x42
	ANDW	$0xFFFFFFF0, R3
	ANDW	$0xFFFFFFF0, R2
	BYTE	$0x18	//lr	%r5,%r1
	BYTE	$0x51
	MOVW	R1, R7
	CMPBLE	R7, $22, L7
	WORD	$0xEC63000F	//risbgn	%r6,%r3,64-64+0,64-64+0+16-1,64-0-16
	BYTE	$0x30
	BYTE	$0x59
	WORD	$0xEC82101F	//risbgn	%r8,%r2,64-64+16,64-64+16+16-1,64-16-16
	BYTE	$0x20
	BYTE	$0x59
L2:
	MOVH	R5, R5
	MOVH	$0x7FEF, R1
	CMPW	R5, R1
	BGT	L1
	WORD	$0xB3C10026	//ldgr	%f2,%r6
	FMUL	F2, F0
	WORD	$0xEC4439BB	//risbg	%r4,%r4,57,128+59,3
	BYTE	$0x03
	BYTE	$0x55
	FMOVD	80(R9), F2
	MOVD	$·logxm<>+0(SB), R7
	ADD	R7, R4
	FMOVD	72(R9), F4
	WORD	$0xED004000	//madb	%f2,%f0,0(%r4)
	BYTE	$0x20
	BYTE	$0x1E
	FMOVD	64(R9), F1
	FMOVD	F2, F0
	FMOVD	56(R9), F2
	WFMADB	V0, V2, V4, V2
	WFMDB	V0, V0, V6
	FMOVD	48(R9), F4
	WFMADB	V0, V2, V4, V2
	FMOVD	40(R9), F4
	WFMADB	V2, V6, V1, V2
	FMOVD	32(R9), F1
	WFMADB	V6, V4, V1, V4
	FMOVD	24(R9), F1
	WFMADB	V6, V2, V1, V2
	FMOVD	16(R9), F1
	WFMADB	V6, V4, V1, V4
	MOVD	$·logxl1<>+0(SB), R1
	FMOVD	8(R9), F1
	WFMADB	V6, V2, V1, V2
	FMOVD	0(R9), F1
	WFMADB	V6, V4, V1, V4
	FMOVD	8(R4), F1
	WFMADB	V0, V2, V4, V2
	WORD	$0xB3C10048	//ldgr	%f4,%r8
	WFMADB	V6, V2, V0, V2
	WORD	$0xED401000	//msdb	%f1,%f4,0(%r1)
	BYTE	$0x10
	BYTE	$0x1F
	MOVD	·logxl2<>+0(SB), R1
	WORD	$0xB3130001	//lcdbr	%f0,%f1
	WORD	$0xB3C10041	//ldgr	%f4,%r1
	WFMADB	V0, V4, V2, V0
L1:
	FMOVD	F0, ret+8(FP)
	RET
L3:
	WORD	$0xB3120000	//ltdbr	%f0,%f0
	BEQ	L20
	BGE	L1
	BVS	L1

	MOVD	$·logxnan<>+0(SB), R1
	FMOVD	0(R1), F0
	BR	L1
L20:
	MOVD	$·logxminf<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET
L17:
	BYTE	$0x18	//lr	%r5,%r1
	BYTE	$0x51
	BR	L2
