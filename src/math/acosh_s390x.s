// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial coefficients and other constants
DATA ·acoshrodataL11<> + 0(SB)/8, $-1.0
DATA ·acoshrodataL11<> + 8(SB)/8, $.41375273347623353626
DATA ·acoshrodataL11<> + 16(SB)/8, $.51487302528619766235E+04
DATA ·acoshrodataL11<> + 24(SB)/8, $-1.67526912689208984375
DATA ·acoshrodataL11<> + 32(SB)/8, $0.181818181818181826E+00
DATA ·acoshrodataL11<> + 40(SB)/8, $-.165289256198351540E-01
DATA ·acoshrodataL11<> + 48(SB)/8, $0.200350613573012186E-02
DATA ·acoshrodataL11<> + 56(SB)/8, $-.273205381970859341E-03
DATA ·acoshrodataL11<> + 64(SB)/8, $0.397389654305194527E-04
DATA ·acoshrodataL11<> + 72(SB)/8, $0.938370938292558173E-06
DATA ·acoshrodataL11<> + 80(SB)/8, $-.602107458843052029E-05
DATA ·acoshrodataL11<> + 88(SB)/8, $0.212881813645679599E-07
DATA ·acoshrodataL11<> + 96(SB)/8, $-.148682720127920854E-06
DATA ·acoshrodataL11<> + 104(SB)/8, $-5.5
DATA ·acoshrodataL11<> + 112(SB)/8, $0x7ff8000000000000      //Nan
GLOBL ·acoshrodataL11<> + 0(SB), RODATA, $120

// Table of log correction terms
DATA ·acoshtab2068<> + 0(SB)/8, $0.585235384085551248E-01
DATA ·acoshtab2068<> + 8(SB)/8, $0.412206153771168640E-01
DATA ·acoshtab2068<> + 16(SB)/8, $0.273839003221648339E-01
DATA ·acoshtab2068<> + 24(SB)/8, $0.166383778368856480E-01
DATA ·acoshtab2068<> + 32(SB)/8, $0.866678223433169637E-02
DATA ·acoshtab2068<> + 40(SB)/8, $0.319831684989627514E-02
DATA ·acoshtab2068<> + 48(SB)/8, $0.0
DATA ·acoshtab2068<> + 56(SB)/8, $-.113006378583725549E-02
DATA ·acoshtab2068<> + 64(SB)/8, $-.367979419636602491E-03
DATA ·acoshtab2068<> + 72(SB)/8, $0.213172484510484979E-02
DATA ·acoshtab2068<> + 80(SB)/8, $0.623271047682013536E-02
DATA ·acoshtab2068<> + 88(SB)/8, $0.118140812789696885E-01
DATA ·acoshtab2068<> + 96(SB)/8, $0.187681358930914206E-01
DATA ·acoshtab2068<> + 104(SB)/8, $0.269985148668178992E-01
DATA ·acoshtab2068<> + 112(SB)/8, $0.364186619761331328E-01
DATA ·acoshtab2068<> + 120(SB)/8, $0.469505379381388441E-01
GLOBL ·acoshtab2068<> + 0(SB), RODATA, $128

// Acosh returns the inverse hyperbolic cosine of the argument.
//
// Special cases are:
//      Acosh(+Inf) = +Inf
//      Acosh(x) = NaN if x < 1
//      Acosh(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT	·acoshAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·acoshrodataL11<>+0(SB), R9
	WORD	$0xB3CD0010	//lgdr %r1, %f0
	WORD	$0xC0295FEF	//iilf	%r2,1609564159
	BYTE	$0xFF
	BYTE	$0xFF
	SRAD	$32, R1
	CMPW	R1, R2
	BGT	L2
	WORD	$0xC0293FEF	//iilf	%r2,1072693247
	BYTE	$0xFF
	BYTE	$0xFF
	CMPW	R1, R2
	BGT	L10
L3:
	WFCEDBS	V0, V0, V2
	BVS	L1
	FMOVD	112(R9), F0
L1:
	FMOVD	F0, ret+8(FP)
	RET
L2:
	WORD	$0xC0297FEF	//iilf	%r2,2146435071
	BYTE	$0xFF
	BYTE	$0xFF
	MOVW	R1, R6
	MOVW	R2, R7
	CMPBGT	R6, R7, L1
	FMOVD	F0, F8
	FMOVD	$0, F0
	WFADB	V0, V8, V0
	WORD	$0xC0398006	//iilf	%r3,2147909631
	BYTE	$0x7F
	BYTE	$0xFF
	WORD	$0xB3CD0050	//lgdr %r5, %f0
	SRAD	$32, R5
	MOVH	$0x0, R1
	SUBW	R5, R3
	FMOVD	$0, F10
	WORD	$0xEC4320AF	//risbg	%r4,%r3,32,128+47,0
	BYTE	$0x00
	BYTE	$0x55
	WORD	$0xEC3339BC	//risbg	%r3,%r3,57,128+60,64-13
	BYTE	$0x33
	BYTE	$0x55
	BYTE	$0x18	//lr	%r2,%r4
	BYTE	$0x24
	WORD	$0xEC14001F	//risbgn	%r1,%r4,64-64+0,64-64+0+32-1,64-0-32
	BYTE	$0x20
	BYTE	$0x59
	SUBW	$0x100000, R2
	SRAW	$8, R2, R2
	ORW	$0x45000000, R2
L5:
	WORD	$0xB3C10001	//ldgr	%f0,%r1
	FMOVD	104(R9), F2
	FMADD	F8, F0, F2
	FMOVD	96(R9), F4
	WFMADB	V10, V0, V2, V0
	FMOVD	88(R9), F6
	FMOVD	80(R9), F2
	WFMADB	V0, V6, V4, V6
	FMOVD	72(R9), F1
	WFMDB	V0, V0, V4
	WFMADB	V0, V1, V2, V1
	FMOVD	64(R9), F2
	WFMADB	V6, V4, V1, V6
	FMOVD	56(R9), F1
	WORD	$0xEC3339BC	//risbg	%r3,%r3,57,128+60,0
	BYTE	$0x00
	BYTE	$0x55
	WFMADB	V0, V2, V1, V2
	FMOVD	48(R9), F1
	WFMADB	V4, V6, V2, V6
	FMOVD	40(R9), F2
	WFMADB	V0, V1, V2, V1
	VLVGF	$0, R2, V2
	WFMADB	V4, V6, V1, V4
	LDEBR	F2, F2
	FMOVD	32(R9), F6
	WFMADB	V0, V4, V6, V4
	FMOVD	24(R9), F1
	FMOVD	16(R9), F6
	MOVD	$·acoshtab2068<>+0(SB), R1
	WFMADB	V2, V1, V6, V2
	FMOVD	0(R3)(R1*1), F3
	WFMADB	V0, V4, V3, V0
	FMOVD	8(R9), F4
	FMADD	F4, F2, F0
	FMOVD	F0, ret+8(FP)
	RET
L10:
	FMOVD	F0, F8
	FMOVD	0(R9), F0
	FMADD	F8, F8, F0
	LTDBR	F0, F0
	FSQRT	F0, F10
L4:
	WFADB	V10, V8, V0
	WORD	$0xC0398006	//iilf	%r3,2147909631
	BYTE	$0x7F
	BYTE	$0xFF
	WORD	$0xB3CD0050	//lgdr %r5, %f0
	SRAD	$32, R5
	MOVH	$0x0, R1
	SUBW	R5, R3
	SRAW	$8, R3, R2
	WORD	$0xEC4320AF	//risbg	%r4,%r3,32,128+47,0
	BYTE	$0x00
	BYTE	$0x55
	ANDW	$0xFFFFFF00, R2
	WORD	$0xEC3339BC	//risbg	%r3,%r3,57,128+60,64-13
	BYTE	$0x33
	BYTE	$0x55
	ORW	$0x45000000, R2
	WORD	$0xEC14001F	//risbgn	%r1,%r4,64-64+0,64-64+0+32-1,64-0-32
	BYTE	$0x20
	BYTE	$0x59
	BR	L5
