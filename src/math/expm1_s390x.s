// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial approximation and other constants
DATA ·expm1rodataL22<> + 0(SB)/8, $-1.0
DATA ·expm1rodataL22<> + 8(SB)/8, $800.0E+00
DATA ·expm1rodataL22<> + 16(SB)/8, $1.0
DATA ·expm1rodataL22<> + 24(SB)/8, $-.231904681384629956E-16
DATA ·expm1rodataL22<> + 32(SB)/8, $0.50000000000000029671E+00
DATA ·expm1rodataL22<> + 40(SB)/8, $0.16666666666666676570E+00
DATA ·expm1rodataL22<> + 48(SB)/8, $0.83333333323590973444E-02
DATA ·expm1rodataL22<> + 56(SB)/8, $0.13889096526400683566E-02
DATA ·expm1rodataL22<> + 64(SB)/8, $0.41666666661701152924E-01
DATA ·expm1rodataL22<> + 72(SB)/8, $0.19841562053987360264E-03
DATA ·expm1rodataL22<> + 80(SB)/8, $-.693147180559945286E+00
DATA ·expm1rodataL22<> + 88(SB)/8, $0.144269504088896339E+01
DATA ·expm1rodataL22<> + 96(SB)/8, $704.0E+00
GLOBL ·expm1rodataL22<> + 0(SB), RODATA, $104

DATA ·expm1xmone<> + 0(SB)/8, $0xbff0000000000000
GLOBL ·expm1xmone<> + 0(SB), RODATA, $8
DATA ·expm1xinf<> + 0(SB)/8, $0x7ff0000000000000
GLOBL ·expm1xinf<> + 0(SB), RODATA, $8
DATA ·expm1x4ff<> + 0(SB)/8, $0x4ff0000000000000
GLOBL ·expm1x4ff<> + 0(SB), RODATA, $8
DATA ·expm1x2ff<> + 0(SB)/8, $0x2ff0000000000000
GLOBL ·expm1x2ff<> + 0(SB), RODATA, $8
DATA ·expm1xaddexp<> + 0(SB)/8, $0xc2f0000100003ff0
GLOBL ·expm1xaddexp<> + 0(SB), RODATA, $8

// Log multipliers table
DATA ·expm1tab<> + 0(SB)/8, $0.0
DATA ·expm1tab<> + 8(SB)/8, $-.171540871271399150E-01
DATA ·expm1tab<> + 16(SB)/8, $-.306597931864376363E-01
DATA ·expm1tab<> + 24(SB)/8, $-.410200970469965021E-01
DATA ·expm1tab<> + 32(SB)/8, $-.486343079978231466E-01
DATA ·expm1tab<> + 40(SB)/8, $-.538226193725835820E-01
DATA ·expm1tab<> + 48(SB)/8, $-.568439602538111520E-01
DATA ·expm1tab<> + 56(SB)/8, $-.579091847395528847E-01
DATA ·expm1tab<> + 64(SB)/8, $-.571909584179366341E-01
DATA ·expm1tab<> + 72(SB)/8, $-.548312665987204407E-01
DATA ·expm1tab<> + 80(SB)/8, $-.509471843643441085E-01
DATA ·expm1tab<> + 88(SB)/8, $-.456353588448863359E-01
DATA ·expm1tab<> + 96(SB)/8, $-.389755254243262365E-01
DATA ·expm1tab<> + 104(SB)/8, $-.310332908285244231E-01
DATA ·expm1tab<> + 112(SB)/8, $-.218623539150173528E-01
DATA ·expm1tab<> + 120(SB)/8, $-.115062908917949451E-01
GLOBL ·expm1tab<> + 0(SB), RODATA, $128

// Expm1 returns e**x - 1, the base-e exponential of x minus 1.
// It is more accurate than Exp(x) - 1 when x is near zero.
//
// Special cases are:
//      Expm1(+Inf) = +Inf
//      Expm1(-Inf) = -1
//      Expm1(NaN) = NaN
// Very large values overflow to -1 or +Inf.
// The algorithm used is minimax polynomial approximation using a table of
// polynomial coefficients determined with a Remez exchange algorithm.

TEXT	·expm1Asm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·expm1rodataL22<>+0(SB), R5
	LTDBR	F0, F0
	BLTU	L20
	FMOVD	F0, F2
L2:
	WORD	$0xED205060	//cdb	%f2,.L23-.L22(%r5)
	BYTE	$0x00
	BYTE	$0x19
	BGE	L16
	BVS	L16
	WFCEDBS	V2, V2, V2
	BVS	LEXITTAGexpm1
	MOVD	$·expm1xaddexp<>+0(SB), R1
	FMOVD	88(R5), F1
	FMOVD	0(R1), F2
	WFMSDB	V0, V1, V2, V1
	FMOVD	80(R5), F6
	WFADB	V1, V2, V4
	FMOVD	72(R5), F2
	FMADD	F6, F4, F0
	FMOVD	64(R5), F3
	FMOVD	56(R5), F6
	FMOVD	48(R5), F5
	FMADD	F2, F0, F6
	WFMADB	V0, V5, V3, V5
	WFMDB	V0, V0, V2
	WORD	$0xB3CD0011	//lgdr	%r1,%f1
	WFMADB	V6, V2, V5, V6
	FMOVD	40(R5), F3
	FMOVD	32(R5), F5
	WFMADB	V0, V3, V5, V3
	FMOVD	24(R5), F5
	WFMADB	V2, V6, V3, V2
	FMADD	F5, F4, F0
	FMOVD	16(R5), F6
	WFMADB	V0, V2, V6, V2
	WORD	$0xEC3139BC	//risbg	%r3,%r1,57,128+60,3
	BYTE	$0x03
	BYTE	$0x55
	WORD	$0xB3130022	//lcdbr	%f2,%f2
	MOVD	$·expm1tab<>+0(SB), R2
	WORD	$0x68432000	//ld	%f4,0(%r3,%r2)
	FMADD	F4, F0, F0
	SLD	$48, R1, R2
	WFMSDB	V2, V0, V4, V0
	WORD	$0xB3C10042	//ldgr	%f4,%r2
	WORD	$0xB3130000	//lcdbr	%f0,%f0
	FSUB	F4, F6
	WFMSDB	V0, V4, V6, V0
	FMOVD	F0, ret+8(FP)
	RET
L16:
	WFCEDBS	V2, V2, V4
	BVS	LEXITTAGexpm1
	WORD	$0xED205008	//cdb	%f2,.L34-.L22(%r5)
	BYTE	$0x00
	BYTE	$0x19
	BLT	L6
	WFCEDBS	V2, V0, V0
	BVS	L7
	MOVD	$·expm1xinf<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET
L20:
	WORD	$0xB3130020	//lcdbr	%f2,%f0
	BR	L2
L6:
	MOVD	$·expm1xaddexp<>+0(SB), R1
	FMOVD	88(R5), F5
	FMOVD	0(R1), F4
	WFMSDB	V0, V5, V4, V5
	FMOVD	80(R5), F3
	WFADB	V5, V4, V1
	VLEG	$0, 48(R5), V16
	WFMADB	V1, V3, V0, V3
	FMOVD	56(R5), F4
	FMOVD	64(R5), F7
	FMOVD	72(R5), F6
	WFMADB	V3, V16, V7, V16
	WFMADB	V3, V6, V4, V6
	WFMDB	V3, V3, V4
	MOVD	$·expm1tab<>+0(SB), R2
	WFMADB	V6, V4, V16, V6
	VLEG	$0, 32(R5), V16
	FMOVD	40(R5), F7
	WFMADB	V3, V7, V16, V7
	VLEG	$0, 24(R5), V16
	WFMADB	V4, V6, V7, V4
	WFMADB	V1, V16, V3, V1
	FMOVD	16(R5), F6
	FMADD	F4, F1, F6
	WORD	$0xB3CD0015	//lgdr	%r1,%f5
	WORD	$0xB3130066	//lcdbr	%f6,%f6
	WORD	$0xEC3139BC	//risbg	%r3,%r1,57,128+60,3
	BYTE	$0x03
	BYTE	$0x55
	WORD	$0x68432000	//ld	%f4,0(%r3,%r2)
	FMADD	F4, F1, F1
	MOVD	$0x4086000000000000, R2
	FMSUB	F1, F6, F4
	WORD	$0xB3130044	//lcdbr	%f4,%f4
	WFCHDBS	V2, V0, V0
	BEQ	L21
	ADDW	$0xF000, R1
	WORD	$0xEC21000F	//risbgn	%r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE	$0x30
	BYTE	$0x59
	WORD	$0xB3C10002	//ldgr	%f0,%r2
	FMADD	F0, F4, F0
	MOVD	$·expm1x4ff<>+0(SB), R3
	FMOVD	0(R5), F4
	FMOVD	0(R3), F2
	WFMADB	V2, V0, V4, V0
	FMOVD	F0, ret+8(FP)
	RET
L7:
	MOVD	$·expm1xmone<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET
L21:
	ADDW	$0x1000, R1
	WORD	$0xEC21000F	//risbgn	%r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE	$0x30
	BYTE	$0x59
	WORD	$0xB3C10002	//ldgr	%f0,%r2
	FMADD	F0, F4, F0
	MOVD	$·expm1x2ff<>+0(SB), R3
	FMOVD	0(R5), F4
	FMOVD	0(R3), F2
	WFMADB	V2, V0, V4, V0
	FMOVD	F0, ret+8(FP)
	RET
LEXITTAGexpm1:
	FMOVD	F0, ret+8(FP)
	RET
