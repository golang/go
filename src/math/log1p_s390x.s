// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Constants
DATA ·log1pxlim<> + 0(SB)/4, $0xfff00000
GLOBL ·log1pxlim<> + 0(SB), RODATA, $4
DATA ·log1pxzero<> + 0(SB)/8, $0.0
GLOBL ·log1pxzero<> + 0(SB), RODATA, $8
DATA ·log1pxminf<> + 0(SB)/8, $0xfff0000000000000
GLOBL ·log1pxminf<> + 0(SB), RODATA, $8
DATA ·log1pxnan<> + 0(SB)/8, $0x7ff8000000000000
GLOBL ·log1pxnan<> + 0(SB), RODATA, $8
DATA ·log1pyout<> + 0(SB)/8, $0x40fce621e71da000
GLOBL ·log1pyout<> + 0(SB), RODATA, $8
DATA ·log1pxout<> + 0(SB)/8, $0x40f1000000000000
GLOBL ·log1pxout<> + 0(SB), RODATA, $8
DATA ·log1pxl2<> + 0(SB)/8, $0xbfda7aecbeba4e46
GLOBL ·log1pxl2<> + 0(SB), RODATA, $8
DATA ·log1pxl1<> + 0(SB)/8, $0x3ffacde700000000
GLOBL ·log1pxl1<> + 0(SB), RODATA, $8
DATA ·log1pxa<> + 0(SB)/8, $5.5
GLOBL ·log1pxa<> + 0(SB), RODATA, $8
DATA ·log1pxmone<> + 0(SB)/8, $-1.0
GLOBL ·log1pxmone<> + 0(SB), RODATA, $8

// Minimax polynomial approximations
DATA ·log1pc8<> + 0(SB)/8, $0.212881813645679599E-07
GLOBL ·log1pc8<> + 0(SB), RODATA, $8
DATA ·log1pc7<> + 0(SB)/8, $-.148682720127920854E-06
GLOBL ·log1pc7<> + 0(SB), RODATA, $8
DATA ·log1pc6<> + 0(SB)/8, $0.938370938292558173E-06
GLOBL ·log1pc6<> + 0(SB), RODATA, $8
DATA ·log1pc5<> + 0(SB)/8, $-.602107458843052029E-05
GLOBL ·log1pc5<> + 0(SB), RODATA, $8
DATA ·log1pc4<> + 0(SB)/8, $0.397389654305194527E-04
GLOBL ·log1pc4<> + 0(SB), RODATA, $8
DATA ·log1pc3<> + 0(SB)/8, $-.273205381970859341E-03
GLOBL ·log1pc3<> + 0(SB), RODATA, $8
DATA ·log1pc2<> + 0(SB)/8, $0.200350613573012186E-02
GLOBL ·log1pc2<> + 0(SB), RODATA, $8
DATA ·log1pc1<> + 0(SB)/8, $-.165289256198351540E-01
GLOBL ·log1pc1<> + 0(SB), RODATA, $8
DATA ·log1pc0<> + 0(SB)/8, $0.181818181818181826E+00
GLOBL ·log1pc0<> + 0(SB), RODATA, $8


// Table of log10 correction terms
DATA ·log1ptab<> + 0(SB)/8, $0.585235384085551248E-01
DATA ·log1ptab<> + 8(SB)/8, $0.412206153771168640E-01
DATA ·log1ptab<> + 16(SB)/8, $0.273839003221648339E-01
DATA ·log1ptab<> + 24(SB)/8, $0.166383778368856480E-01
DATA ·log1ptab<> + 32(SB)/8, $0.866678223433169637E-02
DATA ·log1ptab<> + 40(SB)/8, $0.319831684989627514E-02
DATA ·log1ptab<> + 48(SB)/8, $-.000000000000000000E+00
DATA ·log1ptab<> + 56(SB)/8, $-.113006378583725549E-02
DATA ·log1ptab<> + 64(SB)/8, $-.367979419636602491E-03
DATA ·log1ptab<> + 72(SB)/8, $0.213172484510484979E-02
DATA ·log1ptab<> + 80(SB)/8, $0.623271047682013536E-02
DATA ·log1ptab<> + 88(SB)/8, $0.118140812789696885E-01
DATA ·log1ptab<> + 96(SB)/8, $0.187681358930914206E-01
DATA ·log1ptab<> + 104(SB)/8, $0.269985148668178992E-01
DATA ·log1ptab<> + 112(SB)/8, $0.364186619761331328E-01
DATA ·log1ptab<> + 120(SB)/8, $0.469505379381388441E-01
GLOBL ·log1ptab<> + 0(SB), RODATA, $128

// Log1p returns the natural logarithm of 1 plus its argument x.
// It is more accurate than Log(1 + x) when x is near zero.
//
// Special cases are:
//      Log1p(+Inf) = +Inf
//      Log1p(±0) = ±0
//      Log1p(-1) = -Inf
//      Log1p(x < -1) = NaN
//      Log1p(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT	·log1pAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·log1pxmone<>+0(SB), R1
	MOVD	·log1pxout<>+0(SB), R2
	FMOVD	0(R1), F3
	MOVD	$·log1pxa<>+0(SB), R1
	MOVWZ	·log1pxlim<>+0(SB), R0
	FMOVD	0(R1), F1
	MOVD	$·log1pc8<>+0(SB), R1
	FMOVD	0(R1), F5
	MOVD	$·log1pc7<>+0(SB), R1
	VLEG	$0, 0(R1), V20
	MOVD	$·log1pc6<>+0(SB), R1
	WFSDB	V0, V3, V4
	VLEG	$0, 0(R1), V18
	MOVD	$·log1pc5<>+0(SB), R1
	VLEG	$0, 0(R1), V16
	MOVD	R2, R5
	LGDR	F4, R3
	WORD	$0xC0190006	//iilf	%r1,425983
	BYTE	$0x7F
	BYTE	$0xFF
	SRAD	$32, R3, R3
	SUBW	R3, R1
	SRW	$16, R1, R1
	BYTE	$0x18	//lr	%r4,%r1
	BYTE	$0x41
	RISBGN	$0, $15, $48, R4, R2
	RISBGN	$16, $31, $32, R4, R5
	MOVW	R0, R6
	MOVW	R3, R7
	CMPBGT	R6, R7, L8
	WFCEDBS	V4, V4, V6
	MOVD	$·log1pxzero<>+0(SB), R1
	FMOVD	0(R1), F2
	BVS	LEXITTAGlog1p
	LCDBR	F4, F4
	WFCEDBS	V2, V4, V6
	BEQ	L9
	WFCHDBS	V4, V2, V2
	BEQ	LEXITTAGlog1p
	MOVD	$·log1pxnan<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET

L8:
	LDGR	R2, F2
	FSUB	F4, F3
	FMADD	F2, F4, F1
	MOVD	$·log1pc4<>+0(SB), R2
	LCDBR	F1, F4
	FMOVD	0(R2), F7
	FSUB	F3, F0
	MOVD	$·log1pc3<>+0(SB), R2
	FMOVD	0(R2), F3
	MOVD	$·log1pc2<>+0(SB), R2
	WFMDB	V1, V1, V6
	FMADD	F7, F4, F3
	WFMSDB	V0, V2, V1, V0
	FMOVD	0(R2), F7
	WFMADB	V4, V5, V20, V5
	MOVD	$·log1pc1<>+0(SB), R2
	FMOVD	0(R2), F2
	FMADD	F7, F4, F2
	WFMADB	V4, V18, V16, V4
	FMADD	F3, F6, F2
	WFMADB	V5, V6, V4, V5
	FMUL	F6, F6
	MOVD	$·log1pc0<>+0(SB), R2
	WFMADB	V6, V5, V2, V6
	FMOVD	0(R2), F4
	WFMADB	V0, V6, V4, V6
	RISBGZ	$57, $60, $3, R1, R1
	MOVD	$·log1ptab<>+0(SB), R2
	MOVD	$·log1pxl1<>+0(SB), R3
	WORD	$0x68112000	//ld	%f1,0(%r1,%r2)
	FMOVD	0(R3), F2
	WFMADB	V0, V6, V1, V0
	MOVD	$·log1pyout<>+0(SB), R1
	LDGR	R5, F6
	FMOVD	0(R1), F4
	WFMSDB	V2, V6, V4, V2
	MOVD	$·log1pxl2<>+0(SB), R1
	FMOVD	0(R1), F4
	FMADD	F4, F2, F0
	FMOVD	F0, ret+8(FP)
	RET

L9:
	MOVD	$·log1pxminf<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET


LEXITTAGlog1p:
	FMOVD	F0, ret+8(FP)
	RET

