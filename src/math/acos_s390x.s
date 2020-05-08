// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial coefficients and other constants
DATA ·acosrodataL13<> + 0(SB)/8, $0.314159265358979323E+01   //pi
DATA ·acosrodataL13<> + 8(SB)/8, $-0.0
DATA ·acosrodataL13<> + 16(SB)/8, $0x7ff8000000000000    //Nan
DATA ·acosrodataL13<> + 24(SB)/8, $-1.0
DATA ·acosrodataL13<> + 32(SB)/8, $1.0
DATA ·acosrodataL13<> + 40(SB)/8, $0.166666666666651626E+00
DATA ·acosrodataL13<> + 48(SB)/8, $0.750000000042621169E-01
DATA ·acosrodataL13<> + 56(SB)/8, $0.446428567178116477E-01
DATA ·acosrodataL13<> + 64(SB)/8, $0.303819660378071894E-01
DATA ·acosrodataL13<> + 72(SB)/8, $0.223715011892010405E-01
DATA ·acosrodataL13<> + 80(SB)/8, $0.173659424522364952E-01
DATA ·acosrodataL13<> + 88(SB)/8, $0.137810186504372266E-01
DATA ·acosrodataL13<> + 96(SB)/8, $0.134066870961173521E-01
DATA ·acosrodataL13<> + 104(SB)/8, $-.412335502831898721E-02
DATA ·acosrodataL13<> + 112(SB)/8, $0.867383739532082719E-01
DATA ·acosrodataL13<> + 120(SB)/8, $-.328765950607171649E+00
DATA ·acosrodataL13<> + 128(SB)/8, $0.110401073869414626E+01
DATA ·acosrodataL13<> + 136(SB)/8, $-.270694366992537307E+01
DATA ·acosrodataL13<> + 144(SB)/8, $0.500196500770928669E+01
DATA ·acosrodataL13<> + 152(SB)/8, $-.665866959108585165E+01
DATA ·acosrodataL13<> + 160(SB)/8, $-.344895269334086578E+01
DATA ·acosrodataL13<> + 168(SB)/8, $0.927437952918301659E+00
DATA ·acosrodataL13<> + 176(SB)/8, $0.610487478874645653E+01
DATA ·acosrodataL13<> + 184(SB)/8, $0.157079632679489656e+01
DATA ·acosrodataL13<> + 192(SB)/8, $0.0
GLOBL ·acosrodataL13<> + 0(SB), RODATA, $200

// Acos returns the arccosine, in radians, of the argument.
//
// Special case is:
//      Acos(x) = NaN if x < -1 or x > 1
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT	·acosAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·acosrodataL13<>+0(SB), R9
	LGDR	F0, R12
	FMOVD	F0, F10
	SRAD	$32, R12
	WORD	$0xC0293FE6	//iilf	%r2,1072079005
	BYTE	$0xA0
	BYTE	$0x9D
	WORD	$0xB917001C	//llgtr	%r1,%r12
	CMPW	R1,R2
	BGT	L2
	FMOVD	192(R9), F8
	FMADD	F0, F0, F8
	FMOVD	184(R9), F1
L3:
	WFMDB	V8, V8, V2
	FMOVD	176(R9), F6
	FMOVD	168(R9), F0
	FMOVD	160(R9), F4
	WFMADB	V2, V0, V6, V0
	FMOVD	152(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	144(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	136(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	128(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	120(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	112(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	104(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	96(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	88(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	80(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	72(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	64(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	56(R9), F6
	WFMADB	V2, V4, V6, V4
	FMOVD	48(R9), F6
	WFMADB	V2, V0, V6, V0
	FMOVD	40(R9), F6
	WFMADB	V2, V4, V6, V2
	FMOVD	192(R9), F4
	WFMADB	V8, V0, V2, V0
	WFMADB	V10, V8, V4, V8
	FMADD	F0, F8, F10
	WFSDB	V10, V1, V10
L1:
	FMOVD	F10, ret+8(FP)
	RET

L2:
	WORD	$0xC0293FEF	//iilf	%r2,1072693247
	BYTE	$0xFF
	BYTE	$0xFF
	CMPW	R1, R2
	BLE	L12
L4:
	WORD	$0xED009020	//cdb	%f0,.L34-.L13(%r9)
	BYTE	$0x00
	BYTE	$0x19
	BEQ	L8
	WORD	$0xED009018	//cdb	%f0,.L35-.L13(%r9)
	BYTE	$0x00
	BYTE	$0x19
	BEQ	L9
	WFCEDBS	V10, V10, V0
	BVS	L1
	FMOVD	16(R9), F10
	BR	L1
L12:
	FMOVD	24(R9), F0
	FMADD	F10, F10, F0
	WORD	$0xB3130080	//lcdbr	%f8,%f0
	WORD	$0xED009008	//cdb	%f0,.L37-.L13(%r9)
	BYTE	$0x00
	BYTE	$0x19
	FSQRT	F8, F10
L5:
	MOVW	R12, R4
	CMPBLE	R4, $0, L7
	WORD	$0xB31300AA	//lcdbr	%f10,%f10
	FMOVD	$0, F1
	BR	L3
L9:
	FMOVD	0(R9), F10
	BR	L1
L8:
	FMOVD	$0, F0
	FMOVD	F0, ret+8(FP)
	RET
L7:
	FMOVD	0(R9), F1
	BR	L3
