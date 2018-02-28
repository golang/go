// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial approximation and other constants
DATA ·exprodataL22<> + 0(SB)/8, $800.0E+00
DATA ·exprodataL22<> + 8(SB)/8, $1.0000000000000022e+00
DATA ·exprodataL22<> + 16(SB)/8, $0.500000000000004237e+00
DATA ·exprodataL22<> + 24(SB)/8, $0.166666666630345592e+00
DATA ·exprodataL22<> + 32(SB)/8, $0.138926439368309441e-02
DATA ·exprodataL22<> + 40(SB)/8, $0.833349307718286047e-02
DATA ·exprodataL22<> + 48(SB)/8, $0.416666664838056960e-01
DATA ·exprodataL22<> + 56(SB)/8, $-.231904681384629956E-16
DATA ·exprodataL22<> + 64(SB)/8, $-.693147180559945286E+00
DATA ·exprodataL22<> + 72(SB)/8, $0.144269504088896339E+01
DATA ·exprodataL22<> + 80(SB)/8, $704.0E+00
GLOBL ·exprodataL22<> + 0(SB), RODATA, $88

DATA ·expxinf<> + 0(SB)/8, $0x7ff0000000000000
GLOBL ·expxinf<> + 0(SB), RODATA, $8
DATA ·expx4ff<> + 0(SB)/8, $0x4ff0000000000000
GLOBL ·expx4ff<> + 0(SB), RODATA, $8
DATA ·expx2ff<> + 0(SB)/8, $0x2ff0000000000000
GLOBL ·expx2ff<> + 0(SB), RODATA, $8
DATA ·expxaddexp<> + 0(SB)/8, $0xc2f0000100003fef
GLOBL ·expxaddexp<> + 0(SB), RODATA, $8

// Log multipliers table
DATA ·exptexp<> + 0(SB)/8, $0.442737824274138381E-01
DATA ·exptexp<> + 8(SB)/8, $0.263602189790660309E-01
DATA ·exptexp<> + 16(SB)/8, $0.122565642281703586E-01
DATA ·exptexp<> + 24(SB)/8, $0.143757052860721398E-02
DATA ·exptexp<> + 32(SB)/8, $-.651375034121276075E-02
DATA ·exptexp<> + 40(SB)/8, $-.119317678849450159E-01
DATA ·exptexp<> + 48(SB)/8, $-.150868749549871069E-01
DATA ·exptexp<> + 56(SB)/8, $-.161992609578469234E-01
DATA ·exptexp<> + 64(SB)/8, $-.154492360403337917E-01
DATA ·exptexp<> + 72(SB)/8, $-.129850717389178721E-01
DATA ·exptexp<> + 80(SB)/8, $-.892902649276657891E-02
DATA ·exptexp<> + 88(SB)/8, $-.338202636596794887E-02
DATA ·exptexp<> + 96(SB)/8, $0.357266307045684762E-02
DATA ·exptexp<> + 104(SB)/8, $0.118665304327406698E-01
DATA ·exptexp<> + 112(SB)/8, $0.214434994118118914E-01
DATA ·exptexp<> + 120(SB)/8, $0.322580645161290314E-01
GLOBL ·exptexp<> + 0(SB), RODATA, $128

// Exp returns e**x, the base-e exponential of x.
//
// Special cases are:
//      Exp(+Inf) = +Inf
//      Exp(NaN) = NaN
// Very large values overflow to 0 or +Inf.
// Very small values underflow to 1.
// The algorithm used is minimax polynomial approximation using a table of
// polynomial coefficients determined with a Remez exchange algorithm.

TEXT	·expAsm(SB), NOSPLIT, $0-16
	FMOVD	x+0(FP), F0
	MOVD	$·exprodataL22<>+0(SB), R5
	WORD	$0xB3120000	//ltdbr	%f0,%f0
	BLTU	L20
	FMOVD	F0, F2
L2:
	WORD	$0xED205050	//cdb	%f2,.L23-.L22(%r5)
	BYTE	$0x00
	BYTE	$0x19
	BGE	L16
	BVS	L16
	WFCEDBS	V2, V2, V2
	BVS	LEXITTAGexp
	MOVD	$·expxaddexp<>+0(SB), R1
	FMOVD	72(R5), F6
	FMOVD	0(R1), F2
	WFMSDB	V0, V6, V2, V6
	FMOVD	64(R5), F4
	FADD	F6, F2
	FMOVD	56(R5), F1
	FMADD	F4, F2, F0
	FMOVD	48(R5), F3
	WFMADB	V2, V1, V0, V2
	FMOVD	40(R5), F1
	FMOVD	32(R5), F4
	FMUL	F0, F0
	WFMADB	V2, V4, V1, V4
	WORD	$0xB3CD0016	//lgdr	%r1,%f6
	FMOVD	24(R5), F1
	WFMADB	V2, V3, V1, V3
	FMOVD	16(R5), F1
	WFMADB	V0, V4, V3, V4
	FMOVD	8(R5), F3
	WFMADB	V2, V1, V3, V1
	WORD	$0xEC3139BC	//risbg	%r3,%r1,57,128+60,3
	BYTE	$0x03
	BYTE	$0x55
	WFMADB	V0, V4, V1, V0
	MOVD	$·exptexp<>+0(SB), R2
	WORD	$0x68432000	//ld	%f4,0(%r3,%r2)
	FMADD	F4, F2, F2
	SLD	$48, R1, R2
	WFMADB	V2, V0, V4, V2
	WORD	$0xB3C10002	//ldgr	%f0,%r2
	FMADD	F0, F2, F0
	FMOVD	F0, ret+8(FP)
	RET
L16:
	WFCEDBS	V2, V2, V4
	BVS	LEXITTAGexp
	WORD	$0xED205000	//cdb	%f2,.L33-.L22(%r5)
	BYTE	$0x00
	BYTE	$0x19
	BLT	L6
	WFCEDBS	V2, V0, V0
	BVS	L13
	MOVD	$·expxinf<>+0(SB), R1
	FMOVD	0(R1), F0
	FMOVD	F0, ret+8(FP)
	RET
L20:
	WORD	$0xB3130020	//lcdbr	%f2,%f0
	BR	L2
L6:
	MOVD	$·expxaddexp<>+0(SB), R1
	FMOVD	72(R5), F3
	FMOVD	0(R1), F4
	WFMSDB	V0, V3, V4, V3
	FMOVD	64(R5), F6
	FADD	F3, F4
	FMOVD	56(R5), F5
	WFMADB	V4, V6, V0, V6
	FMOVD	32(R5), F1
	WFMADB	V4, V5, V6, V4
	FMOVD	40(R5), F5
	FMUL	F6, F6
	WFMADB	V4, V1, V5, V1
	FMOVD	48(R5), F7
	WORD	$0xB3CD0013	//lgdr	%r1,%f3
	FMOVD	24(R5), F5
	WFMADB	V4, V7, V5, V7
	FMOVD	16(R5), F5
	WFMADB	V6, V1, V7, V1
	FMOVD	8(R5), F7
	WFMADB	V4, V5, V7, V5
	WORD	$0xEC3139BC	//risbg	%r3,%r1,57,128+60,3
	BYTE	$0x03
	BYTE	$0x55
	WFMADB	V6, V1, V5, V6
	MOVD	$·exptexp<>+0(SB), R2
	WFCHDBS	V2, V0, V0
	WORD	$0x68132000	//ld	%f1,0(%r3,%r2)
	FMADD	F1, F4, F4
	MOVD	$0x4086000000000000, R2
	WFMADB	V4, V6, V1, V4
	BEQ	L21
	ADDW	$0xF000, R1
	WORD	$0xEC21000F	//risbgn	%r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE	$0x30
	BYTE	$0x59
	WORD	$0xB3C10002	//ldgr	%f0,%r2
	FMADD	F0, F4, F0
	MOVD	$·expx4ff<>+0(SB), R3
	FMOVD	0(R3), F2
	FMUL	F2, F0
	FMOVD	F0, ret+8(FP)
	RET
L13:
	FMOVD	$0, F0
	FMOVD	F0, ret+8(FP)
	RET
L21:
	ADDW	$0x1000, R1
	WORD	$0xEC21000F	//risbgn	%r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE	$0x30
	BYTE	$0x59
	WORD	$0xB3C10002	//ldgr	%f0,%r2
	FMADD	F0, F4, F0
	MOVD	$·expx2ff<>+0(SB), R3
	FMOVD	0(R3), F2
	FMUL	F2, F0
	FMOVD	F0, ret+8(FP)
	RET
LEXITTAGexp:
	FMOVD	F0, ret+8(FP)
	RET
