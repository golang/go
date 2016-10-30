// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


#include "textflag.h"

// Constants
DATA sinhrodataL21<>+0(SB)/8, $0.231904681384629956E-16
DATA sinhrodataL21<>+8(SB)/8, $0.693147180559945286E+00
DATA sinhrodataL21<>+16(SB)/8, $704.E0
GLOBL sinhrodataL21<>+0(SB), RODATA, $24
DATA sinhrlog2<>+0(SB)/8, $0x3ff7154760000000
GLOBL sinhrlog2<>+0(SB), RODATA, $8
DATA sinhxinf<>+0(SB)/8, $0x7ff0000000000000
GLOBL sinhxinf<>+0(SB), RODATA, $8
DATA sinhxinit<>+0(SB)/8, $0x3ffb504f333f9de6
GLOBL sinhxinit<>+0(SB), RODATA, $8
DATA sinhxlim1<>+0(SB)/8, $800.E0
GLOBL sinhxlim1<>+0(SB), RODATA, $8
DATA sinhxadd<>+0(SB)/8, $0xc3200001610007fb
GLOBL sinhxadd<>+0(SB), RODATA, $8
DATA sinhx4ff<>+0(SB)/8, $0x4ff0000000000000
GLOBL sinhx4ff<>+0(SB), RODATA, $8

// Minimax polynomial approximations
DATA sinhe0<>+0(SB)/8, $0.11715728752538099300E+01
GLOBL sinhe0<>+0(SB), RODATA, $8
DATA sinhe1<>+0(SB)/8, $0.11715728752538099300E+01
GLOBL sinhe1<>+0(SB), RODATA, $8
DATA sinhe2<>+0(SB)/8, $0.58578643762688526692E+00
GLOBL sinhe2<>+0(SB), RODATA, $8
DATA sinhe3<>+0(SB)/8, $0.19526214587563004497E+00
GLOBL sinhe3<>+0(SB), RODATA, $8
DATA sinhe4<>+0(SB)/8, $0.48815536475176217404E-01
GLOBL sinhe4<>+0(SB), RODATA, $8
DATA sinhe5<>+0(SB)/8, $0.97631072948627397816E-02
GLOBL sinhe5<>+0(SB), RODATA, $8
DATA sinhe6<>+0(SB)/8, $0.16271839297756073153E-02
GLOBL sinhe6<>+0(SB), RODATA, $8
DATA sinhe7<>+0(SB)/8, $0.23245485387271142509E-03
GLOBL sinhe7<>+0(SB), RODATA, $8
DATA sinhe8<>+0(SB)/8, $0.29080955860869629131E-04
GLOBL sinhe8<>+0(SB), RODATA, $8
DATA sinhe9<>+0(SB)/8, $0.32311267157667725278E-05
GLOBL sinhe9<>+0(SB), RODATA, $8

// Sinh returns the hyperbolic sine of the argument.
//
// Special cases are:
//      Sinh(±0) = ±0
//      Sinh(±Inf) = ±Inf
//      Sinh(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT ·sinhAsm(SB),NOSPLIT,$0-16
	FMOVD   x+0(FP), F0
	//specail case Sinh(±0) = ±0
	FMOVD   $(0.0), F1
	FCMPU   F0, F1
	BEQ     sinhIsZero
	//specail case Sinh(±Inf = ±Inf
	FMOVD   $1.797693134862315708145274237317043567981e+308, F1
	FCMPU   F1, F0
	BLEU    sinhIsInf
	FMOVD   $-1.797693134862315708145274237317043567981e+308, F1
	FCMPU   F1, F0
	BGT             sinhIsInf

	MOVD    $sinhrodataL21<>+0(SB), R5
	WORD    $0xB3120000     //ltdbr %f0,%f0
	MOVD    sinhxinit<>+0(SB), R1
	FMOVD   F0, F4
	MOVD    R1, R3
	BLTU    L19
	FMOVD   F0, F2
L2:
	WORD    $0xED205010     //cdb %f2,.L22-.L21(%r5)
	BYTE    $0x00
	BYTE    $0x19
	BGE     L15     //jnl   .L15
	BVS     L15
	WFCEDBS V2, V2, V0
	BEQ     L20
L12:
	FMOVD   F4, F0
	FMOVD   F0, ret+8(FP)
	RET

L15:
	WFCEDBS V2, V2, V0
	BVS     L12
	MOVD    $sinhxlim1<>+0(SB), R2
	FMOVD   0(R2), F0
	WFCHDBS V0, V2, V0
	BEQ     L6
	WFCHEDBS        V4, V2, V6
	MOVD    $sinhxinf<>+0(SB), R1
	FMOVD   0(R1), F0
	BNE     LEXITTAGsinh
	WFCHDBS V2, V4, V2
	BNE     L16
	FNEG    F0, F0
	FMOVD   F0, ret+8(FP)
	RET

L19:
	FNEG    F0, F2
	BR      L2
L6:
	MOVD    $sinhxadd<>+0(SB), R2
	FMOVD   0(R2), F0
	MOVD    sinhrlog2<>+0(SB), R2
	WORD    $0xB3C10062     //ldgr  %f6,%r2
	WFMSDB  V4, V6, V0, V16
	FMOVD   sinhrodataL21<>+8(SB), F6
	WFADB   V0, V16, V0
	FMOVD   sinhrodataL21<>+0(SB), F3
	WFMSDB  V0, V6, V4, V6
	MOVD    $sinhe9<>+0(SB), R2
	WFMADB  V0, V3, V6, V0
	FMOVD   0(R2), F1
	MOVD    $sinhe7<>+0(SB), R2
	WFMDB   V0, V0, V6
	FMOVD   0(R2), F5
	MOVD    $sinhe8<>+0(SB), R2
	FMOVD   0(R2), F3
	MOVD    $sinhe6<>+0(SB), R2
	WFMADB  V6, V1, V5, V1
	FMOVD   0(R2), F5
	MOVD    $sinhe5<>+0(SB), R2
	FMOVD   0(R2), F7
	MOVD    $sinhe3<>+0(SB), R2
	WFMADB  V6, V3, V5, V3
	FMOVD   0(R2), F5
	MOVD    $sinhe4<>+0(SB), R2
	WFMADB  V6, V7, V5, V7
	FMOVD   0(R2), F5
	MOVD    $sinhe2<>+0(SB), R2
	VLEG    $0, 0(R2), V20
	WFMDB   V6, V6, V18
	WFMADB  V6, V5, V20, V5
	WFMADB  V1, V18, V7, V1
	FNEG    F0, F0
	WFMADB  V3, V18, V5, V3
	MOVD    $sinhe1<>+0(SB), R3
	WFCEDBS V2, V4, V2
	FMOVD   0(R3), F5
	MOVD    $sinhe0<>+0(SB), R3
	WFMADB  V6, V1, V5, V1
	FMOVD   0(R3), F5
	VLGVG   $0, V16, R2
	WFMADB  V6, V3, V5, V6
	RLL     $3, R2, R2
	WORD    $0xEC12000F     //risbgn %r1,%r2,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	BEQ     L9
	WFMSDB  V0, V1, V6, V0
	MOVD    $sinhx4ff<>+0(SB), R3
	FNEG    F0, F0
	FMOVD   0(R3), F2
	FMUL    F2, F0
	ANDW    $0xFFFF, R2
	WORD    $0xA53FEFB6     //llill %r3,61366
	SUBW    R2, R3, R2
	WORD    $0xEC12000F     //risbgn %r1,%r2,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WORD    $0xB3C10021     //ldgr %f2,%r1
	FMUL    F2, F0
	FMOVD   F0, ret+8(FP)
	RET

L20:
	MOVD    $sinhxadd<>+0(SB), R2
	FMOVD   0(R2), F2
	MOVD    sinhrlog2<>+0(SB), R2
	WORD    $0xB3C10002     //ldgr  %f0,%r2
	WFMSDB  V4, V0, V2, V6
	FMOVD   sinhrodataL21<>+8(SB), F0
	FADD    F6, F2
	MOVD    $sinhe9<>+0(SB), R2
	FMSUB   F0, F2, F4, F4
	FMOVD   0(R2), F1
	FMOVD   sinhrodataL21<>+0(SB), F3
	MOVD    $sinhe7<>+0(SB), R2
	FMADD   F3, F2, F4, F4
	FMOVD   0(R2), F0
	MOVD    $sinhe8<>+0(SB), R2
	WFMDB   V4, V4, V2
	FMOVD   0(R2), F3
	MOVD    $sinhe6<>+0(SB), R2
	FMOVD   0(R2), F5
	WORD    $0xB3CD0026     //lgdr %r2,%f6
	RLL     $3, R2, R2
	WORD    $0xEC12000F     //risbgn %r1,%r2,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WFMADB  V2, V1, V0, V1
	WORD    $0xB3C10001     //ldgr  %f0,%r1
	MOVD    $sinhe5<>+0(SB), R1
	WFMADB  V2, V3, V5, V3
	FMOVD   0(R1), F5
	MOVD    $sinhe3<>+0(SB), R1
	FMOVD   0(R1), F6
	WFMDB   V2, V2, V7
	WFMADB  V2, V5, V6, V5
	WORD    $0xA7487FB6     //lhi %r4,32694
	FNEG    F4, F4
	ANDW    $0xFFFF, R2
	SUBW    R2, R4, R2
	WORD    $0xEC32000F     //risbgn %r3,%r2,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WORD    $0xB3C10063     //ldgr  %f6,%r3
	WFADB   V0, V6, V16
	MOVD    $sinhe4<>+0(SB), R1
	WFMADB  V1, V7, V5, V1
	WFMDB   V4, V16, V4
	FMOVD   0(R1), F5
	MOVD    $sinhe2<>+0(SB), R1
	VLEG    $0, 0(R1), V16
	MOVD    $sinhe1<>+0(SB), R1
	WFMADB  V2, V5, V16, V5
	VLEG    $0, 0(R1), V16
	WFMADB  V3, V7, V5, V3
	WFMADB  V2, V1, V16, V1
	FSUB    F6, F0
	FMUL    F1, F4
	MOVD    $sinhe0<>+0(SB), R1
	FMOVD   0(R1), F6
	WFMADB  V2, V3, V6, V2
	WFMADB  V0, V2, V4, V0
	FMOVD   F0, ret+8(FP)
	RET

L9:
	WFMADB  V0, V1, V6, V0
	MOVD    $sinhx4ff<>+0(SB), R3
	FMOVD   0(R3), F2
	FMUL    F2, F0
	WORD    $0xA72AF000     //ahi   %r2,-4096
	WORD    $0xEC12000F     //risbgn %r1,%r2,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WORD    $0xB3C10021     //ldgr %f2,%r1
	FMUL    F2, F0
	FMOVD   F0, ret+8(FP)
	RET

L16:
	FMOVD   F0, ret+8(FP)
	RET

LEXITTAGsinh:
sinhIsInf:
sinhIsZero:
	FMOVD   F0, ret+8(FP)
	RET
