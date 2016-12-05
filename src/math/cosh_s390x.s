// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Constants
DATA coshrodataL23<>+0(SB)/8, $0.231904681384629956E-16
DATA coshrodataL23<>+8(SB)/8, $0.693147180559945286E+00
DATA coshrodataL23<>+16(SB)/8, $0.144269504088896339E+01
DATA coshrodataL23<>+24(SB)/8, $704.E0
GLOBL coshrodataL23<>+0(SB), RODATA, $32
DATA coshxinf<>+0(SB)/8, $0x7FF0000000000000
GLOBL coshxinf<>+0(SB), RODATA, $8
DATA coshxlim1<>+0(SB)/8, $800.E0
GLOBL coshxlim1<>+0(SB), RODATA, $8
DATA coshxaddhy<>+0(SB)/8, $0xc2f0000100003fdf
GLOBL coshxaddhy<>+0(SB), RODATA, $8
DATA coshx4ff<>+0(SB)/8, $0x4ff0000000000000
GLOBL coshx4ff<>+0(SB), RODATA, $8
DATA coshe1<>+0(SB)/8, $0x3ff000000000000a
GLOBL coshe1<>+0(SB), RODATA, $8

// Log multiplier table
DATA coshtab<>+0(SB)/8, $0.442737824274138381E-01
DATA coshtab<>+8(SB)/8, $0.263602189790660309E-01
DATA coshtab<>+16(SB)/8, $0.122565642281703586E-01
DATA coshtab<>+24(SB)/8, $0.143757052860721398E-02
DATA coshtab<>+32(SB)/8, $-.651375034121276075E-02
DATA coshtab<>+40(SB)/8, $-.119317678849450159E-01
DATA coshtab<>+48(SB)/8, $-.150868749549871069E-01
DATA coshtab<>+56(SB)/8, $-.161992609578469234E-01
DATA coshtab<>+64(SB)/8, $-.154492360403337917E-01
DATA coshtab<>+72(SB)/8, $-.129850717389178721E-01
DATA coshtab<>+80(SB)/8, $-.892902649276657891E-02
DATA coshtab<>+88(SB)/8, $-.338202636596794887E-02
DATA coshtab<>+96(SB)/8, $0.357266307045684762E-02
DATA coshtab<>+104(SB)/8, $0.118665304327406698E-01
DATA coshtab<>+112(SB)/8, $0.214434994118118914E-01
DATA coshtab<>+120(SB)/8, $0.322580645161290314E-01
GLOBL coshtab<>+0(SB), RODATA, $128

// Minimax polynomial approximations
DATA coshe2<>+0(SB)/8, $0.500000000000004237e+00
GLOBL coshe2<>+0(SB), RODATA, $8
DATA coshe3<>+0(SB)/8, $0.166666666630345592e+00
GLOBL coshe3<>+0(SB), RODATA, $8
DATA coshe4<>+0(SB)/8, $0.416666664838056960e-01
GLOBL coshe4<>+0(SB), RODATA, $8
DATA coshe5<>+0(SB)/8, $0.833349307718286047e-02
GLOBL coshe5<>+0(SB), RODATA, $8
DATA coshe6<>+0(SB)/8, $0.138926439368309441e-02
GLOBL coshe6<>+0(SB), RODATA, $8

// Cosh returns the hyperbolic cosine of x.
//
// Special cases are:
//      Cosh(±0) = 1
//      Cosh(±Inf) = +Inf
//      Cosh(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT ·coshAsm(SB),NOSPLIT,$0-16
	FMOVD   x+0(FP), F0
	MOVD    $coshrodataL23<>+0(SB), R9
	WORD    $0xB3120000     //ltdbr %f0,%f0
	MOVD    $0x4086000000000000, R2
	MOVD    $0x4086000000000000, R3
	BLTU    L19
	FMOVD   F0, F4
L2:
	WORD    $0xED409018     //cdb %f4,.L24-.L23(%r9)
	BYTE    $0x00
	BYTE    $0x19
	BGE     L14     //jnl   .L14
	BVS     L14
	WFCEDBS V4, V4, V2
	BEQ     L20
L1:
	FMOVD   F0, ret+8(FP)
	RET

L14:
	WFCEDBS V4, V4, V2
	BVS     L1
	MOVD    $coshxlim1<>+0(SB), R1
	FMOVD   0(R1), F2
	WFCHEDBS        V4, V2, V2
	BEQ     L21
	MOVD    $coshxaddhy<>+0(SB), R1
	FMOVD   coshrodataL23<>+16(SB), F5
	FMOVD   0(R1), F2
	WFMSDB  V0, V5, V2, V5
	FMOVD   coshrodataL23<>+8(SB), F3
	FADD    F5, F2
	MOVD    $coshe6<>+0(SB), R1
	WFMSDB  V2, V3, V0, V3
	FMOVD   0(R1), F6
	WFMDB   V3, V3, V1
	MOVD    $coshe4<>+0(SB), R1
	FMOVD   coshrodataL23<>+0(SB), F7
	WFMADB  V2, V7, V3, V2
	FMOVD   0(R1), F3
	MOVD    $coshe5<>+0(SB), R1
	WFMADB  V1, V6, V3, V6
	FMOVD   0(R1), F7
	MOVD    $coshe3<>+0(SB), R1
	FMOVD   0(R1), F3
	WFMADB  V1, V7, V3, V7
	FNEG    F2, F3
	WORD    $0xB3CD0015     //lgdr %r1,%f5
	MOVD    $coshe2<>+0(SB), R3
	WFCEDBS V4, V0, V0
	FMOVD   0(R3), F5
	MOVD    $coshe1<>+0(SB), R3
	WFMADB  V1, V6, V5, V6
	FMOVD   0(R3), F5
	WORD    $0xEC21000F     //risbgn %r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WFMADB  V1, V7, V5, V1
	BVS     L22
	WORD    $0xEC4139BC     //risbg %r4,%r1,57,128+60,3
	BYTE    $0x03
	BYTE    $0x55
	MOVD    $coshtab<>+0(SB), R3
	WFMADB  V3, V6, V1, V6
	WORD    $0x68043000     //ld    %f0,0(%r4,%r3)
	FMSUB   F0, F3, F2, F2
	WORD    $0xA71AF000     //ahi   %r1,-4096
	WFMADB  V2, V6, V0, V6
L17:
	WORD    $0xEC21000F     //risbgn %r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WORD    $0xB3C10022     //ldgr %f2,%r2
	FMADD   F2, F6, F2, F2
	MOVD    $coshx4ff<>+0(SB), R1
	FMOVD   0(R1), F0
	FMUL    F2, F0
	FMOVD   F0, ret+8(FP)
	RET

L19:
	FNEG    F0, F4
	BR      L2
L20:
	MOVD    $coshxaddhy<>+0(SB), R1
	FMOVD   coshrodataL23<>+16(SB), F3
	FMOVD   0(R1), F2
	WFMSDB  V0, V3, V2, V3
	FMOVD   coshrodataL23<>+8(SB), F4
	FADD    F3, F2
	MOVD    $coshe6<>+0(SB), R1
	FMSUB   F4, F2, F0, F0
	FMOVD   0(R1), F6
	WFMDB   V0, V0, V1
	MOVD    $coshe4<>+0(SB), R1
	FMOVD   0(R1), F4
	MOVD    $coshe5<>+0(SB), R1
	FMOVD   coshrodataL23<>+0(SB), F5
	WFMADB  V1, V6, V4, V6
	FMADD   F5, F2, F0, F0
	FMOVD   0(R1), F2
	MOVD    $coshe3<>+0(SB), R1
	FMOVD   0(R1), F4
	WFMADB  V1, V2, V4, V2
	MOVD    $coshe2<>+0(SB), R1
	FMOVD   0(R1), F5
	FNEG    F0, F4
	WFMADB  V1, V6, V5, V6
	MOVD    $coshe1<>+0(SB), R1
	FMOVD   0(R1), F5
	WFMADB  V1, V2, V5, V1
	WORD    $0xB3CD0013     //lgdr  %r1,%f3
	MOVD    $coshtab<>+0(SB), R5
	WFMADB  V4, V6, V1, V3
	WORD    $0xEC4139BC     //risbg %r4,%r1,57,128+60,3
	BYTE    $0x03
	BYTE    $0x55
	WFMSDB  V4, V6, V1, V6
	WORD    $0x68145000     //ld %f1,0(%r4,%r5)
	WFMSDB  V4, V1, V0, V2
	WORD    $0xA7487FBE     //lhi %r4,32702
	FMADD   F3, F2, F1, F1
	SUBW    R1, R4
	WORD    $0xECC439BC     //risbg %r12,%r4,57,128+60,3
	BYTE    $0x03
	BYTE    $0x55
	WORD    $0x682C5000     //ld %f2,0(%r12,%r5)
	FMSUB   F2, F4, F0, F0
	WORD    $0xEC21000F     //risbgn %r2,%r1,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WFMADB  V0, V6, V2, V6
	WORD    $0xEC34000F     //risbgn %r3,%r4,64-64+0,64-64+0+16-1,64-0-16
	BYTE    $0x30
	BYTE    $0x59
	WORD    $0xB3C10022     //ldgr %f2,%r2
	WORD    $0xB3C10003     //ldgr %f0,%r3
	FMADD   F2, F1, F2, F2
	FMADD   F0, F6, F0, F0
	FADD    F2, F0
	FMOVD   F0, ret+8(FP)
	RET

L22:
	WORD    $0xA7387FBE     //lhi %r3,32702
	MOVD    $coshtab<>+0(SB), R4
	SUBW    R1, R3
	WFMSDB  V3, V6, V1, V6
	WORD    $0xEC3339BC     //risbg %r3,%r3,57,128+60,3
	BYTE    $0x03
	BYTE    $0x55
	WORD    $0x68034000     //ld %f0,0(%r3,%r4)
	FMSUB   F0, F3, F2, F2
	WORD    $0xA7386FBE     //lhi %r3,28606
	WFMADB  V2, V6, V0, V6
	SUBW    R1, R3, R1
	BR      L17
L21:
	MOVD    $coshxinf<>+0(SB), R1
	FMOVD   0(R1), F0
	FMOVD   F0, ret+8(FP)
	RET

