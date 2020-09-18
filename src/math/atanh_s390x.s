// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial coefficients and other constants
DATA ·atanhrodataL10<> + 0(SB)/8, $.41375273347623353626
DATA ·atanhrodataL10<> + 8(SB)/8, $.51487302528619766235E+04
DATA ·atanhrodataL10<> + 16(SB)/8, $-1.67526912689208984375
DATA ·atanhrodataL10<> + 24(SB)/8, $0.181818181818181826E+00
DATA ·atanhrodataL10<> + 32(SB)/8, $-.165289256198351540E-01
DATA ·atanhrodataL10<> + 40(SB)/8, $0.200350613573012186E-02
DATA ·atanhrodataL10<> + 48(SB)/8, $0.397389654305194527E-04
DATA ·atanhrodataL10<> + 56(SB)/8, $-.273205381970859341E-03
DATA ·atanhrodataL10<> + 64(SB)/8, $0.938370938292558173E-06
DATA ·atanhrodataL10<> + 72(SB)/8, $-.148682720127920854E-06
DATA ·atanhrodataL10<> + 80(SB)/8, $ 0.212881813645679599E-07
DATA ·atanhrodataL10<> + 88(SB)/8, $-.602107458843052029E-05
DATA ·atanhrodataL10<> + 96(SB)/8, $-5.5
DATA ·atanhrodataL10<> + 104(SB)/8, $-0.5
DATA ·atanhrodataL10<> + 112(SB)/8, $0.0
DATA ·atanhrodataL10<> + 120(SB)/8, $0x7ff8000000000000      //Nan
DATA ·atanhrodataL10<> + 128(SB)/8, $-1.0
DATA ·atanhrodataL10<> + 136(SB)/8, $1.0
DATA ·atanhrodataL10<> + 144(SB)/8, $1.0E-20
GLOBL ·atanhrodataL10<> + 0(SB), RODATA, $152

// Table of log correction terms
DATA ·atanhtab2076<> + 0(SB)/8, $0.585235384085551248E-01
DATA ·atanhtab2076<> + 8(SB)/8, $0.412206153771168640E-01
DATA ·atanhtab2076<> + 16(SB)/8, $0.273839003221648339E-01
DATA ·atanhtab2076<> + 24(SB)/8, $0.166383778368856480E-01
DATA ·atanhtab2076<> + 32(SB)/8, $0.866678223433169637E-02
DATA ·atanhtab2076<> + 40(SB)/8, $0.319831684989627514E-02
DATA ·atanhtab2076<> + 48(SB)/8, $0.000000000000000000E+00
DATA ·atanhtab2076<> + 56(SB)/8, $-.113006378583725549E-02
DATA ·atanhtab2076<> + 64(SB)/8, $-.367979419636602491E-03
DATA ·atanhtab2076<> + 72(SB)/8, $0.213172484510484979E-02
DATA ·atanhtab2076<> + 80(SB)/8, $0.623271047682013536E-02
DATA ·atanhtab2076<> + 88(SB)/8, $0.118140812789696885E-01
DATA ·atanhtab2076<> + 96(SB)/8, $0.187681358930914206E-01
DATA ·atanhtab2076<> + 104(SB)/8, $0.269985148668178992E-01
DATA ·atanhtab2076<> + 112(SB)/8, $0.364186619761331328E-01
DATA ·atanhtab2076<> + 120(SB)/8, $0.469505379381388441E-01
GLOBL ·atanhtab2076<> + 0(SB), RODATA, $128

// Table of +/- .5
DATA ·atanhtabh2075<> + 0(SB)/8, $0.5
DATA ·atanhtabh2075<> + 8(SB)/8, $-.5
GLOBL ·atanhtabh2075<> + 0(SB), RODATA, $16

// Atanh returns the inverse hyperbolic tangent of the argument.
//
// Special cases are:
//      Atanh(1) = +Inf
//      Atanh(±0) = ±0
//      Atanh(-1) = -Inf
//      Atanh(x) = NaN if x < -1 or x > 1
//      Atanh(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT    ·atanhAsm(SB), NOSPLIT, $0-16
    FMOVD   x+0(FP), F0
    MOVD    $·atanhrodataL10<>+0(SB), R5
    LGDR    F0, R1
    WORD    $0xC0393FEF //iilf  %r3,1072693247
    BYTE    $0xFF
    BYTE    $0xFF
    SRAD    $32, R1
    WORD    $0xB9170021 //llgtr %r2,%r1
    MOVW    R2, R6
    MOVW    R3, R7
    CMPBGT  R6, R7, L2
    WORD    $0xC0392FFF //iilf  %r3,805306367
    BYTE    $0xFF
    BYTE    $0xFF
    MOVW    R2, R6
    MOVW    R3, R7
    CMPBGT  R6, R7, L9
L3:
    FMOVD   144(R5), F2
    FMADD   F2, F0, F0
L1:
    FMOVD   F0, ret+8(FP)
    RET

L2:
    WORD    $0xED005088 //cdb   %f0,.L12-.L10(%r5)
    BYTE    $0x00
    BYTE    $0x19
    BEQ L5
    WORD    $0xED005080 //cdb   %f0,.L13-.L10(%r5)
    BYTE    $0x00
    BYTE    $0x19
    BEQ L5
    WFCEDBS V0, V0, V2
    BVS L1
    FMOVD   120(R5), F0
    BR  L1
L5:
    WORD    $0xED005070 //ddb   %f0,.L15-.L10(%r5)
    BYTE    $0x00
    BYTE    $0x1D
    FMOVD   F0, ret+8(FP)
    RET

L9:
    FMOVD   F0, F2
    MOVD    $·atanhtabh2075<>+0(SB), R2
    SRW $31, R1, R1
    FMOVD   104(R5), F4
    MOVW    R1, R1
    SLD $3, R1, R1
    WORD    $0x68012000 //ld    %f0,0(%r1,%r2)
    WFMADB  V2, V4, V0, V4
    VLEG    $0, 96(R5), V16
    FDIV    F4, F2
    WORD    $0xC0298006 //iilf  %r2,2147909631
    BYTE    $0x7F
    BYTE    $0xFF
    FMOVD   88(R5), F6
    FMOVD   80(R5), F1
    FMOVD   72(R5), F7
    FMOVD   64(R5), F5
    FMOVD   F2, F4
    WORD    $0xED405088 //adb   %f4,.L12-.L10(%r5)
    BYTE    $0x00
    BYTE    $0x1A
    LGDR    F4, R4
    SRAD    $32, R4
    FMOVD   F4, F3
    WORD    $0xED305088 //sdb   %f3,.L12-.L10(%r5)
    BYTE    $0x00
    BYTE    $0x1B
    SUBW    R4, R2
    WFSDB   V3, V2, V3
    RISBGZ  $32, $47, $0, R2, R1
    SLD $32, R1, R1
    LDGR    R1, F2
    WFMADB  V4, V2, V16, V4
    SRAW    $8, R2, R1
    WFMADB  V4, V5, V6, V5
    WFMDB   V4, V4, V6
    WFMADB  V4, V1, V7, V1
    WFMADB  V2, V3, V4, V2
    WFMADB  V1, V6, V5, V1
    FMOVD   56(R5), F3
    FMOVD   48(R5), F5
    WFMADB  V4, V5, V3, V4
    FMOVD   40(R5), F3
    FMADD   F1, F6, F4
    FMOVD   32(R5), F1
    FMADD   F3, F2, F1
    ANDW    $0xFFFFFF00, R1
    WFMADB  V6, V4, V1, V6
    FMOVD   24(R5), F3
    ORW $0x45000000, R1
    WFMADB  V2, V6, V3, V6
    VLVGF   $0, R1, V4
    LDEBR   F4, F4
    RISBGZ  $57, $60, $51, R2, R2
    MOVD    $·atanhtab2076<>+0(SB), R1
    FMOVD   16(R5), F3
    WORD    $0x68521000 //ld    %f5,0(%r2,%r1)
    FMOVD   8(R5), F1
    WFMADB  V2, V6, V5, V2
    WFMADB  V4, V3, V1, V4
    FMOVD   0(R5), F6
    FMADD   F6, F4, F2
    FMUL    F2, F0
    FMOVD   F0, ret+8(FP)
    RET
