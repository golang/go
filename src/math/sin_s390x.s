// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Various constants
DATA sincosxnan<>+0(SB)/8, $0x7ff8000000000000
GLOBL sincosxnan<>+0(SB), RODATA, $8
DATA sincosxlim<>+0(SB)/8, $0x432921fb54442d19
GLOBL sincosxlim<>+0(SB), RODATA, $8
DATA sincosxadd<>+0(SB)/8, $0xc338000000000000
GLOBL sincosxadd<>+0(SB), RODATA, $8
DATA sincosxpi2l<>+0(SB)/8, $0.108285667392191389e-31
GLOBL sincosxpi2l<>+0(SB), RODATA, $8
DATA sincosxpi2m<>+0(SB)/8, $0.612323399573676480e-16
GLOBL sincosxpi2m<>+0(SB), RODATA, $8
DATA sincosxpi2h<>+0(SB)/8, $0.157079632679489656e+01
GLOBL sincosxpi2h<>+0(SB), RODATA, $8
DATA sincosrpi2<>+0(SB)/8, $0.636619772367581341e+00
GLOBL sincosrpi2<>+0(SB), RODATA, $8

// Minimax polynomial approximations
DATA sincosc0<>+0(SB)/8, $0.100000000000000000E+01
GLOBL sincosc0<>+0(SB), RODATA, $8
DATA sincosc1<>+0(SB)/8, $-.499999999999999833E+00
GLOBL sincosc1<>+0(SB), RODATA, $8
DATA sincosc2<>+0(SB)/8, $0.416666666666625843E-01
GLOBL sincosc2<>+0(SB), RODATA, $8
DATA sincosc3<>+0(SB)/8, $-.138888888885498984E-02
GLOBL sincosc3<>+0(SB), RODATA, $8
DATA sincosc4<>+0(SB)/8, $0.248015871681607202E-04
GLOBL sincosc4<>+0(SB), RODATA, $8
DATA sincosc5<>+0(SB)/8, $-.275572911309937875E-06
GLOBL sincosc5<>+0(SB), RODATA, $8
DATA sincosc6<>+0(SB)/8, $0.208735047247632818E-08
GLOBL sincosc6<>+0(SB), RODATA, $8
DATA sincosc7<>+0(SB)/8, $-.112753632738365317E-10
GLOBL sincosc7<>+0(SB), RODATA, $8
DATA sincoss0<>+0(SB)/8, $0.100000000000000000E+01
GLOBL sincoss0<>+0(SB), RODATA, $8
DATA sincoss1<>+0(SB)/8, $-.166666666666666657E+00
GLOBL sincoss1<>+0(SB), RODATA, $8
DATA sincoss2<>+0(SB)/8, $0.833333333333309209E-02
GLOBL sincoss2<>+0(SB), RODATA, $8
DATA sincoss3<>+0(SB)/8, $-.198412698410701448E-03
GLOBL sincoss3<>+0(SB), RODATA, $8
DATA sincoss4<>+0(SB)/8, $0.275573191453906794E-05
GLOBL sincoss4<>+0(SB), RODATA, $8
DATA sincoss5<>+0(SB)/8, $-.250520918387633290E-07
GLOBL sincoss5<>+0(SB), RODATA, $8
DATA sincoss6<>+0(SB)/8, $0.160571285514715856E-09
GLOBL sincoss6<>+0(SB), RODATA, $8
DATA sincoss7<>+0(SB)/8, $-.753213484933210972E-12
GLOBL sincoss7<>+0(SB), RODATA, $8

// Sin returns the sine of the radian argument x.
//
// Special cases are:
//      Sin(±0) = ±0
//      Sin(±Inf) = NaN
//      Sin(NaN) = NaN
// The algorithm used is minimax polynomial approximation.
// with coefficients determined with a Remez exchange algorithm.

TEXT ·sinAsm(SB),NOSPLIT,$0-16
	FMOVD   x+0(FP), F0
	//special case Sin(±0) = ±0
	FMOVD   $(0.0), F1
	FCMPU   F0, F1
	BEQ     sinIsZero
	WORD    $0xB3120000     //ltdbr %f0,%f0
	BLTU    L17
	FMOVD   F0, F5
L2:
	MOVD    $sincoss7<>+0(SB), R1
	FMOVD   0(R1), F4
	MOVD    $sincoss6<>+0(SB), R1
	FMOVD   0(R1), F1
	MOVD    $sincoss5<>+0(SB), R1
	VLEG    $0, 0(R1), V18
	MOVD    $sincoss4<>+0(SB), R1
	FMOVD   0(R1), F6
	MOVD    $sincoss2<>+0(SB), R1
	VLEG    $0, 0(R1), V16
	MOVD    $sincoss3<>+0(SB), R1
	FMOVD   0(R1), F7
	MOVD    $sincoss1<>+0(SB), R1
	FMOVD   0(R1), F3
	MOVD    $sincoss0<>+0(SB), R1
	FMOVD   0(R1), F2
	WFCHDBS V2, V5, V2
	BEQ     L18
	MOVD    $sincosrpi2<>+0(SB), R1
	FMOVD   0(R1), F3
	MOVD    $sincosxadd<>+0(SB), R1
	FMOVD   0(R1), F2
	WFMSDB  V0, V3, V2, V3
	FMOVD   0(R1), F6
	FADD    F3, F6
	MOVD    $sincosxpi2h<>+0(SB), R1
	FMOVD   0(R1), F2
	FMSUB   F2, F6, F0, F0
	MOVD    $sincosxpi2m<>+0(SB), R1
	FMOVD   0(R1), F4
	FMADD   F4, F6, F0, F0
	MOVD    $sincosxpi2l<>+0(SB), R1
	WFMDB   V0, V0, V1
	FMOVD   0(R1), F7
	WFMDB   V1, V1, V2
	WORD    $0xB3CD0013     //lgdr  %r1,%f3
	MOVD    $sincosxlim<>+0(SB), R2
	WORD    $0xA7110001     //tmll  %r1,1
	BEQ     L6
	FMOVD   0(R2), F0
	WFCHDBS V0, V5, V0
	BNE     L14
	MOVD    $sincosc7<>+0(SB), R2
	FMOVD   0(R2), F0
	MOVD    $sincosc6<>+0(SB), R2
	FMOVD   0(R2), F4
	MOVD    $sincosc5<>+0(SB), R2
	WFMADB  V1, V0, V4, V0
	FMOVD   0(R2), F6
	MOVD    $sincosc4<>+0(SB), R2
	WFMADB  V1, V0, V6, V0
	FMOVD   0(R2), F4
	MOVD    $sincosc2<>+0(SB), R2
	FMOVD   0(R2), F6
	WFMADB  V2, V4, V6, V4
	MOVD    $sincosc3<>+0(SB), R2
	FMOVD   0(R2), F3
	MOVD    $sincosc1<>+0(SB), R2
	WFMADB  V2, V0, V3, V0
	FMOVD   0(R2), F6
	WFMADB  V1, V4, V6, V4
	WORD    $0xA7110002     //tmll  %r1,2
	WFMADB  V2, V0, V4, V0
	MOVD    $sincosc0<>+0(SB), R1
	FMOVD   0(R1), F2
	WFMADB  V1, V0, V2, V0
	BNE     L15
	FMOVD   F0, ret+8(FP)
	RET

L6:
	FMOVD   0(R2), F4
	WFCHDBS V4, V5, V4
	BNE     L14
	MOVD    $sincoss7<>+0(SB), R2
	FMOVD   0(R2), F4
	MOVD    $sincoss6<>+0(SB), R2
	FMOVD   0(R2), F3
	MOVD    $sincoss5<>+0(SB), R2
	WFMADB  V1, V4, V3, V4
	WFMADB  V6, V7, V0, V6
	FMOVD   0(R2), F0
	MOVD    $sincoss4<>+0(SB), R2
	FMADD   F4, F1, F0, F0
	FMOVD   0(R2), F3
	MOVD    $sincoss2<>+0(SB), R2
	FMOVD   0(R2), F4
	MOVD    $sincoss3<>+0(SB), R2
	WFMADB  V2, V3, V4, V3
	FMOVD   0(R2), F4
	MOVD    $sincoss1<>+0(SB), R2
	WFMADB  V2, V0, V4, V0
	FMOVD   0(R2), F4
	WFMADB  V1, V3, V4, V3
	FNEG    F6, F4
	WFMADB  V2, V0, V3, V2
	WFMDB   V4, V1, V0
	WORD    $0xA7110002     //tmll  %r1,2
	WFMSDB  V0, V2, V6, V0
	BNE     L15
	FMOVD   F0, ret+8(FP)
	RET

L14:
	MOVD    $sincosxnan<>+0(SB), R1
	FMOVD   0(R1), F0
	FMOVD   F0, ret+8(FP)
	RET

L18:
	WFMDB   V0, V0, V2
	WFMADB  V2, V4, V1, V4
	WFMDB   V2, V2, V1
	WFMADB  V2, V4, V18, V4
	WFMADB  V1, V6, V16, V6
	WFMADB  V1, V4, V7, V4
	WFMADB  V2, V6, V3, V6
	FMUL    F0, F2
	WFMADB  V1, V4, V6, V4
	FMADD   F4, F2, F0, F0
	FMOVD   F0, ret+8(FP)
	RET

L17:
	FNEG    F0, F5
	BR      L2
L15:
	FNEG    F0, F0
	FMOVD   F0, ret+8(FP)
	RET


sinIsZero:
	FMOVD   F0, ret+8(FP)
	RET

// Cos returns the cosine of the radian argument.
//
// Special cases are:
//      Cos(±Inf) = NaN
//      Cos(NaN) = NaN
// The algorithm used is minimax polynomial approximation.
// with coefficients determined with a Remez exchange algorithm.

TEXT ·cosAsm(SB),NOSPLIT,$0-16
	FMOVD   x+0(FP), F0
	WORD    $0xB3120000     //ltdbr %f0,%f0
	BLTU    L35
	FMOVD   F0, F1
L21:
	MOVD    $sincosc7<>+0(SB), R1
	FMOVD   0(R1), F4
	MOVD    $sincosc6<>+0(SB), R1
	VLEG    $0, 0(R1), V20
	MOVD    $sincosc5<>+0(SB), R1
	VLEG    $0, 0(R1), V18
	MOVD    $sincosc4<>+0(SB), R1
	FMOVD   0(R1), F6
	MOVD    $sincosc2<>+0(SB), R1
	VLEG    $0, 0(R1), V16
	MOVD    $sincosc3<>+0(SB), R1
	FMOVD   0(R1), F7
	MOVD    $sincosc1<>+0(SB), R1
	FMOVD   0(R1), F5
	MOVD    $sincosrpi2<>+0(SB), R1
	FMOVD   0(R1), F2
	MOVD    $sincosxadd<>+0(SB), R1
	FMOVD   0(R1), F3
	MOVD    $sincoss0<>+0(SB), R1
	WFMSDB  V0, V2, V3, V2
	FMOVD   0(R1), F3
	WFCHDBS V3, V1, V3
	WORD    $0xB3CD0012     //lgdr %r1,%f2
	BEQ     L36
	MOVD    $sincosxadd<>+0(SB), R2
	FMOVD   0(R2), F4
	FADD    F2, F4
	MOVD    $sincosxpi2h<>+0(SB), R2
	FMOVD   0(R2), F2
	WFMSDB  V4, V2, V0, V2
	MOVD    $sincosxpi2m<>+0(SB), R2
	FMOVD   0(R2), F0
	WFMADB  V4, V0, V2, V0
	MOVD    $sincosxpi2l<>+0(SB), R2
	WFMDB   V0, V0, V2
	FMOVD   0(R2), F5
	WFMDB   V2, V2, V6
	MOVD    $sincosxlim<>+0(SB), R2
	WORD    $0xA7110001     //tmll %r1,1
	BNE     L25
	FMOVD   0(R2), F0
	WFCHDBS V0, V1, V0
	BNE     L33
	MOVD    $sincosc7<>+0(SB), R2
	FMOVD   0(R2), F0
	MOVD    $sincosc6<>+0(SB), R2
	FMOVD   0(R2), F4
	MOVD    $sincosc5<>+0(SB), R2
	WFMADB  V2, V0, V4, V0
	FMOVD   0(R2), F1
	MOVD    $sincosc4<>+0(SB), R2
	WFMADB  V2, V0, V1, V0
	FMOVD   0(R2), F4
	MOVD    $sincosc2<>+0(SB), R2
	FMOVD   0(R2), F1
	WFMADB  V6, V4, V1, V4
	MOVD    $sincosc3<>+0(SB), R2
	FMOVD   0(R2), F3
	MOVD    $sincosc1<>+0(SB), R2
	WFMADB  V6, V0, V3, V0
	FMOVD   0(R2), F1
	WFMADB  V2, V4, V1, V4
	WORD    $0xA7110002     //tmll %r1,2
	WFMADB  V6, V0, V4, V0
	MOVD    $sincosc0<>+0(SB), R1
	FMOVD   0(R1), F4
	WFMADB  V2, V0, V4, V0
	BNE     L34
	FMOVD   F0, ret+8(FP)
	RET

L25:
	FMOVD   0(R2), F3
	WFCHDBS V3, V1, V1
	BNE     L33
	MOVD    $sincoss7<>+0(SB), R2
	FMOVD   0(R2), F1
	MOVD    $sincoss6<>+0(SB), R2
	FMOVD   0(R2), F3
	MOVD    $sincoss5<>+0(SB), R2
	WFMADB  V2, V1, V3, V1
	FMOVD   0(R2), F3
	MOVD    $sincoss4<>+0(SB), R2
	WFMADB  V2, V1, V3, V1
	FMOVD   0(R2), F3
	MOVD    $sincoss2<>+0(SB), R2
	FMOVD   0(R2), F7
	WFMADB  V6, V3, V7, V3
	MOVD    $sincoss3<>+0(SB), R2
	FMADD   F5, F4, F0, F0
	FMOVD   0(R2), F4
	MOVD    $sincoss1<>+0(SB), R2
	FMADD   F1, F6, F4, F4
	FMOVD   0(R2), F1
	FMADD   F3, F2, F1, F1
	FMUL    F0, F2
	WFMADB  V6, V4, V1, V6
	WORD    $0xA7110002     //tmll  %r1,2
	FMADD   F6, F2, F0, F0
	BNE     L34
	FMOVD   F0, ret+8(FP)
	RET

L33:
	MOVD    $sincosxnan<>+0(SB), R1
	FMOVD   0(R1), F0
	FMOVD   F0, ret+8(FP)
	RET

L36:
	FMUL    F0, F0
	MOVD    $sincosc0<>+0(SB), R1
	WFMDB   V0, V0, V1
	WFMADB  V0, V4, V20, V4
	WFMADB  V1, V6, V16, V6
	WFMADB  V0, V4, V18, V4
	WFMADB  V0, V6, V5, V6
	WFMADB  V1, V4, V7, V4
	FMOVD   0(R1), F2
	WFMADB  V1, V4, V6, V4
	WFMADB  V0, V4, V2, V0
	FMOVD   F0, ret+8(FP)
	RET

L35:
	FNEG    F0, F1
	BR      L21
L34:
	FNEG    F0, F0
	FMOVD   F0, ret+8(FP)
	RET
