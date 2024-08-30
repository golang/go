// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial approximations
DATA tanhrodataL18<>+0(SB)/8, $-1.0
DATA tanhrodataL18<>+8(SB)/8, $-2.0
DATA tanhrodataL18<>+16(SB)/8, $1.0
DATA tanhrodataL18<>+24(SB)/8, $2.0
DATA tanhrodataL18<>+32(SB)/8, $0.20000000000000011868E+01
DATA tanhrodataL18<>+40(SB)/8, $0.13333333333333341256E+01
DATA tanhrodataL18<>+48(SB)/8, $0.26666666663549111502E+00
DATA tanhrodataL18<>+56(SB)/8, $0.66666666658721844678E+00
DATA tanhrodataL18<>+64(SB)/8, $0.88890217768964374821E-01
DATA tanhrodataL18<>+72(SB)/8, $0.25397199429103821138E-01
DATA tanhrodataL18<>+80(SB)/8, $-.346573590279972643E+00
DATA tanhrodataL18<>+88(SB)/8, $20.E0
GLOBL tanhrodataL18<>+0(SB), RODATA, $96

// Constants
DATA tanhrlog2<>+0(SB)/8, $0x4007154760000000
GLOBL tanhrlog2<>+0(SB), RODATA, $8
DATA tanhxadd<>+0(SB)/8, $0xc2f0000100003ff0
GLOBL tanhxadd<>+0(SB), RODATA, $8
DATA tanhxmone<>+0(SB)/8, $-1.0
GLOBL tanhxmone<>+0(SB), RODATA, $8
DATA tanhxzero<>+0(SB)/8, $0
GLOBL tanhxzero<>+0(SB), RODATA, $8

// Polynomial coefficients
DATA tanhtab<>+0(SB)/8, $0.000000000000000000E+00
DATA tanhtab<>+8(SB)/8, $-.171540871271399150E-01
DATA tanhtab<>+16(SB)/8, $-.306597931864376363E-01
DATA tanhtab<>+24(SB)/8, $-.410200970469965021E-01
DATA tanhtab<>+32(SB)/8, $-.486343079978231466E-01
DATA tanhtab<>+40(SB)/8, $-.538226193725835820E-01
DATA tanhtab<>+48(SB)/8, $-.568439602538111520E-01
DATA tanhtab<>+56(SB)/8, $-.579091847395528847E-01
DATA tanhtab<>+64(SB)/8, $-.571909584179366341E-01
DATA tanhtab<>+72(SB)/8, $-.548312665987204407E-01
DATA tanhtab<>+80(SB)/8, $-.509471843643441085E-01
DATA tanhtab<>+88(SB)/8, $-.456353588448863359E-01
DATA tanhtab<>+96(SB)/8, $-.389755254243262365E-01
DATA tanhtab<>+104(SB)/8, $-.310332908285244231E-01
DATA tanhtab<>+112(SB)/8, $-.218623539150173528E-01
DATA tanhtab<>+120(SB)/8, $-.115062908917949451E-01
GLOBL tanhtab<>+0(SB), RODATA, $128

// Tanh returns the hyperbolic tangent of the argument.
//
// Special cases are:
//      Tanh(±0) = ±0
//      Tanh(±Inf) = ±1
//      Tanh(NaN) = NaN
// The algorithm used is minimax polynomial approximation using a table of
// polynomial coefficients determined with a Remez exchange algorithm.

TEXT ·tanhAsm(SB),NOSPLIT,$0-16
	FMOVD   x+0(FP), F0
	// special case Tanh(±0) = ±0
	FMOVD   $(0.0), F1
	FCMPU   F0, F1
	BEQ     tanhIsZero
	MOVD    $tanhrodataL18<>+0(SB), R5
	LTDBR	F0, F0
	MOVD    $0x4034000000000000, R1
	BLTU    L15
	FMOVD   F0, F1
L2:
	MOVD    $tanhxadd<>+0(SB), R2
	FMOVD   0(R2), F2
	MOVD    tanhrlog2<>+0(SB), R2
	LDGR    R2, F4
	WFMSDB  V0, V4, V2, V4
	MOVD    $tanhtab<>+0(SB), R3
	LGDR    F4, R2
	RISBGZ	$57, $60, $3, R2, R4
	WORD    $0xED105058     //cdb %f1,.L19-.L18(%r5)
	BYTE    $0x00
	BYTE    $0x19
	RISBGN	$0, $15, $48, R2, R1
	WORD    $0x68543000     //ld %f5,0(%r4,%r3)
	LDGR    R1, F6
	BLT     L3
	MOVD    $tanhxzero<>+0(SB), R1
	FMOVD   0(R1), F2
	WFCHDBS V0, V2, V4
	BEQ     L9
	WFCHDBS V2, V0, V2
	BNE     L1
	MOVD    $tanhxmone<>+0(SB), R1
	FMOVD   0(R1), F0
	FMOVD   F0, ret+8(FP)
	RET

L3:
	FADD    F4, F2
	FMOVD   tanhrodataL18<>+80(SB), F4
	FMADD   F4, F2, F0
	FMOVD   tanhrodataL18<>+72(SB), F1
	WFMDB   V0, V0, V3
	FMOVD   tanhrodataL18<>+64(SB), F2
	WFMADB  V0, V1, V2, V1
	FMOVD   tanhrodataL18<>+56(SB), F4
	FMOVD   tanhrodataL18<>+48(SB), F2
	WFMADB  V1, V3, V4, V1
	FMOVD   tanhrodataL18<>+40(SB), F4
	WFMADB  V3, V2, V4, V2
	FMOVD   tanhrodataL18<>+32(SB), F4
	WORD    $0xB9270022     //lhr %r2,%r2
	WFMADB  V3, V1, V4, V1
	FMOVD   tanhrodataL18<>+24(SB), F4
	WFMADB  V3, V2, V4, V3
	WFMADB  V0, V5, V0, V2
	WFMADB  V0, V1, V3, V0
	WORD    $0xA7183ECF     //lhi %r1,16079
	WFMADB  V0, V2, V5, V2
	FMUL    F6, F2
	MOVW    R2, R10
	MOVW    R1, R11
	CMPBLE  R10, R11, L16
	FMOVD   F6, F0
	WORD    $0xED005010     //adb %f0,.L28-.L18(%r5)
	BYTE    $0x00
	BYTE    $0x1A
	WORD    $0xA7184330     //lhi %r1,17200
	FADD    F2, F0
	MOVW    R2, R10
	MOVW    R1, R11
	CMPBGT  R10, R11, L17
	WORD    $0xED605010     //sdb %f6,.L28-.L18(%r5)
	BYTE    $0x00
	BYTE    $0x1B
	FADD    F6, F2
	WFDDB   V0, V2, V0
	FMOVD   F0, ret+8(FP)
	RET

L9:
	FMOVD   tanhrodataL18<>+16(SB), F0
L1:
	FMOVD   F0, ret+8(FP)
	RET

L15:
	FNEG    F0, F1
	BR      L2
L16:
	FADD    F6, F2
	FMOVD   tanhrodataL18<>+8(SB), F0
	FMADD   F4, F2, F0
	FMOVD   tanhrodataL18<>+0(SB), F4
	FNEG    F0, F0
	WFMADB  V0, V2, V4, V0
	FMOVD   F0, ret+8(FP)
	RET

L17:
	WFDDB   V0, V4, V0
	FMOVD   tanhrodataL18<>+16(SB), F2
	WFSDB   V0, V2, V0
	FMOVD   F0, ret+8(FP)
	RET

tanhIsZero:      //return ±0
	FMOVD   F0, ret+8(FP)
	RET
