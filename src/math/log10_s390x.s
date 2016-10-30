// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Minimax polynomial coefficients and other constants
DATA log10rodataL19<>+0(SB)/8, $0.000000000000000000E+00
DATA log10rodataL19<>+8(SB)/8, $-1.0
DATA log10rodataL19<>+16(SB)/8, $0x7FF8000000000000   //+NanN
DATA log10rodataL19<>+24(SB)/8, $.15375570329280596749
DATA log10rodataL19<>+32(SB)/8, $.60171950900703668594E+04
DATA log10rodataL19<>+40(SB)/8, $-1.9578460454940795898
DATA log10rodataL19<>+48(SB)/8, $0.78962633073318517310E-01
DATA log10rodataL19<>+56(SB)/8, $-.71784211884836937993E-02
DATA log10rodataL19<>+64(SB)/8, $0.87011165920689940661E-03
DATA log10rodataL19<>+72(SB)/8, $-.11865158981621437541E-03
DATA log10rodataL19<>+80(SB)/8, $0.17258413403018680410E-04
DATA log10rodataL19<>+88(SB)/8, $0.40752932047883484315E-06
DATA log10rodataL19<>+96(SB)/8, $-.26149194688832680410E-05
DATA log10rodataL19<>+104(SB)/8, $0.92453396963875026759E-08
DATA log10rodataL19<>+112(SB)/8, $-.64572084905921579630E-07
DATA log10rodataL19<>+120(SB)/8, $-5.5
DATA log10rodataL19<>+128(SB)/8, $18446744073709551616.
GLOBL log10rodataL19<>+0(SB), RODATA, $136

// Table of log10 correction terms
DATA log10tab2074<>+0(SB)/8, $0.254164497922885069E-01
DATA log10tab2074<>+8(SB)/8, $0.179018857989381839E-01
DATA log10tab2074<>+16(SB)/8, $0.118926768029048674E-01
DATA log10tab2074<>+24(SB)/8, $0.722595568238080033E-02
DATA log10tab2074<>+32(SB)/8, $0.376393570022739135E-02
DATA log10tab2074<>+40(SB)/8, $0.138901135928814326E-02
DATA log10tab2074<>+48(SB)/8, $0
DATA log10tab2074<>+56(SB)/8, $-0.490780466387818203E-03
DATA log10tab2074<>+64(SB)/8, $-0.159811431402137571E-03
DATA log10tab2074<>+72(SB)/8, $0.925796337165100494E-03
DATA log10tab2074<>+80(SB)/8, $0.270683176738357035E-02
DATA log10tab2074<>+88(SB)/8, $0.513079030821304758E-02
DATA log10tab2074<>+96(SB)/8, $0.815089785397996303E-02
DATA log10tab2074<>+104(SB)/8, $0.117253060262419215E-01
DATA log10tab2074<>+112(SB)/8, $0.158164239345343963E-01
DATA log10tab2074<>+120(SB)/8, $0.203903595489229786E-01
GLOBL log10tab2074<>+0(SB), RODATA, $128

// Log10 returns the decimal logarithm of the argument.
//
// Special cases are:
//      Log(+Inf) = +Inf
//      Log(0) = -Inf
//      Log(x < 0) = NaN
//      Log(NaN) = NaN
// The algorithm used is minimax polynomial approximation
// with coefficients determined with a Remez exchange algorithm.

TEXT ·log10Asm(SB),NOSPLIT,$8-16
	FMOVD   x+0(FP), F0
	MOVD    $log10rodataL19<>+0(SB), R9
	FMOVD   F0, x-8(SP)
	WORD    $0xC0298006     //iilf %r2,2147909631
	BYTE    $0x7F
	BYTE    $0xFF
	WORD    $0x5840F008     //l %r4, 8(%r15)
	SUBW    R4, R2, R3
	WORD    $0xEC5320AF     //risbg %r5,%r3,32,128+47,0
	BYTE    $0x00
	BYTE    $0x55
	MOVH    $0x0, R1
	WORD    $0xEC15001F     //risbgn %r1,%r5,64-64+0,64-64+0+32-1,64-0-32
	BYTE    $0x20
	BYTE    $0x59
	WORD    $0xC0590016     //iilf %r5,1507327
	BYTE    $0xFF
	BYTE    $0xFF
	MOVW    R4, R10
	MOVW    R5, R11
	CMPBLE  R10, R11, L2
	WORD    $0xC0297FEF     //iilf %r2,2146435071
	BYTE    $0xFF
	BYTE    $0xFF
	MOVW    R4, R10
	MOVW    R2, R11
	CMPBLE  R10, R11, L16
L3:
L1:
	FMOVD   F0, ret+8(FP)
	RET

L2:
	WORD    $0xB3120000     //ltdbr %f0,%f0
	BLEU    L13
	WORD    $0xED009080     //mdb %f0,.L20-.L19(%r9)
	BYTE    $0x00
	BYTE    $0x1C
	FMOVD   F0, x-8(SP)
	WORD    $0x5B20F008     //s %r2, 8(%r15)
	WORD    $0xEC3239BC     //risbg %r3,%r2,57,128+60,64-13
	BYTE    $0x33
	BYTE    $0x55
	ANDW    $0xFFFF0000, R2
	WORD    $0xEC12001F     //risbgn %r1,%r2,64-64+0,64-64+0+32-1,64-0-32
	BYTE    $0x20
	BYTE    $0x59
	ADDW    $0x4000000, R2
	BLEU    L17
L8:
	SRW     $8, R2, R2
	ORW     $0x45000000, R2
L4:
	FMOVD   log10rodataL19<>+120(SB), F2
	WORD    $0xB3C10041     //ldgr  %f4,%r1
	WFMADB  V4, V0, V2, V0
	FMOVD   log10rodataL19<>+112(SB), F4
	FMOVD   log10rodataL19<>+104(SB), F6
	WFMADB  V0, V6, V4, V6
	FMOVD   log10rodataL19<>+96(SB), F4
	FMOVD   log10rodataL19<>+88(SB), F1
	WFMADB  V0, V1, V4, V1
	WFMDB   V0, V0, V4
	FMOVD   log10rodataL19<>+80(SB), F2
	WFMADB  V6, V4, V1, V6
	FMOVD   log10rodataL19<>+72(SB), F1
	WFMADB  V0, V2, V1, V2
	FMOVD   log10rodataL19<>+64(SB), F1
	WORD    $0xEC3339BC     //risbg %r3,%r3,57,128+60,0
	BYTE    $0x00
	BYTE    $0x55
	WFMADB  V4, V6, V2, V6
	FMOVD   log10rodataL19<>+56(SB), F2
	WFMADB  V0, V1, V2, V1
	VLVGF   $0, R2, V2
	WFMADB  V4, V6, V1, V4
	LDEBR   F2, F2
	FMOVD   log10rodataL19<>+48(SB), F6
	WFMADB  V0, V4, V6, V4
	FMOVD   log10rodataL19<>+40(SB), F1
	FMOVD   log10rodataL19<>+32(SB), F6
	MOVD    $log10tab2074<>+0(SB), R1
	WFMADB  V2, V1, V6, V2
	WORD    $0x68331000     //ld %f3,0(%r3,%r1)
	WFMADB  V0, V4, V3, V0
	FMOVD   log10rodataL19<>+24(SB), F4
	FMADD   F4, F2, F0, F0
	FMOVD   F0, ret+8(FP)
	RET

L16:
	WORD    $0xEC2328B7     //risbg %r2,%r3,40,128+55,64-8
	BYTE    $0x38
	BYTE    $0x55
	WORD    $0xEC3339BC     //risbg %r3,%r3,57,128+60,64-13
	BYTE    $0x33
	BYTE    $0x55
	ORW     $0x45000000, R2
	BR      L4
L13:
	BGE     L18     //jnl .L18
	BVS     L18
	FMOVD   log10rodataL19<>+16(SB), F0
	BR      L1
L17:
	SRAW    $1, R2, R2
	SUBW    $0x40000000, R2
	BR      L8
L18:
	FMOVD   log10rodataL19<>+8(SB), F0
	WORD    $0xED009000     //ddb %f0,.L36-.L19(%r9)
	BYTE    $0x00
	BYTE    $0x1D
	BR      L1
