// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Asin(SB),NOSPLIT,$0
	BR ·asin(SB)

TEXT ·Acos(SB),NOSPLIT,$0
	BR ·acos(SB)

TEXT ·Atan2(SB),NOSPLIT,$0
	BR ·atan2(SB)

TEXT ·Atan(SB),NOSPLIT,$0
	BR ·atan(SB)

TEXT ·Exp2(SB),NOSPLIT,$0
	BR ·exp2(SB)

TEXT ·Expm1(SB),NOSPLIT,$0
	BR ·expm1(SB)

TEXT ·Exp(SB),NOSPLIT,$0
	BR ·exp(SB)

TEXT ·Frexp(SB),NOSPLIT,$0
	BR ·frexp(SB)

TEXT ·Hypot(SB),NOSPLIT,$0
	BR ·hypot(SB)

TEXT ·Ldexp(SB),NOSPLIT,$0
	BR ·ldexp(SB)

TEXT ·Log2(SB),NOSPLIT,$0
	BR ·log2(SB)

TEXT ·Log1p(SB),NOSPLIT,$0
	BR ·log1p(SB)

TEXT ·Log(SB),NOSPLIT,$0
	BR ·log(SB)

TEXT ·Modf(SB),NOSPLIT,$0
	BR ·modf(SB)

TEXT ·Mod(SB),NOSPLIT,$0
	BR ·mod(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	BR ·remainder(SB)

TEXT ·Sincos(SB),NOSPLIT,$0
	BR ·sincos(SB)

TEXT ·Tan(SB),NOSPLIT,$0
	BR ·tan(SB)

//if go assembly use vector instruction
TEXT ·hasVectorFacility(SB),NOSPLIT,$24-1
	MOVD    $x-24(SP), R1
	XC      $24, 0(R1), 0(R1) // clear the storage
	MOVD    $2, R0            // R0 is the number of double words stored -1
	WORD    $0xB2B01000       // STFLE 0(R1)
	XOR     R0, R0            // reset the value of R0
	MOVBZ   z-8(SP), R1
	AND     $0x40, R1
	BEQ     novector
vectorinstalled:
	// check if the vector instruction has been enabled
	VLEIB   $0, $0xF, V16
	VLGVB   $0, V16, R1
	CMPBNE  R1, $0xF, novector
	MOVB    $1, ret+0(FP) // have vx
	RET
novector:
	MOVB    $0, ret+0(FP) // no vx
	RET

TEXT ·Log10(SB),NOSPLIT,$0
	MOVD    log10vectorfacility+0x00(SB),R1
	BR      (R1)

TEXT ·log10TrampolineSetup(SB),NOSPLIT, $0
	MOVB    ·hasVX(SB), R1
	CMPBEQ  R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD    $log10vectorfacility+0x00(SB), R1
	MOVD    $·log10(SB), R2
	MOVD    R2, 0(R1)
	BR      ·log10(SB)
vectorimpl:
	MOVD    $log10vectorfacility+0x00(SB), R1
	MOVD    $·log10Asm(SB), R2
	MOVD    R2, 0(R1)
	BR      ·log10Asm(SB)

GLOBL log10vectorfacility+0x00(SB), NOPTR, $8
DATA log10vectorfacility+0x00(SB)/8, $·log10TrampolineSetup(SB)


TEXT ·Cos(SB),NOSPLIT,$0
	MOVD    cosvectorfacility+0x00(SB),R1
	BR      (R1)

TEXT ·cosTrampolineSetup(SB),NOSPLIT, $0
	MOVB    ·hasVX(SB), R1
	CMPBEQ  R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD    $cosvectorfacility+0x00(SB), R1
	MOVD    $·cos(SB), R2
	MOVD    R2, 0(R1)
	BR      ·cos(SB)
vectorimpl:
	MOVD    $cosvectorfacility+0x00(SB), R1
	MOVD    $·cosAsm(SB), R2
	MOVD    R2, 0(R1)
	BR      ·cosAsm(SB)

GLOBL cosvectorfacility+0x00(SB), NOPTR, $8
DATA cosvectorfacility+0x00(SB)/8, $·cosTrampolineSetup(SB)


TEXT ·Cosh(SB),NOSPLIT,$0
	MOVD    coshvectorfacility+0x00(SB),R1
	BR      (R1)

TEXT ·coshTrampolineSetup(SB),NOSPLIT, $0
	MOVB    ·hasVX(SB), R1
	CMPBEQ  R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD    $coshvectorfacility+0x00(SB), R1
	MOVD    $·cosh(SB), R2
	MOVD    R2, 0(R1)
	BR      ·cosh(SB)
vectorimpl:
	MOVD    $coshvectorfacility+0x00(SB), R1
	MOVD    $·coshAsm(SB), R2
	MOVD    R2, 0(R1)
	BR      ·coshAsm(SB)

GLOBL coshvectorfacility+0x00(SB), NOPTR, $8
DATA coshvectorfacility+0x00(SB)/8, $·coshTrampolineSetup(SB)


TEXT ·Sin(SB),NOSPLIT,$0
	MOVD    sinvectorfacility+0x00(SB),R1
	BR      (R1)

TEXT ·sinTrampolineSetup(SB),NOSPLIT, $0
	MOVB    ·hasVX(SB), R1
	CMPBEQ  R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD    $sinvectorfacility+0x00(SB), R1
	MOVD    $·sin(SB), R2
	MOVD    R2, 0(R1)
	BR      ·sin(SB)
vectorimpl:
	MOVD    $sinvectorfacility+0x00(SB), R1
	MOVD    $·sinAsm(SB), R2
	MOVD    R2, 0(R1)
	BR      ·sinAsm(SB)

GLOBL sinvectorfacility+0x00(SB), NOPTR, $8
DATA sinvectorfacility+0x00(SB)/8, $·sinTrampolineSetup(SB)


TEXT ·Sinh(SB),NOSPLIT,$0
	MOVD    sinhvectorfacility+0x00(SB),R1
	BR      (R1)

TEXT ·sinhTrampolineSetup(SB),NOSPLIT, $0
	MOVB    ·hasVX(SB), R1
	CMPBEQ  R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD    $sinhvectorfacility+0x00(SB), R1
	MOVD    $·sinh(SB), R2
	MOVD    R2, 0(R1)
	BR      ·sinh(SB)
vectorimpl:
	MOVD    $sinhvectorfacility+0x00(SB), R1
	MOVD    $·sinhAsm(SB), R2
	MOVD    R2, 0(R1)
	BR      ·sinhAsm(SB)

GLOBL sinhvectorfacility+0x00(SB), NOPTR, $8
DATA sinhvectorfacility+0x00(SB)/8, $·sinhTrampolineSetup(SB)



TEXT ·Tanh(SB),NOSPLIT,$0
	MOVD    tanhvectorfacility+0x00(SB),R1
	BR      (R1)

TEXT ·tanhTrampolineSetup(SB),NOSPLIT, $0
	MOVB    ·hasVX(SB), R1
	CMPBEQ  R1, $1, vectorimpl      // vectorfacility = 1, vector supported
	MOVD    $tanhvectorfacility+0x00(SB), R1
	MOVD    $·tanh(SB), R2
	MOVD    R2, 0(R1)
	BR      ·tanh(SB)
vectorimpl:
	MOVD    $tanhvectorfacility+0x00(SB), R1
	MOVD    $·tanhAsm(SB), R2
	MOVD    R2, 0(R1)
	BR      ·tanhAsm(SB)

GLOBL tanhvectorfacility+0x00(SB), NOPTR, $8
DATA tanhvectorfacility+0x00(SB)/8, $·tanhTrampolineSetup(SB)


