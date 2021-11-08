// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·archLog10(SB), NOSPLIT, $0
	MOVD ·log10vectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·log10TrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·log10vectorfacility+0x00(SB), R1
	MOVD   $·log10(SB), R2
	MOVD   R2, 0(R1)
	BR     ·log10(SB)

vectorimpl:
	MOVD $·log10vectorfacility+0x00(SB), R1
	MOVD $·log10Asm(SB), R2
	MOVD R2, 0(R1)
	BR   ·log10Asm(SB)

GLOBL ·log10vectorfacility+0x00(SB), NOPTR, $8
DATA ·log10vectorfacility+0x00(SB)/8, $·log10TrampolineSetup(SB)

TEXT ·archCos(SB), NOSPLIT, $0
	MOVD ·cosvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·cosTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·cosvectorfacility+0x00(SB), R1
	MOVD   $·cos(SB), R2
	MOVD   R2, 0(R1)
	BR     ·cos(SB)

vectorimpl:
	MOVD $·cosvectorfacility+0x00(SB), R1
	MOVD $·cosAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·cosAsm(SB)

GLOBL ·cosvectorfacility+0x00(SB), NOPTR, $8
DATA ·cosvectorfacility+0x00(SB)/8, $·cosTrampolineSetup(SB)

TEXT ·archCosh(SB), NOSPLIT, $0
	MOVD ·coshvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·coshTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·coshvectorfacility+0x00(SB), R1
	MOVD   $·cosh(SB), R2
	MOVD   R2, 0(R1)
	BR     ·cosh(SB)

vectorimpl:
	MOVD $·coshvectorfacility+0x00(SB), R1
	MOVD $·coshAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·coshAsm(SB)

GLOBL ·coshvectorfacility+0x00(SB), NOPTR, $8
DATA ·coshvectorfacility+0x00(SB)/8, $·coshTrampolineSetup(SB)

TEXT ·archSin(SB), NOSPLIT, $0
	MOVD ·sinvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·sinTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·sinvectorfacility+0x00(SB), R1
	MOVD   $·sin(SB), R2
	MOVD   R2, 0(R1)
	BR     ·sin(SB)

vectorimpl:
	MOVD $·sinvectorfacility+0x00(SB), R1
	MOVD $·sinAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·sinAsm(SB)

GLOBL ·sinvectorfacility+0x00(SB), NOPTR, $8
DATA ·sinvectorfacility+0x00(SB)/8, $·sinTrampolineSetup(SB)

TEXT ·archSinh(SB), NOSPLIT, $0
	MOVD ·sinhvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·sinhTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·sinhvectorfacility+0x00(SB), R1
	MOVD   $·sinh(SB), R2
	MOVD   R2, 0(R1)
	BR     ·sinh(SB)

vectorimpl:
	MOVD $·sinhvectorfacility+0x00(SB), R1
	MOVD $·sinhAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·sinhAsm(SB)

GLOBL ·sinhvectorfacility+0x00(SB), NOPTR, $8
DATA ·sinhvectorfacility+0x00(SB)/8, $·sinhTrampolineSetup(SB)

TEXT ·archTanh(SB), NOSPLIT, $0
	MOVD ·tanhvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·tanhTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·tanhvectorfacility+0x00(SB), R1
	MOVD   $·tanh(SB), R2
	MOVD   R2, 0(R1)
	BR     ·tanh(SB)

vectorimpl:
	MOVD $·tanhvectorfacility+0x00(SB), R1
	MOVD $·tanhAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·tanhAsm(SB)

GLOBL ·tanhvectorfacility+0x00(SB), NOPTR, $8
DATA ·tanhvectorfacility+0x00(SB)/8, $·tanhTrampolineSetup(SB)

TEXT ·archLog1p(SB), NOSPLIT, $0
	MOVD ·log1pvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·log1pTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·log1pvectorfacility+0x00(SB), R1
	MOVD   $·log1p(SB), R2
	MOVD   R2, 0(R1)
	BR     ·log1p(SB)

vectorimpl:
	MOVD $·log1pvectorfacility+0x00(SB), R1
	MOVD $·log1pAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·log1pAsm(SB)

GLOBL ·log1pvectorfacility+0x00(SB), NOPTR, $8
DATA ·log1pvectorfacility+0x00(SB)/8, $·log1pTrampolineSetup(SB)

TEXT ·archAtanh(SB), NOSPLIT, $0
	MOVD ·atanhvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·atanhTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·atanhvectorfacility+0x00(SB), R1
	MOVD   $·atanh(SB), R2
	MOVD   R2, 0(R1)
	BR     ·atanh(SB)

vectorimpl:
	MOVD $·atanhvectorfacility+0x00(SB), R1
	MOVD $·atanhAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·atanhAsm(SB)

GLOBL ·atanhvectorfacility+0x00(SB), NOPTR, $8
DATA ·atanhvectorfacility+0x00(SB)/8, $·atanhTrampolineSetup(SB)

TEXT ·archAcos(SB), NOSPLIT, $0
	MOVD ·acosvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·acosTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·acosvectorfacility+0x00(SB), R1
	MOVD   $·acos(SB), R2
	MOVD   R2, 0(R1)
	BR     ·acos(SB)

vectorimpl:
	MOVD $·acosvectorfacility+0x00(SB), R1
	MOVD $·acosAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·acosAsm(SB)

GLOBL ·acosvectorfacility+0x00(SB), NOPTR, $8
DATA ·acosvectorfacility+0x00(SB)/8, $·acosTrampolineSetup(SB)

TEXT ·archAsin(SB), NOSPLIT, $0
	MOVD ·asinvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·asinTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·asinvectorfacility+0x00(SB), R1
	MOVD   $·asin(SB), R2
	MOVD   R2, 0(R1)
	BR     ·asin(SB)

vectorimpl:
	MOVD $·asinvectorfacility+0x00(SB), R1
	MOVD $·asinAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·asinAsm(SB)

GLOBL ·asinvectorfacility+0x00(SB), NOPTR, $8
DATA ·asinvectorfacility+0x00(SB)/8, $·asinTrampolineSetup(SB)

TEXT ·archAsinh(SB), NOSPLIT, $0
	MOVD ·asinhvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·asinhTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·asinhvectorfacility+0x00(SB), R1
	MOVD   $·asinh(SB), R2
	MOVD   R2, 0(R1)
	BR     ·asinh(SB)

vectorimpl:
	MOVD $·asinhvectorfacility+0x00(SB), R1
	MOVD $·asinhAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·asinhAsm(SB)

GLOBL ·asinhvectorfacility+0x00(SB), NOPTR, $8
DATA ·asinhvectorfacility+0x00(SB)/8, $·asinhTrampolineSetup(SB)

TEXT ·archAcosh(SB), NOSPLIT, $0
	MOVD ·acoshvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·acoshTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·acoshvectorfacility+0x00(SB), R1
	MOVD   $·acosh(SB), R2
	MOVD   R2, 0(R1)
	BR     ·acosh(SB)

vectorimpl:
	MOVD $·acoshvectorfacility+0x00(SB), R1
	MOVD $·acoshAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·acoshAsm(SB)

GLOBL ·acoshvectorfacility+0x00(SB), NOPTR, $8
DATA ·acoshvectorfacility+0x00(SB)/8, $·acoshTrampolineSetup(SB)

TEXT ·archErf(SB), NOSPLIT, $0
	MOVD ·erfvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·erfTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·erfvectorfacility+0x00(SB), R1
	MOVD   $·erf(SB), R2
	MOVD   R2, 0(R1)
	BR     ·erf(SB)

vectorimpl:
	MOVD $·erfvectorfacility+0x00(SB), R1
	MOVD $·erfAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·erfAsm(SB)

GLOBL ·erfvectorfacility+0x00(SB), NOPTR, $8
DATA ·erfvectorfacility+0x00(SB)/8, $·erfTrampolineSetup(SB)

TEXT ·archErfc(SB), NOSPLIT, $0
	MOVD ·erfcvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·erfcTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·erfcvectorfacility+0x00(SB), R1
	MOVD   $·erfc(SB), R2
	MOVD   R2, 0(R1)
	BR     ·erfc(SB)

vectorimpl:
	MOVD $·erfcvectorfacility+0x00(SB), R1
	MOVD $·erfcAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·erfcAsm(SB)

GLOBL ·erfcvectorfacility+0x00(SB), NOPTR, $8
DATA ·erfcvectorfacility+0x00(SB)/8, $·erfcTrampolineSetup(SB)

TEXT ·archAtan(SB), NOSPLIT, $0
	MOVD ·atanvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·atanTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·atanvectorfacility+0x00(SB), R1
	MOVD   $·atan(SB), R2
	MOVD   R2, 0(R1)
	BR     ·atan(SB)

vectorimpl:
	MOVD $·atanvectorfacility+0x00(SB), R1
	MOVD $·atanAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·atanAsm(SB)

GLOBL ·atanvectorfacility+0x00(SB), NOPTR, $8
DATA ·atanvectorfacility+0x00(SB)/8, $·atanTrampolineSetup(SB)

TEXT ·archAtan2(SB), NOSPLIT, $0
	MOVD ·atan2vectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·atan2TrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·atan2vectorfacility+0x00(SB), R1
	MOVD   $·atan2(SB), R2
	MOVD   R2, 0(R1)
	BR     ·atan2(SB)

vectorimpl:
	MOVD $·atan2vectorfacility+0x00(SB), R1
	MOVD $·atan2Asm(SB), R2
	MOVD R2, 0(R1)
	BR   ·atan2Asm(SB)

GLOBL ·atan2vectorfacility+0x00(SB), NOPTR, $8
DATA ·atan2vectorfacility+0x00(SB)/8, $·atan2TrampolineSetup(SB)

TEXT ·archCbrt(SB), NOSPLIT, $0
	MOVD ·cbrtvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·cbrtTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                // vectorfacility = 1, vector supported
	MOVD   $·cbrtvectorfacility+0x00(SB), R1
	MOVD   $·cbrt(SB), R2
	MOVD   R2, 0(R1)
	BR     ·cbrt(SB)

vectorimpl:
	MOVD $·cbrtvectorfacility+0x00(SB), R1
	MOVD $·cbrtAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·cbrtAsm(SB)

GLOBL ·cbrtvectorfacility+0x00(SB), NOPTR, $8
DATA ·cbrtvectorfacility+0x00(SB)/8, $·cbrtTrampolineSetup(SB)

TEXT ·archLog(SB), NOSPLIT, $0
	MOVD ·logvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·logTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·logvectorfacility+0x00(SB), R1
	MOVD   $·log(SB), R2
	MOVD   R2, 0(R1)
	BR     ·log(SB)

vectorimpl:
	MOVD $·logvectorfacility+0x00(SB), R1
	MOVD $·logAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·logAsm(SB)

GLOBL ·logvectorfacility+0x00(SB), NOPTR, $8
DATA ·logvectorfacility+0x00(SB)/8, $·logTrampolineSetup(SB)

TEXT ·archTan(SB), NOSPLIT, $0
	MOVD ·tanvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·tanTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·tanvectorfacility+0x00(SB), R1
	MOVD   $·tan(SB), R2
	MOVD   R2, 0(R1)
	BR     ·tan(SB)

vectorimpl:
	MOVD $·tanvectorfacility+0x00(SB), R1
	MOVD $·tanAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·tanAsm(SB)

GLOBL ·tanvectorfacility+0x00(SB), NOPTR, $8
DATA ·tanvectorfacility+0x00(SB)/8, $·tanTrampolineSetup(SB)

TEXT ·archExp(SB), NOSPLIT, $0
	MOVD ·expvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·expTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·expvectorfacility+0x00(SB), R1
	MOVD   $·exp(SB), R2
	MOVD   R2, 0(R1)
	BR     ·exp(SB)

vectorimpl:
	MOVD $·expvectorfacility+0x00(SB), R1
	MOVD $·expAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·expAsm(SB)

GLOBL ·expvectorfacility+0x00(SB), NOPTR, $8
DATA ·expvectorfacility+0x00(SB)/8, $·expTrampolineSetup(SB)

TEXT ·archExpm1(SB), NOSPLIT, $0
	MOVD ·expm1vectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·expm1TrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl                 // vectorfacility = 1, vector supported
	MOVD   $·expm1vectorfacility+0x00(SB), R1
	MOVD   $·expm1(SB), R2
	MOVD   R2, 0(R1)
	BR     ·expm1(SB)

vectorimpl:
	MOVD $·expm1vectorfacility+0x00(SB), R1
	MOVD $·expm1Asm(SB), R2
	MOVD R2, 0(R1)
	BR   ·expm1Asm(SB)

GLOBL ·expm1vectorfacility+0x00(SB), NOPTR, $8
DATA ·expm1vectorfacility+0x00(SB)/8, $·expm1TrampolineSetup(SB)

TEXT ·archPow(SB), NOSPLIT, $0
	MOVD ·powvectorfacility+0x00(SB), R1
	BR   (R1)

TEXT ·powTrampolineSetup(SB), NOSPLIT, $0
	MOVB   ·hasVX(SB), R1
	CMPBEQ R1, $1, vectorimpl               // vectorfacility = 1, vector supported
	MOVD   $·powvectorfacility+0x00(SB), R1
	MOVD   $·pow(SB), R2
	MOVD   R2, 0(R1)
	BR     ·pow(SB)

vectorimpl:
	MOVD $·powvectorfacility+0x00(SB), R1
	MOVD $·powAsm(SB), R2
	MOVD R2, 0(R1)
	BR   ·powAsm(SB)

GLOBL ·powvectorfacility+0x00(SB), NOPTR, $8
DATA ·powvectorfacility+0x00(SB)/8, $·powTrampolineSetup(SB)

