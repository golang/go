// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix && (mips64 || mips64le)

#include "textflag.h"

// Called on g0 by cgoCallTrampoline with a PC buffer in R4, its size
// in R5, and runtime.Callers in R6. Call Callers(0, buf) using the
// MIPS64 stack ABI while preserving C callee-saved registers.
#ifndef GOMIPS64_softfloat
TEXT ·trampolineCallers(SB),NOSPLIT,$184-0
#else
TEXT ·trampolineCallers(SB),NOSPLIT,$120-0
#endif
	MOVV	R16, (8*6)(R29)
	MOVV	R17, (8*7)(R29)
	MOVV	R18, (8*8)(R29)
	MOVV	R19, (8*9)(R29)
	MOVV	R20, (8*10)(R29)
	MOVV	R21, (8*11)(R29)
	MOVV	R22, (8*12)(R29)
	MOVV	R23, (8*13)(R29)
	MOVV	RSB, (8*14)(R29)
	MOVV	g, (8*15)(R29)
#ifndef GOMIPS64_softfloat
	MOVD	F24, (8*16)(R29)
	MOVD	F25, (8*17)(R29)
	MOVD	F26, (8*18)(R29)
	MOVD	F27, (8*19)(R29)
	MOVD	F28, (8*20)(R29)
	MOVD	F29, (8*21)(R29)
	MOVD	F30, (8*22)(R29)
	MOVD	F31, (8*23)(R29)
#endif
	MOVV	R0, 8(R29)
	MOVV	R4, 16(R29)
	MOVV	R5, 24(R29)
	MOVV	R5, 32(R29)
	MOVV	R6, R25
	JAL	(R25)
	MOVV	40(R29), R2
#ifndef GOMIPS64_softfloat
	MOVD	(8*16)(R29), F24
	MOVD	(8*17)(R29), F25
	MOVD	(8*18)(R29), F26
	MOVD	(8*19)(R29), F27
	MOVD	(8*20)(R29), F28
	MOVD	(8*21)(R29), F29
	MOVD	(8*22)(R29), F30
	MOVD	(8*23)(R29), F31
#endif
	MOVV	(8*6)(R29), R16
	MOVV	(8*7)(R29), R17
	MOVV	(8*8)(R29), R18
	MOVV	(8*9)(R29), R19
	MOVV	(8*10)(R29), R20
	MOVV	(8*11)(R29), R21
	MOVV	(8*12)(R29), R22
	MOVV	(8*13)(R29), R23
	MOVV	(8*14)(R29), RSB
	MOVV	(8*15)(R29), g
	RET
