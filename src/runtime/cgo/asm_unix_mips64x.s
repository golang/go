// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix && (mips64 || mips64le)

#include "go_asm.h"
#include "textflag.h"

// cgoCallTrampoline is called on the system stack by runtime.asmcgocall with
// a *cgoCallFrame in R4 using the MIPS N64 C ABI.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	R31, -32(R29)
	ADDV	$-32, R29
	MOVV	R31, 0(R29)
	MOVV	R16, 8(R29)
	MOVV	RSB, 16(R29)
	MOVV	R4, R16
	MOVV	cgoCallFrame_fn(R16), R25
	MOVV	cgoCallFrame_a0(R16), R4
	MOVV	cgoCallFrame_a1(R16), R5
	MOVV	cgoCallFrame_a2(R16), R6
	MOVV	cgoCallFrame_a3(R16), R7
	MOVV	cgoCallFrame_a4(R16), R8
	MOVV	cgoCallFrame_a5(R16), R9
	JAL	(R25)
	MOVV	R2, cgoCallFrame_r1(R16)
	MOVV	R0, R2
	MOVV	16(R29), RSB
	MOVV	8(R29), R16
	MOVV	0(R29), R31
	ADDV	$32, R29
	RET

#ifndef GOMIPS64_softfloat
#define CGO_PROLOGUE \
	ADDV	$(-8*24), R29; \
	MOVV	R16, (8*5)(R29); \
	MOVV	R17, (8*6)(R29); \
	MOVV	R18, (8*7)(R29); \
	MOVV	R19, (8*8)(R29); \
	MOVV	R20, (8*9)(R29); \
	MOVV	R21, (8*10)(R29); \
	MOVV	R22, (8*11)(R29); \
	MOVV	R23, (8*12)(R29); \
	MOVV	RSB, (8*13)(R29); \
	MOVV	g, (8*14)(R29); \
	MOVV	R31, (8*15)(R29); \
	MOVD	F24, (8*16)(R29); \
	MOVD	F25, (8*17)(R29); \
	MOVD	F26, (8*18)(R29); \
	MOVD	F27, (8*19)(R29); \
	MOVD	F28, (8*20)(R29); \
	MOVD	F29, (8*21)(R29); \
	MOVD	F30, (8*22)(R29); \
	MOVD	F31, (8*23)(R29); \
	BGEZAL	R0, 1(PC); \
	SRLV	$32, R31, RSB; \
	SLLV	$32, RSB; \
	MOVV	R0, g

#define CGO_EPILOGUE \
	MOVD	(8*16)(R29), F24; \
	MOVD	(8*17)(R29), F25; \
	MOVD	(8*18)(R29), F26; \
	MOVD	(8*19)(R29), F27; \
	MOVD	(8*20)(R29), F28; \
	MOVD	(8*21)(R29), F29; \
	MOVD	(8*22)(R29), F30; \
	MOVD	(8*23)(R29), F31; \
	MOVV	(8*5)(R29), R16; \
	MOVV	(8*6)(R29), R17; \
	MOVV	(8*7)(R29), R18; \
	MOVV	(8*8)(R29), R19; \
	MOVV	(8*9)(R29), R20; \
	MOVV	(8*10)(R29), R21; \
	MOVV	(8*11)(R29), R22; \
	MOVV	(8*12)(R29), R23; \
	MOVV	(8*13)(R29), RSB; \
	MOVV	(8*14)(R29), g; \
	MOVV	(8*15)(R29), R31; \
	ADDV	$(8*24), R29
#else
#define CGO_PROLOGUE \
	ADDV	$(-8*16), R29; \
	MOVV	R16, (8*5)(R29); \
	MOVV	R17, (8*6)(R29); \
	MOVV	R18, (8*7)(R29); \
	MOVV	R19, (8*8)(R29); \
	MOVV	R20, (8*9)(R29); \
	MOVV	R21, (8*10)(R29); \
	MOVV	R22, (8*11)(R29); \
	MOVV	R23, (8*12)(R29); \
	MOVV	RSB, (8*13)(R29); \
	MOVV	g, (8*14)(R29); \
	MOVV	R31, (8*15)(R29); \
	BGEZAL	R0, 1(PC); \
	SRLV	$32, R31, RSB; \
	SLLV	$32, RSB; \
	MOVV	R0, g

#define CGO_EPILOGUE \
	MOVV	(8*5)(R29), R16; \
	MOVV	(8*6)(R29), R17; \
	MOVV	(8*7)(R29), R18; \
	MOVV	(8*8)(R29), R19; \
	MOVV	(8*9)(R29), R20; \
	MOVV	(8*10)(R29), R21; \
	MOVV	(8*11)(R29), R22; \
	MOVV	(8*12)(R29), R23; \
	MOVV	(8*13)(R29), RSB; \
	MOVV	(8*14)(R29), g; \
	MOVV	(8*15)(R29), R31; \
	ADDV	$(8*16), R29
#endif

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	RSB, R14
	MOVV	R31, R15
	BGEZAL	R0, 1(PC)
	SRLV	$32, R31, RSB
	SLLV	$32, RSB
	MOVV	R15, R31
	MOVV	$·runtime_init_done(SB), R12
	MOVW	(R12), R12
	MOVV	$2, R13
	BNE	R12, R13, slow
	// Make the load of runtime_init_done an acquire load.
	SYNC
	MOVV	$·cgo_context_function(SB), R2
	MOVV	(R2), R2
	BNE	R2, R0, slow
	MOVV	R14, RSB
	RET
slow:
	MOVV	R14, RSB
	CGO_PROLOGUE
	JAL	·cgoWaitRuntimeInitDone(SB)
	MOVV	8(R29), R2
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	BEQ	R4, done
	CGO_PROLOGUE
	MOVV	R4, 8(R29)
	JAL	·cgoReleaseContext(SB)
	CGO_EPILOGUE
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	R4, R5
	MOVV	R0, R4
	MOVV	R0, R7
	JMP	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVV	R4, 8(R29)
	JAL	·threadentry(SB)
	CGO_EPILOGUE
	MOVV	R0, R2
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVV	R4, 8(R29)
	MOVV	R5, 16(R29)
	MOVV	R6, 24(R29)
	MOVV	R7, 32(R29)
	JAL	·cgoInit(SB)
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)
