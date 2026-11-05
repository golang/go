// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix && (mips || mipsle)

#include "go_asm.h"
#include "textflag.h"

// cgoCallTrampoline is called on the system stack by runtime.asmcgocall with
// a *cgoCallFrame in R4 using the MIPS O32 C ABI.
// Note: O32 requires the caller to reserve 0(R29)..15(R29) as the callee's
// argument home area, so LR cannot be stored at 0(R29) across the C call.
// runtime.asmcgocall and runtime.asmcgocall_no_g align R29 to 8 bytes.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	ADDU	$-40, R29
	MOVW	R16, 24(R29)
	MOVW	R28, 28(R29)
	MOVW	R31, 32(R29)
	MOVW	R4, R16
	MOVW	cgoCallFrame_fn(R16), R25
	MOVW	cgoCallFrame_a0(R16), R4
	MOVW	cgoCallFrame_a1(R16), R5
	MOVW	cgoCallFrame_a2(R16), R6
	MOVW	cgoCallFrame_a3(R16), R7
	MOVW	cgoCallFrame_a4(R16), R8
	MOVW	cgoCallFrame_a5(R16), R9
	MOVW	R8, 16(R29)
	MOVW	R9, 20(R29)
	JAL	(R25)
	MOVW	R2, cgoCallFrame_r1(R16)
	MOVW	R0, R2
	MOVW	32(R29), R31
	MOVW	28(R29), R28
	MOVW	24(R29), R16
	ADDU	$40, R29
	RET

#ifndef GOMIPS_softfloat
#define CGO_PROLOGUE \
	ADDU	$-112, R29; \
	MOVW	R16, 20(R29); \
	MOVW	R17, 24(R29); \
	MOVW	R18, 28(R29); \
	MOVW	R19, 32(R29); \
	MOVW	R20, 36(R29); \
	MOVW	R21, 40(R29); \
	MOVW	R22, 44(R29); \
	MOVW	R23, 48(R29); \
	MOVW	R28, 52(R29); \
	MOVW	g, 56(R29); \
	MOVW	R31, 60(R29); \
	MOVD	F20, 64(R29); \
	MOVD	F22, 72(R29); \
	MOVD	F24, 80(R29); \
	MOVD	F26, 88(R29); \
	MOVD	F28, 96(R29); \
	MOVD	F30, 104(R29); \
	MOVW	R0, g

#define CGO_EPILOGUE \
	MOVD	64(R29), F20; \
	MOVD	72(R29), F22; \
	MOVD	80(R29), F24; \
	MOVD	88(R29), F26; \
	MOVD	96(R29), F28; \
	MOVD	104(R29), F30; \
	MOVW	20(R29), R16; \
	MOVW	24(R29), R17; \
	MOVW	28(R29), R18; \
	MOVW	32(R29), R19; \
	MOVW	36(R29), R20; \
	MOVW	40(R29), R21; \
	MOVW	44(R29), R22; \
	MOVW	48(R29), R23; \
	MOVW	52(R29), R28; \
	MOVW	56(R29), g; \
	MOVW	60(R29), R31; \
	ADDU	$112, R29
#else
#define CGO_PROLOGUE \
	ADDU	$-64, R29; \
	MOVW	R16, 20(R29); \
	MOVW	R17, 24(R29); \
	MOVW	R18, 28(R29); \
	MOVW	R19, 32(R29); \
	MOVW	R20, 36(R29); \
	MOVW	R21, 40(R29); \
	MOVW	R22, 44(R29); \
	MOVW	R23, 48(R29); \
	MOVW	R28, 52(R29); \
	MOVW	g, 56(R29); \
	MOVW	R31, 60(R29); \
	MOVW	R0, g

#define CGO_EPILOGUE \
	MOVW	20(R29), R16; \
	MOVW	24(R29), R17; \
	MOVW	28(R29), R18; \
	MOVW	32(R29), R19; \
	MOVW	36(R29), R20; \
	MOVW	40(R29), R21; \
	MOVW	44(R29), R22; \
	MOVW	48(R29), R23; \
	MOVW	52(R29), R28; \
	MOVW	56(R29), g; \
	MOVW	60(R29), R31; \
	ADDU	$64, R29
#endif

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	$·runtime_init_done(SB), R8
	MOVW	(R8), R8
	MOVW	$2, R9
	BNE	R8, R9, slow
	// Make the load of runtime_init_done an acquire load.
	SYNC
	MOVW	$·cgo_context_function(SB), R2
	MOVW	(R2), R2
	BNE	R2, R0, slow
	RET
slow:
	CGO_PROLOGUE
	JAL	·cgoWaitRuntimeInitDone(SB)
	MOVW	4(R29), R2
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	BEQ	R4, done
	CGO_PROLOGUE
	MOVW	R4, 4(R29)
	JAL	·cgoReleaseContext(SB)
	CGO_EPILOGUE
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	R4, R5
	MOVW	R0, R4
	MOVW	R0, R7
	JMP	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVW	R4, 4(R29)
	JAL	·threadentry(SB)
	CGO_EPILOGUE
	MOVW	R0, R2
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVW	R4, 4(R29)
	MOVW	R5, 8(R29)
	MOVW	R6, 12(R29)
	MOVW	R7, 16(R29)
	JAL	·cgoInit(SB)
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $4
DATA _cgo_init+0(SB)/4, $x_cgo_init<>(SB)
