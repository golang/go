// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix && arm

#include "go_asm.h"
#include "textflag.h"

// cgoCallTrampoline is called by runtime.asmcgocall or runtime.asmcgocall_no_g
// using the ARM EABI (AAPCS) with R0 pointing to a cgoCallFrame.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	SUB	$16, R13
	MOVW	R4, 8(R13)
	MOVW	R14, 12(R13)
	MOVW	R0, R4
	MOVW	cgoCallFrame_a4(R4), R1
	MOVW	R1, 0(R13)
	MOVW	cgoCallFrame_a5(R4), R1
	MOVW	R1, 4(R13)
	MOVW	cgoCallFrame_fn(R4), R12
	MOVW	cgoCallFrame_a0(R4), R0
	MOVW	cgoCallFrame_a1(R4), R1
	MOVW	cgoCallFrame_a2(R4), R2
	MOVW	cgoCallFrame_a3(R4), R3
	BL	(R12)
	MOVW	R0, cgoCallFrame_r1(R4)
	MOVW	$0, R0
	MOVW	8(R13), R4
	MOVW	12(R13), R14
	ADD	$16, R13
	RET

#define CGO_PROLOGUE(skipfp) \
	SUB	$120, R13; \
	MOVW	R4, 20(R13); \
	MOVW	R5, 24(R13); \
	MOVW	R6, 28(R13); \
	MOVW	R7, 32(R13); \
	MOVW	R8, 36(R13); \
	MOVW	R9, 40(R13); \
	MOVW	g, 44(R13); \
	MOVW	R11, 48(R13); \
	MOVW	R14, 52(R13); \
	MOVB	runtime·goarmsoftfp(SB), R11; \
	CMP	$0, R11; \
	BNE	skipfp; \
	MOVD	F8, 56(R13); \
	MOVD	F9, 64(R13); \
	MOVD	F10, 72(R13); \
	MOVD	F11, 80(R13); \
	MOVD	F12, 88(R13); \
	MOVD	F13, 96(R13); \
	MOVD	F14, 104(R13); \
	MOVD	F15, 112(R13); \
skipfp: \
	MOVW	$0, g

#define CGO_EPILOGUE(skipfp) \
	MOVB	runtime·goarmsoftfp(SB), R11; \
	CMP	$0, R11; \
	BNE	skipfp; \
	MOVD	56(R13), F8; \
	MOVD	64(R13), F9; \
	MOVD	72(R13), F10; \
	MOVD	80(R13), F11; \
	MOVD	88(R13), F12; \
	MOVD	96(R13), F13; \
	MOVD	104(R13), F14; \
	MOVD	112(R13), F15; \
skipfp: \
	MOVW	20(R13), R4; \
	MOVW	24(R13), R5; \
	MOVW	28(R13), R6; \
	MOVW	32(R13), R7; \
	MOVW	36(R13), R8; \
	MOVW	40(R13), R9; \
	MOVW	44(R13), g; \
	MOVW	48(R13), R11; \
	MOVW	52(R13), R14; \
	ADD	$120, R13

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	// Save R11 (REGTMP) in caller-saved R12: GOT loads under -dynlink
	// clobber R11, which is callee-saved in the ARM EABI.
	MOVW	R11, R12
	MOVW	$·runtime_init_done(SB), R1
	MOVW	(R1), R1
	CMP	$2, R1
	BNE	slow
	// Make the load of runtime_init_done an acquire load, as
	// internal/runtime/atomic.Load does. DMB requires ARMv7;
	// use the slow path on older processors.
	// Load runtime·goarm through R2: with -dynlink, a direct load
	// would use R9, which is callee-saved in the ARM EABI.
	MOVW	$runtime·goarm(SB), R2
	MOVB	(R2), R2
	CMP	$7, R2
	BLT	slow
	DMB	MB_ISH
	MOVW	$·cgo_context_function(SB), R0
	MOVW	(R0), R0
	CMP	$0, R0
	BNE	slow
	MOVW	R12, R11
	RET
slow:
	MOVW	R12, R11
	CGO_PROLOGUE(wait_skipfpsave)
	BL	·cgoWaitRuntimeInitDone(SB)
	MOVW	4(R13), R0
	CGO_EPILOGUE(wait_skipfprest)
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	CMP	$0, R0
	BEQ	done
	CGO_PROLOGUE(rel_skipfpsave)
	MOVW	R0, 4(R13)
	BL	·cgoReleaseContext(SB)
	CGO_EPILOGUE(rel_skipfprest)
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	R0, R1
	MOVW	$0, R0
	MOVW	$0, R3
	B	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE(thr_skipfpsave)
	MOVW	R0, 4(R13)
	BL	·threadentry(SB)
	CGO_EPILOGUE(thr_skipfprest)
	MOVW	$0, R0
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE(init_skipfpsave)
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	MOVW	R3, 16(R13)
	BL	·cgoInit(SB)
	CGO_EPILOGUE(init_skipfprest)
	RET
GLOBL _cgo_init(SB), NOPTR, $4
DATA _cgo_init+0(SB)/4, $x_cgo_init<>(SB)
