// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (darwin || (linux && !android)) && arm64

#include "go_asm.h"
#include "textflag.h"
#include "abi_arm64.h"

// cgoCallTrampoline is called by runtime.asmcgocall (or runtime.cgocall)
// using the AAPCS64 C ABI with R0 pointing to a cgoCallFrame.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	// Go unwinders expect the saved LR at SP as soon as SP changes.
	// The C frame pointer needs a separate (FP, LR) record.
	STP.W	(R30, R19), -32(RSP)
	STP	(R29, R30), 16(RSP)
	ADD	$16, RSP, R29
	MOVD	R0, R19
	MOVD	cgoCallFrame_fn(R0), R9
	MOVD	cgoCallFrame_a1(R0), R1
	MOVD	cgoCallFrame_a2(R0), R2
	MOVD	cgoCallFrame_a3(R0), R3
	MOVD	cgoCallFrame_a4(R0), R4
	MOVD	cgoCallFrame_a5(R0), R5
	MOVD	cgoCallFrame_a0(R0), R0
	BL	(R9)
	MOVD	R0, cgoCallFrame_r1(R19)
	MOVD	$0, R0
	MOVD	16(RSP), R29
	LDP.P	32(RSP), (R30, R19)
	RET

#define CGO_PROLOGUE \
	SUB	$208, RSP; \
	SAVE_R19_TO_R28(8*6); \
	SAVE_F8_TO_F15(8*16); \
	STP	(R29, R30), (8*24)(RSP); \
	ADD	$(8*24), RSP, R29; \
	MOVD	ZR, g

#define CGO_EPILOGUE \
	RESTORE_R19_TO_R28(8*6); \
	RESTORE_F8_TO_F15(8*16); \
	LDP	(8*24)(RSP), (R29, R30); \
	ADD	$208, RSP

// Indirectly call Go targets via R9 to avoid the linker's static NOSPLIT
// stack overflow check: these entrypoints are only invoked from C on a
// full-size C/pthread stack after saving host registers in CGO_PROLOGUE.
#define CALL_GO(fn) \
	MOVD	$fn(SB), R9; \
	BL	(R9)

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	// Save R27 in caller-saved R9: global accesses below clobber R27
	// (callee-saved in the C ABI) under -dynlink.
	MOVD	R27, R9
	MOVD	$·runtime_init_done(SB), R1
	LDARW	(R1), R1
	CMPW	$2, R1
	BNE	slow
	MOVD	$·cgo_context_function(SB), R0
	LDAR	(R0), R0
	CBNZ	R0, slow
	MOVD	R9, R27
	RET
slow:
	MOVD	R9, R27
	CGO_PROLOGUE
	CALL_GO(·cgoWaitRuntimeInitDone)
	MOVD	8(RSP), R0
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	CBZ	R0, done
	CGO_PROLOGUE
	MOVD	R0, 8(RSP)
	CALL_GO(·cgoReleaseContext)
	CGO_EPILOGUE
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R0, R1
	MOVD	$0, R0
	MOVD	$0, R3
	JMP	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	CALL_GO(·threadentry<ABIInternal>)
	CGO_EPILOGUE
	MOVD	$0, R0
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	STP	(R0, R1), 8(RSP)
	CALL_GO(·cgoInit)
	CGO_EPILOGUE
	RET

GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)
