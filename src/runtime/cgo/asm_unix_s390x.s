// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix && s390x

#include "go_asm.h"
#include "textflag.h"

// cgoCallTrampoline is called on the system stack by runtime.asmcgocall with
// a *cgoCallFrame in R2 using the Linux z/Architecture ELF ABI.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	STMG	R6, R14, 48(R15)
	MOVD	R2, R10
	MOVD	R14, -168(R15)
	ADD	$-168, R15
	MOVD	R14, 0(R15)
	MOVD	cgoCallFrame_fn(R10), R1
	MOVD	cgoCallFrame_a0(R10), R2
	MOVD	cgoCallFrame_a1(R10), R3
	MOVD	cgoCallFrame_a2(R10), R4
	MOVD	cgoCallFrame_a3(R10), R5
	MOVD	cgoCallFrame_a4(R10), R6
	MOVD	cgoCallFrame_a5(R10), R7
	MOVD	R7, 160(R15)
	BL	R1
	XOR	R0, R0
	MOVD	R2, cgoCallFrame_r1(R10)
	MOVD	$0, R2
	MOVD	0(R15), R14
	ADD	$168, R15
	LMG	48(R15), R6, R14
	RET

#define CGO_PROLOGUE \
	STMG	R6, R14, 48(R15); \
	ADD	$-104, R15; \
	FMOVD	F8, 40(R15); \
	FMOVD	F9, 48(R15); \
	FMOVD	F10, 56(R15); \
	FMOVD	F11, 64(R15); \
	FMOVD	F12, 72(R15); \
	FMOVD	F13, 80(R15); \
	FMOVD	F14, 88(R15); \
	FMOVD	F15, 96(R15); \
	XOR	R0, R0; \
	MOVD	$0, g

#define CGO_EPILOGUE \
	FMOVD	40(R15), F8; \
	FMOVD	48(R15), F9; \
	FMOVD	56(R15), F10; \
	FMOVD	64(R15), F11; \
	FMOVD	72(R15), F12; \
	FMOVD	80(R15), F13; \
	FMOVD	88(R15), F14; \
	FMOVD	96(R15), F15; \
	ADD	$104, R15; \
	LMG	48(R15), R6, R14

// CALL_GO calls a Go function indirectly so that the linker's nosplit
// check does not count the Go function's stack usage: these entrypoints
// are only called from C, on a C stack, after CGO_PROLOGUE.
#define CALL_GO(fn) \
	MOVD	$fn(SB), R1; \
	BL	R1

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$·runtime_init_done(SB), R1
	MOVWZ	(R1), R1
	CMPW	R1, $2
	BNE	slow
	MOVD	$·cgo_context_function(SB), R2
	MOVD	(R2), R2
	CMPBNE	R2, $0, slow
	RET
slow:
	CGO_PROLOGUE
	CALL_GO(·cgoWaitRuntimeInitDone)
	MOVD	8(R15), R2
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R2, $0, done
	CGO_PROLOGUE
	MOVD	R2, 8(R15)
	CALL_GO(·cgoReleaseContext)
	CGO_EPILOGUE
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R2, R3
	MOVD	$0, R2
	MOVD	$0, R5
	JMP	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVD	R2, 8(R15)
	CALL_GO(·threadentry)
	CGO_EPILOGUE
	MOVD	$0, R2
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVD	R2, 8(R15)
	MOVD	R3, 16(R15)
	// runtime.rt0_go does not pass the TLS arguments used by
	// cgoInitTLS on Android; R4 and R5 hold unrelated values.
	MOVD	$0, 24(R15)
	MOVD	$0, 32(R15)
	CALL_GO(·cgoInit)
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)
