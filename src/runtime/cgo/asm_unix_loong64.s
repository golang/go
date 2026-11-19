// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix && loong64

#include "go_asm.h"
#include "textflag.h"
#include "abi_loong64.h"

// cgoCallTrampoline is called by runtime.asmcgocall or runtime.asmcgocall_no_g
// using the LoongArch LP64D C ABI with R4 pointing to a cgoCallFrame.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	R1, -16(R3)
	ADDV	$-16, R3
	MOVV	R1, 0(R3)
	MOVV	R23, 8(R3)
	MOVV	R4, R23
	MOVV	cgoCallFrame_fn(R23), R12
	MOVV	cgoCallFrame_a1(R23), R5
	MOVV	cgoCallFrame_a2(R23), R6
	MOVV	cgoCallFrame_a3(R23), R7
	MOVV	cgoCallFrame_a4(R23), R8
	MOVV	cgoCallFrame_a5(R23), R9
	MOVV	cgoCallFrame_a0(R23), R4
	JAL	(R12)
	MOVV	R4, cgoCallFrame_r1(R23)
	MOVV	R0, R4
	MOVV	8(R3), R23
	MOVV	0(R3), R1
	ADDV	$16, R3
	RET

#define CGO_PROLOGUE \
	ADDV	$(-24*8), R3; \
	SAVE_R22_TO_R31(5*8); \
	SAVE_F24_TO_F31(15*8); \
	MOVV	R1, (23*8)(R3); \
	MOVV	R0, g

#define CGO_EPILOGUE \
	RESTORE_R22_TO_R31(5*8); \
	RESTORE_F24_TO_F31(15*8); \
	MOVV	(23*8)(R3), R1; \
	ADDV	$(24*8), R3

// CALL_GO calls a Go function indirectly so that the linker's nosplit
// check does not count the Go function's stack usage: these entrypoints
// are only called from C, on a C stack, after CGO_PROLOGUE.
#define CALL_GO(fn) \
	MOVV	$fn(SB), R20; \
	JAL	(R20)

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	$·runtime_init_done(SB), R12
	MOVW	(R12), R12
	MOVV	$2, R13
	BNE	R12, R13, slow
	// Make the load of runtime_init_done an acquire load.
	DBAR
	MOVV	$·cgo_context_function(SB), R4
	MOVV	(R4), R4
	BNE	R4, R0, slow
	RET
slow:
	CGO_PROLOGUE
	CALL_GO(·cgoWaitRuntimeInitDone)
	MOVV	8(R3), R4
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	BEQ	R4, R0, done
	CGO_PROLOGUE
	MOVV	R4, 8(R3)
	CALL_GO(·cgoReleaseContext)
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
	MOVV	R4, 8(R3)
	CALL_GO(·threadentry)
	CGO_EPILOGUE
	MOVV	R0, R4
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVV	R4, 8(R3)
	MOVV	R5, 16(R3)
	MOVV	R6, 24(R3)
	MOVV	R7, 32(R3)
	CALL_GO(·cgoInit)
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)

// x_cgo_callers is jumped to from runtime.cgoSigtramp on the signal stack.
TEXT ·x_cgo_callers(SB),NOSPLIT|NOFRAME,$0
	ADDV	$-80, R3
	MOVV	g, 64(R3)		// save fp (R22/g)
	MOVV	R1, 72(R3)
	ADDV	$80, R3, g

	MOVV	R0, 0(R3)		// arg.context = 0
	MOVV	R6, 8(R3)		// arg.sigContext = context
	MOVV	R8, 16(R3)		// arg.buf = cgoCallers
	MOVV	$32, R12
	MOVV	R12, 24(R3)		// arg.max = 32

	MOVV	R4, 32(R3)		// save sig
	MOVV	R5, 40(R3)		// save info
	MOVV	R6, 48(R3)		// save context
	MOVV	R9, 56(R3)		// save sigtramp

	MOVV	R3, R4			// &arg
	JAL	(R7)			// (*cgoTraceback)(&arg)

	MOVV	32(R3), R4		// restore sig
	MOVV	40(R3), R5		// restore info
	MOVV	48(R3), R6		// restore context
	MOVV	56(R3), R12		// restore sigtramp

	MOVV	64(R3), g
	MOVV	72(R3), R1
	ADDV	$80, R3
	JMP	(R12)			// sigtramp(sig, info, context)

GLOBL _cgo_callers(SB), NOPTR, $8
DATA _cgo_callers+0(SB)/8, $·x_cgo_callers(SB)
