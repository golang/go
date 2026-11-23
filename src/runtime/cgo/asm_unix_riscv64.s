// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix && riscv64

#include "go_asm.h"
#include "textflag.h"
#include "abi_riscv64.h"

// cgoCallTrampoline is called by runtime.asmcgocall or runtime.asmcgocall_no_g
// using the RISC-V LP64D C ABI with X10 pointing to a cgoCallFrame.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	// Call through T1 (X6), not T0 (X5): T0 is the alternate link
	// register, so JALR RA, T0 tells the return-address predictor
	// to pop an entry before pushing, as for a coroutine switch.
	MOV	RA, -16(X2)
	ADD	$-16, X2
	MOV	RA, 0(X2)
	MOV	X9, 8(X2)
	MOV	X10, X9
	MOV	cgoCallFrame_fn(X9), X6
	MOV	cgoCallFrame_a1(X9), X11
	MOV	cgoCallFrame_a2(X9), X12
	MOV	cgoCallFrame_a3(X9), X13
	MOV	cgoCallFrame_a4(X9), X14
	MOV	cgoCallFrame_a5(X9), X15
	MOV	cgoCallFrame_a0(X9), X10
	JALR	RA, X6
	MOV	X10, cgoCallFrame_r1(X9)
	MOV	ZERO, X10
	MOV	0(X2), RA
	MOV	8(X2), X9
	ADD	$16, X2
	RET

#define CGO_PROLOGUE \
	ADD	$(-30*8), X2; \
	SAVE_GPR(5*8); \
	MOV	X1, (17*8)(X2); \
	SAVE_FPR(18*8); \
	MOV	ZERO, g

#define CGO_EPILOGUE \
	RESTORE_GPR(5*8); \
	MOV	(17*8)(X2), X1; \
	RESTORE_FPR(18*8); \
	ADD	$(30*8), X2

// CALL_GO calls a Go function indirectly so that the linker's nosplit
// check does not count the Go function's stack usage: these entrypoints
// are only called from C, on a C stack, after CGO_PROLOGUE.
#define CALL_GO(fn) \
	MOV	$fn(SB), X6; \
	JALR	RA, X6

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	MOV	$·runtime_init_done(SB), X5
	MOVW	(X5), X5
	MOV	$2, X6
	BNE	X5, X6, slow
	// Make the load of runtime_init_done an acquire load.
	FENCE
	MOV	$·cgo_context_function(SB), X10
	MOV	(X10), X10
	BNE	X10, ZERO, slow
	RET
slow:
	CGO_PROLOGUE
	CALL_GO(·cgoWaitRuntimeInitDone)
	MOV	8(X2), X10
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	BEQ	X10, ZERO, done
	CGO_PROLOGUE
	MOV	X10, 8(X2)
	CALL_GO(·cgoReleaseContext)
	CGO_EPILOGUE
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOV	X10, X11
	MOV	ZERO, X10
	MOV	ZERO, X13
	JMP	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOV	X10, 8(X2)
	CALL_GO(·threadentry)
	CGO_EPILOGUE
	MOV	ZERO, X10
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOV	X10, 8(X2)
	MOV	X11, 16(X2)
	MOV	X12, 24(X2)
	MOV	X13, 32(X2)
	CALL_GO(·cgoInit)
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)
