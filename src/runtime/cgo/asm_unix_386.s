// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix && 386

#include "go_asm.h"
#include "textflag.h"

// cgoCallTrampoline is called by runtime.asmcgocall or runtime.asmcgocall_no_g
// using the System V i386 C ABI with 4(SP) pointing to a cgoCallFrame.
// Both callers 16-byte align SP before CALL, so SP%16 == 12 on entry.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	PUSHL	BP
	MOVL	SP, BP
	// 4B saved BX + 12B padding + 24B outgoing args; keeps SP 16-byte aligned.
	ADJSP	$40
	MOVL	BX, -4(BP)
	MOVL	8(BP), BX
	MOVL	cgoCallFrame_a0(BX), AX
	MOVL	AX, 0(SP)
	MOVL	cgoCallFrame_a1(BX), AX
	MOVL	AX, 4(SP)
	MOVL	cgoCallFrame_a2(BX), AX
	MOVL	AX, 8(SP)
	MOVL	cgoCallFrame_a3(BX), AX
	MOVL	AX, 12(SP)
	MOVL	cgoCallFrame_a4(BX), AX
	MOVL	AX, 16(SP)
	MOVL	cgoCallFrame_a5(BX), AX
	MOVL	AX, 20(SP)
	MOVL	cgoCallFrame_fn(BX), AX
	CALL	AX
	MOVL	AX, cgoCallFrame_r1(BX)
	XORL	AX, AX
	MOVL	-4(BP), BX
	ADJSP	$-40
	POPL	BP
	RET

#define CGO_PROLOGUE \
	CLD; \
	PUSHL	BP; \
	PUSHL	BX; \
	PUSHL	SI; \
	PUSHL	DI

#define CGO_EPILOGUE \
	POPL	DI; \
	POPL	SI; \
	POPL	BX; \
	POPL	BP

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	LEAL	·runtime_init_done(SB), AX
	MOVL	(AX), AX
	CMPL	AX, $2
	JNE	slow
	LEAL	·cgo_context_function(SB), AX
	MOVL	(AX), AX
	TESTL	AX, AX
	JNE	slow
	RET
slow:
	CGO_PROLOGUE
	ADJSP	$4
	CALL	·cgoWaitRuntimeInitDone(SB)
	MOVL	0(SP), AX
	ADJSP	$-4
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	MOVL	4(SP), AX
	TESTL	AX, AX
	JEQ	done
	CGO_PROLOGUE
	ADJSP	$4
	MOVL	AX, 0(SP)
	CALL	·cgoReleaseContext(SB)
	ADJSP	$-4
	CGO_EPILOGUE
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$20, SP
	ANDL	$~15, SP
	// BX is callee-saved in C, and -dynlink code clobbers it.
	MOVL	BX, -4(BP)
	MOVL	8(BP), AX
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)
	MOVL	$0, 8(SP)
	MOVL	$0, 12(SP)
	CALL	crosscall2(SB)
	MOVL	-4(BP), BX
	MOVL	BP, SP
	POPL	BP
	RET

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVL	20(SP), AX
	ADJSP	$4
	MOVL	AX, 0(SP)
	CALL	·threadentry(SB)
	ADJSP	$-4
	CGO_EPILOGUE
	XORL	AX, AX
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	ADJSP	$16
	MOVL	36(SP), AX
	MOVL	AX, 0(SP)
	MOVL	40(SP), AX
	MOVL	AX, 4(SP)
	MOVL	44(SP), AX
	MOVL	AX, 8(SP)
	MOVL	48(SP), AX
	MOVL	AX, 12(SP)
	CALL	·cgoInit(SB)
	ADJSP	$-16
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $4
DATA _cgo_init+0(SB)/4, $x_cgo_init<>(SB)
