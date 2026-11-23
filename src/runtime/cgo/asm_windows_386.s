// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && windows && 386

#include "go_asm.h"
#include "textflag.h"

// cgoCallTrampoline is called on the system stack by runtime.asmcgocall with
// a *cgoCallFrame at 4(SP). Restoring SP from BP handles both cdecl and
// stdcall callees. It returns TEB.LastErrorValue in AX, so that
// runtime.cgocall returns the error before runtime.exitsyscall can clobber
// it (see cgocallErr).
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	PUSHL	BP
	MOVL	SP, BP
	// runtime.asmcgocall and runtime.asmcgocall_no_g align SP to 16 bytes
	// before CALL, so SP is 16-byte aligned after PUSHL BP and SUBL $40.
	SUBL	$40, SP
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
	MOVL	0x34(FS), AX	// TEB.LastErrorValue
	MOVL	-4(BP), BX
	MOVL	BP, SP
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
	MOVL	·runtime_init_done(SB), AX
	CMPL	AX, $2
	JNE	slow
	MOVL	·cgo_context_function(SB), AX
	TESTL	AX, AX
	JNE	slow
	XORL	AX, AX
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
