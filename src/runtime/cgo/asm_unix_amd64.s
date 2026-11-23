// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix && amd64

#include "go_asm.h"
#include "textflag.h"
#include "abi_amd64.h"

// cgoCallTrampoline is called by runtime.asmcgocall (or runtime.cgocall)
// using the System V AMD64 C ABI with DI pointing to a cgoCallFrame.
// The target function may be variadic: AX is cleared to tell it that no
// arguments are passed in vector registers.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	ADJSP	$8		// align SP before CALL and record the change for unwinding
	MOVQ	DI, BX
	MOVQ	cgoCallFrame_fn(DI), R11
	MOVQ	cgoCallFrame_a1(DI), SI
	MOVQ	cgoCallFrame_a2(DI), DX
	MOVQ	cgoCallFrame_a3(DI), CX
	MOVQ	cgoCallFrame_a4(DI), R8
	MOVQ	cgoCallFrame_a5(DI), R9
	MOVQ	cgoCallFrame_a0(DI), DI
	XORL	AX, AX		// vararg: say "no float args"
	CALL	R11
	MOVQ	AX, cgoCallFrame_r1(BX)
	XORL	AX, AX
	ADJSP	$-8
	POPQ	BX
	POPQ	BP
	RET

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	// Use LEAQ to avoid clobbering R15 (callee-saved in the C ABI)
	// when dynamic linking rewrites global accesses via the GOT.
	LEAQ	·runtime_init_done(SB), AX
	MOVL	(AX), AX
	CMPL	AX, $2
	JNE	slow
	LEAQ	·cgo_context_function(SB), AX
	MOVQ	(AX), AX
	TESTQ	AX, AX
	JNZ	callctx
	RET
slow:
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$8
	CALL	·cgoWaitRuntimeInit(SB)
	MOVQ	0(SP), AX
	ADJSP	$-8
	POP_REGS_HOST_TO_ABI0()
	TESTQ	AX, AX
	JNZ	callctx
	RET
callctx:
	JMP	·x_cgo_call_context(SB)

// x_cgo_call_context calls the registered cgo context function (passed in AX).
// cmd/link emits static unwind info matching the standard frame-pointer prologue
// so callbacks using _Unwind_Backtrace unwind cleanly into the C caller.
TEXT ·x_cgo_call_context(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	SUBQ	$16, SP

	MOVQ	$0, 0(SP)	// arg.context = 0
#ifdef const_msanEnabled
	MOVQ	AX, 8(SP)	// save pfn

	// Unpoison arg on the C stack.
	LEAQ	·msanUnpoisonFn(SB), AX
	MOVQ	(AX), AX
	MOVQ	SP, DI		// &arg
	MOVQ	$8, SI		// sizeof(arg)
	CALL	AX
	MOVQ	8(SP), AX	// pfn
#endif
	MOVQ	SP, DI		// &arg
	CALL	AX

	MOVQ	0(SP), AX	// return arg.context
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	TESTQ	DI, DI
	JEQ	done
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$8
	MOVQ	DI, 0(SP)
	CALL	·cgoReleaseContext(SB)
	ADJSP	$-8
	POP_REGS_HOST_TO_ABI0()
done:
	RET

TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
	MOVQ	DI, SI
	XORL	DI, DI
	XORL	CX, CX
	JMP	crosscall2(SB)

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	PUSH_REGS_HOST_TO_ABI0()
	XORPS	X15, X15
#ifndef GOAMD64_v3
#ifndef GOAMD64_v4
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JNE	2(PC)
#endif
#endif
	VXORPS	X15, X15, X15
	XORQ	R14, R14
	MOVQ	DI, AX
	CALL	·threadentry<ABIInternal>(SB)
	POP_REGS_HOST_TO_ABI0()
	XORL	AX, AX
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$32
	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	MOVQ	DX, 16(SP)
	MOVQ	CX, 24(SP)
	CALL	·cgoInit(SB)
	ADJSP	$-32
	POP_REGS_HOST_TO_ABI0()
	RET

GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)

// x_cgo_callers is jumped to from runtime.cgoSigtramp on the signal stack.
// cmd/link emits static unwind info matching the standard frame-pointer prologue
// so traceback callbacks unwind cleanly into the OS signal trampoline.
TEXT ·x_cgo_callers(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	SUBQ	$80, SP

	MOVQ	$0, 0(SP)	// arg.context = 0
	MOVQ	DX, 8(SP)	// arg.sigContext = context
	MOVQ	R8, 16(SP)	// arg.buf = cgoCallers
	MOVQ	$32, 24(SP)	// arg.max = 32

	MOVQ	DI, 32(SP)	// save sig
	MOVQ	SI, 40(SP)	// save info
	MOVQ	DX, 48(SP)	// save context
	MOVQ	R9, 56(SP)	// save sigtramp
	MOVQ	CX, 64(SP)	// save cgoTraceback

#ifdef const_msanEnabled
	// Unpoison arg on the signal stack (issue #47543).
	LEAQ	·msanUnpoisonFn(SB), AX
	MOVQ	(AX), AX
	MOVQ	SP, DI		// &arg
	MOVQ	$32, SI		// sizeof(arg)
	CALL	AX
#endif
	LEAQ	·tsanAcquireFnPtr(SB), AX
	MOVQ	(AX), AX
	MOVQ	(AX), AX
	TESTQ	AX, AX
	JZ	callers_acq_done
	LEAQ	·cgoSyncPtr(SB), DI
	MOVQ	(DI), DI
	CALL	AX
callers_acq_done:
	MOVQ	SP, DI		// &arg
	MOVQ	64(SP), AX	// cgoTraceback
	CALL	AX		// (*cgoTraceback)(&arg)

	LEAQ	·tsanReleaseFnPtr(SB), AX
	MOVQ	(AX), AX
	MOVQ	(AX), AX
	TESTQ	AX, AX
	JZ	callers_rel_done
	LEAQ	·cgoSyncPtr(SB), DI
	MOVQ	(DI), DI
	CALL	AX
callers_rel_done:
	MOVQ	32(SP), DI	// restore sig
	MOVQ	40(SP), SI	// restore info
	MOVQ	48(SP), DX	// restore context
	MOVQ	56(SP), AX	// restore sigtramp

	MOVQ	BP, SP
	POPQ	BP
	JMP	AX		// sigtramp(sig, info, context)

GLOBL _cgo_callers(SB), NOPTR, $8
DATA _cgo_callers+0(SB)/8, $·x_cgo_callers(SB)

#ifdef GOOS_darwin
#define CGO_DARWIN
#endif
#ifdef GOOS_ios
#define CGO_DARWIN
#endif
#ifdef CGO_DARWIN
// cgoTSanYield calls strncpy(&cgoYieldDst, &cgoYieldSrc, 0) using the C ABI
// so TSAN libc interceptors deliver pending asynchronous signals.
TEXT ·cgoTSanYield(SB),NOSPLIT|NOFRAME,$0-0
	LEAQ	·cgoYieldDst(SB), DI
	LEAQ	·cgoYieldSrc(SB), SI
	XORL	DX, DX
	LEAQ	·libc_strncpy(SB), AX
	MOVQ	(AX), AX
	JMP	AX
#endif
