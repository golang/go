// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (darwin || linux || netbsd || openbsd) && amd64

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
	JNE	slow
	RET
slow:
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$8
	CALL	·cgoWaitRuntimeInitDone(SB)
	MOVQ	0(SP), AX
	ADJSP	$-8
	POP_REGS_HOST_TO_ABI0()
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
