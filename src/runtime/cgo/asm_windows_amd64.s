// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && windows && amd64

#include "go_asm.h"
#include "textflag.h"
#include "abi_amd64.h"

// cgoCallTrampoline is called by runtime.asmcgocall_landingpad or
// runtime.asmcgocall_no_g using the Win64 C ABI with CX pointing to a
// cgoCallFrame. Reserves 32 bytes of Win64 shadow space on a 16-byte aligned
// stack.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	ADJSP	$40		// record shadow space and alignment for unwinding
	MOVQ	CX, BX
	MOVQ	cgoCallFrame_fn(CX), AX
	MOVQ	cgoCallFrame_a3(CX), R9
	MOVQ	cgoCallFrame_a2(CX), R8
	MOVQ	cgoCallFrame_a1(CX), DX
	MOVQ	cgoCallFrame_a0(CX), CX
	CALL	AX
	MOVQ	AX, cgoCallFrame_r1(BX)
	XORL	AX, AX
	ADJSP	$-40
	POPQ	BX
	POPQ	BP
	RET

// On Windows we call the ABIInternal targets directly rather than going
// through an ABI0 wrapper: the compiler-generated ABI0 wrapper loads g from
// static TLS, which may be null on C threads created before a c-shared Go DLL
// is loaded via LoadLibrary.

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT,$0-0
	LEAQ	·runtime_init_done(SB), AX
	MOVL	(AX), AX
	CMPL	AX, $2
	JNE	slow
	LEAQ	·cgo_context_function(SB), AX
	MOVQ	(AX), AX
	TESTQ	AX, AX
	JNE	slow
	XORL	AX, AX
	RET
slow:
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
	CALL	·cgoWaitRuntimeInitDone<ABIInternal>(SB)
	POP_REGS_HOST_TO_ABI0()
	RET

TEXT _cgo_release_context(SB),NOSPLIT,$0-0
	TESTQ	CX, CX
	JEQ	done
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$8		// spill slot for ctxt argument
	MOVQ	CX, AX
	XORPS	X15, X15
#ifndef GOAMD64_v3
#ifndef GOAMD64_v4
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JNE	2(PC)
#endif
#endif
	VXORPS	X15, X15, X15
	XORQ	R14, R14
	CALL	·cgoReleaseContext<ABIInternal>(SB)
	ADJSP	$-8
	POP_REGS_HOST_TO_ABI0()
done:
	RET
