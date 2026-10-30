// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && (darwin || (linux && !android)) && amd64

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
	SUBQ	$8, SP		// keep SP 16-byte aligned before CALL
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
	ADDQ	$8, SP
	POPQ	BX
	POPQ	BP
	RET

TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$8
	MOVQ	DI, 0(SP)
	CALL	·threadentry(SB)
	ADJSP	$-8
	POP_REGS_HOST_TO_ABI0()
	XORL	AX, AX
	RET

TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$16
	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	CALL	·cgoInit(SB)
	ADJSP	$-16
	POP_REGS_HOST_TO_ABI0()
	RET

GLOBL _cgo_init(SB), NOPTR, $8
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)
