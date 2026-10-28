// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && linux && !android && amd64

#include "go_asm.h"
#include "textflag.h"

// cgoSyscallTrampoline is called by runtime.cgocall using the System V AMD64
// C ABI with DI pointing to a cgoSyscallFrame. It calls fn(a0, a1, a2) and,
// if fn returns -1, fetches errno via errnoFn before returning so both calls
// execute on the same OS thread.
TEXT ·cgoSyscallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	SUBQ	$8, SP		// keep SP 16-byte aligned before CALL
	MOVQ	DI, BX
	MOVQ	cgoSyscallFrame_fn(DI), R11
	MOVQ	cgoSyscallFrame_a1(DI), SI
	MOVQ	cgoSyscallFrame_a2(DI), DX
	MOVQ	cgoSyscallFrame_a0(DI), DI
	XORL	AX, AX
	CALL	R11
	MOVLQSX	AX, AX
	CMPQ	AX, $-1
	JNE	ok
	CALL    libc_errno_location(SB)
	MOVLQSX	(AX), AX
ok:
	MOVQ	AX, cgoSyscallFrame_r1(BX)
	XORL	AX, AX
	ADDQ	$8, SP
	POPQ	BX
	POPQ	BP
	RET
