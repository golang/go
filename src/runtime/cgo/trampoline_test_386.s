// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix

#include "textflag.h"

// Called on g0 by cgoCallTrampoline with the PC buffer at 4(SP), its length
// at 8(SP), and runtime.Callers at 12(SP). Call Callers(0, buf) using the
// 386 stack ABI while preserving C callee-saved registers.
TEXT ·trampolineCallers(SB),NOSPLIT|NOFRAME,$0-0
	PUSHL	BP
	MOVL	SP, BP
	ADJSP	$32
	MOVL	BX, 20(SP)
	MOVL	SI, 24(SP)
	MOVL	DI, 28(SP)
	MOVL	$0, 0(SP)
	MOVL	8(BP), AX
	MOVL	AX, 4(SP)
	MOVL	12(BP), AX
	MOVL	AX, 8(SP)
	MOVL	AX, 12(SP)
	MOVL	16(BP), AX
	CALL	AX
	MOVL	16(SP), AX
	MOVL	20(SP), BX
	MOVL	24(SP), SI
	MOVL	28(SP), DI
	ADJSP	$-32
	POPL	BP
	RET
