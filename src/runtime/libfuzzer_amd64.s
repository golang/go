// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build libfuzzer

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Based on race_amd64.s; see commentary there.

#ifdef GOOS_windows
#define RARG0 CX
#define RARG1 DX
#else
#define RARG0 DI
#define RARG1 SI
#endif

// void runtime·libfuzzerCall(fn, arg0, arg1 uintptr)
// Calls C function fn from libFuzzer and passes 2 arguments to it.
TEXT	runtime·libfuzzerCall(SB), NOSPLIT, $0-24
	MOVQ	fn+0(FP), AX
	MOVQ	arg0+8(FP), RARG0
	MOVQ	arg1+16(FP), RARG1

	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	g_m(R14), R13

	// Switch to g0 stack.
	MOVQ	SP, R12		// callee-saved, preserved across the CALL
	MOVQ	m_g0(R13), R10
	CMPQ	R10, R14
	JE	call	// already on g0
	MOVQ	(g_sched+gobuf_sp)(R10), SP
call:
	ANDQ	$~15, SP	// alignment for gcc ABI
	CALL	AX
	MOVQ	R12, SP
	RET
