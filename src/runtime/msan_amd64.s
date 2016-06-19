// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build msan

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// This is like race_amd64.s, but for the msan calls.
// See race_amd64.s for detailed comments.

#ifdef GOOS_windows
#define RARG0 CX
#define RARG1 DX
#define RARG2 R8
#define RARG3 R9
#else
#define RARG0 DI
#define RARG1 SI
#define RARG2 DX
#define RARG3 CX
#endif

// func runtime·domsanread(addr unsafe.Pointer, sz uintptr)
// Called from msanread.
TEXT	runtime·domsanread(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __msan_read_go(void *addr, uintptr_t sz);
	MOVQ	$__msan_read_go(SB), AX
	JMP	msancall<>(SB)

// func runtime·msanwrite(addr unsafe.Pointer, sz uintptr)
// Called from instrumented code.
TEXT	runtime·msanwrite(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __msan_write_go(void *addr, uintptr_t sz);
	MOVQ	$__msan_write_go(SB), AX
	JMP	msancall<>(SB)

// func runtime·msanmalloc(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·msanmalloc(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __msan_malloc_go(void *addr, uintptr_t sz);
	MOVQ	$__msan_malloc_go(SB), AX
	JMP	msancall<>(SB)

// func runtime·msanfree(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·msanfree(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __msan_free_go(void *addr, uintptr_t sz);
	MOVQ	$__msan_free_go(SB), AX
	JMP	msancall<>(SB)

// Switches SP to g0 stack and calls (AX). Arguments already set.
TEXT	msancall<>(SB), NOSPLIT, $0-0
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
