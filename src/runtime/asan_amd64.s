// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan
// +build asan

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// This is like race_amd64.s, but for the asan calls.
// See race_amd64.s for detailed comments.

#ifdef GOOS_windows
#define RARG0 CX
#define RARG1 DX
#else
#define RARG0 DI
#define RARG1 SI
#endif

// Called from intrumented code.
// func runtime·asanread(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanread(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __asan_read_go(void *addr, uintptr_t sz);
	MOVQ	$__asan_read_go(SB), AX
	JMP	asancall<>(SB)

// func runtime·asanwrite(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanwrite(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __asan_write_go(void *addr, uintptr_t sz);
	MOVQ	$__asan_write_go(SB), AX
	JMP	asancall<>(SB)

// func runtime·asanunpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanunpoison(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __asan_unpoison_go(void *addr, uintptr_t sz);
	MOVQ	$__asan_unpoison_go(SB), AX
	JMP	asancall<>(SB)

// func runtime·asanpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanpoison(SB), NOSPLIT, $0-16
	MOVQ	addr+0(FP), RARG0
	MOVQ	size+8(FP), RARG1
	// void __asan_poison_go(void *addr, uintptr_t sz);
	MOVQ	$__asan_poison_go(SB), AX
	JMP	asancall<>(SB)

// Switches SP to g0 stack and calls (AX). Arguments already set.
TEXT	asancall<>(SB), NOSPLIT, $0-0
	get_tls(R12)
	MOVQ	g(R12), R14
	MOVQ	SP, R12		// callee-saved, preserved across the CALL
	CMPQ	R14, $0
	JE	call	// no g; still on a system stack

	MOVQ	g_m(R14), R13
	// Switch to g0 stack.
	MOVQ	m_g0(R13), R10
	CMPQ	R10, R14
	JE	call	// already on g0

	MOVQ	(g_sched+gobuf_sp)(R10), SP
call:
	ANDQ	$~15, SP	// alignment for gcc ABI
	CALL	AX
	MOVQ	R12, SP
	RET
