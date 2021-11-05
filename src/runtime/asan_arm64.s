// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan
// +build asan

#include "go_asm.h"
#include "textflag.h"

#define RARG0 R0
#define RARG1 R1
#define FARG R3

// Called from instrumented code.
// func runtime·asanread(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanread(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	size+8(FP), RARG1
	// void __asan_read_go(void *addr, uintptr_t sz);
	MOVD	$__asan_read_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·asanwrite(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanwrite(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	size+8(FP), RARG1
	// void __asan_write_go(void *addr, uintptr_t sz);
	MOVD	$__asan_write_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·asanunpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanunpoison(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	size+8(FP), RARG1
	// void __asan_unpoison_go(void *addr, uintptr_t sz);
	MOVD	$__asan_unpoison_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·asanpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanpoison(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	size+8(FP), RARG1
	// void __asan_poison_go(void *addr, uintptr_t sz);
	MOVD	$__asan_poison_go(SB), FARG
	JMP	asancall<>(SB)

// Switches SP to g0 stack and calls (FARG). Arguments already set.
TEXT	asancall<>(SB), NOSPLIT, $0-0
	MOVD	RSP, R19                  // callee-saved
	CBZ	g, g0stack                // no g, still on a system stack
	MOVD	g_m(g), R10
	MOVD	m_g0(R10), R11
	CMP	R11, g
	BEQ	g0stack

	MOVD	(g_sched+gobuf_sp)(R11), R4
	MOVD	R4, RSP

g0stack:
	BL	(FARG)
	MOVD	R19, RSP
	RET
