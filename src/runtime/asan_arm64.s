// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

#include "go_asm.h"
#include "textflag.h"

#define RARG0 R0
#define RARG1 R1
#define RARG2 R2
#define RARG3 R3
#define FARG R4

// Called from instrumented code.
// func runtime·doasanread(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanread(SB), NOSPLIT, $0-32
	MOVD	addr+0(FP), RARG0
	MOVD	size+8(FP), RARG1
	MOVD	sp+16(FP), RARG2
	MOVD	pc+24(FP), RARG3
	// void __asan_read_go(void *addr, uintptr_t sz, void *sp, void *pc);
	MOVD	$__asan_read_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·doasanwrite(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanwrite(SB), NOSPLIT, $0-32
	MOVD	addr+0(FP), RARG0
	MOVD	size+8(FP), RARG1
	MOVD	sp+16(FP), RARG2
	MOVD	pc+24(FP), RARG3
	// void __asan_write_go(void *addr, uintptr_t sz, void *sp, void *pc);
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

	MOVD	(g_sched+gobuf_sp)(R11), R5
	MOVD	R5, RSP

g0stack:
	BL	(FARG)
	MOVD	R19, RSP
	RET
