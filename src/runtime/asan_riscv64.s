// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

#include "go_asm.h"
#include "textflag.h"

// Called from instrumented code.
// func runtime·doasanread(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanread(SB), NOSPLIT, $0-32
	MOV	addr+0(FP), X10
	MOV	sz+8(FP), X11
	MOV	sp+16(FP), X12
	MOV	pc+24(FP), X13
	// void __asan_read_go(void *addr, uintptr_t sz);
	MOV	$__asan_read_go(SB), X14
	JMP	asancall<>(SB)

// func runtime·doasanwrite(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanwrite(SB), NOSPLIT, $0-32
	MOV	addr+0(FP), X10
	MOV	sz+8(FP), X11
	MOV	sp+16(FP), X12
	MOV	pc+24(FP), X13
	// void __asan_write_go(void *addr, uintptr_t sz);
	MOV	$__asan_write_go(SB), X14
	JMP	asancall<>(SB)

// func runtime·asanunpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanunpoison(SB), NOSPLIT, $0-16
	MOV	addr+0(FP), X10
	MOV	sz+8(FP), X11
	// void __asan_unpoison_go(void *addr, uintptr_t sz);
	MOV	$__asan_unpoison_go(SB), X14
	JMP	asancall<>(SB)

// func runtime·asanpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanpoison(SB), NOSPLIT, $0-16
	MOV	addr+0(FP), X10
	MOV	sz+8(FP), X11
	// void __asan_poison_go(void *addr, uintptr_t sz);
	MOV	$__asan_poison_go(SB), X14
	JMP	asancall<>(SB)

// func runtime·asanregisterglobals(addr unsafe.Pointer, n uintptr)
TEXT	runtime·asanregisterglobals(SB), NOSPLIT, $0-16
	MOV	addr+0(FP), X10
	MOV	n+8(FP), X11
	// void __asan_register_globals_go(void *addr, uintptr_t n);
	MOV	$__asan_register_globals_go(SB), X14
	JMP	asancall<>(SB)

// Switches SP to g0 stack and calls (X14). Arguments already set.
TEXT	asancall<>(SB), NOSPLIT, $0-0
	MOV	X2, X8		// callee-saved
	BEQZ	g, g0stack	// no g, still on a system stack
	MOV	g_m(g), X21
	MOV	m_g0(X21), X21
	BEQ	X21, g, g0stack

	MOV	(g_sched+gobuf_sp)(X21), X2

g0stack:
	JALR	RA, X14
	MOV	X8, X2
	RET
