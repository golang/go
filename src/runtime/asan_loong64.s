// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

#include "go_asm.h"
#include "textflag.h"

#define RARG0 R4
#define RARG1 R5
#define RARG2 R6
#define RARG3 R7
#define FARG  R8

// Called from instrumented code.
// func runtime·doasanread(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanread(SB), NOSPLIT, $0-32
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	MOVV	sp+16(FP), RARG2
	MOVV	pc+24(FP), RARG3
	// void __asan_read_go(void *addr, uintptr_t sz, void *sp, void *pc);
	MOVV	$__asan_read_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·doasanwrite(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanwrite(SB), NOSPLIT, $0-32
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	MOVV	sp+16(FP), RARG2
	MOVV	pc+24(FP), RARG3
	// void __asan_write_go(void *addr, uintptr_t sz, void *sp, void *pc);
	MOVV	$__asan_write_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·asanunpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanunpoison(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	// void __asan_unpoison_go(void *addr, uintptr_t sz);
	MOVV	$__asan_unpoison_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·asanpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanpoison(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	// void __asan_poison_go(void *addr, uintptr_t sz);
	MOVV	$__asan_poison_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·asanregisterglobals(addr unsafe.Pointer, n uintptr)
TEXT	runtime·asanregisterglobals(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	n+8(FP), RARG1
	// void __asan_register_globals_go(void *addr, uintptr_t n);
	MOVV	$__asan_register_globals_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·lsanregisterrootregion(addr unsafe.Pointer, n uintptr)
TEXT	runtime·lsanregisterrootregion(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	n+8(FP), RARG1
	// void __lsan_register_root_region_go(void *addr, uintptr_t n);
	MOVV	$__lsan_register_root_region_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·lsanunregisterrootregion(addr unsafe.Pointer, n uintptr)
TEXT	runtime·lsanunregisterrootregion(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	n+8(FP), RARG1
	// void __lsan_unregister_root_region_go(void *addr, uintptr_t n);
	MOVV	$__lsan_unregister_root_region_go(SB), FARG
	JMP	asancall<>(SB)

// func runtime·lsandoleakcheck()
TEXT	runtime·lsandoleakcheck(SB), NOSPLIT, $0-0
	// void __lsan_do_leak_check_go(void);
	MOVV	$__lsan_do_leak_check_go(SB), FARG
	JMP	asancall<>(SB)

// Switches SP to g0 stack and calls (FARG). Arguments already set.
TEXT	asancall<>(SB), NOSPLIT, $0-0
	MOVV	R3, R23         // callee-saved
	BEQ	g, call         // no g, still on a system stack
	MOVV	g_m(g), R14

	// Switch to g0 stack if we aren't already on g0 or gsignal.
	MOVV	m_gsignal(R14), R15
	BEQ	R15, g, call

	MOVV	m_g0(R14), R15
	BEQ	R15, g, call

	MOVV	(g_sched+gobuf_sp)(R15), R9
	MOVV	R9, R3

call:
	JAL	(FARG)
	MOVV	R23, R3
	RET
