// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

#include "go_asm.h"
#include "textflag.h"

#define RARG0 R3
#define RARG1 R4
#define RARG2 R5
#define RARG3 R6
#define FARG R12

// Called from instrumented code.
// func runtime·doasanread(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanread(SB),NOSPLIT|NOFRAME,$0-32
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	MOVD	sp+16(FP), RARG2
	MOVD	pc+24(FP), RARG3
	// void __asan_read_go(void *addr, uintptr_t sz, void *sp, void *pc);
	MOVD	$__asan_read_go(SB), FARG
	BR	asancall<>(SB)

// func runtime·doasanwrite(addr unsafe.Pointer, sz, sp, pc uintptr)
TEXT	runtime·doasanwrite(SB),NOSPLIT|NOFRAME,$0-32
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	MOVD	sp+16(FP), RARG2
	MOVD	pc+24(FP), RARG3
	// void __asan_write_go(void *addr, uintptr_t sz, void *sp, void *pc);
	MOVD	$__asan_write_go(SB), FARG
	BR	asancall<>(SB)

// func runtime·asanunpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanunpoison(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	// void __asan_unpoison_go(void *addr, uintptr_t sz);
	MOVD	$__asan_unpoison_go(SB), FARG
	BR	asancall<>(SB)

// func runtime·asanpoison(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·asanpoison(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	// void __asan_poison_go(void *addr, uintptr_t sz);
	MOVD	$__asan_poison_go(SB), FARG
	BR	asancall<>(SB)

// func runtime·asanregisterglobals(addr unsafe.Pointer, n uintptr)
TEXT	runtime·asanregisterglobals(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	addr+0(FP), RARG0
	MOVD	n+8(FP), RARG1
	// void __asan_register_globals_go(void *addr, uintptr_t n);
	MOVD	$__asan_register_globals_go(SB), FARG
	BR	asancall<>(SB)

// func runtime·lsanregisterrootregion(addr unsafe.Pointer, n uintptr)
TEXT	runtime·lsanregisterrootregion(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	addr+0(FP), RARG0
	MOVD	n+8(FP), RARG1
	// void __lsan_register_root_region_go(void *addr, uintptr_t n);
	MOVD	$__lsan_register_root_region_go(SB), FARG
	BR	asancall<>(SB)

// func runtime·lsandoleakcheck()
TEXT	runtime·lsandoleakcheck(SB), NOSPLIT|NOFRAME, $0-0
	// void __lsan_do_leak_check_go(void);
	MOVD	$__lsan_do_leak_check_go(SB), FARG
	BR	asancall<>(SB)

// Switches SP to g0 stack and calls (FARG). Arguments already set.
TEXT	asancall<>(SB), NOSPLIT, $0-0
	// LR saved in generated prologue
	// Get info from the current goroutine
	MOVD	runtime·tls_g(SB), R10  // g offset in TLS
	MOVD	0(R10), g
	MOVD	g_m(g), R7		// m for g
	MOVD	R1, R16			// callee-saved, preserved across C call
	MOVD	m_g0(R7), R10		// g0 for m
	CMP	R10, g			// same g0?
	BEQ	call			// already on g0
	MOVD	(g_sched+gobuf_sp)(R10), R1 // switch R1
call:
	// prepare frame for C ABI
	SUB	$32, R1			// create frame for callee saving LR, CR, R2 etc.
	RLDCR	$0, R1, $~15, R1	// align SP to 16 bytes
	MOVD	FARG, CTR		// address of function to be called
	MOVD	R0, 0(R1)		// clear back chain pointer
	BL	(CTR)
	MOVD	$0, R0			// C code can clobber R0 set it back to 0
	MOVD	R16, R1			// restore R1;
	MOVD	runtime·tls_g(SB), R10	// find correct g
	MOVD	0(R10), g
	RET

// tls_g, g value for each thread in TLS
GLOBL runtime·tls_g+0(SB), TLSBSS+DUPOK, $8
