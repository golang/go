// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build msan

#include "go_asm.h"
#include "textflag.h"

#define RARG0 R4
#define RARG1 R5
#define RARG2 R6
#define FARG  R7

// func runtime·domsanread(addr unsafe.Pointer, sz uintptr)
// Called from msanread.
TEXT	runtime·domsanread(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	// void __msan_read_go(void *addr, uintptr_t sz);
	MOVV	$__msan_read_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanwrite(addr unsafe.Pointer, sz uintptr)
// Called from instrumented code.
TEXT	runtime·msanwrite(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	// void __msan_write_go(void *addr, uintptr_t sz);
	MOVV	$__msan_write_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanmalloc(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·msanmalloc(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	// void __msan_malloc_go(void *addr, uintptr_t sz);
	MOVV	$__msan_malloc_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanfree(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·msanfree(SB), NOSPLIT, $0-16
	MOVV	addr+0(FP), RARG0
	MOVV	sz+8(FP), RARG1
	// void __msan_free_go(void *addr, uintptr_t sz);
	MOVV	$__msan_free_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanmove(dst, src unsafe.Pointer, sz uintptr)
TEXT	runtime·msanmove(SB), NOSPLIT, $0-24
	MOVV	dst+0(FP), RARG0
	MOVV	src+8(FP), RARG1
	MOVV	sz+16(FP), RARG2
	// void __msan_memmove_go(void *dst, void *src, uintptr_t sz);
	MOVV	$__msan_memmove_go(SB), FARG
	JMP	msancall<>(SB)

// Switches SP to g0 stack and calls (FARG). Arguments already set.
TEXT	msancall<>(SB), NOSPLIT, $0-0
	MOVV	R3, R23         // callee-saved
	BEQ	g, g0stack      // no g, still on a system stack
	MOVV	g_m(g), R14
	MOVV	m_g0(R14), R15
	BEQ	R15, g, g0stack

	MOVV	(g_sched+gobuf_sp)(R15), R9
	MOVV	R9, R3

g0stack:
	JAL	(FARG)
	MOVV	R23, R3
	RET
