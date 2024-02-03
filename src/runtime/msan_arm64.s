// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build msan

#include "go_asm.h"
#include "textflag.h"

#define RARG0 R0
#define RARG1 R1
#define RARG2 R2
#define FARG R3

// func runtime·domsanread(addr unsafe.Pointer, sz uintptr)
// Called from msanread.
TEXT	runtime·domsanread(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	// void __msan_read_go(void *addr, uintptr_t sz);
	MOVD	$__msan_read_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanwrite(addr unsafe.Pointer, sz uintptr)
// Called from instrumented code.
TEXT	runtime·msanwrite(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	// void __msan_write_go(void *addr, uintptr_t sz);
	MOVD	$__msan_write_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanmalloc(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·msanmalloc(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	// void __msan_malloc_go(void *addr, uintptr_t sz);
	MOVD	$__msan_malloc_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanfree(addr unsafe.Pointer, sz uintptr)
TEXT	runtime·msanfree(SB), NOSPLIT, $0-16
	MOVD	addr+0(FP), RARG0
	MOVD	sz+8(FP), RARG1
	// void __msan_free_go(void *addr, uintptr_t sz);
	MOVD	$__msan_free_go(SB), FARG
	JMP	msancall<>(SB)

// func runtime·msanmove(dst, src unsafe.Pointer, sz uintptr)
TEXT	runtime·msanmove(SB), NOSPLIT, $0-24
	MOVD	dst+0(FP), RARG0
	MOVD	src+8(FP), RARG1
	MOVD	sz+16(FP), RARG2
	// void __msan_memmove(void *dst, void *src, uintptr_t sz);
	MOVD	$__msan_memmove(SB), FARG
	JMP	msancall<>(SB)

// Switches SP to g0 stack and calls (FARG). Arguments already set.
TEXT	msancall<>(SB), NOSPLIT, $0-0
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
