// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "amd64/asm.h"

TEXT	_rt0_amd64(SB),7,$-8

	// copy arguments forward on an even stack
	MOVQ	0(SP), AX		// argc
	LEAQ	8(SP), BX		// argv
	SUBQ	$(4*8+7), SP		// 2args 2auto
	ANDQ	$~7, SP
	MOVQ	AX, 16(SP)
	MOVQ	BX, 24(SP)

	// set the per-goroutine and per-mach registers
	LEAQ	m0(SB), m
	LEAQ	g0(SB), g
	MOVQ	g, m_g0(m)		// m has pointer to its g0

	// create istack out of the given (operating system) stack
	LEAQ	(-8192+104)(SP), AX
	MOVQ	AX, g_stackguard(g)
	MOVQ	SP, g_stackbase(g)

	CLD				// convention is D is always left cleared
	CALL	check(SB)

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	args(SB)
	CALL	osinit(SB)
	CALL	schedinit(SB)

	// create a new goroutine to start program
	PUSHQ	$mainstart(SB)		// entry
	PUSHQ	$0			// arg size
	CALL	sys·newproc(SB)
	POPQ	AX
	POPQ	AX

	// start this M
	CALL	mstart(SB)

	CALL	notok(SB)		// never returns
	RET

TEXT mainstart(SB),7,$0
	CALL	main·init(SB)
	CALL	initdone(SB)
	CALL	main·main(SB)
	PUSHQ	$0
	CALL	exit(SB)
	POPQ	AX
	CALL	notok(SB)
	RET

TEXT	breakpoint(SB),7,$0
	BYTE	$0xcc
	RET

/*
 *  go-routine
 */

// uintptr gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT gosave(SB), 7, $0
	MOVQ	8(SP), AX		// gobuf
	LEAQ	8(SP), BX		// caller's SP
	MOVQ	BX, gobuf_sp(AX)
	MOVQ	0(SP), BX		// caller's PC
	MOVQ	BX, gobuf_pc(AX)
	MOVQ	g, gobuf_g(AX)
	MOVL	$0, AX			// return 0
	RET

// void gogo(Gobuf*, uintptr)
// restore state from Gobuf; longjmp
TEXT gogo(SB), 7, $0
	MOVQ	16(SP), AX		// return 2nd arg
	MOVQ	8(SP), BX		// gobuf
	MOVQ	gobuf_g(BX), g
	MOVQ	0(g), CX		// make sure g != nil
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_pc(BX), BX
	JMP	BX

// void gogocall(Gobuf*, void (*fn)(void))
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
TEXT gogocall(SB), 7, $0
	MOVQ	16(SP), AX		// fn
	MOVQ	8(SP), BX		// gobuf
	MOVQ	gobuf_g(BX), g
	MOVQ	0(g), CX		// make sure g != nil
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_pc(BX), BX
	PUSHQ	BX
	JMP	AX
	POPQ	BX	// not reached

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
TEXT sys·morestack(SB),7,$0
	// Called from f.
	// Set m->morebuf to f's caller.
	MOVQ	8(SP), AX	// f's caller's PC
	MOVQ	AX, (m_morebuf+gobuf_pc)(m)
	LEAQ	16(SP), AX	// f's caller's SP
	MOVQ	AX, (m_morebuf+gobuf_sp)(m)
	MOVQ	g, (m_morebuf+gobuf_g)(m)

	// Set m->morepc to f's PC.
	MOVQ	0(SP), AX
	MOVQ	AX, m_morepc(m)

	// Call newstack on m's scheduling stack.
	MOVQ	m_g0(m), g
	MOVQ	(m_sched+gobuf_sp)(m), SP
	CALL	newstack(SB)
	MOVQ	$0, 0x1003	// crash if newstack returns
	RET

// Return point when leaving stack.
TEXT sys·lessstack(SB), 7, $0
	// Save return value in m->cret
	MOVQ	AX, m_cret(m)

	// Call oldstack on m's scheduling stack.
	MOVQ	m_g0(m), g
	MOVQ	(m_sched+gobuf_sp)(m), SP
	CALL	oldstack(SB)
	MOVQ	$0, 0x1004	// crash if oldstack returns
	RET

// morestack trampolines
TEXT	sys·morestack00+0(SB),7,$0
	MOVQ	$0, AX
	MOVQ	AX, m_morearg(m)
	MOVQ	$sys·morestack+0(SB), AX
	JMP	AX

TEXT	sys·morestack01+0(SB),7,$0
	SHLQ	$32, AX
	MOVQ	AX, m_morearg(m)
	MOVQ	$sys·morestack+0(SB), AX
	JMP	AX

TEXT	sys·morestack10+0(SB),7,$0
	MOVLQZX	AX, AX
	MOVQ	AX, m_morearg(m)
	MOVQ	$sys·morestack+0(SB), AX
	JMP	AX

TEXT	sys·morestack11+0(SB),7,$0
	MOVQ	AX, m_morearg(m)
	MOVQ	$sys·morestack+0(SB), AX
	JMP	AX

// subcases of morestack01
// with const of 8,16,...48
TEXT	sys·morestack8(SB),7,$0
	PUSHQ	$1
	MOVQ	$sys·morestackx(SB), AX
	JMP	AX

TEXT	sys·morestack16(SB),7,$0
	PUSHQ	$2
	MOVQ	$sys·morestackx(SB), AX
	JMP	AX

TEXT	sys·morestack24(SB),7,$0
	PUSHQ	$3
	MOVQ	$sys·morestackx(SB), AX
	JMP	AX

TEXT	sys·morestack32(SB),7,$0
	PUSHQ	$4
	MOVQ	$sys·morestackx(SB), AX
	JMP	AX

TEXT	sys·morestack40(SB),7,$0
	PUSHQ	$5
	MOVQ	$sys·morestackx(SB), AX
	JMP	AX

TEXT	sys·morestack48(SB),7,$0
	PUSHQ	$6
	MOVQ	$sys·morestackx(SB), AX
	JMP	AX

TEXT	sys·morestackx(SB),7,$0
	POPQ	AX
	SHLQ	$35, AX
	MOVQ	AX, m_morearg(m)
	MOVQ	$sys·morestack(SB), AX
	JMP	AX

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT cas(SB), 7, $0
	MOVQ	8(SP), BX
	MOVL	16(SP), AX
	MOVL	20(SP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ 3(PC)
	MOVL	$0, AX
	RET
	MOVL	$1, AX
	RET

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT jmpdefer(SB), 7, $0
	MOVQ	8(SP), AX	// fn
	MOVQ	16(SP), BX	// caller sp
	LEAQ	-8(BX), SP	// caller sp after CALL
	SUBQ	$5, (SP)	// return to CALL again
	JMP	AX	// but first run the deferred function
