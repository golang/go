// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "amd64/asm.h"

TEXT _rt0_amd64(SB),7,$-8
	// copy arguments forward on an even stack
	MOVQ	0(DI), AX		// argc
	LEAQ	8(DI), BX		// argv
	SUBQ	$(4*8+7), SP		// 2args 2auto
	ANDQ	$~15, SP
	MOVQ	AX, 16(SP)
	MOVQ	BX, 24(SP)

	// if there is an initcgo, call it.
	MOVQ	initcgo(SB), AX
	TESTQ	AX, AX
	JZ	needtls
	CALL	AX
	JMP ok

needtls:
	LEAQ	runtime·tls0(SB), DI
	CALL	runtime·settls(SB)

	// store through it, to make sure it works
	get_tls(BX)
	MOVQ	$0x123, g(BX)
	MOVQ	runtime·tls0(SB), AX
	CMPQ	AX, $0x123
	JEQ 2(PC)
	MOVL	AX, 0	// abort
ok:
	// set the per-goroutine and per-mach "registers"
	get_tls(BX)
	LEAQ	runtime·g0(SB), CX
	MOVQ	CX, g(BX)
	LEAQ	runtime·m0(SB), AX
	MOVQ	AX, m(BX)

	// save m->g0 = g0
	MOVQ	CX, m_g0(AX)

	// create istack out of the given (operating system) stack
	LEAQ	(-8192+104)(SP), AX
	MOVQ	AX, g_stackguard(CX)
	MOVQ	SP, g_stackbase(CX)

	CLD				// convention is D is always left cleared
	CALL	runtime·check(SB)

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	runtime·args(SB)
	CALL	runtime·osinit(SB)
	CALL	runtime·schedinit(SB)

	// create a new goroutine to start program
	PUSHQ	$runtime·mainstart(SB)		// entry
	PUSHQ	$0			// arg size
	CALL	runtime·newproc(SB)
	POPQ	AX
	POPQ	AX

	// start this M
	CALL	runtime·mstart(SB)

	CALL	runtime·notok(SB)		// never returns
	RET

TEXT runtime·mainstart(SB),7,$0
	CALL	main·init(SB)
	CALL	runtime·initdone(SB)
	CALL	main·main(SB)
	PUSHQ	$0
	CALL	runtime·exit(SB)
	POPQ	AX
	CALL	runtime·notok(SB)
	RET

TEXT runtime·breakpoint(SB),7,$0
	BYTE	$0xcc
	RET

/*
 *  go-routine
 */

// uintptr gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), 7, $0
	MOVQ	8(SP), AX		// gobuf
	LEAQ	8(SP), BX		// caller's SP
	MOVQ	BX, gobuf_sp(AX)
	MOVQ	0(SP), BX		// caller's PC
	MOVQ	BX, gobuf_pc(AX)
	get_tls(CX)
	MOVQ	g(CX), BX
	MOVQ	BX, gobuf_g(AX)
	MOVL	$0, AX			// return 0
	RET

// void gogo(Gobuf*, uintptr)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), 7, $0
	MOVQ	16(SP), AX		// return 2nd arg
	MOVQ	8(SP), BX		// gobuf
	MOVQ	gobuf_g(BX), DX
	MOVQ	0(DX), CX		// make sure g != nil
	get_tls(CX)
	MOVQ	DX, g(CX)
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_pc(BX), BX
	JMP	BX

// void gogocall(Gobuf*, void (*fn)(void))
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
TEXT runtime·gogocall(SB), 7, $0
	MOVQ	16(SP), AX		// fn
	MOVQ	8(SP), BX		// gobuf
	MOVQ	gobuf_g(BX), DX
	get_tls(CX)
	MOVQ	DX, g(CX)
	MOVQ	0(DX), CX	// make sure g != nil
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_pc(BX), BX
	PUSHQ	BX
	JMP	AX
	POPQ	BX	// not reached

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// Caller has already done get_tls(CX); MOVQ m(CX), BX.
TEXT runtime·morestack(SB),7,$0
	// Cannot grow scheduler stack (m->g0).
	MOVQ	m_g0(BX), SI
	CMPQ	g(CX), SI
	JNE	2(PC)
	INT	$3

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVQ	8(SP), AX	// f's caller's PC
	MOVQ	AX, (m_morebuf+gobuf_pc)(BX)
	LEAQ	16(SP), AX	// f's caller's SP
	MOVQ	AX, (m_morebuf+gobuf_sp)(BX)
	MOVQ	AX, m_moreargp(BX)
	get_tls(CX)
	MOVQ	g(CX), SI
	MOVQ	SI, (m_morebuf+gobuf_g)(BX)

	// Set m->morepc to f's PC.
	MOVQ	0(SP), AX
	MOVQ	AX, m_morepc(BX)

	// Call newstack on m's scheduling stack.
	MOVQ	m_g0(BX), BP
	MOVQ	BP, g(CX)
	MOVQ	(m_sched+gobuf_sp)(BX), SP
	CALL	runtime·newstack(SB)
	MOVQ	$0, 0x1003	// crash if newstack returns
	RET

// Called from reflection library.  Mimics morestack,
// reuses stack growth code to create a frame
// with the desired args running the desired function.
//
// func call(fn *byte, arg *byte, argsize uint32).
TEXT reflect·call(SB), 7, $0
	get_tls(CX)
	MOVQ	m(CX), BX

	// Save our caller's state as the PC and SP to
	// restore when returning from f.
	MOVQ	0(SP), AX	// our caller's PC
	MOVQ	AX, (m_morebuf+gobuf_pc)(BX)
	LEAQ	8(SP), AX	// our caller's SP
	MOVQ	AX, (m_morebuf+gobuf_sp)(BX)
	MOVQ	g(CX), AX
	MOVQ	AX, (m_morebuf+gobuf_g)(BX)

	// Set up morestack arguments to call f on a new stack.
	// We set f's frame size to 1, as a hint to newstack
	// that this is a call from reflect·call.
	// If it turns out that f needs a larger frame than
	// the default stack, f's usual stack growth prolog will
	// allocate a new segment (and recopy the arguments).
	MOVQ	8(SP), AX	// fn
	MOVQ	16(SP), DX	// arg frame
	MOVL	24(SP), CX	// arg size

	MOVQ	AX, m_morepc(BX)	// f's PC
	MOVQ	DX, m_moreargp(BX)	// argument frame pointer
	MOVL	CX, m_moreargsize(BX)	// f's argument size
	MOVL	$1, m_moreframesize(BX)	// f's frame size

	// Call newstack on m's scheduling stack.
	MOVQ	m_g0(BX), BP
	get_tls(CX)
	MOVQ	BP, g(CX)
	MOVQ	(m_sched+gobuf_sp)(BX), SP
	CALL	runtime·newstack(SB)
	MOVQ	$0, 0x1103	// crash if newstack returns
	RET

// Return point when leaving stack.
TEXT runtime·lessstack(SB), 7, $0
	// Save return value in m->cret
	get_tls(CX)
	MOVQ	m(CX), BX
	MOVQ	AX, m_cret(BX)

	// Call oldstack on m's scheduling stack.
	MOVQ	m_g0(BX), DX
	MOVQ	DX, g(CX)
	MOVQ	(m_sched+gobuf_sp)(BX), SP
	CALL	runtime·oldstack(SB)
	MOVQ	$0, 0x1004	// crash if oldstack returns
	RET

// morestack trampolines
TEXT runtime·morestack00(SB),7,$0
	get_tls(CX)
	MOVQ	m(CX), BX
	MOVQ	$0, AX
	MOVQ	AX, m_moreframesize(BX)
	MOVQ	$runtime·morestack(SB), AX
	JMP	AX

TEXT runtime·morestack01(SB),7,$0
	get_tls(CX)
	MOVQ	m(CX), BX
	SHLQ	$32, AX
	MOVQ	AX, m_moreframesize(BX)
	MOVQ	$runtime·morestack(SB), AX
	JMP	AX

TEXT runtime·morestack10(SB),7,$0
	get_tls(CX)
	MOVQ	m(CX), BX
	MOVLQZX	AX, AX
	MOVQ	AX, m_moreframesize(BX)
	MOVQ	$runtime·morestack(SB), AX
	JMP	AX

TEXT runtime·morestack11(SB),7,$0
	get_tls(CX)
	MOVQ	m(CX), BX
	MOVQ	AX, m_moreframesize(BX)
	MOVQ	$runtime·morestack(SB), AX
	JMP	AX

// subcases of morestack01
// with const of 8,16,...48
TEXT runtime·morestack8(SB),7,$0
	PUSHQ	$1
	MOVQ	$morestack<>(SB), AX
	JMP	AX

TEXT runtime·morestack16(SB),7,$0
	PUSHQ	$2
	MOVQ	$morestack<>(SB), AX
	JMP	AX

TEXT runtime·morestack24(SB),7,$0
	PUSHQ	$3
	MOVQ	$morestack<>(SB), AX
	JMP	AX

TEXT runtime·morestack32(SB),7,$0
	PUSHQ	$4
	MOVQ	$morestack<>(SB), AX
	JMP	AX

TEXT runtime·morestack40(SB),7,$0
	PUSHQ	$5
	MOVQ	$morestack<>(SB), AX
	JMP	AX

TEXT runtime·morestack48(SB),7,$0
	PUSHQ	$6
	MOVQ	$morestack<>(SB), AX
	JMP	AX

TEXT morestack<>(SB),7,$0
	get_tls(CX)
	MOVQ	m(CX), BX
	POPQ	AX
	SHLQ	$35, AX
	MOVQ	AX, m_moreframesize(BX)
	MOVQ	$runtime·morestack(SB), AX
	JMP	AX

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime·cas(SB), 7, $0
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

// bool casp(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime·casp(SB), 7, $0
	MOVQ	8(SP), BX
	MOVQ	16(SP), AX
	MOVQ	24(SP), CX
	LOCK
	CMPXCHGQ	CX, 0(BX)
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
TEXT runtime·jmpdefer(SB), 7, $0
	MOVQ	8(SP), AX	// fn
	MOVQ	16(SP), BX	// caller sp
	LEAQ	-8(BX), SP	// caller sp after CALL
	SUBQ	$5, (SP)	// return to CALL again
	JMP	AX	// but first run the deferred function

// runcgo(void(*fn)(void*), void *arg)
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
TEXT runtime·runcgo(SB),7,$32
	MOVQ	fn+0(FP), R12
	MOVQ	arg+8(FP), R13
	MOVQ	SP, CX

	// Figure out if we need to switch to m->g0 stack.
	get_tls(DI)
	MOVQ	m(DI), DX
	MOVQ	m_g0(DX), SI
	CMPQ	g(DI), SI
	JEQ	2(PC)
	MOVQ	(m_sched+gobuf_sp)(DX), SP

	// Now on a scheduling stack (a pthread-created stack).
	SUBQ	$32, SP
	ANDQ	$~15, SP	// alignment for gcc ABI
	MOVQ	g(DI), BP
	MOVQ	BP, 16(SP)
	MOVQ	SI, g(DI)
	MOVQ	CX, 8(SP)
	MOVQ	R13, DI		// DI = first argument in AMD64 ABI
	CALL	R12

	// Restore registers, g, stack pointer.
	get_tls(DI)
	MOVQ	16(SP), SI
	MOVQ	SI, g(DI)
	MOVQ	8(SP), SP
	RET

// runcgocallback(G *g1, void* sp, void (*fn)(void))
// Switch to g1 and sp, call fn, switch back.  fn's arguments are on
// the new stack.
TEXT runtime·runcgocallback(SB),7,$48
	MOVQ	g1+0(FP), DX
	MOVQ	sp+8(FP), AX
	MOVQ	fp+16(FP), BX

	// We are running on m's scheduler stack.  Save current SP
	// into m->sched.sp so that a recursive call to runcgo doesn't
	// clobber our stack, and also so that we can restore
	// the SP when the call finishes.  Reusing m->sched.sp
	// for this purpose depends on the fact that there is only
	// one possible gosave of m->sched.
	get_tls(CX)
	MOVQ	DX, g(CX)
	MOVQ	m(CX), CX
	MOVQ	SP, (m_sched+gobuf_sp)(CX)

	// Set new SP, call fn
	MOVQ	AX, SP
	CALL	BX

	// Restore old g and SP, return
	get_tls(CX)
	MOVQ	m(CX), DX
	MOVQ	m_g0(DX), BX
	MOVQ	BX, g(CX)
	MOVQ	(m_sched+gobuf_sp)(DX), SP
	RET

// check that SP is in range [g->stackbase, g->stackguard)
TEXT runtime·stackcheck(SB), 7, $0
	get_tls(CX)
	MOVQ	g(CX), AX
	CMPQ	g_stackbase(AX), SP
	JHI	2(PC)
	INT	$3
	CMPQ	SP, g_stackguard(AX)
	JHI	2(PC)
	INT	$3
	RET

TEXT runtime·memclr(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 addr
	MOVQ	16(SP), CX		// arg 2 count
	ADDQ	$7, CX
	SHRQ	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	RET

TEXT runtime·getcallerpc(SB),7,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	-8(AX),AX		// get calling pc
	RET

TEXT runtime·setcallerpc(SB),7,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	x+8(FP), BX
	MOVQ	BX, -8(AX)		// set calling pc
	RET

TEXT runtime·getcallersp(SB),7,$0
	MOVQ	sp+0(FP), AX
	RET

GLOBL runtime·tls0(SB), $64
