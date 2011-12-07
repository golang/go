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
	
	// create istack out of the given (operating system) stack.
	// initcgo may update stackguard.
	MOVQ	$runtime·g0(SB), DI
	LEAQ	(-8192+104)(SP), BX
	MOVQ	BX, g_stackguard(DI)
	MOVQ	SP, g_stackbase(DI)

	// if there is an initcgo, call it.
	MOVQ	initcgo(SB), AX
	TESTQ	AX, AX
	JZ	needtls
	CALL	AX  // g0 already in DI
	CMPL	runtime·iswindows(SB), $0
	JEQ ok

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
	PUSHQ	$runtime·main(SB)		// entry
	PUSHQ	$0			// arg size
	CALL	runtime·newproc(SB)
	POPQ	AX
	POPQ	AX

	// start this M
	CALL	runtime·mstart(SB)

	CALL	runtime·notok(SB)		// never returns
	RET

TEXT runtime·breakpoint(SB),7,$0
	BYTE	$0xcc
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
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

// void mcall(void (*fn)(G*))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), 7, $0
	MOVQ	fn+0(FP), DI
	
	get_tls(CX)
	MOVQ	g(CX), AX	// save state in g->gobuf
	MOVQ	0(SP), BX	// caller's PC
	MOVQ	BX, (g_sched+gobuf_pc)(AX)
	LEAQ	8(SP), BX	// caller's SP
	MOVQ	BX, (g_sched+gobuf_sp)(AX)
	MOVQ	AX, (g_sched+gobuf_g)(AX)

	// switch to m->g0 & its stack, call fn
	MOVQ	m(CX), BX
	MOVQ	m_g0(BX), SI
	CMPQ	SI, AX	// if g == m->g0 call badmcall
	JNE	2(PC)
	CALL	runtime·badmcall(SB)
	MOVQ	SI, g(CX)	// g = m->g0
	MOVQ	(g_sched+gobuf_sp)(SI), SP	// sp = m->g0->gobuf.sp
	PUSHQ	AX
	CALL	DI
	POPQ	AX
	CALL	runtime·badmcall2(SB)
	RET

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

	// Call newstack on m->g0's stack.
	MOVQ	m_g0(BX), BP
	MOVQ	BP, g(CX)
	MOVQ	(g_sched+gobuf_sp)(BP), SP
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

	// Call newstack on m->g0's stack.
	MOVQ	m_g0(BX), BP
	get_tls(CX)
	MOVQ	BP, g(CX)
	MOVQ	(g_sched+gobuf_sp)(BP), SP
	CALL	runtime·newstack(SB)
	MOVQ	$0, 0x1103	// crash if newstack returns
	RET

// Return point when leaving stack.
TEXT runtime·lessstack(SB), 7, $0
	// Save return value in m->cret
	get_tls(CX)
	MOVQ	m(CX), BX
	MOVQ	AX, m_cret(BX)

	// Call oldstack on m->g0's stack.
	MOVQ	m_g0(BX), BP
	MOVQ	BP, g(CX)
	MOVQ	(g_sched+gobuf_sp)(BP), SP
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

// uint32 xadd(uint32 volatile *val, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT runtime·xadd(SB), 7, $0
	MOVQ	8(SP), BX
	MOVL	16(SP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BX)
	ADDL	CX, AX
	RET

TEXT runtime·xchg(SB), 7, $0
	MOVQ	8(SP), BX
	MOVL	16(SP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT runtime·procyield(SB),7,$0
	MOVL	8(SP), AX
again:
	PAUSE
	SUBL	$1, AX
	JNZ	again
	RET

TEXT runtime·atomicstorep(SB), 7, $0
	MOVQ	8(SP), BX
	MOVQ	16(SP), AX
	XCHGQ	AX, 0(BX)
	RET

TEXT runtime·atomicstore(SB), 7, $0
	MOVQ	8(SP), BX
	MOVL	16(SP), AX
	XCHGL	AX, 0(BX)
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

// Dummy function to use in saved gobuf.PC,
// to match SP pointing at a return address.
// The gobuf.PC is unused by the contortions here
// but setting it to return will make the traceback code work.
TEXT return<>(SB),7,$0
	RET

// asmcgocall(void(*fn)(void*), void *arg)
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.c for more details.
TEXT runtime·asmcgocall(SB),7,$0
	MOVQ	fn+0(FP), AX
	MOVQ	arg+8(FP), BX
	MOVQ	SP, DX

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	get_tls(CX)
	MOVQ	m(CX), BP
	MOVQ	m_g0(BP), SI
	MOVQ	g(CX), DI
	CMPQ	SI, DI
	JEQ	6(PC)
	MOVQ	SP, (g_sched+gobuf_sp)(DI)
	MOVQ	$return<>(SB), (g_sched+gobuf_pc)(DI)
	MOVQ	DI, (g_sched+gobuf_g)(DI)
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), SP

	// Now on a scheduling stack (a pthread-created stack).
	SUBQ	$48, SP
	ANDQ	$~15, SP	// alignment for gcc ABI
	MOVQ	DI, 32(SP)	// save g
	MOVQ	DX, 24(SP)	// save SP
	MOVQ	BX, DI		// DI = first argument in AMD64 ABI
	MOVQ	BX, CX		// CX = first argument in Win64
	CALL	AX

	// Restore registers, g, stack pointer.
	get_tls(CX)
	MOVQ	32(SP), DI
	MOVQ	DI, g(CX)
	MOVQ	24(SP), SP
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// See cgocall.c for more details.
TEXT runtime·cgocallback(SB),7,$24
	MOVQ	fn+0(FP), AX
	MOVQ	frame+8(FP), BX
	MOVQ	framesize+16(FP), DX

	// Save current m->g0->sched.sp on stack and then set it to SP.
	get_tls(CX)
	MOVQ	m(CX), BP
	MOVQ	m_g0(BP), SI
	PUSHQ	(g_sched+gobuf_sp)(SI)
	MOVQ	SP, (g_sched+gobuf_sp)(SI)

	// Switch to m->curg stack and call runtime.cgocallbackg
	// with the three arguments.  Because we are taking over
	// the execution of m->curg but *not* resuming what had
	// been running, we need to save that information (m->curg->gobuf)
	// so that we can restore it when we're done. 
	// We can restore m->curg->gobuf.sp easily, because calling
	// runtime.cgocallbackg leaves SP unchanged upon return.
	// To save m->curg->gobuf.pc, we push it onto the stack.
	// This has the added benefit that it looks to the traceback
	// routine like cgocallbackg is going to return to that
	// PC (because we defined cgocallbackg to have
	// a frame size of 24, the same amount that we use below),
	// so that the traceback will seamlessly trace back into
	// the earlier calls.
	MOVQ	m_curg(BP), SI
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), DI  // prepare stack as DI

	// Push gobuf.pc
	MOVQ	(g_sched+gobuf_pc)(SI), BP
	SUBQ	$8, DI
	MOVQ	BP, 0(DI)

	// Push arguments to cgocallbackg.
	// Frame size here must match the frame size above
	// to trick traceback routines into doing the right thing.
	SUBQ	$24, DI
	MOVQ	AX, 0(DI)
	MOVQ	BX, 8(DI)
	MOVQ	DX, 16(DI)
	
	// Switch stack and make the call.
	MOVQ	DI, SP
	CALL	runtime·cgocallbackg(SB)

	// Restore g->gobuf (== m->curg->gobuf) from saved values.
	get_tls(CX)
	MOVQ	g(CX), SI
	MOVQ	24(SP), BP
	MOVQ	BP, (g_sched+gobuf_pc)(SI)
	LEAQ	(24+8)(SP), DI
	MOVQ	DI, (g_sched+gobuf_sp)(SI)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVQ	m(CX), BP
	MOVQ	m_g0(BP), SI
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), SP
	POPQ	(g_sched+gobuf_sp)(SI)

	// Done!
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
	MOVQ	CX, BX
	ANDQ	$7, BX
	SHRQ	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	MOVQ	BX, CX
	REP
	STOSB
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
