// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "386/asm.h"

TEXT _rt0_386(SB),7,$0
	// copy arguments forward on an even stack
	MOVL	0(SP), AX		// argc
	LEAL	4(SP), BX		// argv
	SUBL	$128, SP		// plenty of scratch
	ANDL	$~15, SP
	MOVL	AX, 120(SP)		// save argc, argv away
	MOVL	BX, 124(SP)

	// if there is an initcgo, call it to let it
	// initialize and to set up GS.  if not,
	// we set up GS ourselves.
	MOVL	initcgo(SB), AX
	TESTL	AX, AX
	JZ	3(PC)
	CALL	AX
	JMP	ok

	// set up %gs
	CALL	ldt0setup(SB)

	// store through it, to make sure it works
	get_tls(BX)
	MOVL	$0x123, g(BX)
	MOVL	tls0(SB), AX
	CMPL	AX, $0x123
	JEQ	ok
	MOVL	AX, 0	// abort
ok:
	// set up m and g "registers"
	get_tls(BX)
	LEAL	g0(SB), CX
	MOVL	CX, g(BX)
	LEAL	m0(SB), AX
	MOVL	AX, m(BX)

	// save m->g0 = g0
	MOVL	CX, m_g0(AX)

	// create istack out of the OS stack
	LEAL	(-8192+104)(SP), AX	// TODO: 104?
	MOVL	AX, g_stackguard(CX)
	MOVL	SP, g_stackbase(CX)
	CALL	emptyfunc(SB)	// fault if stack check is wrong

	// convention is D is always cleared
	CLD

	CALL	check(SB)

	// saved argc, argv
	MOVL	120(SP), AX
	MOVL	AX, 0(SP)
	MOVL	124(SP), AX
	MOVL	AX, 4(SP)
	CALL	args(SB)
	CALL	osinit(SB)
	CALL	schedinit(SB)

	// create a new goroutine to start program
	PUSHL	$mainstart(SB)	// entry
	PUSHL	$0	// arg size
	CALL	runtime·newproc(SB)
	POPL	AX
	POPL	AX

	// start this M
	CALL	mstart(SB)

	INT $3
	RET

TEXT mainstart(SB),7,$0
	CALL	main·init(SB)
	CALL	initdone(SB)
	CALL	main·main(SB)
	PUSHL	$0
	CALL	exit(SB)
	POPL	AX
	INT $3
	RET

TEXT	breakpoint(SB),7,$0
	INT $3
	RET

/*
 *  go-routine
 */

// uintptr gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT gosave(SB), 7, $0
	MOVL	4(SP), AX		// gobuf
	LEAL	4(SP), BX		// caller's SP
	MOVL	BX, gobuf_sp(AX)
	MOVL	0(SP), BX		// caller's PC
	MOVL	BX, gobuf_pc(AX)
	get_tls(CX)
	MOVL	g(CX), BX
	MOVL	BX, gobuf_g(AX)
	MOVL	$0, AX			// return 0
	RET

// void gogo(Gobuf*, uintptr)
// restore state from Gobuf; longjmp
TEXT gogo(SB), 7, $0
	MOVL	8(SP), AX		// return 2nd arg
	MOVL	4(SP), BX		// gobuf
	MOVL	gobuf_g(BX), DX
	MOVL	0(DX), CX		// make sure g != nil
	get_tls(CX)
	MOVL	DX, g(CX)
	MOVL	gobuf_sp(BX), SP	// restore SP
	MOVL	gobuf_pc(BX), BX
	JMP	BX

// void gogocall(Gobuf*, void (*fn)(void))
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
TEXT gogocall(SB), 7, $0
	MOVL	8(SP), AX		// fn
	MOVL	4(SP), BX		// gobuf
	MOVL	gobuf_g(BX), DX
	get_tls(CX)
	MOVL	DX, g(CX)
	MOVL	0(DX), CX		// make sure g != nil
	MOVL	gobuf_sp(BX), SP	// restore SP
	MOVL	gobuf_pc(BX), BX
	PUSHL	BX
	JMP	AX
	POPL	BX	// not reached

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
TEXT runtime·morestack(SB),7,$0
	// Cannot grow scheduler stack (m->g0).
	get_tls(CX)
	MOVL	m(CX), BX
	MOVL	m_g0(BX), SI
	CMPL	g(CX), SI
	JNE	2(PC)
	INT	$3

	// frame size in DX
	// arg size in AX
	// Save in m.
	MOVL	DX, m_moreframe(BX)
	MOVL	AX, m_moreargs(BX)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVL	4(SP), DI	// f's caller's PC
	MOVL	DI, (m_morebuf+gobuf_pc)(BX)
	LEAL	8(SP), CX	// f's caller's SP
	MOVL	CX, (m_morebuf+gobuf_sp)(BX)
	MOVL	CX, (m_morefp)(BX)
	get_tls(CX)
	MOVL	g(CX), SI
	MOVL	SI, (m_morebuf+gobuf_g)(BX)

	// Set m->morepc to f's PC.
	MOVL	0(SP), AX
	MOVL	AX, m_morepc(BX)

	// Call newstack on m's scheduling stack.
	MOVL	m_g0(BX), BP
	MOVL	BP, g(CX)
	MOVL	(m_sched+gobuf_sp)(BX), SP
	CALL	newstack(SB)
	MOVL	$0, 0x1003	// crash if newstack returns
	RET

// Called from reflection library.  Mimics morestack,
// reuses stack growth code to create a frame
// with the desired args running the desired function.
//
// func call(fn *byte, arg *byte, argsize uint32).
TEXT reflect·call(SB), 7, $0
	get_tls(CX)
	MOVL	m(CX), BX

	// Save our caller's state as the PC and SP to
	// restore when returning from f.
	MOVL	0(SP), AX	// our caller's PC
	MOVL	AX, (m_morebuf+gobuf_pc)(BX)
	LEAL	4(SP), AX	// our caller's SP
	MOVL	AX, (m_morebuf+gobuf_sp)(BX)
	MOVL	g(CX), AX
	MOVL	AX, (m_morebuf+gobuf_g)(BX)

	// Set up morestack arguments to call f on a new stack.
	// We set f's frame size to zero, meaning
	// allocate a standard sized stack segment.
	// If it turns out that f needs a larger frame than this,
	// f's usual stack growth prolog will allocate
	// a new segment (and recopy the arguments).
	MOVL	4(SP), AX	// fn
	MOVL	8(SP), DX	// arg frame
	MOVL	12(SP), CX	// arg size

	MOVL	AX, m_morepc(BX)	// f's PC
	MOVL	DX, m_morefp(BX)	// argument frame pointer
	MOVL	CX, m_moreargs(BX)	// f's argument size
	MOVL	$0, m_moreframe(BX)	// f's frame size

	// Call newstack on m's scheduling stack.
	MOVL	m_g0(BX), BP
	get_tls(CX)
	MOVL	BP, g(CX)
	MOVL	(m_sched+gobuf_sp)(BX), SP
	CALL	newstack(SB)
	MOVL	$0, 0x1103	// crash if newstack returns
	RET


// Return point when leaving stack.
TEXT runtime·lessstack(SB), 7, $0
	// Save return value in m->cret
	get_tls(CX)
	MOVL	m(CX), BX
	MOVL	AX, m_cret(BX)

	// Call oldstack on m's scheduling stack.
	MOVL	m_g0(BX), DX
	MOVL	DX, g(CX)
	MOVL	(m_sched+gobuf_sp)(BX), SP
	CALL	oldstack(SB)
	MOVL	$0, 0x1004	// crash if oldstack returns
	RET


// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT cas(SB), 7, $0
	MOVL	4(SP), BX
	MOVL	8(SP), AX
	MOVL	12(SP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ 3(PC)
	MOVL	$0, AX
	RET
	MOVL	$1, AX
	RET

// bool casp(void **p, void *old, void *new)
// Atomically:
//	if(*p == old){
//		*p = new;
//		return 1;
//	}else
//		return 0;
TEXT casp(SB), 7, $0
	MOVL	4(SP), BX
	MOVL	8(SP), AX
	MOVL	12(SP), CX
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
	MOVL	4(SP), AX	// fn
	MOVL	8(SP), BX	// caller sp
	LEAL	-4(BX), SP	// caller sp after CALL
	SUBL	$5, (SP)	// return to CALL again
	JMP	AX	// but first run the deferred function

TEXT	runtime·memclr(SB),7,$0
	MOVL	4(SP), DI		// arg 1 addr
	MOVL	8(SP), CX		// arg 2 count
	ADDL	$3, CX
	SHRL	$2, CX
	MOVL	$0, AX
	CLD
	REP
	STOSL
	RET

TEXT	runtime·getcallerpc+0(SB),7,$0
	MOVL	x+0(FP),AX		// addr of first arg
	MOVL	-4(AX),AX		// get calling pc
	RET

TEXT	runtime·setcallerpc+0(SB),7,$0
	MOVL	x+0(FP),AX		// addr of first arg
	MOVL	x+4(FP), BX
	MOVL	BX, -4(AX)		// set calling pc
	RET

TEXT ldt0setup(SB),7,$16
	// set up ldt 7 to point at tls0
	// ldt 1 would be fine on Linux, but on OS X, 7 is as low as we can go.
	// the entry number is just a hint.  setldt will set up GS with what it used.
	MOVL	$7, 0(SP)
	LEAL	tls0(SB), AX
	MOVL	AX, 4(SP)
	MOVL	$32, 8(SP)	// sizeof(tls array)
	CALL	setldt(SB)
	RET

TEXT emptyfunc(SB),0,$0
	RET

TEXT	abort(SB),7,$0
	INT $0x3

// runcgo(void(*fn)(void*), void *arg)
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
TEXT	runcgo(SB),7,$16
	MOVL	fn+0(FP), AX
	MOVL	arg+4(FP), BX
	MOVL	SP, CX

	// Figure out if we need to switch to m->g0 stack.
	get_tls(DI)
	MOVL	m(DI), DX
	MOVL	m_g0(DX), SI
	CMPL	g(DI), SI
	JEQ	2(PC)
	MOVL	(m_sched+gobuf_sp)(DX), SP

	// Now on a scheduling stack (a pthread-created stack).
	SUBL	$16, SP
	ANDL	$~15, SP	// alignment for gcc ABI
	MOVL	CX, 4(SP)
	MOVL	BX, 0(SP)
	CALL	AX
	
	// Back; switch to original stack, re-establish
	// "DF is clear" invariant.
	CLD
	MOVL	4(SP), SP
	RET

// check that SP is in range [g->stackbase, g->stackguard)
TEXT stackcheck(SB), 7, $0
	get_tls(CX)
	MOVL g(CX), AX
	CMPL g_stackbase(AX), SP
	JHI 2(PC)
	INT $3
	CMPL SP, g_stackguard(AX)
	JHI 2(PC)
	INT $3
	RET


GLOBL m0(SB), $1024
GLOBL g0(SB), $1024
GLOBL tls0(SB), $32
GLOBL initcgo(SB), $4

