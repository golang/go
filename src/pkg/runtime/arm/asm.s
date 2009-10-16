// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "arm/asm.h"

// using frame size $-4 means do not save LR on stack.
TEXT _rt0_arm(SB),7,$-4
	MOVW $setR12(SB), R12

	// copy arguments forward on an even stack
	// use R13 instead of SP to avoid linker rewriting the offsets
	MOVW	0(R13), R0		// argc
	MOVW	$4(R13), R1		// argv
	SUB	$128, R13		// plenty of scratch
	AND	$~7, R13
	MOVW	R0, 120(R13)		// save argc, argv away
	MOVW	R1, 124(R13)

	// set up m and g registers
	// g is R10, m is R9
	MOVW	$g0(SB), g
	MOVW	$m0(SB), m

	// save m->g0 = g0
	MOVW	g, m_g0(m)

	// create istack out of the OS stack
	MOVW	$(-8192+104)(R13), R0
	MOVW	R0, g_stackguard(g)	// (w 104b guard)
	MOVW	R13, g_stackbase(g)
	BL	emptyfunc(SB)	// fault if stack check is wrong

	BL	check(SB)

	// saved argc, argv
	MOVW	120(R13), R0
	MOVW	R0, 4(R13)
	MOVW	124(R13), R1
	MOVW	R1, 8(R13)
	BL	args(SB)
	BL	osinit(SB)
	BL	schedinit(SB)

	// create a new goroutine to start program
	MOVW	$mainstart(SB), R0
	MOVW.W	R0, -4(R13)
	MOVW	$8, R0
	MOVW.W	R0, -4(R13)
	MOVW	$0, R0
	MOVW.W	R0, -4(R13)	// push $0 as guard
	BL	runtime·newproc(SB)
	MOVW	$12(R13), R13	// pop args and LR

	// start this M
	BL	mstart(SB)

	MOVW	$0, R0
	SWI	$0x00900001
	B	_dep_dummy(SB)	// Never reached


TEXT mainstart(SB),7,$0
	BL	main·init(SB)
	BL	initdone(SB)
	BL	main·main(SB)
	MOVW	$0, R0
	MOVW.W	R0, -4(SP)
	MOVW.W	R14, -4(SP)	// Push link as well
	BL	exit(SB)
	MOVW	$8(SP), SP	// pop args and LR
	RET

// TODO(kaib): remove these once linker works properly
// pull in dummy dependencies
TEXT _dep_dummy(SB),7,$0
	BL	_div(SB)
	BL	_divu(SB)
	BL	_mod(SB)
	BL	_modu(SB)
	BL	_modu(SB)

TEXT	breakpoint(SB),7,$0
	BL	abort(SB)
//	BYTE $0xcc
//	RET

/*
 *  go-routine
 */

// uintptr gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT gosave(SB), 7, $0
	MOVW	0(FP), R0
	MOVW	SP, gobuf_sp(R0)
	MOVW	LR, gobuf_pc(R0)
	MOVW	g, gobuf_g(R0)
	MOVW	$0, R0			// return 0
	RET

// void gogo(Gobuf*, uintptr)
// restore state from Gobuf; longjmp
TEXT	gogo(SB), 7, $0
	MOVW	0(FP), R1			// gobuf
	MOVW	4(FP), R0		// return 2nd arg
	MOVW	gobuf_g(R1), g
	MOVW	0(g), R2		// make sure g != nil
	MOVW	gobuf_sp(R1), SP	// restore SP
	MOVW	gobuf_pc(R1), PC

// void gogocall(Gobuf*, void (*fn)(void))
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
// using frame size $-4 means do not save LR on stack.
TEXT gogocall(SB), 7, $-4
	MOVW	0(FP), R0
	MOVW	4(FP), R1		// fn
	MOVW	gobuf_g(R0), g
	MOVW	0(g), R2		// make sure g != nil
	MOVW	gobuf_sp(R0), SP	// restore SP
	MOVW	gobuf_pc(R0), LR
	MOVW	gobuf_r0(R0), R0
	MOVW	R1, PC

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// R1 frame size
// R2 arg size
// R3 prolog's LR
// using frame size $-4 means do not save LR on stack.
TEXT runtime·morestack(SB),7,$-4
	// Cannot grow scheduler stack (m->g0).
	MOVW	m_g0(m), R4
	CMP	g, R4
	BNE	2(PC)
	BL	abort(SB)

	// Save in m.
	MOVW	R1, m_moreframe(m)
	MOVW	R2, m_moreargs(m)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVW	R3, (m_morebuf+gobuf_pc)(m) // f's caller's PC
	MOVW	SP, (m_morebuf+gobuf_sp)(m) // f's caller's SP
	MOVW	SP, m_morefp(m) // f's caller's SP
	MOVW	g, (m_morebuf+gobuf_g)(m)
	MOVW	R0, (m_morebuf+gobuf_r0)(m)

	// Set m->morepc to f's PC.
	MOVW	LR, m_morepc(m)

	// Call newstack on m's scheduling stack.
	MOVW	m_g0(m), g
	MOVW	(m_sched+gobuf_sp)(m), SP
	B	newstack(SB)

// Called from reflection library.  Mimics morestack,
// reuses stack growth code to create a frame
// with the desired args running the desired function.
//
// func call(fn *byte, arg *byte, argsize uint32).
TEXT reflect·call(SB), 7, $-4
	// Save our caller's state as the PC and SP to
	// restore when returning from f.
	MOVW	LR, (m_morebuf+gobuf_pc)(m)	// our caller's PC
	MOVW	SP, (m_morebuf+gobuf_sp)(m)	// our caller's SP
	MOVW	R0, (m_morebuf+gobuf_r0)(m)
	MOVW	g,  (m_morebuf+gobuf_g)(m)

	// Set up morestack arguments to call f on a new stack.
	// We set f's frame size to zero, meaning
	// allocate a standard sized stack segment.
	// If it turns out that f needs a larger frame than this,
	// f's usual stack growth prolog will allocate
	// a new segment (and recopy the arguments).
	MOVW	4(SP), R0	// fn
	MOVW	8(SP), R1	// arg frame
	MOVW	12(SP), R2	// arg size

	MOVW	R0, m_morepc(m)	// f's PC
	MOVW	R1, m_morefp(m)	// argument frame pointer
	MOVW	R2, m_moreargs(m)	// f's argument size
	MOVW	$0, R3
	MOVW	R3, m_moreframe(m)	// f's frame size

	// Call newstack on m's scheduling stack.
	MOVW	m_g0(m), g
	MOVW	(m_sched+gobuf_sp)(m), SP
	B	newstack(SB)

// Return point when leaving stack.
// using frame size $-4 means do not save LR on stack.
TEXT runtime·lessstack(SB), 7, $-4
	// Save return value in m->cret
	MOVW	R0, m_cret(m)

	// Call oldstack on m's scheduling stack.
	MOVW	m_g0(m), g
	MOVW	(m_sched+gobuf_sp)(m), SP
	B	oldstack(SB)

// Optimization to make inline stack splitting code smaller
// R0 is original first argument
// R2 is argsize
// R3 is LR for f (f's caller's PC)
// using frame size $-4 means do not save LR on stack.
TEXT runtime·morestackx(SB), 7, $-4
	MOVW	$0, R1		// set frame size
	B	runtime·morestack(SB)


// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 4 bytes to get back to BL deferreturn
// 3. B to fn
TEXT jmpdefer(SB), 7, $0
	MOVW	0(SP), LR
	MOVW	$-4(LR), LR	// BL deferreturn
	MOVW	4(SP), R0		// fn
	MOVW	8(SP), R1
	MOVW	$-4(R1), SP	// correct for sp pointing to arg0, past stored lr
	B		(R0)

TEXT	runtime·memclr(SB),7,$20
	MOVW	0(FP), R0
	MOVW	$0, R1		// c = 0
	MOVW	R1, -16(SP)
	MOVW	4(FP), R1	// n
	MOVW	R1, -12(SP)
	MOVW	m, -8(SP)	// Save m and g
	MOVW	g, -4(SP)
	BL	memset(SB)
	MOVW	-8(SP), m	// Restore m and g, memset clobbers them
	MOVW	-4(SP), g
	RET

TEXT	runtime·getcallerpc+0(SB),7,$0
	BL	abort(SB)
//	MOVL	x+0(FP),AX		// addr of first arg
//	MOVL	-4(AX),AX		// get calling pc
//	RET

TEXT	runtime·setcallerpc+0(SB),7,$0
	BL	abort(SB)
//	MOVL	x+0(FP),AX		// addr of first arg
//	MOVL	x+4(FP), BX
//	MOVL	BX, -4(AX)		// set calling pc
//	RET

// runcgo(void(*fn)(void*), void *arg)
// Just call fn(arg), but first align the stack
// appropriately for the gcc ABI.
// TODO(kaib): figure out the arm-gcc ABI
TEXT	runcgo(SB),7,$16
	BL	abort(SB)
//	MOVL	fn+0(FP), AX
//	MOVL	arg+4(FP), BX
//	MOVL	SP, CX
//	ANDL	$~15, SP	// alignment for gcc ABI
//	MOVL	CX, 4(SP)
//	MOVL	BX, 0(SP)
//	CALL	AX
//	MOVL	4(SP), SP
//	RET

TEXT emptyfunc(SB),0,$0
	RET

TEXT abort(SB),7,$0
	MOVW	$0, R0
	MOVW	(R0), R1

