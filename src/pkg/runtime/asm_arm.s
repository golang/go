// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

// using frame size $-4 means do not save LR on stack.
TEXT _rt0_arm(SB),7,$-4
	MOVW	$0xcafebabe, R12

	// copy arguments forward on an even stack
	// use R13 instead of SP to avoid linker rewriting the offsets
	MOVW	0(R13), R0		// argc
	MOVW	$4(R13), R1		// argv
	SUB	$64, R13		// plenty of scratch
	AND	$~7, R13
	MOVW	R0, 60(R13)		// save argc, argv away
	MOVW	R1, 64(R13)

	// set up m and g registers
	// g is R10, m is R9
	MOVW	$runtime·g0(SB), g
	MOVW	$runtime·m0(SB), m

	// save m->g0 = g0
	MOVW	g, m_g0(m)

	// create istack out of the OS stack
	MOVW	$(-8192+104)(R13), R0
	MOVW	R0, g_stackguard(g)	// (w 104b guard)
	MOVW	R13, g_stackbase(g)
	BL	runtime·emptyfunc(SB)	// fault if stack check is wrong

	BL	runtime·check(SB)

	// saved argc, argv
	MOVW	60(R13), R0
	MOVW	R0, 4(R13)
	MOVW	64(R13), R1
	MOVW	R1, 8(R13)
	BL	runtime·args(SB)
	BL	runtime·osinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVW	$runtime·main(SB), R0
	MOVW.W	R0, -4(R13)
	MOVW	$8, R0
	MOVW.W	R0, -4(R13)
	MOVW	$0, R0
	MOVW.W	R0, -4(R13)	// push $0 as guard
	BL	runtime·newproc(SB)
	MOVW	$12(R13), R13	// pop args and LR

	// start this M
	BL	runtime·mstart(SB)

	MOVW	$1234, R0
	MOVW	$1000, R1
	MOVW	R0, (R1)	// fail hard
	B	runtime·_dep_dummy(SB)	// Never reached

// TODO(kaib): remove these once i actually understand how the linker removes symbols
// pull in dummy dependencies
TEXT runtime·_dep_dummy(SB),7,$0
	BL	_div(SB)
	BL	_divu(SB)
	BL	_mod(SB)
	BL	_modu(SB)
	BL	_modu(SB)
	BL	_sfloat(SB)

TEXT runtime·breakpoint(SB),7,$0
	// no breakpoint yet; let program exit
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), 7, $-4
	MOVW	0(FP), R0		// gobuf
	MOVW	SP, gobuf_sp(R0)
	MOVW	LR, gobuf_pc(R0)
	MOVW	g, gobuf_g(R0)
	RET

// void gogo(Gobuf*, uintptr)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), 7, $-4
	MOVW	0(FP), R1		// gobuf
	MOVW	4(FP), R0		// return 2nd arg
	MOVW	gobuf_g(R1), g
	MOVW	0(g), R2		// make sure g != nil
	MOVW	gobuf_sp(R1), SP	// restore SP
	MOVW	gobuf_pc(R1), PC

// void gogocall(Gobuf*, void (*fn)(void))
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
// using frame size $-4 means do not save LR on stack.
TEXT runtime·gogocall(SB), 7, $-4
	MOVW	0(FP), R0		// gobuf
	MOVW	4(FP), R1		// fn
	MOVW	8(FP), R2		// fp offset
	MOVW	gobuf_g(R0), g
	MOVW	0(g), R3		// make sure g != nil
	MOVW	gobuf_sp(R0), SP	// restore SP
	MOVW	gobuf_pc(R0), LR
	MOVW	R1, PC

// void mcall(void (*fn)(G*))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), 7, $-4
	MOVW	fn+0(FP), R0

	// Save caller state in g->gobuf.
	MOVW	SP, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	g, (g_sched+gobuf_g)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVW	g, R1
	MOVW	m_g0(m), g
	CMP	g, R1
	BL.EQ	runtime·badmcall(SB)
	MOVW	(g_sched+gobuf_sp)(g), SP
	SUB	$8, SP
	MOVW	R1, 4(SP)
	BL	(R0)
	BL	runtime·badmcall2(SB)
	RET

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// R1 frame size
// R2 arg size
// R3 prolog's LR
// NB. we do not save R0 because we've forced 5c to pass all arguments
// on the stack.
// using frame size $-4 means do not save LR on stack.
TEXT runtime·morestack(SB),7,$-4
	// Cannot grow scheduler stack (m->g0).
	MOVW	m_g0(m), R4
	CMP	g, R4
	BL.EQ	runtime·abort(SB)

	// Save in m.
	MOVW	R1, m_moreframesize(m)
	MOVW	R2, m_moreargsize(m)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVW	R3, (m_morebuf+gobuf_pc)(m)	// f's caller's PC
	MOVW	SP, (m_morebuf+gobuf_sp)(m)	// f's caller's SP
	MOVW	$4(SP), R3			// f's argument pointer
	MOVW	R3, m_moreargp(m)	
	MOVW	g, (m_morebuf+gobuf_g)(m)

	// Set m->morepc to f's PC.
	MOVW	LR, m_morepc(m)

	// Call newstack on m->g0's stack.
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), SP
	B	runtime·newstack(SB)

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
	MOVW	g,  (m_morebuf+gobuf_g)(m)

	// Set up morestack arguments to call f on a new stack.
	// We set f's frame size to 1, as a hint to newstack
	// that this is a call from reflect·call.
	// If it turns out that f needs a larger frame than
	// the default stack, f's usual stack growth prolog will
	// allocate a new segment (and recopy the arguments).
	MOVW	4(SP), R0			// fn
	MOVW	8(SP), R1			// arg frame
	MOVW	12(SP), R2			// arg size

	MOVW	R0, m_morepc(m)			// f's PC
	MOVW	R1, m_moreargp(m)		// f's argument pointer
	MOVW	R2, m_moreargsize(m)		// f's argument size
	MOVW	$1, R3
	MOVW	R3, m_moreframesize(m)		// f's frame size

	// Call newstack on m->g0's stack.
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), SP
	B	runtime·newstack(SB)

// Return point when leaving stack.
// using frame size $-4 means do not save LR on stack.
TEXT runtime·lessstack(SB), 7, $-4
	// Save return value in m->cret
	MOVW	R0, m_cret(m)

	// Call oldstack on m->g0's stack.
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), SP
	B	runtime·oldstack(SB)

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 4 bytes to get back to BL deferreturn
// 3. B to fn
TEXT runtime·jmpdefer(SB), 7, $0
	MOVW	0(SP), LR
	MOVW	$-4(LR), LR	// BL deferreturn
	MOVW	fn+0(FP), R0
	MOVW	argp+4(FP), SP
	MOVW	$-4(SP), SP	// SP is 4 below argp, due to saved LR
	B		(R0)

TEXT	runtime·asmcgocall(SB),7,$0
	B	runtime·cgounimpl(SB)

TEXT	runtime·cgocallback(SB),7,$0
	B	runtime·cgounimpl(SB)

TEXT runtime·memclr(SB),7,$20
	MOVW	0(FP), R0
	MOVW	$0, R1		// c = 0
	MOVW	R1, -16(SP)
	MOVW	4(FP), R1	// n
	MOVW	R1, -12(SP)
	MOVW	m, -8(SP)	// Save m and g
	MOVW	g, -4(SP)
	BL	runtime·memset(SB)
	MOVW	-8(SP), m	// Restore m and g, memset clobbers them
	MOVW	-4(SP), g
	RET

TEXT runtime·getcallerpc(SB),7,$-4
	MOVW	0(SP), R0
	RET

TEXT runtime·setcallerpc(SB),7,$-4
	MOVW	x+4(FP), R0
	MOVW	R0, 0(SP)
	RET

TEXT runtime·getcallersp(SB),7,$-4
	MOVW	0(FP), R0
	MOVW	$-4(R0), R0
	RET

TEXT runtime·emptyfunc(SB),0,$0
	RET

TEXT runtime·abort(SB),7,$-4
	MOVW	$0, R0
	MOVW	(R0), R1

// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
//
// To implement runtime·cas in ../$GOOS/arm/sys.s
// using the native instructions, use:
//
//	TEXT runtime·cas(SB),7,$0
//		B	runtime·armcas(SB)
//
TEXT runtime·armcas(SB),7,$0
	MOVW	valptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casl:
	LDREX	(R1), R0
	CMP		R0, R2
	BNE		casfail
	STREX	R3, (R1), R0
	CMP		$0, R0
	BNE		casl
	MOVW	$1, R0
	RET
casfail:
	MOVW	$0, R0
	RET
