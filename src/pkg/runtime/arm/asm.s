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
	BL	sys·newproc(SB)
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
	MOVW	SP, gobuf_sp(R0)
	MOVW	LR, gobuf_pc(R0)
	MOVW	g, gobuf_g(R0)
	MOVW	$0, R0			// return 0
	RET

// void gogo(Gobuf*, uintptr)
// restore state from Gobuf; longjmp
TEXT	gogo(SB), 7, $0
	MOVW	R0, R1			// gobuf
	MOVW	8(SP), R0		// return 2nd arg
	MOVW	gobuf_g(R1), g
	MOVW	0(g), R2		// make sure g != nil
	MOVW	gobuf_sp(R1), SP	// restore SP
	MOVW	gobuf_pc(R1), PC

// void gogocall(Gobuf*, void (*fn)(void))
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
// TODO(kaib): add R0 to gobuf so it can be restored properly
// using frame size $-4 means do not save LR on stack.
TEXT gogocall(SB), 7, $-4
	MOVW	8(SP), R1		// fn
	MOVW	gobuf_g(R0), g
	MOVW	0(g), R2		// make sure g != nil
	MOVW	gobuf_sp(R0), SP	// restore SP
	MOVW	gobuf_pc(R0), LR
	MOVW	R1, PC

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// R1 frame size
// R2 arg size
// R3 prolog's LR
// using frame size $-4 means do not save LR on stack.
TEXT sys·morestack(SB),7,$-4
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
	MOVW	g, (m_morebuf+gobuf_g)(m)

	// Set m->morepc to f's PC.
	MOVW	LR, m_morepc(m)

	// Call newstack on m's scheduling stack.
	MOVW	m_g0(m), g
	MOVW	(m_sched+gobuf_sp)(m), SP
	B	newstack(SB)

// Return point when leaving stack.
// using frame size $-4 means do not save LR on stack.
TEXT sys·lessstack(SB), 7, $-4
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
TEXT sys·morestackx(SB), 7, $-4
	MOVW	R0, 0(FP)	// Save arg0
	MOVW	$0, R1		// set frame size
	B	sys·morestack(SB)

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
#define	LDREX(a,r)	WORD	$(0xe<<28|0x01900f9f | (a)<<16 | (r)<<12)
#define	STREX(a,v,r)	WORD	$(0xe<<28|0x01800f90 | (a)<<16 | (r)<<12 | (v)<<0)

TEXT	cas+0(SB),0,$12		/* r0 holds p */
	MOVW	ov+4(FP), R1
	MOVW	nv+8(FP), R2
spin:
/*	LDREX	0(R0),R3	*/
	LDREX(0,3)
	CMP.S	R3, R1
	BNE	fail
/*	STREX	0(R0),R2,R4	*/
	STREX(0,2,4)
	CMP.S	$0, R4
	BNE	spin
	MOVW	$1, R0
	RET
fail:
	MOVW	$0, R0
	RET

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT jmpdefer(SB), 7, $0
	BL	abort(SB)
//	MOVL	4(SP), AX	// fn
//	MOVL	8(SP), BX	// caller sp
//	LEAL	-4(BX), SP	// caller sp after CALL
//	SUBL	$5, (SP)	// return to CALL again
//	JMP	AX	// but first run the deferred function

TEXT	sys·memclr(SB),7,$20
// R0 = addr and passes implicitly to memset
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

TEXT	sys·getcallerpc+0(SB),7,$0
	BL	abort(SB)
//	MOVL	x+0(FP),AX		// addr of first arg
//	MOVL	-4(AX),AX		// get calling pc
//	RET

TEXT	sys·setcallerpc+0(SB),7,$0
	BL	abort(SB)
//	MOVL	x+0(FP),AX		// addr of first arg
//	MOVL	x+4(FP), BX
//	MOVL	BX, -4(AX)		// set calling pc
//	RET

TEXT emptyfunc(SB),0,$0
	RET

TEXT abort(SB),7,$0
	MOVW	$0, R0
	MOVW	(R0), R1

