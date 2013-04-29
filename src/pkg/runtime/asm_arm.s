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

	// if there is an _cgo_init, call it.
	MOVW	_cgo_init(SB), R2
	CMP	$0, R2
	MOVW.NE	g, R0 // first argument of _cgo_init is g
	BL.NE	(R2) // will clobber R0-R3

	BL	runtime·checkgoarm(SB)
	BL	runtime·check(SB)

	// saved argc, argv
	MOVW	60(R13), R0
	MOVW	R0, 4(R13)
	MOVW	64(R13), R1
	MOVW	R1, 8(R13)
	BL	runtime·args(SB)
	BL	runtime·osinit(SB)
	BL	runtime·hashinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVW	$runtime·main·f(SB), R0
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

DATA	runtime·main·f+0(SB)/4,$runtime·main(SB)
GLOBL	runtime·main·f(SB),8,$4

TEXT runtime·breakpoint(SB),7,$0
	// gdb won't skip this breakpoint instruction automatically,
	// so you must manually "set $pc+=4" to skip it and continue.
	WORD    $0xe1200071 // BKPT 0x0001
	RET

GLOBL runtime·goarm(SB), $4
TEXT runtime·asminit(SB),7,$0
	// disable runfast (flush-to-zero) mode of vfp if runtime.goarm > 5
	MOVW runtime·goarm(SB), R11
	CMP $5, R11
	BLE 4(PC)
	WORD $0xeef1ba10	// vmrs r11, fpscr
	BIC $(1<<24), R11
	WORD $0xeee1ba10	// vmsr fpscr, r11
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
	MOVW	gobuf_g(R1), g
	MOVW	0(g), R2		// make sure g != nil
	MOVW	_cgo_save_gm(SB), R2
	CMP 	$0, R2 // if in Cgo, we have to save g and m
	BL.NE	(R2) // this call will clobber R0
	MOVW	4(FP), R0		// return 2nd arg
	MOVW	gobuf_sp(R1), SP	// restore SP
	MOVW	gobuf_pc(R1), PC

// void gogocall(Gobuf*, void (*fn)(void), uintptr r7)
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
// using frame size $-4 means do not save LR on stack.
TEXT runtime·gogocall(SB), 7, $-4
	MOVW	0(FP), R3		// gobuf
	MOVW	4(FP), R1		// fn
	MOVW	gobuf_g(R3), g
	MOVW	0(g), R0		// make sure g != nil
	MOVW	_cgo_save_gm(SB), R0
	CMP 	$0, R0 // if in Cgo, we have to save g and m
	BL.NE	(R0) // this call will clobber R0
	MOVW	8(FP), R7	// context
	MOVW	gobuf_sp(R3), SP	// restore SP
	MOVW	gobuf_pc(R3), LR
	MOVW	R1, PC

// void gogocallfn(Gobuf*, FuncVal*)
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
// using frame size $-4 means do not save LR on stack.
TEXT runtime·gogocallfn(SB), 7, $-4
	MOVW	0(FP), R3		// gobuf
	MOVW	4(FP), R1		// fn
	MOVW	gobuf_g(R3), g
	MOVW	0(g), R0		// make sure g != nil
	MOVW	_cgo_save_gm(SB), R0
	CMP 	$0, R0 // if in Cgo, we have to save g and m
	BL.NE	(R0) // this call will clobber R0
	MOVW	gobuf_sp(R3), SP	// restore SP
	MOVW	gobuf_pc(R3), LR
	MOVW	R1, R7
	MOVW	0(R1), PC

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
	MOVW	R7, m_cret(m) // function context
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
	MOVW	fn+0(FP), R7
	MOVW	argp+4(FP), SP
	MOVW	$-4(SP), SP	// SP is 4 below argp, due to saved LR
	MOVW	0(R7), R1
	B	(R1)

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
TEXT	runtime·asmcgocall(SB),7,$0
	MOVW	fn+0(FP), R1
	MOVW	arg+4(FP), R0
	MOVW	R13, R2
	MOVW	g, R5

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	MOVW	m_g0(m), R3
	CMP	R3, g
	BEQ	7(PC)
	MOVW	R13, (g_sched + gobuf_sp)(g)
	MOVW	$return<>(SB), R4
	MOVW	R4, (g_sched+gobuf_pc)(g)
	MOVW	g, (g_sched+gobuf_g)(g)
	MOVW	R3, g
	MOVW	(g_sched+gobuf_sp)(g), R13

	// Now on a scheduling stack (a pthread-created stack).
	SUB	$24, R13
	BIC	$0x7, R13	// alignment for gcc ABI
	MOVW	R5, 20(R13) // save old g
	MOVW	R2, 16(R13)	// save old SP
	// R0 already contains the first argument
	BL	(R1)

	// Restore registers, g, stack pointer.
	MOVW	20(R13), g
	MOVW	16(R13), R13
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),7,$12
	MOVW	$fn+0(FP), R0
	MOVW	R0, 4(R13)
	MOVW	frame+4(FP), R0
	MOVW	R0, 8(R13)
	MOVW	framesize+8(FP), R0
	MOVW	R0, 12(R13)
	MOVW	$runtime·cgocallback_gofunc(SB), R0
	BL	(R0)
	RET

// cgocallback_gofunc(void (*fn)(void*), void *frame, uintptr framesize)
// See cgocall.c for more details.
TEXT	runtime·cgocallback_gofunc(SB),7,$12
	// Load m and g from thread-local storage.
	MOVW	_cgo_load_gm(SB), R0
	CMP	$0, R0
	BL.NE	(R0)

	// If m is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	MOVW	m, savedm-12(SP)
	CMP	$0, m
	B.NE havem
	MOVW	$runtime·needm(SB), R0
	BL	(R0)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	MOVW	m_g0(m), R3
	MOVW	(g_sched+gobuf_sp)(R3), R4
	MOVW.W	R4, -4(R13)
	MOVW	R13, (g_sched+gobuf_sp)(R3)

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
	// a frame size of 12, the same amount that we use below),
	// so that the traceback will seamlessly trace back into
	// the earlier calls.
	MOVW	fn+4(FP), R0
	MOVW	frame+8(FP), R1
	MOVW	framesize+12(FP), R2

	MOVW	m_curg(m), g
	MOVW	(g_sched+gobuf_sp)(g), R4 // prepare stack as R4

	// Push gobuf.pc
	MOVW	(g_sched+gobuf_pc)(g), R5
	MOVW.W	R5, -16(R4)

	// Push arguments to cgocallbackg.
	// Frame size here must match the frame size above
	// to trick traceback routines into doing the right thing.
	MOVW	R0, 4(R4)
	MOVW	R1, 8(R4)
	MOVW	R2, 12(R4)
	
	// Switch stack and make the call.
	MOVW	R4, R13
	BL	runtime·cgocallbackg(SB)

	// Restore g->gobuf (== m->curg->gobuf) from saved values.
	MOVW	0(R13), R5
	MOVW	R5, (g_sched+gobuf_pc)(g)
	ADD	$(12+4), R13, R4
	MOVW	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), R13
	// POP R6
	MOVW	0(R13), R6
	ADD	$4, R13
	MOVW	R6, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	MOVW	savedm-12(SP), R6
	CMP	$0, R6
	B.NE	3(PC)
	MOVW	$runtime·dropm(SB), R0
	BL	(R0)

	// Done!
	RET

// void setmg(M*, G*); set m and g. for use by needm.
TEXT runtime·setmg(SB), 7, $0
	MOVW	mm+0(FP), m
	MOVW	gg+4(FP), g

	// Save m and g to thread-local storage.
	MOVW	_cgo_save_gm(SB), R0
	CMP	$0, R0
	BL.NE	(R0)

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
// To implement runtime·cas in sys_$GOOS_arm.s
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

TEXT runtime·stackguard(SB),7,$0
	MOVW	R13, R1
	MOVW	g_stackguard(g), R2
	MOVW	R1, sp+0(FP)
	MOVW	R2, limit+4(FP)
	RET

// AES hashing not implemented for ARM
TEXT runtime·aeshash(SB),7,$-4
	MOVW	$0, R0
	MOVW	(R0), R1
TEXT runtime·aeshash32(SB),7,$-4
	MOVW	$0, R0
	MOVW	(R0), R1
TEXT runtime·aeshash64(SB),7,$-4
	MOVW	$0, R0
	MOVW	(R0), R1
TEXT runtime·aeshashstr(SB),7,$-4
	MOVW	$0, R0
	MOVW	(R0), R1

TEXT runtime·memeq(SB),7,$-4
	MOVW	a+0(FP), R1
	MOVW	b+4(FP), R2
	MOVW	n+8(FP), R3
	ADD	R1, R3, R6
	MOVW	$1, R0
_next:
	CMP	R1, R6
	RET.EQ
	MOVBU.P	1(R1), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	BEQ	_next

	MOVW	$0, R0
	RET
