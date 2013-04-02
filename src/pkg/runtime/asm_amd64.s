// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

TEXT _rt0_amd64(SB),7,$-8
	// copy arguments forward on an even stack
	MOVQ	DI, AX		// argc
	MOVQ	SI, BX		// argv
	SUBQ	$(4*8+7), SP		// 2args 2auto
	ANDQ	$~15, SP
	MOVQ	AX, 16(SP)
	MOVQ	BX, 24(SP)
	
	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVQ	$runtime·g0(SB), DI
	LEAQ	(-64*1024+104)(SP), BX
	MOVQ	BX, g_stackguard(DI)
	MOVQ	SP, g_stackbase(DI)

	// find out information about the processor we're on
	MOVQ	$0, AX
	CPUID
	CMPQ	AX, $0
	JE	nocpuinfo
	MOVQ	$1, AX
	CPUID
	MOVL	CX, runtime·cpuid_ecx(SB)
	MOVL	DX, runtime·cpuid_edx(SB)
nocpuinfo:	
	
	// if there is an _cgo_init, call it.
	MOVQ	_cgo_init(SB), AX
	TESTQ	AX, AX
	JZ	needtls
	// g0 already in DI
	MOVQ	DI, CX	// Win64 uses CX for first parameter
	MOVQ	$setmg_gcc<>(SB), SI
	CALL	AX
	CMPL	runtime·iswindows(SB), $0
	JEQ ok

needtls:
	// skip TLS setup on Plan 9
	CMPL	runtime·isplan9(SB), $1
	JEQ ok

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
	CALL	runtime·hashinit(SB)
	CALL	runtime·schedinit(SB)

	// create a new goroutine to start program
	PUSHQ	$runtime·main·f(SB)		// entry
	PUSHQ	$0			// arg size
	CALL	runtime·newproc(SB)
	POPQ	AX
	POPQ	AX

	// start this M
	CALL	runtime·mstart(SB)

	MOVL	$0xf1, 0xf1  // crash
	RET

DATA	runtime·main·f+0(SB)/8,$runtime·main(SB)
GLOBL	runtime·main·f(SB),8,$8

TEXT runtime·breakpoint(SB),7,$0
	BYTE	$0xcc
	RET

TEXT runtime·asminit(SB),7,$0
	// No per-thread init.
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

// void gogocall(Gobuf*, void (*fn)(void), uintptr r0)
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
TEXT runtime·gogocall(SB), 7, $0
	MOVQ	24(SP), DX	// context
	MOVQ	16(SP), AX		// fn
	MOVQ	8(SP), BX		// gobuf
	MOVQ	gobuf_g(BX), DI
	get_tls(CX)
	MOVQ	DI, g(CX)
	MOVQ	0(DI), CX	// make sure g != nil
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_pc(BX), BX
	PUSHQ	BX
	JMP	AX
	POPQ	BX	// not reached

// void gogocallfn(Gobuf*, FuncVal*)
// restore state from Gobuf but then call fn.
// (call fn, returning to state in Gobuf)
TEXT runtime·gogocallfn(SB), 7, $0
	MOVQ	16(SP), DX		// fn
	MOVQ	8(SP), BX		// gobuf
	MOVQ	gobuf_g(BX), AX
	get_tls(CX)
	MOVQ	AX, g(CX)
	MOVQ	0(AX), CX	// make sure g != nil
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_pc(BX), BX
	PUSHQ	BX
	MOVQ	0(DX), BX
	JMP	BX
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
	
	MOVQ	DX, m_cret(BX)

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

// bool	runtime·cas64(uint64 *val, uint64 *old, uint64 new)
// Atomically:
//	if(*val == *old){
//		*val = new;
//		return 1;
//	} else {
//		*old = *val
//		return 0;
//	}
TEXT runtime·cas64(SB), 7, $0
	MOVQ	8(SP), BX
	MOVQ	16(SP), BP
	MOVQ	0(BP), AX
	MOVQ	24(SP), CX
	LOCK
	CMPXCHGQ	CX, 0(BX)
	JNZ	cas64_fail
	MOVL	$1, AX
	RET
cas64_fail:
	MOVQ	AX, 0(BP)
	MOVL	$0, AX
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

TEXT runtime·xadd64(SB), 7, $0
	MOVQ	8(SP), BX
	MOVQ	16(SP), AX
	MOVQ	AX, CX
	LOCK
	XADDQ	AX, 0(BX)
	ADDQ	CX, AX
	RET

TEXT runtime·xchg(SB), 7, $0
	MOVQ	8(SP), BX
	MOVL	16(SP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT runtime·xchg64(SB), 7, $0
	MOVQ	8(SP), BX
	MOVQ	16(SP), AX
	XCHGQ	AX, 0(BX)
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

TEXT runtime·atomicstore64(SB), 7, $0
	MOVQ	8(SP), BX
	MOVQ	16(SP), AX
	XCHGQ	AX, 0(BX)
	RET

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT runtime·jmpdefer(SB), 7, $0
	MOVQ	8(SP), DX	// fn
	MOVQ	16(SP), BX	// caller sp
	LEAQ	-8(BX), SP	// caller sp after CALL
	SUBQ	$5, (SP)	// return to CALL again
	MOVQ	0(DX), BX
	JMP	BX	// but first run the deferred function

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
	// Make sure we have enough room for 4 stack-backed fast-call
	// registers as per windows amd64 calling convention.
	SUBQ	$64, SP
	ANDQ	$~15, SP	// alignment for gcc ABI
	MOVQ	DI, 48(SP)	// save g
	MOVQ	DX, 40(SP)	// save SP
	MOVQ	BX, DI		// DI = first argument in AMD64 ABI
	MOVQ	BX, CX		// CX = first argument in Win64
	CALL	AX

	// Restore registers, g, stack pointer.
	get_tls(CX)
	MOVQ	48(SP), DI
	MOVQ	DI, g(CX)
	MOVQ	40(SP), SP
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),7,$24
	LEAQ	fn+0(FP), AX
	MOVQ	AX, 0(SP)
	MOVQ	frame+8(FP), AX
	MOVQ	AX, 8(SP)
	MOVQ	framesize+16(FP), AX
	MOVQ	AX, 16(SP)
	MOVQ	$runtime·cgocallback_gofunc(SB), AX
	CALL	AX
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize)
// See cgocall.c for more details.
TEXT runtime·cgocallback_gofunc(SB),7,$24
	// If m is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call through AX.
	get_tls(CX)
#ifdef GOOS_windows
	CMPQ	CX, $0
	JNE	3(PC)
	PUSHQ	$0
	JMP	needm
#endif
	MOVQ	m(CX), BP
	PUSHQ	BP
	CMPQ	BP, $0
	JNE	havem
needm:
	MOVQ	$runtime·needm(SB), AX
	CALL	AX
	get_tls(CX)
	MOVQ	m(CX), BP

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
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
	MOVQ	fn+0(FP), AX
	MOVQ	frame+8(FP), BX
	MOVQ	framesize+16(FP), DX

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
	
	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	POPQ	BP
	CMPQ	BP, $0
	JNE 3(PC)
	MOVQ	$runtime·dropm(SB), AX
	CALL	AX

	// Done!
	RET

// void setmg(M*, G*); set m and g. for use by needm.
TEXT runtime·setmg(SB), 7, $0
	MOVQ	mm+0(FP), AX
#ifdef GOOS_windows
	CMPQ	AX, $0
	JNE	settls
	MOVQ	$0, 0x28(GS)
	RET
settls:
	LEAQ	m_tls(AX), AX
	MOVQ	AX, 0x28(GS)
#endif
	get_tls(CX)
	MOVQ	mm+0(FP), AX
	MOVQ	AX, m(CX)
	MOVQ	gg+8(FP), BX
	MOVQ	BX, g(CX)
	RET

// void setmg_gcc(M*, G*); set m and g called from gcc.
TEXT setmg_gcc<>(SB),7,$0
	get_tls(AX)
	MOVQ	DI, m(AX)
	MOVQ	SI, g(AX)
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

// int64 runtime·cputicks(void)
TEXT runtime·cputicks(SB),7,$0
	RDTSC
	SHLQ	$32, DX
	ADDQ	DX, AX
	RET

TEXT runtime·stackguard(SB),7,$0
	MOVQ	SP, DX
	MOVQ	DX, sp+0(FP)
	get_tls(CX)
	MOVQ	g(CX), BX
	MOVQ	g_stackguard(BX), DX
	MOVQ	DX, limit+8(FP)
	RET

GLOBL runtime·tls0(SB), $64

// hash function using AES hardware instructions
TEXT runtime·aeshash(SB),7,$0
	MOVQ	8(SP), DX	// ptr to hash value
	MOVQ	16(SP), CX	// size
	MOVQ	24(SP), AX	// ptr to data
	JMP	runtime·aeshashbody(SB)

TEXT runtime·aeshashstr(SB),7,$0
	MOVQ	8(SP), DX	// ptr to hash value
	MOVQ	24(SP), AX	// ptr to string struct
	MOVQ	8(AX), CX	// length of string
	MOVQ	(AX), AX	// string data
	JMP	runtime·aeshashbody(SB)

// AX: data
// CX: length
// DX: ptr to seed input / hash output
TEXT runtime·aeshashbody(SB),7,$0
	MOVQ	(DX), X0	// seed to low 64 bits of xmm0
	PINSRQ	$1, CX, X0	// size to high 64 bits of xmm0
	MOVO	runtime·aeskeysched+0(SB), X2
	MOVO	runtime·aeskeysched+16(SB), X3
aesloop:
	CMPQ	CX, $16
	JB	aesloopend
	MOVOU	(AX), X1
	AESENC	X2, X0
	AESENC	X1, X0
	SUBQ	$16, CX
	ADDQ	$16, AX
	JMP	aesloop
aesloopend:
	TESTQ	CX, CX
	JE	finalize	// no partial block

	TESTQ	$16, AX
	JNE	highpartial

	// address ends in 0xxxx.  16 bytes loaded
	// at this address won't cross a page boundary, so
	// we can load it directly.
	MOVOU	(AX), X1
	ADDQ	CX, CX
	PAND	masks(SB)(CX*8), X1
	JMP	partial
highpartial:
	// address ends in 1xxxx.  Might be up against
	// a page boundary, so load ending at last byte.
	// Then shift bytes down using pshufb.
	MOVOU	-16(AX)(CX*1), X1
	ADDQ	CX, CX
	PSHUFB	shifts(SB)(CX*8), X1
partial:
	// incorporate partial block into hash
	AESENC	X3, X0
	AESENC	X1, X0
finalize:	
	// finalize hash
	AESENC	X2, X0
	AESENC	X3, X0
	AESENC	X2, X0
	MOVQ	X0, (DX)
	RET

TEXT runtime·aeshash32(SB),7,$0
	MOVQ	8(SP), DX	// ptr to hash value
	MOVQ	24(SP), AX	// ptr to data
	MOVQ	(DX), X0	// seed
	PINSRD	$2, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+0(SB), X0
	MOVQ	X0, (DX)
	RET

TEXT runtime·aeshash64(SB),7,$0
	MOVQ	8(SP), DX	// ptr to hash value
	MOVQ	24(SP), AX	// ptr to data
	MOVQ	(DX), X0	// seed
	PINSRQ	$1, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+0(SB), X0
	MOVQ	X0, (DX)
	RET

// simple mask to get rid of data in the high part of the register.
TEXT masks(SB),7,$0
	QUAD $0x0000000000000000
	QUAD $0x0000000000000000
	QUAD $0x00000000000000ff
	QUAD $0x0000000000000000
	QUAD $0x000000000000ffff
	QUAD $0x0000000000000000
	QUAD $0x0000000000ffffff
	QUAD $0x0000000000000000
	QUAD $0x00000000ffffffff
	QUAD $0x0000000000000000
	QUAD $0x000000ffffffffff
	QUAD $0x0000000000000000
	QUAD $0x0000ffffffffffff
	QUAD $0x0000000000000000
	QUAD $0x00ffffffffffffff
	QUAD $0x0000000000000000
	QUAD $0xffffffffffffffff
	QUAD $0x0000000000000000
	QUAD $0xffffffffffffffff
	QUAD $0x00000000000000ff
	QUAD $0xffffffffffffffff
	QUAD $0x000000000000ffff
	QUAD $0xffffffffffffffff
	QUAD $0x0000000000ffffff
	QUAD $0xffffffffffffffff
	QUAD $0x00000000ffffffff
	QUAD $0xffffffffffffffff
	QUAD $0x000000ffffffffff
	QUAD $0xffffffffffffffff
	QUAD $0x0000ffffffffffff
	QUAD $0xffffffffffffffff
	QUAD $0x00ffffffffffffff

	// these are arguments to pshufb.  They move data down from
	// the high bytes of the register to the low bytes of the register.
	// index is how many bytes to move.
TEXT shifts(SB),7,$0
	QUAD $0x0000000000000000
	QUAD $0x0000000000000000
	QUAD $0xffffffffffffff0f
	QUAD $0xffffffffffffffff
	QUAD $0xffffffffffff0f0e
	QUAD $0xffffffffffffffff
	QUAD $0xffffffffff0f0e0d
	QUAD $0xffffffffffffffff
	QUAD $0xffffffff0f0e0d0c
	QUAD $0xffffffffffffffff
	QUAD $0xffffff0f0e0d0c0b
	QUAD $0xffffffffffffffff
	QUAD $0xffff0f0e0d0c0b0a
	QUAD $0xffffffffffffffff
	QUAD $0xff0f0e0d0c0b0a09
	QUAD $0xffffffffffffffff
	QUAD $0x0f0e0d0c0b0a0908
	QUAD $0xffffffffffffffff
	QUAD $0x0e0d0c0b0a090807
	QUAD $0xffffffffffffff0f
	QUAD $0x0d0c0b0a09080706
	QUAD $0xffffffffffff0f0e
	QUAD $0x0c0b0a0908070605
	QUAD $0xffffffffff0f0e0d
	QUAD $0x0b0a090807060504
	QUAD $0xffffffff0f0e0d0c
	QUAD $0x0a09080706050403
	QUAD $0xffffff0f0e0d0c0b
	QUAD $0x0908070605040302
	QUAD $0xffff0f0e0d0c0b0a
	QUAD $0x0807060504030201
	QUAD $0xff0f0e0d0c0b0a09

TEXT runtime·memeq(SB),7,$0
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	count+16(FP), BX
	JMP	runtime·memeqbody(SB)


TEXT bytes·Equal(SB),7,$0
	MOVQ	a_len+8(FP), BX
	MOVQ	b_len+32(FP), CX
	XORQ	AX, AX
	CMPQ	BX, CX
	JNE	eqret
	MOVQ	a+0(FP), SI
	MOVQ	b+24(FP), DI
	CALL	runtime·memeqbody(SB)
eqret:
	MOVB	AX, ret+48(FP)
	RET

// a in SI
// b in DI
// count in BX
TEXT runtime·memeqbody(SB),7,$0
	XORQ	AX, AX

	CMPQ	BX, $8
	JB	small
	
	// 64 bytes at a time using xmm registers
hugeloop:
	CMPQ	BX, $64
	JB	bigloop
	MOVOU	(SI), X0
	MOVOU	(DI), X1
	MOVOU	16(SI), X2
	MOVOU	16(DI), X3
	MOVOU	32(SI), X4
	MOVOU	32(DI), X5
	MOVOU	48(SI), X6
	MOVOU	48(DI), X7
	PCMPEQB	X1, X0
	PCMPEQB	X3, X2
	PCMPEQB	X5, X4
	PCMPEQB	X7, X6
	PAND	X2, X0
	PAND	X6, X4
	PAND	X4, X0
	PMOVMSKB X0, DX
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$64, BX
	CMPL	DX, $0xffff
	JEQ	hugeloop
	RET

	// 8 bytes at a time using 64-bit register
bigloop:
	CMPQ	BX, $8
	JBE	leftover
	MOVQ	(SI), CX
	MOVQ	(DI), DX
	ADDQ	$8, SI
	ADDQ	$8, DI
	SUBQ	$8, BX
	CMPQ	CX, DX
	JEQ	bigloop
	RET

	// remaining 0-8 bytes
leftover:
	MOVQ	-8(SI)(BX*1), CX
	MOVQ	-8(DI)(BX*1), DX
	CMPQ	CX, DX
	SETEQ	AX
	RET

small:
	CMPQ	BX, $0
	JEQ	equal

	LEAQ	0(BX*8), CX
	NEGQ	CX

	CMPB	SI, $0xf8
	JA	si_high

	// load at SI won't cross a page boundary.
	MOVQ	(SI), SI
	JMP	si_finish
si_high:
	// address ends in 11111xxx.  Load up to bytes we want, move to correct position.
	MOVQ	-8(SI)(BX*1), SI
	SHRQ	CX, SI
si_finish:

	// same for DI.
	CMPB	DI, $0xf8
	JA	di_high
	MOVQ	(DI), DI
	JMP	di_finish
di_high:
	MOVQ	-8(DI)(BX*1), DI
	SHRQ	CX, DI
di_finish:

	SUBQ	SI, DI
	SHLQ	CX, DI
equal:
	SETEQ	AX
	RET
