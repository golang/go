// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

#define	REGCTXT	R22

TEXT runtime·rt0_go(SB),NOSPLIT,$0
	// R29 = stack; R1 = argc; R2 = argv

	// initialize essential registers
	JAL	runtime·reginit(SB)

	ADDV	$-24, R29
	MOVW	R1, 8(R29) // argc
	MOVV	R2, 16(R29) // argv

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVV	$runtime·g0(SB), g
	MOVV	$(-64*1024), R28
	ADDV	R28, R29, R1
	MOVV	R1, g_stackguard0(g)
	MOVV	R1, g_stackguard1(g)
	MOVV	R1, (g_stack+stack_lo)(g)
	MOVV	R29, (g_stack+stack_hi)(g)

	// no cgo yet

nocgo:
	// update stackguard after _cgo_init
	MOVV	(g_stack+stack_lo)(g), R1
	ADDV	$const__StackGuard, R1
	MOVV	R1, g_stackguard0(g)
	MOVV	R1, g_stackguard1(g)

	// set the per-goroutine and per-mach "registers"
	MOVV	$runtime·m0(SB), R1

	// save m->g0 = g0
	MOVV	g, m_g0(R1)
	// save m0 to g0->m
	MOVV	R1, g_m(g)

	JAL	runtime·check(SB)

	// args are already prepared
	JAL	runtime·args(SB)
	JAL	runtime·osinit(SB)
	JAL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVV	$runtime·mainPC(SB), R1		// entry
	ADDV	$-24, R29
	MOVV	R1, 16(R29)
	MOVV	R0, 8(R29)
	MOVV	R0, 0(R29)
	JAL	runtime·newproc(SB)
	ADDV	$24, R29

	// start this M
	JAL	runtime·mstart(SB)

	MOVV	R0, 1(R0)
	RET

DATA	runtime·mainPC+0(SB)/8,$runtime·main(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT,$-8-0
	MOVV	R0, 2(R0) // TODO: TD
	RET

TEXT runtime·asminit(SB),NOSPLIT,$-8-0
	RET

TEXT _cgo_reginit(SB),NOSPLIT,$-8-0
	// crosscall_ppc64 and crosscall2 need to reginit, but can't
	// get at the 'runtime.reginit' symbol.
	JMP	runtime·reginit(SB)

TEXT runtime·reginit(SB),NOSPLIT,$-8-0
	// initialize essential FP registers
	MOVD	$0.5, F26
	SUBD	F26, F26, F24
	ADDD	F26, F26, F28
	ADDD	F28, F28, F30
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT, $-8-8
	MOVV	buf+0(FP), R1
	MOVV	R29, gobuf_sp(R1)
	MOVV	R31, gobuf_pc(R1)
	MOVV	g, gobuf_g(R1)
	MOVV	R0, gobuf_lr(R1)
	MOVV	R0, gobuf_ret(R1)
	MOVV	R0, gobuf_ctxt(R1)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $-8-8
	MOVV	buf+0(FP), R3
	MOVV	gobuf_g(R3), g	// make sure g is not nil
	JAL	runtime·save_g(SB)

	MOVV	0(g), R2
	MOVV	gobuf_sp(R3), R29
	MOVV	gobuf_lr(R3), R31
	MOVV	gobuf_ret(R3), R1
	MOVV	gobuf_ctxt(R3), REGCTXT
	MOVV	R0, gobuf_sp(R3)
	MOVV	R0, gobuf_ret(R3)
	MOVV	R0, gobuf_lr(R3)
	MOVV	R0, gobuf_ctxt(R3)
	MOVV	gobuf_pc(R3), R4
	JMP	(R4)

// void mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $-8-8
	// Save caller state in g->sched
	MOVV	R29, (g_sched+gobuf_sp)(g)
	MOVV	R31, (g_sched+gobuf_pc)(g)
	MOVV	R0, (g_sched+gobuf_lr)(g)
	MOVV	g, (g_sched+gobuf_g)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVV	g, R1
	MOVV	g_m(g), R3
	MOVV	m_g0(R3), g
	JAL	runtime·save_g(SB)
	BNE	g, R1, 2(PC)
	JMP	runtime·badmcall(SB)
	MOVV	fn+0(FP), REGCTXT			// context
	MOVV	0(REGCTXT), R4			// code pointer
	MOVV	(g_sched+gobuf_sp)(g), R29	// sp = m->g0->sched.sp
	ADDV	$-16, R29
	MOVV	R1, 8(R29)
	MOVV	R0, 0(R29)
	JAL	(R4)
	JMP	runtime·badmcall2(SB)

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack.  We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	UNDEF
	JAL	(R31)	// make sure this function is not leaf
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	MOVV	fn+0(FP), R1	// R1 = fn
	MOVV	R1, REGCTXT		// context
	MOVV	g_m(g), R2	// R2 = m

	MOVV	m_gsignal(R2), R3	// R3 = gsignal
	BEQ	g, R3, noswitch

	MOVV	m_g0(R2), R3	// R3 = g0
	BEQ	g, R3, noswitch

	MOVV	m_curg(R2), R4
	BEQ	g, R4, switch

	// Bad: g is not gsignal, not g0, not curg. What is it?
	// Hide call from linker nosplit analysis.
	MOVV	$runtime·badsystemstack(SB), R4
	JAL	(R4)

switch:
	// save our state in g->sched.  Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVV	$runtime·systemstack_switch(SB), R4
	ADDV	$8, R4	// get past prologue
	MOVV	R4, (g_sched+gobuf_pc)(g)
	MOVV	R29, (g_sched+gobuf_sp)(g)
	MOVV	R0, (g_sched+gobuf_lr)(g)
	MOVV	g, (g_sched+gobuf_g)(g)

	// switch to g0
	MOVV	R3, g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R1
	// make it look like mstart called systemstack on g0, to stop traceback
	ADDV	$-8, R1
	MOVV	$runtime·mstart(SB), R2
	MOVV	R2, 0(R1)
	MOVV	R1, R29

	// call target function
	MOVV	0(REGCTXT), R4	// code pointer
	JAL	(R4)

	// switch back to g
	MOVV	g_m(g), R1
	MOVV	m_curg(R1), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R29
	MOVV	R0, (g_sched+gobuf_sp)(g)
	RET

noswitch:
	// already on m stack, just call directly
	MOVV	0(REGCTXT), R4	// code pointer
	JAL	(R4)
	RET

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// Caller has already loaded:
// R1: framesize, R2: argsize, R3: LR
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT,$-8-0
	// Cannot grow scheduler stack (m->g0).
	MOVV	g_m(g), R7
	MOVV	m_g0(R7), R8
	BNE	g, R8, 2(PC)
	JAL	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVV	m_gsignal(R7), R8
	BNE	g, R8, 2(PC)
	JAL	runtime·abort(SB)

	// Called from f.
	// Set g->sched to context in f.
	MOVV	REGCTXT, (g_sched+gobuf_ctxt)(g)
	MOVV	R29, (g_sched+gobuf_sp)(g)
	MOVV	R31, (g_sched+gobuf_pc)(g)
	MOVV	R3, (g_sched+gobuf_lr)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVV	R3, (m_morebuf+gobuf_pc)(R7)	// f's caller's PC
	MOVV	R29, (m_morebuf+gobuf_sp)(R7)	// f's caller's SP
	MOVV	g, (m_morebuf+gobuf_g)(R7)

	// Call newstack on m->g0's stack.
	MOVV	m_g0(R7), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R29
	JAL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT,$-8-0
	MOVV	R0, REGCTXT
	JMP	runtime·morestack(SB)

TEXT runtime·stackBarrier(SB),NOSPLIT,$0
	// We came here via a RET to an overwritten LR.
	// R1 may be live. Other registers are available.

	// Get the original return PC, g.stkbar[g.stkbarPos].savedLRVal.
	MOVV	(g_stkbar+slice_array)(g), R2
	MOVV	g_stkbarPos(g), R3
	MOVV	$stkbar__size, R4
	MULVU	R3, R4
	MOVV	LO, R4
	ADDV	R2, R4
	MOVV	stkbar_savedLRVal(R4), R4
	// Record that this stack barrier was hit.
	ADDV	$1, R3
	MOVV	R3, g_stkbarPos(g)
	// Jump to the original return PC.
	JMP	(R4)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVV	$MAXSIZE, R28;		\
	SGTU	R1, R28, R28;		\
	BNE	R28, 3(PC);			\
	MOVV	$NAME(SB), R4;	\
	JMP	(R4)
// Note: can't just "BR NAME(SB)" - bad inlining results.

TEXT reflect·call(SB), NOSPLIT, $0-0
	JMP	·reflectcall(SB)

TEXT ·reflectcall(SB), NOSPLIT, $-8-32
	MOVWU argsize+24(FP), R1
	// NOTE(rsc): No call16, because CALLFN needs four words
	// of argument space to invoke callwritebarrier.
	DISPATCH(runtime·call32, 32)
	DISPATCH(runtime·call64, 64)
	DISPATCH(runtime·call128, 128)
	DISPATCH(runtime·call256, 256)
	DISPATCH(runtime·call512, 512)
	DISPATCH(runtime·call1024, 1024)
	DISPATCH(runtime·call2048, 2048)
	DISPATCH(runtime·call4096, 4096)
	DISPATCH(runtime·call8192, 8192)
	DISPATCH(runtime·call16384, 16384)
	DISPATCH(runtime·call32768, 32768)
	DISPATCH(runtime·call65536, 65536)
	DISPATCH(runtime·call131072, 131072)
	DISPATCH(runtime·call262144, 262144)
	DISPATCH(runtime·call524288, 524288)
	DISPATCH(runtime·call1048576, 1048576)
	DISPATCH(runtime·call2097152, 2097152)
	DISPATCH(runtime·call4194304, 4194304)
	DISPATCH(runtime·call8388608, 8388608)
	DISPATCH(runtime·call16777216, 16777216)
	DISPATCH(runtime·call33554432, 33554432)
	DISPATCH(runtime·call67108864, 67108864)
	DISPATCH(runtime·call134217728, 134217728)
	DISPATCH(runtime·call268435456, 268435456)
	DISPATCH(runtime·call536870912, 536870912)
	DISPATCH(runtime·call1073741824, 1073741824)
	MOVV	$runtime·badreflectcall(SB), R4
	JMP	(R4)

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-24;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVV	arg+16(FP), R1;			\
	MOVWU	argsize+24(FP), R2;			\
	MOVV	R29, R3;				\
	ADDV	$8, R3;			\
	ADDV	R3, R2;				\
	BEQ	R3, R2, 6(PC);				\
	MOVBU	(R1), R4;			\
	ADDV	$1, R1;			\
	MOVBU	R4, (R3);			\
	ADDV	$1, R3;			\
	JMP	-5(PC);				\
	/* call function */			\
	MOVV	f+8(FP), REGCTXT;			\
	MOVV	(REGCTXT), R4;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	JAL	(R4);				\
	/* copy return values back */		\
	MOVV	arg+16(FP), R1;			\
	MOVWU	n+24(FP), R2;			\
	MOVWU	retoffset+28(FP), R4;		\
	MOVV	R29, R3;				\
	ADDV	R4, R3; 			\
	ADDV	R4, R1;				\
	SUBVU	R4, R2;				\
	ADDV	$8, R3;			\
	ADDV	R3, R2;				\
loop:						\
	BEQ	R3, R2, end;				\
	MOVBU	(R3), R4;			\
	ADDV	$1, R3;			\
	MOVBU	R4, (R1);			\
	ADDV	$1, R1;			\
	JMP	loop;				\
end:						\
	/* execute write barrier updates */	\
	MOVV	argtype+0(FP), R5;		\
	MOVV	arg+16(FP), R1;			\
	MOVWU	n+24(FP), R2;			\
	MOVWU	retoffset+28(FP), R4;		\
	MOVV	R5, 8(R29);			\
	MOVV	R1, 16(R29);			\
	MOVV	R2, 24(R29);			\
	MOVV	R4, 32(R29);			\
	JAL	runtime·callwritebarrier(SB);	\
	RET

CALLFN(·call16, 16)
CALLFN(·call32, 32)
CALLFN(·call64, 64)
CALLFN(·call128, 128)
CALLFN(·call256, 256)
CALLFN(·call512, 512)
CALLFN(·call1024, 1024)
CALLFN(·call2048, 2048)
CALLFN(·call4096, 4096)
CALLFN(·call8192, 8192)
CALLFN(·call16384, 16384)
CALLFN(·call32768, 32768)
CALLFN(·call65536, 65536)
CALLFN(·call131072, 131072)
CALLFN(·call262144, 262144)
CALLFN(·call524288, 524288)
CALLFN(·call1048576, 1048576)
CALLFN(·call2097152, 2097152)
CALLFN(·call4194304, 4194304)
CALLFN(·call8388608, 8388608)
CALLFN(·call16777216, 16777216)
CALLFN(·call33554432, 33554432)
CALLFN(·call67108864, 67108864)
CALLFN(·call134217728, 134217728)
CALLFN(·call268435456, 268435456)
CALLFN(·call536870912, 536870912)
CALLFN(·call1073741824, 1073741824)

TEXT runtime·procyield(SB),NOSPLIT,$0-0
	RET

// void jmpdefer(fv, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 8 bytes to get back to JAL deferreturn
// 3. JMP to fn
TEXT runtime·jmpdefer(SB), NOSPLIT, $-8-16
	MOVV	0(R29), R31
	ADDV	$-8, R31

	MOVV	fv+0(FP), REGCTXT
	MOVV	argp+8(FP), R29
	ADDV	$-8, R29
	NOR	R0, R0	// prevent scheduling
	MOVV	0(REGCTXT), R4
	JMP	(R4)

// Save state of caller into g->sched. Smashes R31.
TEXT gosave<>(SB),NOSPLIT,$-8
	MOVV	R31, (g_sched+gobuf_pc)(g)
	MOVV	R29, (g_sched+gobuf_sp)(g)
	MOVV	R0, (g_sched+gobuf_lr)(g)
	MOVV	R0, (g_sched+gobuf_ret)(g)
	MOVV	R0, (g_sched+gobuf_ctxt)(g)
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	UNDEF	// no cgo yet
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$24-24
	MOVV	$fn+0(FP), R1
	MOVV	R1, 8(R29)
	MOVV	frame+8(FP), R1
	MOVV	R1, 16(R29)
	MOVV	framesize+16(FP), R1
	MOVV	R1, 24(R29)
	MOVV	$runtime·cgocallback_gofunc(SB), R1
	JAL	(R1)
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize)
// See cgocall.go for more details.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$16-24
	NO_LOCAL_POINTERS

	// Load m and g from thread-local storage.
	MOVB	runtime·iscgo(SB), R1
	BEQ	R1, nocgo
	JAL	runtime·load_g(SB)
nocgo:

	// If g is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	BEQ	g, needm

	MOVV	g_m(g), R3
	MOVV	R3, savedm-8(SP)
	JMP	havem

needm:
	MOVV	g, savedm-8(SP) // g is zero, so is m.
	MOVV	$runtime·needm(SB), R4
	JAL	(R4)

	// Set m->sched.sp = SP, so that if a panic happens
	// during the function we are about to execute, it will
	// have a valid SP to run on the g0 stack.
	// The next few lines (after the havem label)
	// will save this SP onto the stack and then write
	// the same SP back to m->sched.sp. That seems redundant,
	// but if an unrecovered panic happens, unwindm will
	// restore the g->sched.sp from the stack location
	// and then systemstack will try to use it. If we don't set it here,
	// that restored SP will be uninitialized (typically 0) and
	// will not be usable.
	MOVV	g_m(g), R1
	MOVV	m_g0(R1), R1
	MOVV	R29, (g_sched+gobuf_sp)(R1)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 8(R29) aka savedsp-16(SP).
	MOVV	m_g0(R3), R1
	MOVV	(g_sched+gobuf_sp)(R1), R2
	MOVV	R2, savedsp-16(SP)
	MOVV	R29, (g_sched+gobuf_sp)(R1)

	// Switch to m->curg stack and call runtime.cgocallbackg.
	// Because we are taking over the execution of m->curg
	// but *not* resuming what had been running, we need to
	// save that information (m->curg->sched) so we can restore it.
	// We can restore m->curg->sched.sp easily, because calling
	// runtime.cgocallbackg leaves SP unchanged upon return.
	// To save m->curg->sched.pc, we push it onto the stack.
	// This has the added benefit that it looks to the traceback
	// routine like cgocallbackg is going to return to that
	// PC (because the frame we allocate below has the same
	// size as cgocallback_gofunc's frame declared above)
	// so that the traceback will seamlessly trace back into
	// the earlier calls.
	//
	// In the new goroutine, -16(SP) and -8(SP) are unused.
	MOVV	m_curg(R3), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R2 // prepare stack as R2
	MOVV	(g_sched+gobuf_pc)(g), R3
	MOVV	R3, -24(R2)
	MOVV	$-24(R2), R29
	JAL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVV	0(R29), R3
	MOVV	R3, (g_sched+gobuf_pc)(g)
	MOVV	$24(R29), R2
	MOVV	R2, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVV	g_m(g), R3
	MOVV	m_g0(R3), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R29
	MOVV	savedsp-16(SP), R2
	MOVV	R2, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	MOVV	savedm-8(SP), R3
	BNE	R3, droppedm
	MOVV	$runtime·dropm(SB), R4
	JAL	(R4)
droppedm:

	// Done!
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-8
	MOVV	gg+0(FP), g
	// This only happens if iscgo, so jump straight to save_g
	JAL	runtime·save_g(SB)
	RET

// void setg_gcc(G*); set g in C TLS.
// Must obey the gcc calling convention.
TEXT setg_gcc<>(SB),NOSPLIT,$-8-0
	UNDEF	// no cgo yet
	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$8-16
	MOVV	16(R29), R1		// LR saved by caller
	MOVV	runtime·stackBarrierPC(SB), R2
	BNE	R1, R2, nobar
	// Get original return PC.
	JAL	runtime·nextBarrierPC(SB)
	MOVV	8(R29), R1
nobar:
	MOVV	R1, ret+8(FP)
	RET

TEXT runtime·setcallerpc(SB),NOSPLIT,$8-16
	MOVV	pc+8(FP), R1
	MOVV	16(R29), R2
	MOVV	runtime·stackBarrierPC(SB), R3
	BEQ	R2, R3, setbar
	MOVV	R1, 16(R29)		// set LR in caller
	RET
setbar:
	// Set the stack barrier return PC.
	MOVV	R1, 8(R29)
	JAL	runtime·setNextBarrierPC(SB)
	RET

TEXT runtime·getcallersp(SB),NOSPLIT,$0-16
	MOVV	argp+0(FP), R1
	ADDV	$-8, R1
	MOVV	R1, ret+8(FP)
	RET

TEXT runtime·abort(SB),NOSPLIT,$-8-0
	MOVW	(R0), R0
	UNDEF

// memhash_varlen(p unsafe.Pointer, h seed) uintptr
// redirects to memhash(p, h, size) using the size
// stored in the closure.
TEXT runtime·memhash_varlen(SB),NOSPLIT,$40-24
	GO_ARGS
	NO_LOCAL_POINTERS
	MOVV	p+0(FP), R1
	MOVV	h+8(FP), R2
	MOVV	8(REGCTXT), R3
	MOVV	R1, 8(R29)
	MOVV	R2, 16(R29)
	MOVV	R3, 24(R29)
	JAL	runtime·memhash(SB)
	MOVV	32(R29), R1
	MOVV	R1, ret+16(FP)
	RET

// AES hashing not implemented for mips64
TEXT runtime·aeshash(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1
TEXT runtime·aeshash32(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1
TEXT runtime·aeshash64(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1
TEXT runtime·aeshashstr(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1

TEXT runtime·memeq(SB),NOSPLIT,$-8-25
	MOVV	a+0(FP), R1
	MOVV	b+8(FP), R2
	MOVV	size+16(FP), R3
	ADDV	R1, R3, R4
loop:
	BNE	R1, R4, test
	MOVV	$1, R1
	MOVB	R1, ret+24(FP)
	RET
test:
	MOVBU	(R1), R6
	ADDV	$1, R1
	MOVBU	(R2), R7
	ADDV	$1, R2
	BEQ	R6, R7, loop

	MOVB	R0, ret+24(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$40-17
	MOVV	a+0(FP), R1
	MOVV	b+8(FP), R2
	BEQ	R1, R2, eq
	MOVV	8(REGCTXT), R3    // compiler stores size at offset 8 in the closure
	MOVV	R1, 8(R29)
	MOVV	R2, 16(R29)
	MOVV	R3, 24(R29)
	JAL	runtime·memeq(SB)
	MOVBU	32(R29), R1
	MOVB	R1, ret+16(FP)
	RET
eq:
	MOVV	$1, R1
	MOVB	R1, ret+16(FP)
	RET

// eqstring tests whether two strings are equal.
// The compiler guarantees that strings passed
// to eqstring have equal length.
// See runtime_test.go:eqstring_generic for
// equivalent Go code.
TEXT runtime·eqstring(SB),NOSPLIT,$0-33
	MOVV	s1str+0(FP), R1
	MOVV	s2str+16(FP), R2
	MOVV	$1, R3
	MOVB	R3, ret+32(FP)
	BNE	R1, R2, 2(PC)
	RET
	MOVV	s1len+8(FP), R3
	ADDV	R1, R3, R4
loop:
	BNE	R1, R4, 2(PC)
	RET
	MOVBU	(R1), R6
	ADDV	$1, R1
	MOVBU	(R2), R7
	ADDV	$1, R2
	BEQ	R6, R7, loop
	MOVB	R0, ret+32(FP)
	RET

// TODO: share code with memeq?
TEXT bytes·Equal(SB),NOSPLIT,$0-49
	MOVV	a_len+8(FP), R3
	MOVV	b_len+32(FP), R4
	BNE	R3, R4, noteq		// unequal lengths are not equal

	MOVV	a+0(FP), R1
	MOVV	b+24(FP), R2
	ADDV	R1, R3		// end

loop:
	BEQ	R1, R3, equal		// reached the end
	MOVBU	(R1), R6
	ADDV	$1, R1
	MOVBU	(R2), R7
	ADDV	$1, R2
	BEQ	R6, R7, loop

noteq:
	MOVB	R0, ret+48(FP)
	RET

equal:
	MOVV	$1, R1
	MOVB	R1, ret+48(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-40
	MOVV	s+0(FP), R1
	MOVV	s_len+8(FP), R2
	MOVBU	c+24(FP), R3	// byte to find
	MOVV	R1, R4		// store base for later
	ADDV	R1, R2		// end
	ADDV	$-1, R1

loop:
	ADDV	$1, R1
	BEQ	R1, R2, notfound
	MOVBU	(R1), R5
	BNE	R3, R5, loop

	SUBV	R4, R1		// remove base
	MOVV	R1, ret+32(FP)
	RET

notfound:
	MOVV	$-1, R1
	MOVV	R1, ret+32(FP)
	RET

TEXT strings·IndexByte(SB),NOSPLIT,$0-32
	MOVV	p+0(FP), R1
	MOVV	b_len+8(FP), R2
	MOVBU	c+16(FP), R3	// byte to find
	MOVV	R1, R4		// store base for later
	ADDV	R1, R2		// end
	ADDV	$-1, R1

loop:
	ADDV	$1, R1
	BEQ	R1, R2, notfound
	MOVBU	(R1), R5
	BNE	R3, R5, loop

	SUBV	R4, R1		// remove base
	MOVV	R1, ret+24(FP)
	RET

notfound:
	MOVV	$-1, R1
	MOVV	R1, ret+24(FP)
	RET

TEXT runtime·fastrand1(SB), NOSPLIT, $0-4
	MOVV	g_m(g), R2
	MOVWU	m_fastrand(R2), R1
	ADDU	R1, R1
	BGEZ	R1, 2(PC)
	XOR	$0x88888eef, R1
	MOVW	R1, m_fastrand(R2)
	MOVW	R1, ret+0(FP)
	RET

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVW	$0, R1
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$-8
	UNDEF	// no cgo yet
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT,$-8-0
	NOR	R0, R0	// NOP
	JAL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	NOR	R0, R0	// NOP

TEXT runtime·prefetcht0(SB),NOSPLIT,$0-8
	RET

TEXT runtime·prefetcht1(SB),NOSPLIT,$0-8
	RET

TEXT runtime·prefetcht2(SB),NOSPLIT,$0-8
	RET

TEXT runtime·prefetchnta(SB),NOSPLIT,$0-8
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R1
	MOVB	R1, ret+0(FP)
	RET
