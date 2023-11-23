// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

#define	REGCTXT	R29

TEXT runtime·rt0_go(SB),NOSPLIT|TOPFRAME,$0
	// R3 = stack; R4 = argc; R5 = argv

	ADDV	$-24, R3
	MOVW	R4, 8(R3) // argc
	MOVV	R5, 16(R3) // argv

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVV	$runtime·g0(SB), g
	MOVV	$(-64*1024), R30
	ADDV	R30, R3, R19
	MOVV	R19, g_stackguard0(g)
	MOVV	R19, g_stackguard1(g)
	MOVV	R19, (g_stack+stack_lo)(g)
	MOVV	R3, (g_stack+stack_hi)(g)

	// if there is a _cgo_init, call it using the gcc ABI.
	MOVV	_cgo_init(SB), R25
	BEQ	R25, nocgo

	MOVV	R0, R7	// arg 3: not used
	MOVV	R0, R6	// arg 2: not used
	MOVV	$setg_gcc<>(SB), R5	// arg 1: setg
	MOVV	g, R4	// arg 0: G
	JAL	(R25)

nocgo:
	// update stackguard after _cgo_init
	MOVV	(g_stack+stack_lo)(g), R19
	ADDV	$const_stackGuard, R19
	MOVV	R19, g_stackguard0(g)
	MOVV	R19, g_stackguard1(g)

	// set the per-goroutine and per-mach "registers"
	MOVV	$runtime·m0(SB), R19

	// save m->g0 = g0
	MOVV	g, m_g0(R19)
	// save m0 to g0->m
	MOVV	R19, g_m(g)

	JAL	runtime·check(SB)

	// args are already prepared
	JAL	runtime·args(SB)
	JAL	runtime·osinit(SB)
	JAL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVV	$runtime·mainPC(SB), R19		// entry
	ADDV	$-16, R3
	MOVV	R19, 8(R3)
	MOVV	R0, 0(R3)
	JAL	runtime·newproc(SB)
	ADDV	$16, R3

	// start this M
	JAL	runtime·mstart(SB)

	MOVV	R0, 1(R0)
	RET

DATA	runtime·mainPC+0(SB)/8,$runtime·main<ABIInternal>(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT|NOFRAME,$0-0
	BREAK
	RET

TEXT runtime·asminit(SB),NOSPLIT|NOFRAME,$0-0
	RET

TEXT runtime·mstart(SB),NOSPLIT|TOPFRAME,$0
	JAL     runtime·mstart0(SB)
	RET // not reached

// func cputicks() int64
TEXT runtime·cputicks(SB),NOSPLIT,$0-8
	RDTIMED	R0, R4
	MOVV	R4, ret+0(FP)
	RET

/*
 *  go-routine
 */

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT|NOFRAME, $0-8
	MOVV	buf+0(FP), R4
	MOVV	gobuf_g(R4), R5
	MOVV	0(R5), R0	// make sure g != nil
	JMP	gogo<>(SB)

TEXT gogo<>(SB), NOSPLIT|NOFRAME, $0
	MOVV	R5, g
	JAL	runtime·save_g(SB)

	MOVV	gobuf_sp(R4), R3
	MOVV	gobuf_lr(R4), R1
	MOVV	gobuf_ret(R4), R19
	MOVV	gobuf_ctxt(R4), REGCTXT
	MOVV	R0, gobuf_sp(R4)
	MOVV	R0, gobuf_ret(R4)
	MOVV	R0, gobuf_lr(R4)
	MOVV	R0, gobuf_ctxt(R4)
	MOVV	gobuf_pc(R4), R6
	JMP	(R6)

// void mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-8
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R4, REGCTXT
#else
	MOVV	fn+0(FP), REGCTXT
#endif

	// Save caller state in g->sched
	MOVV	R3, (g_sched+gobuf_sp)(g)
	MOVV	R1, (g_sched+gobuf_pc)(g)
	MOVV	R0, (g_sched+gobuf_lr)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVV	g, R4		// arg = g
	MOVV	g_m(g), R20
	MOVV	m_g0(R20), g
	JAL	runtime·save_g(SB)
	BNE	g, R4, 2(PC)
	JMP	runtime·badmcall(SB)
	MOVV	0(REGCTXT), R20			// code pointer
	MOVV	(g_sched+gobuf_sp)(g), R3	// sp = m->g0->sched.sp
	ADDV	$-16, R3
	MOVV	R4, 8(R3)
	MOVV	R0, 0(R3)
	JAL	(R20)
	JMP	runtime·badmcall2(SB)

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack. We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	UNDEF
	JAL	(R1)	// make sure this function is not leaf
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	MOVV	fn+0(FP), R19	// R19 = fn
	MOVV	R19, REGCTXT		// context
	MOVV	g_m(g), R4	// R4 = m

	MOVV	m_gsignal(R4), R5	// R5 = gsignal
	BEQ	g, R5, noswitch

	MOVV	m_g0(R4), R5	// R5 = g0
	BEQ	g, R5, noswitch

	MOVV	m_curg(R4), R6
	BEQ	g, R6, switch

	// Bad: g is not gsignal, not g0, not curg. What is it?
	// Hide call from linker nosplit analysis.
	MOVV	$runtime·badsystemstack(SB), R7
	JAL	(R7)
	JAL	runtime·abort(SB)

switch:
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	JAL	gosave_systemstack_switch<>(SB)

	// switch to g0
	MOVV	R5, g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R19
	MOVV	R19, R3

	// call target function
	MOVV	0(REGCTXT), R6	// code pointer
	JAL	(R6)

	// switch back to g
	MOVV	g_m(g), R4
	MOVV	m_curg(R4), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R3
	MOVV	R0, (g_sched+gobuf_sp)(g)
	RET

noswitch:
	// already on m stack, just call directly
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVV	0(REGCTXT), R4	// code pointer
	MOVV	0(R3), R1	// restore LR
	ADDV	$8, R3
	JMP	(R4)

// func switchToCrashStack0(fn func())
TEXT runtime·switchToCrashStack0(SB), NOSPLIT, $0-8
	MOVV	fn+0(FP), REGCTXT	// context register
	MOVV	g_m(g), R4	// curm

	// set g to gcrash
	MOVV	$runtime·gcrash(SB), g	// g = &gcrash
	JAL	runtime·save_g(SB)
	MOVV	R4, g_m(g)	// g.m = curm
	MOVV	g, m_g0(R4)	// curm.g0 = g

	// switch to crashstack
	MOVV	(g_stack+stack_hi)(g), R4
	ADDV	$(-4*8), R4, R3

	// call target function
	MOVV	0(REGCTXT), R6
	JAL	(R6)

	// should never return
	JAL	runtime·abort(SB)
	UNDEF

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// Caller has already loaded:
// loong64: R31: LR
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT|NOFRAME,$0-0
	// Called from f.
	// Set g->sched to context in f.
	MOVV	R3, (g_sched+gobuf_sp)(g)
	MOVV	R1, (g_sched+gobuf_pc)(g)
	MOVV	R31, (g_sched+gobuf_lr)(g)
	MOVV	REGCTXT, (g_sched+gobuf_ctxt)(g)

	// Cannot grow scheduler stack (m->g0).
	MOVV	g_m(g), R7
	MOVV	m_g0(R7), R8
	BNE	g, R8, 3(PC)
	JAL	runtime·badmorestackg0(SB)
	JAL	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVV	m_gsignal(R7), R8
	BNE	g, R8, 3(PC)
	JAL	runtime·badmorestackgsignal(SB)
	JAL	runtime·abort(SB)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVV	R31, (m_morebuf+gobuf_pc)(R7)	// f's caller's PC
	MOVV	R3, (m_morebuf+gobuf_sp)(R7)	// f's caller's SP
	MOVV	g, (m_morebuf+gobuf_g)(R7)

	// Call newstack on m->g0's stack.
	MOVV	m_g0(R7), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R3
	// Create a stack frame on g0 to call newstack.
	MOVV	R0, -8(R3)	// Zero saved LR in frame
	ADDV	$-8, R3
	JAL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT|NOFRAME,$0-0
	// Force SPWRITE. This function doesn't actually write SP,
	// but it is called with a special calling convention where
	// the caller doesn't save LR on stack but passes it as a
	// register (R5), and the unwinder currently doesn't understand.
	// Make it SPWRITE to stop unwinding. (See issue 54332)
	MOVV    R3, R3

	MOVV	R0, REGCTXT
	JMP	runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(stackArgsType *_type, f *FuncVal, stackArgs *byte, stackArgsSize, stackRetOffset, frameSize uint32, regArgs *abi.RegArgs).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVV	$MAXSIZE, R30;		\
	SGTU	R19, R30, R30;		\
	BNE	R30, 3(PC);			\
	MOVV	$NAME(SB), R4;	\
	JMP	(R4)
// Note: can't just "BR NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT|NOFRAME, $0-48
	MOVWU frameSize+32(FP), R19
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
TEXT NAME(SB), WRAPPER, $MAXSIZE-48;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVV	arg+16(FP), R4;			\
	MOVWU	argsize+24(FP), R5;			\
	MOVV	R3, R12;				\
	ADDV	$8, R12;			\
	ADDV	R12, R5;				\
	BEQ	R12, R5, 6(PC);				\
	MOVBU	(R4), R6;			\
	ADDV	$1, R4;			\
	MOVBU	R6, (R12);			\
	ADDV	$1, R12;			\
	JMP	-5(PC);				\
	/* set up argument registers */		\
	MOVV	regArgs+40(FP), R25;		\
	JAL	·unspillArgs(SB);		\
	/* call function */			\
	MOVV	f+8(FP), REGCTXT;			\
	MOVV	(REGCTXT), R25;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	JAL	(R25);				\
	/* copy return values back */		\
	MOVV	regArgs+40(FP), R25;		\
	JAL	·spillArgs(SB);		\
	MOVV	argtype+0(FP), R7;		\
	MOVV	arg+16(FP), R4;			\
	MOVWU	n+24(FP), R5;			\
	MOVWU	retoffset+28(FP), R6;		\
	ADDV	$8, R3, R12;				\
	ADDV	R6, R12; 			\
	ADDV	R6, R4;				\
	SUBVU	R6, R5;				\
	JAL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $40-0
	NO_LOCAL_POINTERS
	MOVV	R7, 8(R3)
	MOVV	R4, 16(R3)
	MOVV	R12, 24(R3)
	MOVV	R5, 32(R3)
	MOVV	R25, 40(R3)
	JAL	runtime·reflectcallmove(SB)
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

// Save state of caller into g->sched.
// but using fake PC from systemstack_switch.
// Must only be called from functions with no locals ($0)
// or else unwinding from systemstack_switch is incorrect.
// Smashes R19.
TEXT gosave_systemstack_switch<>(SB),NOSPLIT|NOFRAME,$0
	MOVV    $runtime·systemstack_switch(SB), R19
	ADDV	$8, R19
	MOVV	R19, (g_sched+gobuf_pc)(g)
	MOVV	R3, (g_sched+gobuf_sp)(g)
	MOVV	R0, (g_sched+gobuf_lr)(g)
	MOVV	R0, (g_sched+gobuf_ret)(g)
	// Assert ctxt is zero. See func save.
	MOVV	(g_sched+gobuf_ctxt)(g), R19
	BEQ	R19, 2(PC)
	JAL	runtime·abort(SB)
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	MOVV	fn+0(FP), R25
	MOVV	arg+8(FP), R4

	MOVV	R3, R12	// save original stack pointer
	MOVV	g, R13

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	MOVV	g_m(g), R5
	MOVV	m_gsignal(R5), R6
	BEQ	R6, g, g0
	MOVV	m_g0(R5), R6
	BEQ	R6, g, g0

	JAL	gosave_systemstack_switch<>(SB)
	MOVV	R6, g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R3

	// Now on a scheduling stack (a pthread-created stack).
g0:
	// Save room for two of our pointers.
	ADDV	$-16, R3
	MOVV	R13, 0(R3)	// save old g on stack
	MOVV	(g_stack+stack_hi)(R13), R13
	SUBVU	R12, R13
	MOVV	R13, 8(R3)	// save depth in old g stack (can't just save SP, as stack might be copied during a callback)
	JAL	(R25)

	// Restore g, stack pointer. R4 is return value.
	MOVV	0(R3), g
	JAL	runtime·save_g(SB)
	MOVV	(g_stack+stack_hi)(g), R5
	MOVV	8(R3), R6
	SUBVU	R6, R5
	MOVV	R5, R3

	MOVW	R4, ret+16(FP)
	RET

// func cgocallback(fn, frame unsafe.Pointer, ctxt uintptr)
// See cgocall.go for more details.
TEXT ·cgocallback(SB),NOSPLIT,$24-24
	NO_LOCAL_POINTERS

	// Skip cgocallbackg, just dropm when fn is nil, and frame is the saved g.
	// It is used to dropm while thread is exiting.
	MOVV    fn+0(FP), R5
	BNE	R5, loadg
	// Restore the g from frame.
	MOVV    frame+8(FP), g
	JMP	dropm

loadg:
	// Load m and g from thread-local storage.
	MOVB	runtime·iscgo(SB), R19
	BEQ	R19, nocgo
	JAL	runtime·load_g(SB)
nocgo:

	// If g is nil, Go did not create the current thread,
	// or if this thread never called into Go on pthread platforms.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	BEQ	g, needm

	MOVV	g_m(g), R12
	MOVV	R12, savedm-8(SP)
	JMP	havem

needm:
	MOVV	g, savedm-8(SP) // g is zero, so is m.
	MOVV	$runtime·needAndBindM(SB), R4
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
	MOVV	g_m(g), R12
	MOVV	m_g0(R12), R19
	MOVV	R3, (g_sched+gobuf_sp)(R19)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 8(R29) aka savedsp-16(SP).
	MOVV	m_g0(R12), R19
	MOVV	(g_sched+gobuf_sp)(R19), R13
	MOVV	R13, savedsp-24(SP) // must match frame size
	MOVV	R3, (g_sched+gobuf_sp)(R19)

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
	MOVV	m_curg(R12), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R13 // prepare stack as R13
	MOVV	(g_sched+gobuf_pc)(g), R4
	MOVV	R4, -(24+8)(R13) // "saved LR"; must match frame size
	MOVV    fn+0(FP), R5
	MOVV    frame+8(FP), R6
	MOVV    ctxt+16(FP), R7
	MOVV	$-(24+8)(R13), R3
	MOVV    R5, 8(R3)
	MOVV    R6, 16(R3)
	MOVV    R7, 24(R3)
	JAL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVV	0(R3), R4
	MOVV	R4, (g_sched+gobuf_pc)(g)
	MOVV	$(24+8)(R3), R13 // must match frame size
	MOVV	R13, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVV	g_m(g), R12
	MOVV	m_g0(R12), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R3
	MOVV	savedsp-24(SP), R13 // must match frame size
	MOVV	R13, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m,
	// 1. for the duration of the call on non-pthread platforms,
	// 2. or the duration of the C thread alive on pthread platforms.
	// If the m on entry wasn't nil,
	// 1. the thread might be a Go thread,
	// 2. or it wasn't the first call from a C thread on pthread platforms,
	//    since then we skip dropm to resue the m in the first call.
	MOVV	savedm-8(SP), R12
	BNE	R12, droppedm

	// Skip dropm to reuse it in the next call, when a pthread key has been created.
	MOVV	_cgo_pthread_key_created(SB), R12
	// It means cgo is disabled when _cgo_pthread_key_created is a nil pointer, need dropm.
	BEQ	R12, dropm
	MOVV    (R12), R12
	BNE	R12, droppedm

dropm:
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

// void setg_gcc(G*); set g called from gcc with g in R19
TEXT setg_gcc<>(SB),NOSPLIT,$0-0
	MOVV	R19, g
	JAL	runtime·save_g(SB)
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R0
	UNDEF

// AES hashing not implemented for loong64
TEXT runtime·memhash<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-32
	JMP	runtime·memhashFallback<ABIInternal>(SB)
TEXT runtime·strhash<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·strhashFallback<ABIInternal>(SB)
TEXT runtime·memhash32<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash32Fallback<ABIInternal>(SB)
TEXT runtime·memhash64<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash64Fallback<ABIInternal>(SB)

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVW	$0, R19
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$16
	// g (R22) and REGTMP (R30)  might be clobbered by load_g. They
	// are callee-save in the gcc calling convention, so save them.
	MOVV	R30, savedREGTMP-16(SP)
	MOVV	g, savedG-8(SP)

	JAL	runtime·load_g(SB)
	MOVV	g_m(g), R19
	MOVV	m_curg(R19), R19
	MOVV	(g_stack+stack_hi)(R19), R4 // return value in R4

	MOVV	savedG-8(SP), g
	MOVV	savedREGTMP-16(SP), R30
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT|NOFRAME|TOPFRAME,$0-0
	NOOP
	JAL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	NOOP

// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-0
	ADDV	$-0x10, R3
	MOVV	R30, 8(R3) // The access to global variables below implicitly uses R30, which is callee-save
	MOVV	runtime·lastmoduledatap(SB), R12
	MOVV	R4, moduledata_next(R12)
	MOVV	R4, runtime·lastmoduledatap(SB)
	MOVV	8(R3), R30
	ADDV	$0x10, R3
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R19
	MOVB	R19, ret+0(FP)
	RET

#ifdef GOEXPERIMENT_regabiargs
// spillArgs stores return values from registers to a *internal/abi.RegArgs in R25.
TEXT ·spillArgs(SB),NOSPLIT,$0-0
	MOVV	R4, (0*8)(R25)
	MOVV	R5, (1*8)(R25)
	MOVV	R6, (2*8)(R25)
	MOVV	R7, (3*8)(R25)
	MOVV	R8, (4*8)(R25)
	MOVV	R9, (5*8)(R25)
	MOVV	R10, (6*8)(R25)
	MOVV	R11, (7*8)(R25)
	MOVV	R12, (8*8)(R25)
	MOVV	R13, (9*8)(R25)
	MOVV	R14, (10*8)(R25)
	MOVV	R15, (11*8)(R25)
	MOVV	R16, (12*8)(R25)
	MOVV	R17, (13*8)(R25)
	MOVV	R18, (14*8)(R25)
	MOVV	R19, (15*8)(R25)
	MOVD	F0, (16*8)(R25)
	MOVD	F1, (17*8)(R25)
	MOVD	F2, (18*8)(R25)
	MOVD	F3, (19*8)(R25)
	MOVD	F4, (20*8)(R25)
	MOVD	F5, (21*8)(R25)
	MOVD	F6, (22*8)(R25)
	MOVD	F7, (23*8)(R25)
	MOVD	F8, (24*8)(R25)
	MOVD	F9, (25*8)(R25)
	MOVD	F10, (26*8)(R25)
	MOVD	F11, (27*8)(R25)
	MOVD	F12, (28*8)(R25)
	MOVD	F13, (29*8)(R25)
	MOVD	F14, (30*8)(R25)
	MOVD	F15, (31*8)(R25)
	RET

// unspillArgs loads args into registers from a *internal/abi.RegArgs in R25.
TEXT ·unspillArgs(SB),NOSPLIT,$0-0
	MOVV	(0*8)(R25), R4
	MOVV	(1*8)(R25), R5
	MOVV	(2*8)(R25), R6
	MOVV	(3*8)(R25), R7
	MOVV	(4*8)(R25), R8
	MOVV	(5*8)(R25), R9
	MOVV	(6*8)(R25), R10
	MOVV	(7*8)(R25), R11
	MOVV	(8*8)(R25), R12
	MOVV	(9*8)(R25), R13
	MOVV	(10*8)(R25), R14
	MOVV	(11*8)(R25), R15
	MOVV	(12*8)(R25), R16
	MOVV	(13*8)(R25), R17
	MOVV	(14*8)(R25), R18
	MOVV	(15*8)(R25), R19
	MOVD	(16*8)(R25), F0
	MOVD	(17*8)(R25), F1
	MOVD	(18*8)(R25), F2
	MOVD	(19*8)(R25), F3
	MOVD	(20*8)(R25), F4
	MOVD	(21*8)(R25), F5
	MOVD	(22*8)(R25), F6
	MOVD	(23*8)(R25), F7
	MOVD	(24*8)(R25), F8
	MOVD	(25*8)(R25), F9
	MOVD	(26*8)(R25), F10
	MOVD	(27*8)(R25), F11
	MOVD	(28*8)(R25), F12
	MOVD	(29*8)(R25), F13
	MOVD	(30*8)(R25), F14
	MOVD	(31*8)(R25), F15
	RET
#else
TEXT ·spillArgs(SB),NOSPLIT,$0-0
	RET

TEXT ·unspillArgs(SB),NOSPLIT,$0-0
	RET
#endif

// gcWriteBarrier informs the GC about heap pointer writes.
//
// gcWriteBarrier does NOT follow the Go ABI. It accepts the
// number of bytes of buffer needed in R29, and returns a pointer
// to the buffer space in R29.
// It clobbers R30 (the linker temp register).
// The act of CALLing gcWriteBarrier will clobber R1 (LR).
// It does not clobber any other general-purpose registers,
// but may clobber others (e.g., floating point registers).
TEXT gcWriteBarrier<>(SB),NOSPLIT,$216
	// Save the registers clobbered by the fast path.
	MOVV	R19, 208(R3)
	MOVV	R13, 216(R3)
retry:
	MOVV	g_m(g), R19
	MOVV	m_p(R19), R19
	MOVV	(p_wbBuf+wbBuf_next)(R19), R13
	MOVV	(p_wbBuf+wbBuf_end)(R19), R30 // R30 is linker temp register
	// Increment wbBuf.next position.
	ADDV	R29, R13
	// Is the buffer full?
	BLTU	R30, R13, flush
	// Commit to the larger buffer.
	MOVV	R13, (p_wbBuf+wbBuf_next)(R19)
	// Make return value (the original next position)
	SUBV	R29, R13, R29
	// Restore registers.
	MOVV	208(R3), R19
	MOVV	216(R3), R13
	RET

flush:
	// Save all general purpose registers since these could be
	// clobbered by wbBufFlush and were not saved by the caller.
	MOVV	R27, 8(R3)
	MOVV	R28, 16(R3)
	// R1 is LR, which was saved by the prologue.
	MOVV	R2, 24(R3)
	// R3 is SP.
	MOVV	R4, 32(R3)
	MOVV	R5, 40(R3)
	MOVV	R6, 48(R3)
	MOVV	R7, 56(R3)
	MOVV	R8, 64(R3)
	MOVV	R9, 72(R3)
	MOVV	R10, 80(R3)
	MOVV	R11, 88(R3)
	MOVV	R12, 96(R3)
	// R13 already saved
	MOVV	R14, 104(R3)
	MOVV	R15, 112(R3)
	MOVV	R16, 120(R3)
	MOVV	R17, 128(R3)
	MOVV	R18, 136(R3)
	// R19 already saved
	MOVV	R20, 144(R3)
	MOVV	R21, 152(R3)
	// R22 is g.
	MOVV	R23, 160(R3)
	MOVV	R24, 168(R3)
	MOVV	R25, 176(R3)
	MOVV	R26, 184(R3)
	// R27 already saved
	// R28 already saved.
	MOVV	R29, 192(R3)
	// R30 is tmp register.
	MOVV	R31, 200(R3)

	CALL	runtime·wbBufFlush(SB)

	MOVV	8(R3), R27
	MOVV	16(R3), R28
	MOVV	24(R3), R2
	MOVV	32(R3), R4
	MOVV	40(R3), R5
	MOVV	48(R3), R6
	MOVV	56(R3), R7
	MOVV	64(R3), R8
	MOVV	72(R3), R9
	MOVV	80(R3), R10
	MOVV	88(R3), R11
	MOVV	96(R3), R12
	MOVV	104(R3), R14
	MOVV	112(R3), R15
	MOVV	120(R3), R16
	MOVV	128(R3), R17
	MOVV	136(R3), R18
	MOVV	144(R3), R20
	MOVV	152(R3), R21
	MOVV	160(R3), R23
	MOVV	168(R3), R24
	MOVV	176(R3), R25
	MOVV	184(R3), R26
	MOVV	192(R3), R29
	MOVV	200(R3), R31
	JMP	retry

TEXT runtime·gcWriteBarrier1<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$8, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier2<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$16, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier3<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$24, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier4<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$32, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier5<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$40, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier6<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$48, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier7<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$56, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier8<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$64, R29
	JMP	gcWriteBarrier<>(SB)

// Note: these functions use a special calling convention to save generated code space.
// Arguments are passed in registers, but the space for those arguments are allocated
// in the caller's stack frame. These stubs write the args into that stack space and
// then tail call to the corresponding runtime handler.
// The tail call makes these stubs disappear in backtraces.
TEXT runtime·panicIndex<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R20, R4
	MOVV	R21, R5
#else
	MOVV	R20, x+0(FP)
	MOVV	R21, y+8(FP)
#endif
	JMP	runtime·goPanicIndex<ABIInternal>(SB)
TEXT runtime·panicIndexU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R20, R4
	MOVV	R21, R5
#else
	MOVV	R20, x+0(FP)
	MOVV	R21, y+8(FP)
#endif
	JMP	runtime·goPanicIndexU<ABIInternal>(SB)
TEXT runtime·panicSliceAlen<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R21, R4
	MOVV	R23, R5
#else
	MOVV	R21, x+0(FP)
	MOVV	R23, y+8(FP)
#endif
	JMP	runtime·goPanicSliceAlen<ABIInternal>(SB)
TEXT runtime·panicSliceAlenU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R21, R4
	MOVV	R23, R5
#else
	MOVV	R21, x+0(FP)
	MOVV	R23, y+8(FP)
#endif
	JMP	runtime·goPanicSliceAlenU<ABIInternal>(SB)
TEXT runtime·panicSliceAcap<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R21, R4
	MOVV	R23, R5
#else
	MOVV	R21, x+0(FP)
	MOVV	R23, y+8(FP)
#endif
	JMP	runtime·goPanicSliceAcap<ABIInternal>(SB)
TEXT runtime·panicSliceAcapU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R21, R4
	MOVV	R23, R5
#else
	MOVV	R21, x+0(FP)
	MOVV	R23, y+8(FP)
#endif
	JMP	runtime·goPanicSliceAcapU<ABIInternal>(SB)
TEXT runtime·panicSliceB<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R20, R4
	MOVV	R21, R5
#else
	MOVV	R20, x+0(FP)
	MOVV	R21, y+8(FP)
#endif
	JMP	runtime·goPanicSliceB<ABIInternal>(SB)
TEXT runtime·panicSliceBU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R20, R4
	MOVV	R21, R5
#else
	MOVV	R20, x+0(FP)
	MOVV	R21, y+8(FP)
#endif
	JMP	runtime·goPanicSliceBU<ABIInternal>(SB)
TEXT runtime·panicSlice3Alen<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R23, R4
	MOVV	R24, R5
#else
	MOVV	R23, x+0(FP)
	MOVV	R24, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3Alen<ABIInternal>(SB)
TEXT runtime·panicSlice3AlenU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R23, R4
	MOVV	R24, R5
#else
	MOVV	R23, x+0(FP)
	MOVV	R24, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3AlenU<ABIInternal>(SB)
TEXT runtime·panicSlice3Acap<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R23, R4
	MOVV	R24, R5
#else
	MOVV	R23, x+0(FP)
	MOVV	R24, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3Acap<ABIInternal>(SB)
TEXT runtime·panicSlice3AcapU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R23, R4
	MOVV	R24, R5
#else
	MOVV	R23, x+0(FP)
	MOVV	R24, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3AcapU<ABIInternal>(SB)
TEXT runtime·panicSlice3B<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R21, R4
	MOVV	R23, R5
#else
	MOVV	R21, x+0(FP)
	MOVV	R23, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3B<ABIInternal>(SB)
TEXT runtime·panicSlice3BU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R21, R4
	MOVV	R23, R5
#else
	MOVV	R21, x+0(FP)
	MOVV	R23, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3BU<ABIInternal>(SB)
TEXT runtime·panicSlice3C<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R20, R4
	MOVV	R21, R5
#else
	MOVV	R20, x+0(FP)
	MOVV	R21, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3C<ABIInternal>(SB)
TEXT runtime·panicSlice3CU<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R20, R4
	MOVV	R21, R5
#else
	MOVV	R20, x+0(FP)
	MOVV	R21, y+8(FP)
#endif
	JMP	runtime·goPanicSlice3CU<ABIInternal>(SB)
TEXT runtime·panicSliceConvert<ABIInternal>(SB),NOSPLIT,$0-16
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R23, R4
	MOVV	R24, R5
#else
	MOVV	R23, x+0(FP)
	MOVV	R24, y+8(FP)
#endif
	JMP	runtime·goPanicSliceConvert<ABIInternal>(SB)
