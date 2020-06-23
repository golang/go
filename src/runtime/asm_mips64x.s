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
	// R29 = stack; R4 = argc; R5 = argv

	ADDV	$-24, R29
	MOVW	R4, 8(R29) // argc
	MOVV	R5, 16(R29) // argv

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVV	$runtime·g0(SB), g
	MOVV	$(-64*1024), R23
	ADDV	R23, R29, R1
	MOVV	R1, g_stackguard0(g)
	MOVV	R1, g_stackguard1(g)
	MOVV	R1, (g_stack+stack_lo)(g)
	MOVV	R29, (g_stack+stack_hi)(g)

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

TEXT runtime·breakpoint(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	R0, 2(R0) // TODO: TD
	RET

TEXT runtime·asminit(SB),NOSPLIT|NOFRAME,$0-0
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT|NOFRAME, $0-8
	MOVV	buf+0(FP), R1
	MOVV	R29, gobuf_sp(R1)
	MOVV	R31, gobuf_pc(R1)
	MOVV	g, gobuf_g(R1)
	MOVV	R0, gobuf_lr(R1)
	MOVV	R0, gobuf_ret(R1)
	// Assert ctxt is zero. See func save.
	MOVV	gobuf_ctxt(R1), R1
	BEQ	R1, 2(PC)
	JAL	runtime·badctxt(SB)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $16-8
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
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT|NOFRAME, $0-8
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
// of the G stack. We need to distinguish the routine that
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
	JAL	runtime·abort(SB)

switch:
	// save our state in g->sched. Pretend to
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
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVV	0(REGCTXT), R4	// code pointer
	MOVV	0(R29), R31	// restore LR
	ADDV	$8, R29
	JMP	(R4)

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
TEXT runtime·morestack(SB),NOSPLIT|NOFRAME,$0-0
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
	// Set g->sched to context in f.
	MOVV	R29, (g_sched+gobuf_sp)(g)
	MOVV	R31, (g_sched+gobuf_pc)(g)
	MOVV	R3, (g_sched+gobuf_lr)(g)
	MOVV	REGCTXT, (g_sched+gobuf_ctxt)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVV	R3, (m_morebuf+gobuf_pc)(R7)	// f's caller's PC
	MOVV	R29, (m_morebuf+gobuf_sp)(R7)	// f's caller's SP
	MOVV	g, (m_morebuf+gobuf_g)(R7)

	// Call newstack on m->g0's stack.
	MOVV	m_g0(R7), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R29
	// Create a stack frame on g0 to call newstack.
	MOVV	R0, -8(R29)	// Zero saved LR in frame
	ADDV	$-8, R29
	JAL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT|NOFRAME,$0-0
	MOVV	R0, REGCTXT
	JMP	runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVV	$MAXSIZE, R23;		\
	SGTU	R1, R23, R23;		\
	BNE	R23, 3(PC);			\
	MOVV	$NAME(SB), R4;	\
	JMP	(R4)
// Note: can't just "BR NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT|NOFRAME, $0-32
	MOVWU argsize+24(FP), R1
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
	MOVV	argtype+0(FP), R5;		\
	MOVV	arg+16(FP), R1;			\
	MOVWU	n+24(FP), R2;			\
	MOVWU	retoffset+28(FP), R4;		\
	ADDV	$8, R29, R3;				\
	ADDV	R4, R3; 			\
	ADDV	R4, R1;				\
	SUBVU	R4, R2;				\
	JAL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $32-0
	MOVV	R5, 8(R29)
	MOVV	R1, 16(R29)
	MOVV	R3, 24(R29)
	MOVV	R2, 32(R29)
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

// void jmpdefer(fv, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 8 bytes to get back to JAL deferreturn
// 3. JMP to fn
TEXT runtime·jmpdefer(SB), NOSPLIT|NOFRAME, $0-16
	MOVV	0(R29), R31
	ADDV	$-8, R31

	MOVV	fv+0(FP), REGCTXT
	MOVV	argp+8(FP), R29
	ADDV	$-8, R29
	NOR	R0, R0	// prevent scheduling
	MOVV	0(REGCTXT), R4
	JMP	(R4)

// Save state of caller into g->sched. Smashes R1.
TEXT gosave<>(SB),NOSPLIT|NOFRAME,$0
	MOVV	R31, (g_sched+gobuf_pc)(g)
	MOVV	R29, (g_sched+gobuf_sp)(g)
	MOVV	R0, (g_sched+gobuf_lr)(g)
	MOVV	R0, (g_sched+gobuf_ret)(g)
	// Assert ctxt is zero. See func save.
	MOVV	(g_sched+gobuf_ctxt)(g), R1
	BEQ	R1, 2(PC)
	JAL	runtime·badctxt(SB)
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	MOVV	fn+0(FP), R25
	MOVV	arg+8(FP), R4

	MOVV	R29, R3	// save original stack pointer
	MOVV	g, R2

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	MOVV	g_m(g), R5
	MOVV	m_g0(R5), R6
	BEQ	R6, g, g0

	JAL	gosave<>(SB)
	MOVV	R6, g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R29

	// Now on a scheduling stack (a pthread-created stack).
g0:
	// Save room for two of our pointers.
	ADDV	$-16, R29
	MOVV	R2, 0(R29)	// save old g on stack
	MOVV	(g_stack+stack_hi)(R2), R2
	SUBVU	R3, R2
	MOVV	R2, 8(R29)	// save depth in old g stack (can't just save SP, as stack might be copied during a callback)
	JAL	(R25)

	// Restore g, stack pointer. R2 is return value.
	MOVV	0(R29), g
	JAL	runtime·save_g(SB)
	MOVV	(g_stack+stack_hi)(g), R5
	MOVV	8(R29), R6
	SUBVU	R6, R5
	MOVV	R5, R29

	MOVW	R2, ret+16(FP)
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize, uintptr ctxt)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$32-32
	MOVV	$fn+0(FP), R1
	MOVV	R1, 8(R29)
	MOVV	frame+8(FP), R1
	MOVV	R1, 16(R29)
	MOVV	framesize+16(FP), R1
	MOVV	R1, 24(R29)
	MOVV	ctxt+24(FP), R1
	MOVV	R1, 32(R29)
	MOVV	$runtime·cgocallback_gofunc(SB), R1
	JAL	(R1)
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize, uintptr ctxt)
// See cgocall.go for more details.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$16-32
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
	MOVV	g_m(g), R3
	MOVV	m_g0(R3), R1
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
	// In the new goroutine, -8(SP) is unused (where SP refers to
	// m->curg's SP while we're setting it up, before we've adjusted it).
	MOVV	m_curg(R3), g
	JAL	runtime·save_g(SB)
	MOVV	(g_sched+gobuf_sp)(g), R2 // prepare stack as R2
	MOVV	(g_sched+gobuf_pc)(g), R4
	MOVV	R4, -24(R2)
	MOVV    ctxt+24(FP), R1
	MOVV    R1, -16(R2)
	MOVV	$-24(R2), R29
	JAL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVV	0(R29), R4
	MOVV	R4, (g_sched+gobuf_pc)(g)
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

// void setg_gcc(G*); set g called from gcc with g in R1
TEXT setg_gcc<>(SB),NOSPLIT,$0-0
	MOVV	R1, g
	JAL	runtime·save_g(SB)
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R0
	UNDEF

// AES hashing not implemented for mips64
TEXT runtime·memhash(SB),NOSPLIT|NOFRAME,$0-32
	JMP	runtime·memhashFallback(SB)
TEXT runtime·strhash(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·strhashFallback(SB)
TEXT runtime·memhash32(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash32Fallback(SB)
TEXT runtime·memhash64(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash64Fallback(SB)

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVW	$0, R1
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$16
	// g (R30) and REGTMP (R23)  might be clobbered by load_g. They
	// are callee-save in the gcc calling convention, so save them.
	MOVV	R23, savedR23-16(SP)
	MOVV	g, savedG-8(SP)

	JAL	runtime·load_g(SB)
	MOVV	g_m(g), R1
	MOVV	m_curg(R1), R1
	MOVV	(g_stack+stack_hi)(R1), R2 // return value in R2

	MOVV	savedG-8(SP), g
	MOVV	savedR23-16(SP), R23
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT|NOFRAME|TOPFRAME,$0-0
	NOR	R0, R0	// NOP
	JAL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	NOR	R0, R0	// NOP

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R1
	MOVB	R1, ret+0(FP)
	RET

// gcWriteBarrier performs a heap pointer write and informs the GC.
//
// gcWriteBarrier does NOT follow the Go ABI. It takes two arguments:
// - R20 is the destination of the write
// - R21 is the value being written at R20.
// It clobbers R23 (the linker temp register).
// The act of CALLing gcWriteBarrier will clobber R31 (LR).
// It does not clobber any other general-purpose registers,
// but may clobber others (e.g., floating point registers).
TEXT runtime·gcWriteBarrier(SB),NOSPLIT,$192
	// Save the registers clobbered by the fast path.
	MOVV	R1, 184(R29)
	MOVV	R2, 192(R29)
	MOVV	g_m(g), R1
	MOVV	m_p(R1), R1
	MOVV	(p_wbBuf+wbBuf_next)(R1), R2
	// Increment wbBuf.next position.
	ADDV	$16, R2
	MOVV	R2, (p_wbBuf+wbBuf_next)(R1)
	MOVV	(p_wbBuf+wbBuf_end)(R1), R1
	MOVV	R1, R23		// R23 is linker temp register
	// Record the write.
	MOVV	R21, -16(R2)	// Record value
	MOVV	(R20), R1	// TODO: This turns bad writes into bad reads.
	MOVV	R1, -8(R2)	// Record *slot
	// Is the buffer full?
	BEQ	R2, R23, flush
ret:
	MOVV	184(R29), R1
	MOVV	192(R29), R2
	// Do the write.
	MOVV	R21, (R20)
	RET

flush:
	// Save all general purpose registers since these could be
	// clobbered by wbBufFlush and were not saved by the caller.
	MOVV	R20, 8(R29)	// Also first argument to wbBufFlush
	MOVV	R21, 16(R29)	// Also second argument to wbBufFlush
	// R1 already saved
	// R2 already saved
	MOVV	R3, 24(R29)
	MOVV	R4, 32(R29)
	MOVV	R5, 40(R29)
	MOVV	R6, 48(R29)
	MOVV	R7, 56(R29)
	MOVV	R8, 64(R29)
	MOVV	R9, 72(R29)
	MOVV	R10, 80(R29)
	MOVV	R11, 88(R29)
	MOVV	R12, 96(R29)
	MOVV	R13, 104(R29)
	MOVV	R14, 112(R29)
	MOVV	R15, 120(R29)
	MOVV	R16, 128(R29)
	MOVV	R17, 136(R29)
	MOVV	R18, 144(R29)
	MOVV	R19, 152(R29)
	// R20 already saved
	// R21 already saved.
	MOVV	R22, 160(R29)
	// R23 is tmp register.
	MOVV	R24, 168(R29)
	MOVV	R25, 176(R29)
	// R26 is reserved by kernel.
	// R27 is reserved by kernel.
	// R28 is REGSB (not modified by Go code).
	// R29 is SP.
	// R30 is g.
	// R31 is LR, which was saved by the prologue.

	// This takes arguments R20 and R21.
	CALL	runtime·wbBufFlush(SB)

	MOVV	8(R29), R20
	MOVV	16(R29), R21
	MOVV	24(R29), R3
	MOVV	32(R29), R4
	MOVV	40(R29), R5
	MOVV	48(R29), R6
	MOVV	56(R29), R7
	MOVV	64(R29), R8
	MOVV	72(R29), R9
	MOVV	80(R29), R10
	MOVV	88(R29), R11
	MOVV	96(R29), R12
	MOVV	104(R29), R13
	MOVV	112(R29), R14
	MOVV	120(R29), R15
	MOVV	128(R29), R16
	MOVV	136(R29), R17
	MOVV	144(R29), R18
	MOVV	152(R29), R19
	MOVV	160(R29), R22
	MOVV	168(R29), R24
	MOVV	176(R29), R25
	JMP	ret

// Note: these functions use a special calling convention to save generated code space.
// Arguments are passed in registers, but the space for those arguments are allocated
// in the caller's stack frame. These stubs write the args into that stack space and
// then tail call to the corresponding runtime handler.
// The tail call makes these stubs disappear in backtraces.
TEXT runtime·panicIndex(SB),NOSPLIT,$0-16
	MOVV	R1, x+0(FP)
	MOVV	R2, y+8(FP)
	JMP	runtime·goPanicIndex(SB)
TEXT runtime·panicIndexU(SB),NOSPLIT,$0-16
	MOVV	R1, x+0(FP)
	MOVV	R2, y+8(FP)
	JMP	runtime·goPanicIndexU(SB)
TEXT runtime·panicSliceAlen(SB),NOSPLIT,$0-16
	MOVV	R2, x+0(FP)
	MOVV	R3, y+8(FP)
	JMP	runtime·goPanicSliceAlen(SB)
TEXT runtime·panicSliceAlenU(SB),NOSPLIT,$0-16
	MOVV	R2, x+0(FP)
	MOVV	R3, y+8(FP)
	JMP	runtime·goPanicSliceAlenU(SB)
TEXT runtime·panicSliceAcap(SB),NOSPLIT,$0-16
	MOVV	R2, x+0(FP)
	MOVV	R3, y+8(FP)
	JMP	runtime·goPanicSliceAcap(SB)
TEXT runtime·panicSliceAcapU(SB),NOSPLIT,$0-16
	MOVV	R2, x+0(FP)
	MOVV	R3, y+8(FP)
	JMP	runtime·goPanicSliceAcapU(SB)
TEXT runtime·panicSliceB(SB),NOSPLIT,$0-16
	MOVV	R1, x+0(FP)
	MOVV	R2, y+8(FP)
	JMP	runtime·goPanicSliceB(SB)
TEXT runtime·panicSliceBU(SB),NOSPLIT,$0-16
	MOVV	R1, x+0(FP)
	MOVV	R2, y+8(FP)
	JMP	runtime·goPanicSliceBU(SB)
TEXT runtime·panicSlice3Alen(SB),NOSPLIT,$0-16
	MOVV	R3, x+0(FP)
	MOVV	R4, y+8(FP)
	JMP	runtime·goPanicSlice3Alen(SB)
TEXT runtime·panicSlice3AlenU(SB),NOSPLIT,$0-16
	MOVV	R3, x+0(FP)
	MOVV	R4, y+8(FP)
	JMP	runtime·goPanicSlice3AlenU(SB)
TEXT runtime·panicSlice3Acap(SB),NOSPLIT,$0-16
	MOVV	R3, x+0(FP)
	MOVV	R4, y+8(FP)
	JMP	runtime·goPanicSlice3Acap(SB)
TEXT runtime·panicSlice3AcapU(SB),NOSPLIT,$0-16
	MOVV	R3, x+0(FP)
	MOVV	R4, y+8(FP)
	JMP	runtime·goPanicSlice3AcapU(SB)
TEXT runtime·panicSlice3B(SB),NOSPLIT,$0-16
	MOVV	R2, x+0(FP)
	MOVV	R3, y+8(FP)
	JMP	runtime·goPanicSlice3B(SB)
TEXT runtime·panicSlice3BU(SB),NOSPLIT,$0-16
	MOVV	R2, x+0(FP)
	MOVV	R3, y+8(FP)
	JMP	runtime·goPanicSlice3BU(SB)
TEXT runtime·panicSlice3C(SB),NOSPLIT,$0-16
	MOVV	R1, x+0(FP)
	MOVV	R2, y+8(FP)
	JMP	runtime·goPanicSlice3C(SB)
TEXT runtime·panicSlice3CU(SB),NOSPLIT,$0-16
	MOVV	R1, x+0(FP)
	MOVV	R2, y+8(FP)
	JMP	runtime·goPanicSlice3CU(SB)
