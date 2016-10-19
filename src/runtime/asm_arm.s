// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// using frame size $-4 means do not save LR on stack.
TEXT runtime·rt0_go(SB),NOSPLIT,$-4
	MOVW	$0xcafebabe, R12

	// copy arguments forward on an even stack
	// use R13 instead of SP to avoid linker rewriting the offsets
	MOVW	0(R13), R0		// argc
	MOVW	4(R13), R1		// argv
	SUB	$64, R13		// plenty of scratch
	AND	$~7, R13
	MOVW	R0, 60(R13)		// save argc, argv away
	MOVW	R1, 64(R13)

	// set up g register
	// g is R10
	MOVW	$runtime·g0(SB), g
	MOVW	$runtime·m0(SB), R8

	// save m->g0 = g0
	MOVW	g, m_g0(R8)
	// save g->m = m0
	MOVW	R8, g_m(g)

	// create istack out of the OS stack
	// (1MB of system stack is available on iOS and Android)
	MOVW	$(-64*1024+104)(R13), R0
	MOVW	R0, g_stackguard0(g)
	MOVW	R0, g_stackguard1(g)
	MOVW	R0, (g_stack+stack_lo)(g)
	MOVW	R13, (g_stack+stack_hi)(g)

	BL	runtime·emptyfunc(SB)	// fault if stack check is wrong

	BL	runtime·_initcgo(SB)	// will clobber R0-R3

	// update stackguard after _cgo_init
	MOVW	(g_stack+stack_lo)(g), R0
	ADD	$const__StackGuard, R0
	MOVW	R0, g_stackguard0(g)
	MOVW	R0, g_stackguard1(g)

	BL	runtime·check(SB)

	// saved argc, argv
	MOVW	60(R13), R0
	MOVW	R0, 4(R13)
	MOVW	64(R13), R1
	MOVW	R1, 8(R13)
	BL	runtime·args(SB)
	BL	runtime·checkgoarm(SB)
	BL	runtime·osinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVW	$runtime·mainPC(SB), R0
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

DATA	runtime·mainPC+0(SB)/4,$runtime·main(SB)
GLOBL	runtime·mainPC(SB),RODATA,$4

TEXT runtime·breakpoint(SB),NOSPLIT,$0-0
	// gdb won't skip this breakpoint instruction automatically,
	// so you must manually "set $pc+=4" to skip it and continue.
#ifdef GOOS_nacl
	WORD	$0xe125be7f	// BKPT 0x5bef, NACL_INSTR_ARM_BREAKPOINT
#else
#ifdef GOOS_plan9
	WORD	$0xD1200070	// undefined instruction used as armv5 breakpoint in Plan 9
#else
	WORD	$0xe7f001f0	// undefined instruction that gdb understands is a software breakpoint
#endif
#endif
	RET

TEXT runtime·asminit(SB),NOSPLIT,$0-0
	// disable runfast (flush-to-zero) mode of vfp if runtime.goarm > 5
	MOVB	runtime·goarm(SB), R11
	CMP	$5, R11
	BLE	4(PC)
	WORD	$0xeef1ba10	// vmrs r11, fpscr
	BIC	$(1<<24), R11
	WORD	$0xeee1ba10	// vmsr fpscr, r11
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB),NOSPLIT,$-4-4
	MOVW	buf+0(FP), R0
	MOVW	R13, gobuf_sp(R0)
	MOVW	LR, gobuf_pc(R0)
	MOVW	g, gobuf_g(R0)
	MOVW	$0, R11
	MOVW	R11, gobuf_lr(R0)
	MOVW	R11, gobuf_ret(R0)
	// Assert ctxt is zero. See func save.
	MOVW	gobuf_ctxt(R0), R0
	CMP	R0, R11
	B.EQ	2(PC)
	CALL	runtime·badctxt(SB)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB),NOSPLIT,$8-4
	MOVW	buf+0(FP), R1

	// If ctxt is not nil, invoke deletion barrier before overwriting.
	MOVW	gobuf_ctxt(R1), R0
	CMP	$0, R0
	B.EQ	nilctxt
	MOVW	$gobuf_ctxt(R1), R0
	MOVW	R0, 4(R13)
	MOVW	$0, R0
	MOVW	R0, 8(R13)
	BL	runtime·writebarrierptr_prewrite(SB)
	MOVW	buf+0(FP), R1

nilctxt:
	MOVW	gobuf_g(R1), R0
	BL	setg<>(SB)

	// NOTE: We updated g above, and we are about to update SP.
	// Until LR and PC are also updated, the g/SP/LR/PC quadruple
	// are out of sync and must not be used as the basis of a traceback.
	// Sigprof skips the traceback when SP is not within g's bounds,
	// and when the PC is inside this function, runtime.gogo.
	// Since we are about to update SP, until we complete runtime.gogo
	// we must not leave this function. In particular, no calls
	// after this point: it must be straight-line code until the
	// final B instruction.
	// See large comment in sigprof for more details.
	MOVW	gobuf_sp(R1), R13	// restore SP==R13
	MOVW	gobuf_lr(R1), LR
	MOVW	gobuf_ret(R1), R0
	MOVW	gobuf_ctxt(R1), R7
	MOVW	$0, R11
	MOVW	R11, gobuf_sp(R1)	// clear to help garbage collector
	MOVW	R11, gobuf_ret(R1)
	MOVW	R11, gobuf_lr(R1)
	MOVW	R11, gobuf_ctxt(R1)
	MOVW	gobuf_pc(R1), R11
	CMP	R11, R11 // set condition codes for == test, needed by stack split
	B	(R11)

// func mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB),NOSPLIT,$-4-4
	// Save caller state in g->sched.
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	$0, R11
	MOVW	R11, (g_sched+gobuf_lr)(g)
	MOVW	g, (g_sched+gobuf_g)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVW	g, R1
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R0
	BL	setg<>(SB)
	CMP	g, R1
	B.NE	2(PC)
	B	runtime·badmcall(SB)
	MOVB	runtime·iscgo(SB), R11
	CMP	$0, R11
	BL.NE	runtime·save_g(SB)
	MOVW	fn+0(FP), R0
	MOVW	(g_sched+gobuf_sp)(g), R13
	SUB	$8, R13
	MOVW	R1, 4(R13)
	MOVW	R0, R7
	MOVW	0(R0), R0
	BL	(R0)
	B	runtime·badmcall2(SB)
	RET

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack. We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB),NOSPLIT,$0-0
	MOVW	$0, R0
	BL	(R0) // clobber lr to ensure push {lr} is kept
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB),NOSPLIT,$0-4
	MOVW	fn+0(FP), R0	// R0 = fn
	MOVW	g_m(g), R1	// R1 = m

	MOVW	m_gsignal(R1), R2	// R2 = gsignal
	CMP	g, R2
	B.EQ	noswitch

	MOVW	m_g0(R1), R2	// R2 = g0
	CMP	g, R2
	B.EQ	noswitch

	MOVW	m_curg(R1), R3
	CMP	g, R3
	B.EQ	switch

	// Bad: g is not gsignal, not g0, not curg. What is it?
	// Hide call from linker nosplit analysis.
	MOVW	$runtime·badsystemstack(SB), R0
	BL	(R0)

switch:
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVW	$runtime·systemstack_switch(SB), R3
#ifdef GOOS_nacl
	ADD	$4, R3, R3 // get past nacl-insert bic instruction
#endif
	ADD	$4, R3, R3 // get past push {lr}
	MOVW	R3, (g_sched+gobuf_pc)(g)
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_lr)(g)
	MOVW	g, (g_sched+gobuf_g)(g)

	// switch to g0
	MOVW	R0, R5
	MOVW	R2, R0
	BL	setg<>(SB)
	MOVW	R5, R0
	MOVW	(g_sched+gobuf_sp)(R2), R3
	// make it look like mstart called systemstack on g0, to stop traceback
	SUB	$4, R3, R3
	MOVW	$runtime·mstart(SB), R4
	MOVW	R4, 0(R3)
	MOVW	R3, R13

	// call target function
	MOVW	R0, R7
	MOVW	0(R0), R0
	BL	(R0)

	// switch back to g
	MOVW	g_m(g), R1
	MOVW	m_curg(R1), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R13
	MOVW	$0, R3
	MOVW	R3, (g_sched+gobuf_sp)(g)
	RET

noswitch:
	MOVW	R0, R7
	MOVW	0(R0), R0
	BL	(R0)
	RET

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// R1 frame size
// R3 prolog's LR
// NB. we do not save R0 because we've forced 5c to pass all arguments
// on the stack.
// using frame size $-4 means do not save LR on stack.
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT,$-4-0
	// Cannot grow scheduler stack (m->g0).
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R4
	CMP	g, R4
	BNE	3(PC)
	BL	runtime·badmorestackg0(SB)
	B	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVW	m_gsignal(R8), R4
	CMP	g, R4
	BNE	3(PC)
	BL	runtime·badmorestackgsignal(SB)
	B	runtime·abort(SB)

	// Called from f.
	// Set g->sched to context in f.
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	R3, (g_sched+gobuf_lr)(g)
	// newstack will fill gobuf.ctxt.

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVW	R3, (m_morebuf+gobuf_pc)(R8)	// f's caller's PC
	MOVW	R13, (m_morebuf+gobuf_sp)(R8)	// f's caller's SP
	MOVW	$4(R13), R3			// f's argument pointer
	MOVW	g, (m_morebuf+gobuf_g)(R8)

	// Call newstack on m->g0's stack.
	MOVW	m_g0(R8), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R13
	MOVW	$0, R0
	MOVW.W	R0, -8(R13)	// create a call frame on g0
	MOVW	R7, 4(R13)	// ctxt argument
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	RET

TEXT runtime·morestack_noctxt(SB),NOSPLIT,$-4-0
	MOVW	$0, R7
	B runtime·morestack(SB)

TEXT runtime·stackBarrier(SB),NOSPLIT,$0
	// We came here via a RET to an overwritten LR.
	// R0 may be live. Other registers are available.

	// Get the original return PC, g.stkbar[g.stkbarPos].savedLRVal.
	MOVW	(g_stkbar+slice_array)(g), R4
	MOVW	g_stkbarPos(g), R5
	MOVW	$stkbar__size, R6
	MUL	R5, R6
	ADD	R4, R6
	MOVW	stkbar_savedLRVal(R6), R6
	// Record that this stack barrier was hit.
	ADD	$1, R5
	MOVW	R5, g_stkbarPos(g)
	// Jump to the original return PC.
	B	(R6)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMP	$MAXSIZE, R0;		\
	B.HI	3(PC);			\
	MOVW	$NAME(SB), R1;		\
	B	(R1)

TEXT reflect·call(SB), NOSPLIT, $0-0
	B	·reflectcall(SB)

TEXT ·reflectcall(SB),NOSPLIT,$-4-20
	MOVW	argsize+12(FP), R0
	DISPATCH(runtime·call16, 16)
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
	MOVW	$runtime·badreflectcall(SB), R1
	B	(R1)

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-20;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVW	argptr+8(FP), R0;		\
	MOVW	argsize+12(FP), R2;		\
	ADD	$4, R13, R1;			\
	CMP	$0, R2;				\
	B.EQ	5(PC);				\
	MOVBU.P	1(R0), R5;			\
	MOVBU.P R5, 1(R1);			\
	SUB	$1, R2, R2;			\
	B	-5(PC);				\
	/* call function */			\
	MOVW	f+4(FP), R7;			\
	MOVW	(R7), R0;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	BL	(R0);				\
	/* copy return values back */		\
	MOVW	argtype+0(FP), R4;		\
	MOVW	argptr+8(FP), R0;		\
	MOVW	argsize+12(FP), R2;		\
	MOVW	retoffset+16(FP), R3;		\
	ADD	$4, R13, R1;			\
	ADD	R3, R1;				\
	ADD	R3, R0;				\
	SUB	R3, R2;				\
	BL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $16-0
	MOVW	R4, 4(R13)
	MOVW	R0, 8(R13)
	MOVW	R1, 12(R13)
	MOVW	R2, 16(R13)
	BL	runtime·reflectcallmove(SB)
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

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 4 bytes to get back to BL deferreturn
// 3. B to fn
// TODO(rsc): Push things on stack and then use pop
// to load all registers simultaneously, so that a profiling
// interrupt can never see mismatched SP/LR/PC.
// (And double-check that pop is atomic in that way.)
TEXT runtime·jmpdefer(SB),NOSPLIT,$0-8
	MOVW	0(R13), LR
	MOVW	$-4(LR), LR	// BL deferreturn
	MOVW	fv+0(FP), R7
	MOVW	argp+4(FP), R13
	MOVW	$-4(R13), R13	// SP is 4 below argp, due to saved LR
	MOVW	0(R7), R1
	B	(R1)

// Save state of caller into g->sched. Smashes R11.
TEXT gosave<>(SB),NOSPLIT,$-4
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	$0, R11
	MOVW	R11, (g_sched+gobuf_lr)(g)
	MOVW	R11, (g_sched+gobuf_ret)(g)
	MOVW	R11, (g_sched+gobuf_ctxt)(g)
	// Assert ctxt is zero. See func save.
	MOVW	(g_sched+gobuf_ctxt)(g), R11
	CMP	$0, R11
	B.EQ	2(PC)
	CALL	runtime·badctxt(SB)
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-12
	MOVW	fn+0(FP), R1
	MOVW	arg+4(FP), R0

	MOVW	R13, R2
	MOVW	g, R4

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R3
	CMP	R3, g
	BEQ	g0
	BL	gosave<>(SB)
	MOVW	R0, R5
	MOVW	R3, R0
	BL	setg<>(SB)
	MOVW	R5, R0
	MOVW	(g_sched+gobuf_sp)(g), R13

	// Now on a scheduling stack (a pthread-created stack).
g0:
	SUB	$24, R13
	BIC	$0x7, R13	// alignment for gcc ABI
	MOVW	R4, 20(R13) // save old g
	MOVW	(g_stack+stack_hi)(R4), R4
	SUB	R2, R4
	MOVW	R4, 16(R13)	// save depth in stack (can't just save SP, as stack might be copied during a callback)
	BL	(R1)

	// Restore registers, g, stack pointer.
	MOVW	R0, R5
	MOVW	20(R13), R0
	BL	setg<>(SB)
	MOVW	(g_stack+stack_hi)(g), R1
	MOVW	16(R13), R2
	SUB	R2, R1
	MOVW	R5, R0
	MOVW	R1, R13

	MOVW	R0, ret+8(FP)
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize, uintptr ctxt)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$16-16
	MOVW	$fn+0(FP), R0
	MOVW	R0, 4(R13)
	MOVW	frame+4(FP), R0
	MOVW	R0, 8(R13)
	MOVW	framesize+8(FP), R0
	MOVW	R0, 12(R13)
	MOVW	ctxt+12(FP), R0
	MOVW	R0, 16(R13)
	MOVW	$runtime·cgocallback_gofunc(SB), R0
	BL	(R0)
	RET

// cgocallback_gofunc(void (*fn)(void*), void *frame, uintptr framesize, uintptr ctxt)
// See cgocall.go for more details.
TEXT	·cgocallback_gofunc(SB),NOSPLIT,$8-16
	NO_LOCAL_POINTERS
	
	// Load m and g from thread-local storage.
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BL.NE	runtime·load_g(SB)

	// If g is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	CMP	$0, g
	B.EQ	needm

	MOVW	g_m(g), R8
	MOVW	R8, savedm-4(SP)
	B	havem

needm:
	MOVW	g, savedm-4(SP) // g is zero, so is m.
	MOVW	$runtime·needm(SB), R0
	BL	(R0)

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
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R3
	MOVW	R13, (g_sched+gobuf_sp)(R3)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 4(R13) aka savedsp-8(SP).
	MOVW	m_g0(R8), R3
	MOVW	(g_sched+gobuf_sp)(R3), R4
	MOVW	R4, savedsp-8(SP)
	MOVW	R13, (g_sched+gobuf_sp)(R3)

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
	// In the new goroutine, -4(SP) is unused (where SP refers to
	// m->curg's SP while we're setting it up, before we've adjusted it).
	MOVW	m_curg(R8), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R4 // prepare stack as R4
	MOVW	(g_sched+gobuf_pc)(g), R5
	MOVW	R5, -12(R4)
	MOVW	ctxt+12(FP), R0
	MOVW	R0, -8(R4)
	MOVW	$-12(R4), R13
	BL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVW	0(R13), R5
	MOVW	R5, (g_sched+gobuf_pc)(g)
	MOVW	$12(R13), R4
	MOVW	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R13
	MOVW	savedsp-8(SP), R4
	MOVW	R4, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	MOVW	savedm-4(SP), R6
	CMP	$0, R6
	B.NE	3(PC)
	MOVW	$runtime·dropm(SB), R0
	BL	(R0)

	// Done!
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB),NOSPLIT,$-4-4
	MOVW	gg+0(FP), R0
	B	setg<>(SB)

TEXT setg<>(SB),NOSPLIT,$-4-0
	MOVW	R0, g

	// Save g to thread-local storage.
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	B.EQ	2(PC)
	B	runtime·save_g(SB)

	MOVW	g, R0
	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$4-8
	MOVW	8(R13), R0		// LR saved by caller
	MOVW	runtime·stackBarrierPC(SB), R1
	CMP	R0, R1
	BNE	nobar
	// Get original return PC.
	BL	runtime·nextBarrierPC(SB)
	MOVW	4(R13), R0
nobar:
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·setcallerpc(SB),NOSPLIT,$4-8
	MOVW	pc+4(FP), R0
	MOVW	8(R13), R1
	MOVW	runtime·stackBarrierPC(SB), R2
	CMP	R1, R2
	BEQ	setbar
	MOVW	R0, 8(R13)		// set LR in caller
	RET
setbar:
	// Set the stack barrier return PC.
	MOVW	R0, 4(R13)
	BL	runtime·setNextBarrierPC(SB)
	RET

TEXT runtime·emptyfunc(SB),0,$0-0
	RET

TEXT runtime·abort(SB),NOSPLIT,$-4-0
	MOVW	$0, R0
	MOVW	(R0), R1

// armPublicationBarrier is a native store/store barrier for ARMv7+.
// On earlier ARM revisions, armPublicationBarrier is a no-op.
// This will not work on SMP ARMv6 machines, if any are in use.
// To implement publicationBarrier in sys_$GOOS_arm.s using the native
// instructions, use:
//
//	TEXT ·publicationBarrier(SB),NOSPLIT,$-4-0
//		B	runtime·armPublicationBarrier(SB)
//
TEXT runtime·armPublicationBarrier(SB),NOSPLIT,$-4-0
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	WORD $0xf57ff05e	// DMB ST
	RET

// AES hashing not implemented for ARM
TEXT runtime·aeshash(SB),NOSPLIT,$-4-0
	MOVW	$0, R0
	MOVW	(R0), R1
TEXT runtime·aeshash32(SB),NOSPLIT,$-4-0
	MOVW	$0, R0
	MOVW	(R0), R1
TEXT runtime·aeshash64(SB),NOSPLIT,$-4-0
	MOVW	$0, R0
	MOVW	(R0), R1
TEXT runtime·aeshashstr(SB),NOSPLIT,$-4-0
	MOVW	$0, R0
	MOVW	(R0), R1

// memhash_varlen(p unsafe.Pointer, h seed) uintptr
// redirects to memhash(p, h, size) using the size
// stored in the closure.
TEXT runtime·memhash_varlen(SB),NOSPLIT,$16-12
	GO_ARGS
	NO_LOCAL_POINTERS
	MOVW	p+0(FP), R0
	MOVW	h+4(FP), R1
	MOVW	4(R7), R2
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·memhash(SB)
	MOVW	16(R13), R0
	MOVW	R0, ret+8(FP)
	RET

// memequal(p, q unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$-4-13
	MOVW	a+0(FP), R1
	MOVW	b+4(FP), R2
	MOVW	size+8(FP), R3
	ADD	R1, R3, R6
	MOVW	$1, R0
	MOVB	R0, ret+12(FP)
	CMP	R1, R2
	RET.EQ
loop:
	CMP	R1, R6
	RET.EQ
	MOVBU.P	1(R1), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	BEQ	loop

	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$16-9
	MOVW	a+0(FP), R0
	MOVW	b+4(FP), R1
	CMP	R0, R1
	BEQ	eq
	MOVW	4(R7), R2    // compiler stores size at offset 4 in the closure
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·memequal(SB)
	MOVB	16(R13), R0
	MOVB	R0, ret+8(FP)
	RET
eq:
	MOVW	$1, R0
	MOVB	R0, ret+8(FP)
	RET

TEXT runtime·cmpstring(SB),NOSPLIT,$-4-20
	MOVW	s1_base+0(FP), R2
	MOVW	s1_len+4(FP), R0
	MOVW	s2_base+8(FP), R3
	MOVW	s2_len+12(FP), R1
	ADD	$20, R13, R7
	B	runtime·cmpbody(SB)

TEXT bytes·Compare(SB),NOSPLIT,$-4-28
	MOVW	s1+0(FP), R2
	MOVW	s1+4(FP), R0
	MOVW	s2+12(FP), R3
	MOVW	s2+16(FP), R1
	ADD	$28, R13, R7
	B	runtime·cmpbody(SB)

// On entry:
// R0 is the length of s1
// R1 is the length of s2
// R2 points to the start of s1
// R3 points to the start of s2
// R7 points to return value (-1/0/1 will be written here)
//
// On exit:
// R4, R5, and R6 are clobbered
TEXT runtime·cmpbody(SB),NOSPLIT,$-4-0
	CMP	R2, R3
	BEQ	samebytes
	CMP 	R0, R1
	MOVW 	R0, R6
	MOVW.LT	R1, R6	// R6 is min(R0, R1)

	ADD	R2, R6	// R2 is current byte in s1, R6 is last byte in s1 to compare
loop:
	CMP	R2, R6
	BEQ	samebytes // all compared bytes were the same; compare lengths
	MOVBU.P	1(R2), R4
	MOVBU.P	1(R3), R5
	CMP	R4, R5
	BEQ	loop
	// bytes differed
	MOVW.LT	$1, R0
	MOVW.GT	$-1, R0
	MOVW	R0, (R7)
	RET
samebytes:
	CMP	R0, R1
	MOVW.LT	$1, R0
	MOVW.GT	$-1, R0
	MOVW.EQ	$0, R0
	MOVW	R0, (R7)
	RET

// eqstring tests whether two strings are equal.
// The compiler guarantees that strings passed
// to eqstring have equal length.
// See runtime_test.go:eqstring_generic for
// equivalent Go code.
TEXT runtime·eqstring(SB),NOSPLIT,$-4-17
	MOVW	s1_base+0(FP), R2
	MOVW	s2_base+8(FP), R3
	MOVW	$1, R8
	MOVB	R8, ret+16(FP)
	CMP	R2, R3
	RET.EQ
	MOVW	s1_len+4(FP), R0
	ADD	R2, R0, R6
loop:
	CMP	R2, R6
	RET.EQ
	MOVBU.P	1(R2), R4
	MOVBU.P	1(R3), R5
	CMP	R4, R5
	BEQ	loop
	MOVW	$0, R8
	MOVB	R8, ret+16(FP)
	RET

// TODO: share code with memequal?
TEXT bytes·Equal(SB),NOSPLIT,$0-25
	MOVW	a_len+4(FP), R1
	MOVW	b_len+16(FP), R3
	
	CMP	R1, R3		// unequal lengths are not equal
	B.NE	notequal

	MOVW	a+0(FP), R0
	MOVW	b+12(FP), R2
	ADD	R0, R1		// end

loop:
	CMP	R0, R1
	B.EQ	equal		// reached the end
	MOVBU.P	1(R0), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	B.EQ	loop

notequal:
	MOVW	$0, R0
	MOVBU	R0, ret+24(FP)
	RET

equal:
	MOVW	$1, R0
	MOVBU	R0, ret+24(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-20
	MOVW	s+0(FP), R0
	MOVW	s_len+4(FP), R1
	MOVBU	c+12(FP), R2	// byte to find
	MOVW	R0, R4		// store base for later
	ADD	R0, R1		// end

_loop:
	CMP	R0, R1
	B.EQ	_notfound
	MOVBU.P	1(R0), R3
	CMP	R2, R3
	B.NE	_loop

	SUB	$1, R0		// R0 will be one beyond the position we want
	SUB	R4, R0		// remove base
	MOVW    R0, ret+16(FP)
	RET

_notfound:
	MOVW	$-1, R0
	MOVW	R0, ret+16(FP)
	RET

TEXT strings·IndexByte(SB),NOSPLIT,$0-16
	MOVW	s+0(FP), R0
	MOVW	s_len+4(FP), R1
	MOVBU	c+8(FP), R2	// byte to find
	MOVW	R0, R4		// store base for later
	ADD	R0, R1		// end

_sib_loop:
	CMP	R0, R1
	B.EQ	_sib_notfound
	MOVBU.P	1(R0), R3
	CMP	R2, R3
	B.NE	_sib_loop

	SUB	$1, R0		// R0 will be one beyond the position we want
	SUB	R4, R0		// remove base
	MOVW	R0, ret+12(FP)
	RET

_sib_notfound:
	MOVW	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·fastrand(SB),NOSPLIT,$-4-4
	MOVW	g_m(g), R1
	MOVW	m_fastrand(R1), R0
	ADD.S	R0, R0
	EOR.MI	$0x88888eef, R0
	MOVW	R0, m_fastrand(R1)
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·return0(SB),NOSPLIT,$0
	MOVW	$0, R0
	RET

TEXT runtime·procyield(SB),NOSPLIT,$-4
	MOVW	cycles+0(FP), R1
	MOVW	$0, R0
yieldloop:
	CMP	R0, R1
	B.NE	2(PC)
	RET
	SUB	$1, R1
	B yieldloop

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$8
	// R11 and g register are clobbered by load_g. They are
	// callee-save in the gcc calling convention, so save them here.
	MOVW	R11, saveR11-4(SP)
	MOVW	g, saveG-8(SP)
	
	BL	runtime·load_g(SB)
	MOVW	g_m(g), R0
	MOVW	m_curg(R0), R0
	MOVW	(g_stack+stack_hi)(R0), R0
	
	MOVW	saveG-8(SP), g
	MOVW	saveR11-4(SP), R11
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT,$-4-0
	MOVW	R0, R0	// NOP
	BL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	MOVW	R0, R0	// NOP

TEXT runtime·prefetcht0(SB),NOSPLIT,$0-4
	RET

TEXT runtime·prefetcht1(SB),NOSPLIT,$0-4
	RET

TEXT runtime·prefetcht2(SB),NOSPLIT,$0-4
	RET

TEXT runtime·prefetchnta(SB),NOSPLIT,$0-4
	RET

// x -> x/1000000, x%1000000, called from Go with args, results on stack.
TEXT runtime·usplit(SB),NOSPLIT,$0-12
	MOVW	x+0(FP), R0
	CALL	runtime·usplitR0(SB)
	MOVW	R0, q+4(FP)
	MOVW	R1, r+8(FP)
	RET

// R0, R1 = R0/1000000, R0%1000000
TEXT runtime·usplitR0(SB),NOSPLIT,$0
	// magic multiply to avoid software divide without available m.
	// see output of go tool compile -S for x/1000000.
	MOVW	R0, R3
	MOVW	$1125899907, R1
	MULLU	R1, R0, (R0, R1)
	MOVW	R0>>18, R0
	MOVW	$1000000, R1
	MULU	R0, R1
	SUB	R1, R3, R1
	RET

TEXT runtime·sigreturn(SB),NOSPLIT,$0-0
	RET

#ifndef GOOS_nacl
// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-4
	MOVW	R9, saver9-4(SP) // The access to global variables below implicitly uses R9, which is callee-save
	MOVW	runtime·lastmoduledatap(SB), R1
	MOVW	R0, moduledata_next(R1)
	MOVW	R0, runtime·lastmoduledatap(SB)
	MOVW	saver9-4(SP), R9
	RET
#endif

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R3
	MOVB	R3, ret+0(FP)
	RET
