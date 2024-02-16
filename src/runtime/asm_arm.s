// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// _rt0_arm is common startup code for most ARM systems when using
// internal linking. This is the entry point for the program from the
// kernel for an ordinary -buildmode=exe program. The stack holds the
// number of arguments and the C-style argv.
TEXT _rt0_arm(SB),NOSPLIT|NOFRAME,$0
	MOVW	(R13), R0	// argc
	MOVW	$4(R13), R1		// argv
	B	runtime·rt0_go(SB)

// main is common startup code for most ARM systems when using
// external linking. The C startup code will call the symbol "main"
// passing argc and argv in the usual C ABI registers R0 and R1.
TEXT main(SB),NOSPLIT|NOFRAME,$0
	B	runtime·rt0_go(SB)

// _rt0_arm_lib is common startup code for most ARM systems when
// using -buildmode=c-archive or -buildmode=c-shared. The linker will
// arrange to invoke this function as a global constructor (for
// c-archive) or when the shared library is loaded (for c-shared).
// We expect argc and argv to be passed in the usual C ABI registers
// R0 and R1.
TEXT _rt0_arm_lib(SB),NOSPLIT,$104
	// Preserve callee-save registers. Raspberry Pi's dlopen(), for example,
	// actually cares that R11 is preserved.
	MOVW	R4, 12(R13)
	MOVW	R5, 16(R13)
	MOVW	R6, 20(R13)
	MOVW	R7, 24(R13)
	MOVW	R8, 28(R13)
	MOVW	g, 32(R13)
	MOVW	R11, 36(R13)

	// Skip floating point registers on goarmsoftfp != 0.
	MOVB    runtime·goarmsoftfp(SB), R11
	CMP	$0, R11
	BNE     skipfpsave
	MOVD	F8, (40+8*0)(R13)
	MOVD	F9, (40+8*1)(R13)
	MOVD	F10, (40+8*2)(R13)
	MOVD	F11, (40+8*3)(R13)
	MOVD	F12, (40+8*4)(R13)
	MOVD	F13, (40+8*5)(R13)
	MOVD	F14, (40+8*6)(R13)
	MOVD	F15, (40+8*7)(R13)
skipfpsave:
	// Save argc/argv.
	MOVW	R0, _rt0_arm_lib_argc<>(SB)
	MOVW	R1, _rt0_arm_lib_argv<>(SB)

	MOVW	$0, g // Initialize g.

	// Synchronous initialization.
	CALL	runtime·libpreinit(SB)

	// Create a new thread to do the runtime initialization.
	MOVW	_cgo_sys_thread_create(SB), R2
	CMP	$0, R2
	BEQ	nocgo
	MOVW	$_rt0_arm_lib_go<>(SB), R0
	MOVW	$0, R1
	BL	(R2)
	B	rr
nocgo:
	MOVW	$0x800000, R0                     // stacksize = 8192KB
	MOVW	$_rt0_arm_lib_go<>(SB), R1  // fn
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	BL	runtime·newosproc0(SB)
rr:
	// Restore callee-save registers and return.
	MOVB    runtime·goarmsoftfp(SB), R11
	CMP     $0, R11
	BNE     skipfprest
	MOVD	(40+8*0)(R13), F8
	MOVD	(40+8*1)(R13), F9
	MOVD	(40+8*2)(R13), F10
	MOVD	(40+8*3)(R13), F11
	MOVD	(40+8*4)(R13), F12
	MOVD	(40+8*5)(R13), F13
	MOVD	(40+8*6)(R13), F14
	MOVD	(40+8*7)(R13), F15
skipfprest:
	MOVW	12(R13), R4
	MOVW	16(R13), R5
	MOVW	20(R13), R6
	MOVW	24(R13), R7
	MOVW	28(R13), R8
	MOVW	32(R13), g
	MOVW	36(R13), R11
	RET

// _rt0_arm_lib_go initializes the Go runtime.
// This is started in a separate thread by _rt0_arm_lib.
TEXT _rt0_arm_lib_go<>(SB),NOSPLIT,$8
	MOVW	_rt0_arm_lib_argc<>(SB), R0
	MOVW	_rt0_arm_lib_argv<>(SB), R1
	B	runtime·rt0_go(SB)

DATA _rt0_arm_lib_argc<>(SB)/4,$0
GLOBL _rt0_arm_lib_argc<>(SB),NOPTR,$4
DATA _rt0_arm_lib_argv<>(SB)/4,$0
GLOBL _rt0_arm_lib_argv<>(SB),NOPTR,$4

// using NOFRAME means do not save LR on stack.
// argc is in R0, argv is in R1.
TEXT runtime·rt0_go(SB),NOSPLIT|NOFRAME|TOPFRAME,$0
	MOVW	$0xcafebabe, R12

	// copy arguments forward on an even stack
	// use R13 instead of SP to avoid linker rewriting the offsets
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

#ifdef GOOS_openbsd
	// Save g to TLS so that it is available from signal trampoline.
	BL	runtime·save_g(SB)
#endif

	BL	runtime·_initcgo(SB)	// will clobber R0-R3

	// update stackguard after _cgo_init
	MOVW	(g_stack+stack_lo)(g), R0
	ADD	$const_stackGuard, R0
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
	SUB	$8, R13
	MOVW	$runtime·mainPC(SB), R0
	MOVW	R0, 4(R13)	// arg 1: fn
	MOVW	$0, R0
	MOVW	R0, 0(R13)	// dummy LR
	BL	runtime·newproc(SB)
	ADD	$8, R13	// pop args and LR

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
#ifdef GOOS_plan9
	WORD	$0xD1200070	// undefined instruction used as armv5 breakpoint in Plan 9
#else
	WORD	$0xe7f001f0	// undefined instruction that gdb understands is a software breakpoint
#endif
	RET

TEXT runtime·asminit(SB),NOSPLIT,$0-0
	// disable runfast (flush-to-zero) mode of vfp if runtime.goarmsoftfp == 0
	MOVB	runtime·goarmsoftfp(SB), R11
	CMP	$0, R11
	BNE	4(PC)
	WORD	$0xeef1ba10	// vmrs r11, fpscr
	BIC	$(1<<24), R11
	WORD	$0xeee1ba10	// vmsr fpscr, r11
	RET

TEXT runtime·mstart(SB),NOSPLIT|TOPFRAME,$0
	BL	runtime·mstart0(SB)
	RET // not reached

/*
 *  go-routine
 */

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	buf+0(FP), R1
	MOVW	gobuf_g(R1), R0
	MOVW	0(R0), R2	// make sure g != nil
	B	gogo<>(SB)

TEXT gogo<>(SB),NOSPLIT|NOFRAME,$0
	BL	setg<>(SB)
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
TEXT runtime·mcall(SB),NOSPLIT|NOFRAME,$0-4
	// Save caller state in g->sched.
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	$0, R11
	MOVW	R11, (g_sched+gobuf_lr)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVW	g, R1
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R0
	BL	setg<>(SB)
	CMP	g, R1
	B.NE	2(PC)
	B	runtime·badmcall(SB)
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
	B	runtime·abort(SB)

switch:
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	BL	gosave_systemstack_switch<>(SB)

	// switch to g0
	MOVW	R0, R5
	MOVW	R2, R0
	BL	setg<>(SB)
	MOVW	R5, R0
	MOVW	(g_sched+gobuf_sp)(R2), R13

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
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVW	R0, R7
	MOVW	0(R0), R0
	MOVW.P	4(R13), R14	// restore LR
	B	(R0)

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// R3 prolog's LR
// using NOFRAME means do not save LR on stack.
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT|NOFRAME,$0-0
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
	MOVW	R7, (g_sched+gobuf_ctxt)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVW	R3, (m_morebuf+gobuf_pc)(R8)	// f's caller's PC
	MOVW	R13, (m_morebuf+gobuf_sp)(R8)	// f's caller's SP
	MOVW	g, (m_morebuf+gobuf_g)(R8)

	// Call newstack on m->g0's stack.
	MOVW	m_g0(R8), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R13
	MOVW	$0, R0
	MOVW.W  R0, -4(R13)	// create a call frame on g0 (saved LR)
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	RET

TEXT runtime·morestack_noctxt(SB),NOSPLIT|NOFRAME,$0-0
	// Force SPWRITE. This function doesn't actually write SP,
	// but it is called with a special calling convention where
	// the caller doesn't save LR on stack but passes it as a
	// register (R3), and the unwinder currently doesn't understand.
	// Make it SPWRITE to stop unwinding. (See issue 54332)
	MOVW	R13, R13

	MOVW	$0, R7
	B runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(stackArgsType *_type, f *FuncVal, stackArgs *byte, stackArgsSize, stackRetOffset, frameSize uint32, regArgs *abi.RegArgs).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMP	$MAXSIZE, R0;		\
	B.HI	3(PC);			\
	MOVW	$NAME(SB), R1;		\
	B	(R1)

TEXT ·reflectcall(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	frameSize+20(FP), R0
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
TEXT NAME(SB), WRAPPER, $MAXSIZE-28;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVW	stackArgs+8(FP), R0;		\
	MOVW	stackArgsSize+12(FP), R2;		\
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
	MOVW	stackArgsType+0(FP), R4;		\
	MOVW	stackArgs+8(FP), R0;		\
	MOVW	stackArgsSize+12(FP), R2;		\
	MOVW	stackArgsRetOffset+16(FP), R3;		\
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
TEXT callRet<>(SB), NOSPLIT, $20-0
	MOVW	R4, 4(R13)
	MOVW	R0, 8(R13)
	MOVW	R1, 12(R13)
	MOVW	R2, 16(R13)
	MOVW	$0, R7
	MOVW	R7, 20(R13)
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

// Save state of caller into g->sched,
// but using fake PC from systemstack_switch.
// Must only be called from functions with no locals ($0)
// or else unwinding from systemstack_switch is incorrect.
// Smashes R11.
TEXT gosave_systemstack_switch<>(SB),NOSPLIT|NOFRAME,$0
	MOVW	$runtime·systemstack_switch(SB), R11
	ADD	$4, R11 // get past push {lr}
	MOVW	R11, (g_sched+gobuf_pc)(g)
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	$0, R11
	MOVW	R11, (g_sched+gobuf_lr)(g)
	MOVW	R11, (g_sched+gobuf_ret)(g)
	// Assert ctxt is zero. See func save.
	MOVW	(g_sched+gobuf_ctxt)(g), R11
	TST	R11, R11
	B.EQ	2(PC)
	BL	runtime·abort(SB)
	RET

// func asmcgocall_no_g(fn, arg unsafe.Pointer)
// Call fn(arg) aligned appropriately for the gcc ABI.
// Called on a system stack, and there may be no g yet (during needm).
TEXT ·asmcgocall_no_g(SB),NOSPLIT,$0-8
	MOVW	fn+0(FP), R1
	MOVW	arg+4(FP), R0
	MOVW	R13, R2
	SUB	$32, R13
	BIC	$0x7, R13	// alignment for gcc ABI
	MOVW	R2, 8(R13)
	BL	(R1)
	MOVW	8(R13), R2
	MOVW	R2, R13
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-12
	MOVW	fn+0(FP), R1
	MOVW	arg+4(FP), R0

	MOVW	R13, R2
	CMP	$0, g
	BEQ nosave
	MOVW	g, R4

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already. Or we might already
	// be on the m->gsignal stack.
	MOVW	g_m(g), R8
	MOVW	m_gsignal(R8), R3
	CMP	R3, g
	BEQ	nosave
	MOVW	m_g0(R8), R3
	CMP	R3, g
	BEQ	nosave
	BL	gosave_systemstack_switch<>(SB)
	MOVW	R0, R5
	MOVW	R3, R0
	BL	setg<>(SB)
	MOVW	R5, R0
	MOVW	(g_sched+gobuf_sp)(g), R13

	// Now on a scheduling stack (a pthread-created stack).
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

nosave:
	// Running on a system stack, perhaps even without a g.
	// Having no g can happen during thread creation or thread teardown
	// (see needm/dropm on Solaris, for example).
	// This code is like the above sequence but without saving/restoring g
	// and without worrying about the stack moving out from under us
	// (because we're on a system stack, not a goroutine stack).
	// The above code could be used directly if already on a system stack,
	// but then the only path through this code would be a rare case on Solaris.
	// Using this code for all "already on system stack" calls exercises it more,
	// which should help keep it correct.
	SUB	$24, R13
	BIC	$0x7, R13	// alignment for gcc ABI
	// save null g in case someone looks during debugging.
	MOVW	$0, R4
	MOVW	R4, 20(R13)
	MOVW	R2, 16(R13)	// Save old stack pointer.
	BL	(R1)
	// Restore stack pointer.
	MOVW	16(R13), R2
	MOVW	R2, R13
	MOVW	R0, ret+8(FP)
	RET

// cgocallback(fn, frame unsafe.Pointer, ctxt uintptr)
// See cgocall.go for more details.
TEXT	·cgocallback(SB),NOSPLIT,$12-12
	NO_LOCAL_POINTERS

	// Skip cgocallbackg, just dropm when fn is nil, and frame is the saved g.
	// It is used to dropm while thread is exiting.
	MOVW	fn+0(FP), R1
	CMP	$0, R1
	B.NE	loadg
	// Restore the g from frame.
	MOVW	frame+4(FP), g
	B	dropm

loadg:
	// Load m and g from thread-local storage.
#ifdef GOOS_openbsd
	BL	runtime·load_g(SB)
#else
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BL.NE	runtime·load_g(SB)
#endif

	// If g is nil, Go did not create the current thread,
	// or if this thread never called into Go on pthread platforms.
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
	MOVW	$runtime·needAndBindM(SB), R0
	BL	(R0)

	// Set m->g0->sched.sp = SP, so that if a panic happens
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
	// NOTE: unwindm knows that the saved g->sched.sp is at 4(R13) aka savedsp-12(SP).
	MOVW	m_g0(R8), R3
	MOVW	(g_sched+gobuf_sp)(R3), R4
	MOVW	R4, savedsp-12(SP)	// must match frame size
	MOVW	R13, (g_sched+gobuf_sp)(R3)

	// Switch to m->curg stack and call runtime.cgocallbackg.
	// Because we are taking over the execution of m->curg
	// but *not* resuming what had been running, we need to
	// save that information (m->curg->sched) so we can restore it.
	// We can restore m->curg->sched.sp easily, because calling
	// runtime.cgocallbackg leaves SP unchanged upon return.
	// To save m->curg->sched.pc, we push it onto the curg stack and
	// open a frame the same size as cgocallback's g0 frame.
	// Once we switch to the curg stack, the pushed PC will appear
	// to be the return PC of cgocallback, so that the traceback
	// will seamlessly trace back into the earlier calls.
	MOVW	m_curg(R8), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R4 // prepare stack as R4
	MOVW	(g_sched+gobuf_pc)(g), R5
	MOVW	R5, -(12+4)(R4)	// "saved LR"; must match frame size
	// Gather our arguments into registers.
	MOVW	fn+0(FP), R1
	MOVW	frame+4(FP), R2
	MOVW	ctxt+8(FP), R3
	MOVW	$-(12+4)(R4), R13	// switch stack; must match frame size
	MOVW	R1, 4(R13)
	MOVW	R2, 8(R13)
	MOVW	R3, 12(R13)
	BL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVW	0(R13), R5
	MOVW	R5, (g_sched+gobuf_pc)(g)
	MOVW	$(12+4)(R13), R4	// must match frame size
	MOVW	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVW	g_m(g), R8
	MOVW	m_g0(R8), R0
	BL	setg<>(SB)
	MOVW	(g_sched+gobuf_sp)(g), R13
	MOVW	savedsp-12(SP), R4	// must match frame size
	MOVW	R4, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m,
	// 1. for the duration of the call on non-pthread platforms,
	// 2. or the duration of the C thread alive on pthread platforms.
	// If the m on entry wasn't nil,
	// 1. the thread might be a Go thread,
	// 2. or it wasn't the first call from a C thread on pthread platforms,
	//    since then we skip dropm to reuse the m in the first call.
	MOVW	savedm-4(SP), R6
	CMP	$0, R6
	B.NE	done

	// Skip dropm to reuse it in the next call, when a pthread key has been created.
	MOVW	_cgo_pthread_key_created(SB), R6
	// It means cgo is disabled when _cgo_pthread_key_created is a nil pointer, need dropm.
	CMP	$0, R6
	B.EQ	dropm
	MOVW	(R6), R6
	CMP	$0, R6
	B.NE	done

dropm:
	MOVW	$runtime·dropm(SB), R0
	BL	(R0)

done:
	// Done!
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	gg+0(FP), R0
	B	setg<>(SB)

TEXT setg<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	R0, g

	// Save g to thread-local storage.
#ifdef GOOS_windows
	B	runtime·save_g(SB)
#else
#ifdef GOOS_openbsd
	B	runtime·save_g(SB)
#else
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	B.EQ	2(PC)
	B	runtime·save_g(SB)

	MOVW	g, R0
	RET
#endif
#endif

TEXT runtime·emptyfunc(SB),0,$0-0
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	$0, R0
	MOVW	(R0), R1

// armPublicationBarrier is a native store/store barrier for ARMv7+.
// On earlier ARM revisions, armPublicationBarrier is a no-op.
// This will not work on SMP ARMv6 machines, if any are in use.
// To implement publicationBarrier in sys_$GOOS_arm.s using the native
// instructions, use:
//
//	TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
//		B	runtime·armPublicationBarrier(SB)
//
TEXT runtime·armPublicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	DMB	MB_ST
	RET

// AES hashing not implemented for ARM
TEXT runtime·memhash(SB),NOSPLIT|NOFRAME,$0-16
	JMP	runtime·memhashFallback(SB)
TEXT runtime·strhash(SB),NOSPLIT|NOFRAME,$0-12
	JMP	runtime·strhashFallback(SB)
TEXT runtime·memhash32(SB),NOSPLIT|NOFRAME,$0-12
	JMP	runtime·memhash32Fallback(SB)
TEXT runtime·memhash64(SB),NOSPLIT|NOFRAME,$0-12
	JMP	runtime·memhash64Fallback(SB)

TEXT runtime·return0(SB),NOSPLIT,$0
	MOVW	$0, R0
	RET

TEXT runtime·procyield(SB),NOSPLIT|NOFRAME,$0
	MOVW	cycles+0(FP), R1
	MOVW	$0, R0
yieldloop:
	WORD	$0xe320f001	// YIELD (NOP pre-ARMv6K)
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
TEXT runtime·goexit(SB),NOSPLIT|NOFRAME|TOPFRAME,$0-0
	MOVW	R0, R0	// NOP
	BL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	MOVW	R0, R0	// NOP

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

// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-0
	MOVW	R9, saver9-4(SP) // The access to global variables below implicitly uses R9, which is callee-save
	MOVW	R11, saver11-8(SP) // Likewise, R11 is the temp register, but callee-save in C ABI
	MOVW	runtime·lastmoduledatap(SB), R1
	MOVW	R0, moduledata_next(R1)
	MOVW	R0, runtime·lastmoduledatap(SB)
	MOVW	saver11-8(SP), R11
	MOVW	saver9-4(SP), R9
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R3
	MOVB	R3, ret+0(FP)
	RET

// gcWriteBarrier informs the GC about heap pointer writes.
//
// gcWriteBarrier does NOT follow the Go ABI. It accepts the
// number of bytes of buffer needed in R8, and returns a pointer
// to the buffer space in R8.
// It clobbers condition codes.
// It does not clobber any other general-purpose registers,
// but may clobber others (e.g., floating point registers).
// The act of CALLing gcWriteBarrier will clobber R14 (LR).
TEXT gcWriteBarrier<>(SB),NOSPLIT|NOFRAME,$0
	// Save the registers clobbered by the fast path.
	MOVM.DB.W	[R0,R1], (R13)
retry:
	MOVW	g_m(g), R0
	MOVW	m_p(R0), R0
	MOVW	(p_wbBuf+wbBuf_next)(R0), R1
	MOVW	(p_wbBuf+wbBuf_end)(R0), R11
	// Increment wbBuf.next position.
	ADD	R8, R1
	// Is the buffer full?
	CMP	R11, R1
	BHI	flush
	// Commit to the larger buffer.
	MOVW	R1, (p_wbBuf+wbBuf_next)(R0)
	// Make return value (the original next position)
	SUB	R8, R1, R8
	// Restore registers.
	MOVM.IA.W	(R13), [R0,R1]
	RET

flush:
	// Save all general purpose registers since these could be
	// clobbered by wbBufFlush and were not saved by the caller.
	//
	// R0 and R1 were saved at entry.
	// R10 is g, so preserved.
	// R11 is linker temp, so no need to save.
	// R13 is stack pointer.
	// R15 is PC.
	MOVM.DB.W	[R2-R9,R12], (R13)
	// Save R14 (LR) because the fast path above doesn't save it,
	// but needs it to RET.
	MOVM.DB.W	[R14], (R13)

	CALL	runtime·wbBufFlush(SB)

	MOVM.IA.W	(R13), [R14]
	MOVM.IA.W	(R13), [R2-R9,R12]
	JMP	retry

TEXT runtime·gcWriteBarrier1<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$4, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier2<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$8, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier3<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$12, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier4<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$16, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier5<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$20, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier6<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$24, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier7<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$28, R8
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier8<ABIInternal>(SB),NOSPLIT,$0
	MOVW	$32, R8
	JMP	gcWriteBarrier<>(SB)

// Note: these functions use a special calling convention to save generated code space.
// Arguments are passed in registers, but the space for those arguments are allocated
// in the caller's stack frame. These stubs write the args into that stack space and
// then tail call to the corresponding runtime handler.
// The tail call makes these stubs disappear in backtraces.
TEXT runtime·panicIndex(SB),NOSPLIT,$0-8
	MOVW	R0, x+0(FP)
	MOVW	R1, y+4(FP)
	JMP	runtime·goPanicIndex(SB)
TEXT runtime·panicIndexU(SB),NOSPLIT,$0-8
	MOVW	R0, x+0(FP)
	MOVW	R1, y+4(FP)
	JMP	runtime·goPanicIndexU(SB)
TEXT runtime·panicSliceAlen(SB),NOSPLIT,$0-8
	MOVW	R1, x+0(FP)
	MOVW	R2, y+4(FP)
	JMP	runtime·goPanicSliceAlen(SB)
TEXT runtime·panicSliceAlenU(SB),NOSPLIT,$0-8
	MOVW	R1, x+0(FP)
	MOVW	R2, y+4(FP)
	JMP	runtime·goPanicSliceAlenU(SB)
TEXT runtime·panicSliceAcap(SB),NOSPLIT,$0-8
	MOVW	R1, x+0(FP)
	MOVW	R2, y+4(FP)
	JMP	runtime·goPanicSliceAcap(SB)
TEXT runtime·panicSliceAcapU(SB),NOSPLIT,$0-8
	MOVW	R1, x+0(FP)
	MOVW	R2, y+4(FP)
	JMP	runtime·goPanicSliceAcapU(SB)
TEXT runtime·panicSliceB(SB),NOSPLIT,$0-8
	MOVW	R0, x+0(FP)
	MOVW	R1, y+4(FP)
	JMP	runtime·goPanicSliceB(SB)
TEXT runtime·panicSliceBU(SB),NOSPLIT,$0-8
	MOVW	R0, x+0(FP)
	MOVW	R1, y+4(FP)
	JMP	runtime·goPanicSliceBU(SB)
TEXT runtime·panicSlice3Alen(SB),NOSPLIT,$0-8
	MOVW	R2, x+0(FP)
	MOVW	R3, y+4(FP)
	JMP	runtime·goPanicSlice3Alen(SB)
TEXT runtime·panicSlice3AlenU(SB),NOSPLIT,$0-8
	MOVW	R2, x+0(FP)
	MOVW	R3, y+4(FP)
	JMP	runtime·goPanicSlice3AlenU(SB)
TEXT runtime·panicSlice3Acap(SB),NOSPLIT,$0-8
	MOVW	R2, x+0(FP)
	MOVW	R3, y+4(FP)
	JMP	runtime·goPanicSlice3Acap(SB)
TEXT runtime·panicSlice3AcapU(SB),NOSPLIT,$0-8
	MOVW	R2, x+0(FP)
	MOVW	R3, y+4(FP)
	JMP	runtime·goPanicSlice3AcapU(SB)
TEXT runtime·panicSlice3B(SB),NOSPLIT,$0-8
	MOVW	R1, x+0(FP)
	MOVW	R2, y+4(FP)
	JMP	runtime·goPanicSlice3B(SB)
TEXT runtime·panicSlice3BU(SB),NOSPLIT,$0-8
	MOVW	R1, x+0(FP)
	MOVW	R2, y+4(FP)
	JMP	runtime·goPanicSlice3BU(SB)
TEXT runtime·panicSlice3C(SB),NOSPLIT,$0-8
	MOVW	R0, x+0(FP)
	MOVW	R1, y+4(FP)
	JMP	runtime·goPanicSlice3C(SB)
TEXT runtime·panicSlice3CU(SB),NOSPLIT,$0-8
	MOVW	R0, x+0(FP)
	MOVW	R1, y+4(FP)
	JMP	runtime·goPanicSlice3CU(SB)
TEXT runtime·panicSliceConvert(SB),NOSPLIT,$0-8
	MOVW	R2, x+0(FP)
	MOVW	R3, y+4(FP)
	JMP	runtime·goPanicSliceConvert(SB)

// Extended versions for 64-bit indexes.
TEXT runtime·panicExtendIndex(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R0, lo+4(FP)
	MOVW	R1, y+8(FP)
	JMP	runtime·goPanicExtendIndex(SB)
TEXT runtime·panicExtendIndexU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R0, lo+4(FP)
	MOVW	R1, y+8(FP)
	JMP	runtime·goPanicExtendIndexU(SB)
TEXT runtime·panicExtendSliceAlen(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R1, lo+4(FP)
	MOVW	R2, y+8(FP)
	JMP	runtime·goPanicExtendSliceAlen(SB)
TEXT runtime·panicExtendSliceAlenU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R1, lo+4(FP)
	MOVW	R2, y+8(FP)
	JMP	runtime·goPanicExtendSliceAlenU(SB)
TEXT runtime·panicExtendSliceAcap(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R1, lo+4(FP)
	MOVW	R2, y+8(FP)
	JMP	runtime·goPanicExtendSliceAcap(SB)
TEXT runtime·panicExtendSliceAcapU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R1, lo+4(FP)
	MOVW	R2, y+8(FP)
	JMP	runtime·goPanicExtendSliceAcapU(SB)
TEXT runtime·panicExtendSliceB(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R0, lo+4(FP)
	MOVW	R1, y+8(FP)
	JMP	runtime·goPanicExtendSliceB(SB)
TEXT runtime·panicExtendSliceBU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R0, lo+4(FP)
	MOVW	R1, y+8(FP)
	JMP	runtime·goPanicExtendSliceBU(SB)
TEXT runtime·panicExtendSlice3Alen(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R2, lo+4(FP)
	MOVW	R3, y+8(FP)
	JMP	runtime·goPanicExtendSlice3Alen(SB)
TEXT runtime·panicExtendSlice3AlenU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R2, lo+4(FP)
	MOVW	R3, y+8(FP)
	JMP	runtime·goPanicExtendSlice3AlenU(SB)
TEXT runtime·panicExtendSlice3Acap(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R2, lo+4(FP)
	MOVW	R3, y+8(FP)
	JMP	runtime·goPanicExtendSlice3Acap(SB)
TEXT runtime·panicExtendSlice3AcapU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R2, lo+4(FP)
	MOVW	R3, y+8(FP)
	JMP	runtime·goPanicExtendSlice3AcapU(SB)
TEXT runtime·panicExtendSlice3B(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R1, lo+4(FP)
	MOVW	R2, y+8(FP)
	JMP	runtime·goPanicExtendSlice3B(SB)
TEXT runtime·panicExtendSlice3BU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R1, lo+4(FP)
	MOVW	R2, y+8(FP)
	JMP	runtime·goPanicExtendSlice3BU(SB)
TEXT runtime·panicExtendSlice3C(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R0, lo+4(FP)
	MOVW	R1, y+8(FP)
	JMP	runtime·goPanicExtendSlice3C(SB)
TEXT runtime·panicExtendSlice3CU(SB),NOSPLIT,$0-12
	MOVW	R4, hi+0(FP)
	MOVW	R0, lo+4(FP)
	MOVW	R1, y+8(FP)
	JMP	runtime·goPanicExtendSlice3CU(SB)
