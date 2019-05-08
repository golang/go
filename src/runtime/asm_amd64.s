// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// _rt0_amd64 is common startup code for most amd64 systems when using
// internal linking. This is the entry point for the program from the
// kernel for an ordinary -buildmode=exe program. The stack holds the
// number of arguments and the C-style argv.
TEXT _rt0_amd64(SB),NOSPLIT,$-8
	MOVQ	0(SP), DI	// argc
	LEAQ	8(SP), SI	// argv
	JMP	runtime·rt0_go(SB)

// main is common startup code for most amd64 systems when using
// external linking. The C startup code will call the symbol "main"
// passing argc and argv in the usual C ABI registers DI and SI.
TEXT main(SB),NOSPLIT,$-8
	JMP	runtime·rt0_go(SB)

// _rt0_amd64_lib is common startup code for most amd64 systems when
// using -buildmode=c-archive or -buildmode=c-shared. The linker will
// arrange to invoke this function as a global constructor (for
// c-archive) or when the shared library is loaded (for c-shared).
// We expect argc and argv to be passed in the usual C ABI registers
// DI and SI.
TEXT _rt0_amd64_lib(SB),NOSPLIT,$0x50
	// Align stack per ELF ABI requirements.
	MOVQ	SP, AX
	ANDQ	$~15, SP
	// Save C ABI callee-saved registers, as caller may need them.
	MOVQ	BX, 0x10(SP)
	MOVQ	BP, 0x18(SP)
	MOVQ	R12, 0x20(SP)
	MOVQ	R13, 0x28(SP)
	MOVQ	R14, 0x30(SP)
	MOVQ	R15, 0x38(SP)
	MOVQ	AX, 0x40(SP)

	MOVQ	DI, _rt0_amd64_lib_argc<>(SB)
	MOVQ	SI, _rt0_amd64_lib_argv<>(SB)

	// Synchronous initialization.
	CALL	runtime·libpreinit(SB)

	// Create a new thread to finish Go runtime initialization.
	MOVQ	_cgo_sys_thread_create(SB), AX
	TESTQ	AX, AX
	JZ	nocgo
	MOVQ	$_rt0_amd64_lib_go(SB), DI
	MOVQ	$0, SI
	CALL	AX
	JMP	restore

nocgo:
	MOVQ	$0x800000, 0(SP)		// stacksize
	MOVQ	$_rt0_amd64_lib_go(SB), AX
	MOVQ	AX, 8(SP)			// fn
	CALL	runtime·newosproc0(SB)

restore:
	MOVQ	0x10(SP), BX
	MOVQ	0x18(SP), BP
	MOVQ	0x20(SP), R12
	MOVQ	0x28(SP), R13
	MOVQ	0x30(SP), R14
	MOVQ	0x38(SP), R15
	MOVQ	0x40(SP), SP
	RET

// _rt0_amd64_lib_go initializes the Go runtime.
// This is started in a separate thread by _rt0_amd64_lib.
TEXT _rt0_amd64_lib_go(SB),NOSPLIT,$0
	MOVQ	_rt0_amd64_lib_argc<>(SB), DI
	MOVQ	_rt0_amd64_lib_argv<>(SB), SI
	JMP	runtime·rt0_go(SB)

DATA _rt0_amd64_lib_argc<>(SB)/8, $0
GLOBL _rt0_amd64_lib_argc<>(SB),NOPTR, $8
DATA _rt0_amd64_lib_argv<>(SB)/8, $0
GLOBL _rt0_amd64_lib_argv<>(SB),NOPTR, $8

TEXT runtime·rt0_go(SB),NOSPLIT,$0
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
	MOVQ	BX, g_stackguard0(DI)
	MOVQ	BX, g_stackguard1(DI)
	MOVQ	BX, (g_stack+stack_lo)(DI)
	MOVQ	SP, (g_stack+stack_hi)(DI)

	// find out information about the processor we're on
	MOVL	$0, AX
	CPUID
	MOVL	AX, SI
	CMPL	AX, $0
	JE	nocpuinfo

	// Figure out how to serialize RDTSC.
	// On Intel processors LFENCE is enough. AMD requires MFENCE.
	// Don't know about the rest, so let's do MFENCE.
	CMPL	BX, $0x756E6547  // "Genu"
	JNE	notintel
	CMPL	DX, $0x49656E69  // "ineI"
	JNE	notintel
	CMPL	CX, $0x6C65746E  // "ntel"
	JNE	notintel
	MOVB	$1, runtime·isIntel(SB)
	MOVB	$1, runtime·lfenceBeforeRdtsc(SB)
notintel:

	// Load EAX=1 cpuid flags
	MOVL	$1, AX
	CPUID
	MOVL	AX, runtime·processorVersionInfo(SB)

nocpuinfo:
	// if there is an _cgo_init, call it.
	MOVQ	_cgo_init(SB), AX
	TESTQ	AX, AX
	JZ	needtls
	// arg 1: g0, already in DI
	MOVQ	$setg_gcc<>(SB), SI // arg 2: setg_gcc
#ifdef GOOS_android
	MOVQ	$runtime·tls_g(SB), DX 	// arg 3: &tls_g
	// arg 4: TLS base, stored in slot 0 (Android's TLS_SLOT_SELF).
	// Compensate for tls_g (+16).
	MOVQ	-16(TLS), CX
#else
	MOVQ	$0, DX	// arg 3, 4: not used when using platform's TLS
	MOVQ	$0, CX
#endif
#ifdef GOOS_windows
	// Adjust for the Win64 calling convention.
	MOVQ	CX, R9 // arg 4
	MOVQ	DX, R8 // arg 3
	MOVQ	SI, DX // arg 2
	MOVQ	DI, CX // arg 1
#endif
	CALL	AX

	// update stackguard after _cgo_init
	MOVQ	$runtime·g0(SB), CX
	MOVQ	(g_stack+stack_lo)(CX), AX
	ADDQ	$const__StackGuard, AX
	MOVQ	AX, g_stackguard0(CX)
	MOVQ	AX, g_stackguard1(CX)

#ifndef GOOS_windows
	JMP ok
#endif
needtls:
#ifdef GOOS_plan9
	// skip TLS setup on Plan 9
	JMP ok
#endif
#ifdef GOOS_solaris
	// skip TLS setup on Solaris
	JMP ok
#endif
#ifdef GOOS_illumos
	// skip TLS setup on illumos
	JMP ok
#endif
#ifdef GOOS_darwin
	// skip TLS setup on Darwin
	JMP ok
#endif

	LEAQ	runtime·m0+m_tls(SB), DI
	CALL	runtime·settls(SB)

	// store through it, to make sure it works
	get_tls(BX)
	MOVQ	$0x123, g(BX)
	MOVQ	runtime·m0+m_tls(SB), AX
	CMPQ	AX, $0x123
	JEQ 2(PC)
	CALL	runtime·abort(SB)
ok:
	// set the per-goroutine and per-mach "registers"
	get_tls(BX)
	LEAQ	runtime·g0(SB), CX
	MOVQ	CX, g(BX)
	LEAQ	runtime·m0(SB), AX

	// save m->g0 = g0
	MOVQ	CX, m_g0(AX)
	// save m0 to g0->m
	MOVQ	AX, g_m(CX)

	CLD				// convention is D is always left cleared
	CALL	runtime·check(SB)

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	runtime·args(SB)
	CALL	runtime·osinit(SB)
	CALL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVQ	$runtime·mainPC(SB), AX		// entry
	PUSHQ	AX
	PUSHQ	$0			// arg size
	CALL	runtime·newproc(SB)
	POPQ	AX
	POPQ	AX

	// start this M
	CALL	runtime·mstart(SB)

	CALL	runtime·abort(SB)	// mstart should never return
	RET

	// Prevent dead-code elimination of debugCallV1, which is
	// intended to be called by debuggers.
	MOVQ	$runtime·debugCallV1(SB), AX
	RET

DATA	runtime·mainPC+0(SB)/8,$runtime·main(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT,$0-0
	BYTE	$0xcc
	RET

TEXT runtime·asminit(SB),NOSPLIT,$0-0
	// No per-thread init.
	RET

/*
 *  go-routine
 */

// func gosave(buf *gobuf)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT, $0-8
	MOVQ	buf+0(FP), AX		// gobuf
	LEAQ	buf+0(FP), BX		// caller's SP
	MOVQ	BX, gobuf_sp(AX)
	MOVQ	0(SP), BX		// caller's PC
	MOVQ	BX, gobuf_pc(AX)
	MOVQ	$0, gobuf_ret(AX)
	MOVQ	BP, gobuf_bp(AX)
	// Assert ctxt is zero. See func save.
	MOVQ	gobuf_ctxt(AX), BX
	TESTQ	BX, BX
	JZ	2(PC)
	CALL	runtime·badctxt(SB)
	get_tls(CX)
	MOVQ	g(CX), BX
	MOVQ	BX, gobuf_g(AX)
	RET

// func gogo(buf *gobuf)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $16-8
	MOVQ	buf+0(FP), BX		// gobuf
	MOVQ	gobuf_g(BX), DX
	MOVQ	0(DX), CX		// make sure g != nil
	get_tls(CX)
	MOVQ	DX, g(CX)
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_ret(BX), AX
	MOVQ	gobuf_ctxt(BX), DX
	MOVQ	gobuf_bp(BX), BP
	MOVQ	$0, gobuf_sp(BX)	// clear to help garbage collector
	MOVQ	$0, gobuf_ret(BX)
	MOVQ	$0, gobuf_ctxt(BX)
	MOVQ	$0, gobuf_bp(BX)
	MOVQ	gobuf_pc(BX), BX
	JMP	BX

// func mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $0-8
	MOVQ	fn+0(FP), DI

	get_tls(CX)
	MOVQ	g(CX), AX	// save state in g->sched
	MOVQ	0(SP), BX	// caller's PC
	MOVQ	BX, (g_sched+gobuf_pc)(AX)
	LEAQ	fn+0(FP), BX	// caller's SP
	MOVQ	BX, (g_sched+gobuf_sp)(AX)
	MOVQ	AX, (g_sched+gobuf_g)(AX)
	MOVQ	BP, (g_sched+gobuf_bp)(AX)

	// switch to m->g0 & its stack, call fn
	MOVQ	g(CX), BX
	MOVQ	g_m(BX), BX
	MOVQ	m_g0(BX), SI
	CMPQ	SI, AX	// if g == m->g0 call badmcall
	JNE	3(PC)
	MOVQ	$runtime·badmcall(SB), AX
	JMP	AX
	MOVQ	SI, g(CX)	// g = m->g0
	MOVQ	(g_sched+gobuf_sp)(SI), SP	// sp = m->g0->sched.sp
	PUSHQ	AX
	MOVQ	DI, DX
	MOVQ	0(DI), DI
	CALL	DI
	POPQ	AX
	MOVQ	$runtime·badmcall2(SB), AX
	JMP	AX
	RET

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack. We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	MOVQ	fn+0(FP), DI	// DI = fn
	get_tls(CX)
	MOVQ	g(CX), AX	// AX = g
	MOVQ	g_m(AX), BX	// BX = m

	CMPQ	AX, m_gsignal(BX)
	JEQ	noswitch

	MOVQ	m_g0(BX), DX	// DX = g0
	CMPQ	AX, DX
	JEQ	noswitch

	CMPQ	AX, m_curg(BX)
	JNE	bad

	// switch stacks
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVQ	$runtime·systemstack_switch(SB), SI
	MOVQ	SI, (g_sched+gobuf_pc)(AX)
	MOVQ	SP, (g_sched+gobuf_sp)(AX)
	MOVQ	AX, (g_sched+gobuf_g)(AX)
	MOVQ	BP, (g_sched+gobuf_bp)(AX)

	// switch to g0
	MOVQ	DX, g(CX)
	MOVQ	(g_sched+gobuf_sp)(DX), BX
	// make it look like mstart called systemstack on g0, to stop traceback
	SUBQ	$8, BX
	MOVQ	$runtime·mstart(SB), DX
	MOVQ	DX, 0(BX)
	MOVQ	BX, SP

	// call target function
	MOVQ	DI, DX
	MOVQ	0(DI), DI
	CALL	DI

	// switch back to g
	get_tls(CX)
	MOVQ	g(CX), AX
	MOVQ	g_m(AX), BX
	MOVQ	m_curg(BX), AX
	MOVQ	AX, g(CX)
	MOVQ	(g_sched+gobuf_sp)(AX), SP
	MOVQ	$0, (g_sched+gobuf_sp)(AX)
	RET

noswitch:
	// already on m stack; tail call the function
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVQ	DI, DX
	MOVQ	0(DI), DI
	JMP	DI

bad:
	// Bad: g is not gsignal, not g0, not curg. What is it?
	MOVQ	$runtime·badsystemstack(SB), AX
	CALL	AX
	INT	$3


/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT,$0-0
	// Cannot grow scheduler stack (m->g0).
	get_tls(CX)
	MOVQ	g(CX), BX
	MOVQ	g_m(BX), BX
	MOVQ	m_g0(BX), SI
	CMPQ	g(CX), SI
	JNE	3(PC)
	CALL	runtime·badmorestackg0(SB)
	CALL	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVQ	m_gsignal(BX), SI
	CMPQ	g(CX), SI
	JNE	3(PC)
	CALL	runtime·badmorestackgsignal(SB)
	CALL	runtime·abort(SB)

	// Called from f.
	// Set m->morebuf to f's caller.
	NOP	SP	// tell vet SP changed - stop checking offsets
	MOVQ	8(SP), AX	// f's caller's PC
	MOVQ	AX, (m_morebuf+gobuf_pc)(BX)
	LEAQ	16(SP), AX	// f's caller's SP
	MOVQ	AX, (m_morebuf+gobuf_sp)(BX)
	get_tls(CX)
	MOVQ	g(CX), SI
	MOVQ	SI, (m_morebuf+gobuf_g)(BX)

	// Set g->sched to context in f.
	MOVQ	0(SP), AX // f's PC
	MOVQ	AX, (g_sched+gobuf_pc)(SI)
	MOVQ	SI, (g_sched+gobuf_g)(SI)
	LEAQ	8(SP), AX // f's SP
	MOVQ	AX, (g_sched+gobuf_sp)(SI)
	MOVQ	BP, (g_sched+gobuf_bp)(SI)
	MOVQ	DX, (g_sched+gobuf_ctxt)(SI)

	// Call newstack on m->g0's stack.
	MOVQ	m_g0(BX), BX
	MOVQ	BX, g(CX)
	MOVQ	(g_sched+gobuf_sp)(BX), SP
	CALL	runtime·newstack(SB)
	CALL	runtime·abort(SB)	// crash if newstack returns
	RET

// morestack but not preserving ctxt.
TEXT runtime·morestack_noctxt(SB),NOSPLIT,$0
	MOVL	$0, DX
	JMP	runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMPQ	CX, $MAXSIZE;		\
	JA	3(PC);			\
	MOVQ	$NAME(SB), AX;		\
	JMP	AX
// Note: can't just "JMP NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT, $0-32
	MOVLQZX argsize+24(FP), CX
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
	MOVQ	$runtime·badreflectcall(SB), AX
	JMP	AX

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-32;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVQ	argptr+16(FP), SI;		\
	MOVLQZX argsize+24(FP), CX;		\
	MOVQ	SP, DI;				\
	REP;MOVSB;				\
	/* call function */			\
	MOVQ	f+8(FP), DX;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	CALL	(DX);				\
	/* copy return values back */		\
	MOVQ	argtype+0(FP), DX;		\
	MOVQ	argptr+16(FP), DI;		\
	MOVLQZX	argsize+24(FP), CX;		\
	MOVLQZX	retoffset+28(FP), BX;		\
	MOVQ	SP, SI;				\
	ADDQ	BX, DI;				\
	ADDQ	BX, SI;				\
	SUBQ	BX, CX;				\
	CALL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $32-0
	NO_LOCAL_POINTERS
	MOVQ	DX, 0(SP)
	MOVQ	DI, 8(SP)
	MOVQ	SI, 16(SP)
	MOVQ	CX, 24(SP)
	CALL	runtime·reflectcallmove(SB)
	RET

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
	MOVL	cycles+0(FP), AX
again:
	PAUSE
	SUBL	$1, AX
	JNZ	again
	RET


TEXT ·publicationBarrier(SB),NOSPLIT,$0-0
	// Stores are already ordered on x86, so this is just a
	// compile barrier.
	RET

// func jmpdefer(fv *funcval, argp uintptr)
// argp is a caller SP.
// called from deferreturn.
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT runtime·jmpdefer(SB), NOSPLIT, $0-16
	MOVQ	fv+0(FP), DX	// fn
	MOVQ	argp+8(FP), BX	// caller sp
	LEAQ	-8(BX), SP	// caller sp after CALL
	MOVQ	-8(SP), BP	// restore BP as if deferreturn returned (harmless if framepointers not in use)
	SUBQ	$5, (SP)	// return to CALL again
	MOVQ	0(DX), BX
	JMP	BX	// but first run the deferred function

// Save state of caller into g->sched. Smashes R8, R9.
TEXT gosave<>(SB),NOSPLIT,$0
	get_tls(R8)
	MOVQ	g(R8), R8
	MOVQ	0(SP), R9
	MOVQ	R9, (g_sched+gobuf_pc)(R8)
	LEAQ	8(SP), R9
	MOVQ	R9, (g_sched+gobuf_sp)(R8)
	MOVQ	$0, (g_sched+gobuf_ret)(R8)
	MOVQ	BP, (g_sched+gobuf_bp)(R8)
	// Assert ctxt is zero. See func save.
	MOVQ	(g_sched+gobuf_ctxt)(R8), R9
	TESTQ	R9, R9
	JZ	2(PC)
	CALL	runtime·badctxt(SB)
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	MOVQ	fn+0(FP), AX
	MOVQ	arg+8(FP), BX

	MOVQ	SP, DX

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	get_tls(CX)
	MOVQ	g(CX), R8
	CMPQ	R8, $0
	JEQ	nosave
	MOVQ	g_m(R8), R8
	MOVQ	m_g0(R8), SI
	MOVQ	g(CX), DI
	CMPQ	SI, DI
	JEQ	nosave
	MOVQ	m_gsignal(R8), SI
	CMPQ	SI, DI
	JEQ	nosave

	// Switch to system stack.
	MOVQ	m_g0(R8), SI
	CALL	gosave<>(SB)
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), SP

	// Now on a scheduling stack (a pthread-created stack).
	// Make sure we have enough room for 4 stack-backed fast-call
	// registers as per windows amd64 calling convention.
	SUBQ	$64, SP
	ANDQ	$~15, SP	// alignment for gcc ABI
	MOVQ	DI, 48(SP)	// save g
	MOVQ	(g_stack+stack_hi)(DI), DI
	SUBQ	DX, DI
	MOVQ	DI, 40(SP)	// save depth in stack (can't just save SP, as stack might be copied during a callback)
	MOVQ	BX, DI		// DI = first argument in AMD64 ABI
	MOVQ	BX, CX		// CX = first argument in Win64
	CALL	AX

	// Restore registers, g, stack pointer.
	get_tls(CX)
	MOVQ	48(SP), DI
	MOVQ	(g_stack+stack_hi)(DI), SI
	SUBQ	40(SP), SI
	MOVQ	DI, g(CX)
	MOVQ	SI, SP

	MOVL	AX, ret+16(FP)
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
	SUBQ	$64, SP
	ANDQ	$~15, SP
	MOVQ	$0, 48(SP)		// where above code stores g, in case someone looks during debugging
	MOVQ	DX, 40(SP)	// save original stack pointer
	MOVQ	BX, DI		// DI = first argument in AMD64 ABI
	MOVQ	BX, CX		// CX = first argument in Win64
	CALL	AX
	MOVQ	40(SP), SI	// restore original stack pointer
	MOVQ	SI, SP
	MOVL	AX, ret+16(FP)
	RET

// func cgocallback(fn, frame unsafe.Pointer, framesize, ctxt uintptr)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$32-32
	LEAQ	fn+0(FP), AX
	MOVQ	AX, 0(SP)
	MOVQ	frame+8(FP), AX
	MOVQ	AX, 8(SP)
	MOVQ	framesize+16(FP), AX
	MOVQ	AX, 16(SP)
	MOVQ	ctxt+24(FP), AX
	MOVQ	AX, 24(SP)
	MOVQ	$runtime·cgocallback_gofunc(SB), AX
	CALL	AX
	RET

// func cgocallback_gofunc(fn, frame, framesize, ctxt uintptr)
// See cgocall.go for more details.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$16-32
	NO_LOCAL_POINTERS

	// If g is nil, Go did not create the current thread.
	// Call needm to obtain one m for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call through AX.
	get_tls(CX)
#ifdef GOOS_windows
	MOVL	$0, BX
	CMPQ	CX, $0
	JEQ	2(PC)
#endif
	MOVQ	g(CX), BX
	CMPQ	BX, $0
	JEQ	needm
	MOVQ	g_m(BX), BX
	MOVQ	BX, R8 // holds oldm until end of function
	JMP	havem
needm:
	MOVQ	$0, 0(SP)
	MOVQ	$runtime·needm(SB), AX
	CALL	AX
	MOVQ	0(SP), R8
	get_tls(CX)
	MOVQ	g(CX), BX
	MOVQ	g_m(BX), BX

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
	MOVQ	m_g0(BX), SI
	MOVQ	SP, (g_sched+gobuf_sp)(SI)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 0(SP).
	MOVQ	m_g0(BX), SI
	MOVQ	(g_sched+gobuf_sp)(SI), AX
	MOVQ	AX, 0(SP)
	MOVQ	SP, (g_sched+gobuf_sp)(SI)

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
	// In the new goroutine, 8(SP) holds the saved R8.
	MOVQ	m_curg(BX), SI
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), DI  // prepare stack as DI
	MOVQ	(g_sched+gobuf_pc)(SI), BX
	MOVQ	BX, -8(DI)
	// Compute the size of the frame, including return PC and, if
	// GOEXPERIMENT=framepointer, the saved base pointer
	MOVQ	ctxt+24(FP), BX
	LEAQ	fv+0(FP), AX
	SUBQ	SP, AX
	SUBQ	AX, DI
	MOVQ	DI, SP

	MOVQ	R8, 8(SP)
	MOVQ	BX, 0(SP)
	CALL	runtime·cgocallbackg(SB)
	MOVQ	8(SP), R8

	// Compute the size of the frame again. FP and SP have
	// completely different values here than they did above,
	// but only their difference matters.
	LEAQ	fv+0(FP), AX
	SUBQ	SP, AX

	// Restore g->sched (== m->curg->sched) from saved values.
	get_tls(CX)
	MOVQ	g(CX), SI
	MOVQ	SP, DI
	ADDQ	AX, DI
	MOVQ	-8(DI), BX
	MOVQ	BX, (g_sched+gobuf_pc)(SI)
	MOVQ	DI, (g_sched+gobuf_sp)(SI)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVQ	g(CX), BX
	MOVQ	g_m(BX), BX
	MOVQ	m_g0(BX), SI
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), SP
	MOVQ	0(SP), AX
	MOVQ	AX, (g_sched+gobuf_sp)(SI)

	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	CMPQ	R8, $0
	JNE 3(PC)
	MOVQ	$runtime·dropm(SB), AX
	CALL	AX

	// Done!
	RET

// func setg(gg *g)
// set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-8
	MOVQ	gg+0(FP), BX
#ifdef GOOS_windows
	CMPQ	BX, $0
	JNE	settls
	MOVQ	$0, 0x28(GS)
	RET
settls:
	MOVQ	g_m(BX), AX
	LEAQ	m_tls(AX), AX
	MOVQ	AX, 0x28(GS)
#endif
	get_tls(CX)
	MOVQ	BX, g(CX)
	RET

// void setg_gcc(G*); set g called from gcc.
TEXT setg_gcc<>(SB),NOSPLIT,$0
	get_tls(AX)
	MOVQ	DI, g(AX)
	RET

TEXT runtime·abort(SB),NOSPLIT,$0-0
	INT	$3
loop:
	JMP	loop

// check that SP is in range [g->stack.lo, g->stack.hi)
TEXT runtime·stackcheck(SB), NOSPLIT, $0-0
	get_tls(CX)
	MOVQ	g(CX), AX
	CMPQ	(g_stack+stack_hi)(AX), SP
	JHI	2(PC)
	CALL	runtime·abort(SB)
	CMPQ	SP, (g_stack+stack_lo)(AX)
	JHI	2(PC)
	CALL	runtime·abort(SB)
	RET

// func cputicks() int64
TEXT runtime·cputicks(SB),NOSPLIT,$0-0
	CMPB	runtime·lfenceBeforeRdtsc(SB), $1
	JNE	mfence
	LFENCE
	JMP	done
mfence:
	MFENCE
done:
	RDTSC
	SHLQ	$32, DX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET

// func aeshash(p unsafe.Pointer, h, s uintptr) uintptr
// hash function using AES hardware instructions
TEXT runtime·aeshash(SB),NOSPLIT,$0-32
	MOVQ	p+0(FP), AX	// ptr to data
	MOVQ	s+16(FP), CX	// size
	LEAQ	ret+24(FP), DX
	JMP	aeshashbody<>(SB)

// func aeshashstr(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·aeshashstr(SB),NOSPLIT,$0-24
	MOVQ	p+0(FP), AX	// ptr to string struct
	MOVQ	8(AX), CX	// length of string
	MOVQ	(AX), AX	// string data
	LEAQ	ret+16(FP), DX
	JMP	aeshashbody<>(SB)

// AX: data
// CX: length
// DX: address to put return value
TEXT aeshashbody<>(SB),NOSPLIT,$0-0
	// Fill an SSE register with our seeds.
	MOVQ	h+8(FP), X0			// 64 bits of per-table hash seed
	PINSRW	$4, CX, X0			// 16 bits of length
	PSHUFHW $0, X0, X0			// repeat length 4 times total
	MOVO	X0, X1				// save unscrambled seed
	PXOR	runtime·aeskeysched(SB), X0	// xor in per-process seed
	AESENC	X0, X0				// scramble seed

	CMPQ	CX, $16
	JB	aes0to15
	JE	aes16
	CMPQ	CX, $32
	JBE	aes17to32
	CMPQ	CX, $64
	JBE	aes33to64
	CMPQ	CX, $128
	JBE	aes65to128
	JMP	aes129plus

aes0to15:
	TESTQ	CX, CX
	JE	aes0

	ADDQ	$16, AX
	TESTW	$0xff0, AX
	JE	endofpage

	// 16 bytes loaded at this address won't cross
	// a page boundary, so we can load it directly.
	MOVOU	-16(AX), X1
	ADDQ	CX, CX
	MOVQ	$masks<>(SB), AX
	PAND	(AX)(CX*8), X1
final1:
	PXOR	X0, X1	// xor data with seed
	AESENC	X1, X1	// scramble combo 3 times
	AESENC	X1, X1
	AESENC	X1, X1
	MOVQ	X1, (DX)
	RET

endofpage:
	// address ends in 1111xxxx. Might be up against
	// a page boundary, so load ending at last byte.
	// Then shift bytes down using pshufb.
	MOVOU	-32(AX)(CX*1), X1
	ADDQ	CX, CX
	MOVQ	$shifts<>(SB), AX
	PSHUFB	(AX)(CX*8), X1
	JMP	final1

aes0:
	// Return scrambled input seed
	AESENC	X0, X0
	MOVQ	X0, (DX)
	RET

aes16:
	MOVOU	(AX), X1
	JMP	final1

aes17to32:
	// make second starting seed
	PXOR	runtime·aeskeysched+16(SB), X1
	AESENC	X1, X1

	// load data to be hashed
	MOVOU	(AX), X2
	MOVOU	-16(AX)(CX*1), X3

	// xor with seed
	PXOR	X0, X2
	PXOR	X1, X3

	// scramble 3 times
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X2, X2
	AESENC	X3, X3

	// combine results
	PXOR	X3, X2
	MOVQ	X2, (DX)
	RET

aes33to64:
	// make 3 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	PXOR	runtime·aeskeysched+16(SB), X1
	PXOR	runtime·aeskeysched+32(SB), X2
	PXOR	runtime·aeskeysched+48(SB), X3
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3

	MOVOU	(AX), X4
	MOVOU	16(AX), X5
	MOVOU	-32(AX)(CX*1), X6
	MOVOU	-16(AX)(CX*1), X7

	PXOR	X0, X4
	PXOR	X1, X5
	PXOR	X2, X6
	PXOR	X3, X7

	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	PXOR	X6, X4
	PXOR	X7, X5
	PXOR	X5, X4
	MOVQ	X4, (DX)
	RET

aes65to128:
	// make 7 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	MOVO	X1, X4
	MOVO	X1, X5
	MOVO	X1, X6
	MOVO	X1, X7
	PXOR	runtime·aeskeysched+16(SB), X1
	PXOR	runtime·aeskeysched+32(SB), X2
	PXOR	runtime·aeskeysched+48(SB), X3
	PXOR	runtime·aeskeysched+64(SB), X4
	PXOR	runtime·aeskeysched+80(SB), X5
	PXOR	runtime·aeskeysched+96(SB), X6
	PXOR	runtime·aeskeysched+112(SB), X7
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	// load data
	MOVOU	(AX), X8
	MOVOU	16(AX), X9
	MOVOU	32(AX), X10
	MOVOU	48(AX), X11
	MOVOU	-64(AX)(CX*1), X12
	MOVOU	-48(AX)(CX*1), X13
	MOVOU	-32(AX)(CX*1), X14
	MOVOU	-16(AX)(CX*1), X15

	// xor with seed
	PXOR	X0, X8
	PXOR	X1, X9
	PXOR	X2, X10
	PXOR	X3, X11
	PXOR	X4, X12
	PXOR	X5, X13
	PXOR	X6, X14
	PXOR	X7, X15

	// scramble 3 times
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	// combine results
	PXOR	X12, X8
	PXOR	X13, X9
	PXOR	X14, X10
	PXOR	X15, X11
	PXOR	X10, X8
	PXOR	X11, X9
	PXOR	X9, X8
	MOVQ	X8, (DX)
	RET

aes129plus:
	// make 7 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	MOVO	X1, X4
	MOVO	X1, X5
	MOVO	X1, X6
	MOVO	X1, X7
	PXOR	runtime·aeskeysched+16(SB), X1
	PXOR	runtime·aeskeysched+32(SB), X2
	PXOR	runtime·aeskeysched+48(SB), X3
	PXOR	runtime·aeskeysched+64(SB), X4
	PXOR	runtime·aeskeysched+80(SB), X5
	PXOR	runtime·aeskeysched+96(SB), X6
	PXOR	runtime·aeskeysched+112(SB), X7
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	// start with last (possibly overlapping) block
	MOVOU	-128(AX)(CX*1), X8
	MOVOU	-112(AX)(CX*1), X9
	MOVOU	-96(AX)(CX*1), X10
	MOVOU	-80(AX)(CX*1), X11
	MOVOU	-64(AX)(CX*1), X12
	MOVOU	-48(AX)(CX*1), X13
	MOVOU	-32(AX)(CX*1), X14
	MOVOU	-16(AX)(CX*1), X15

	// xor in seed
	PXOR	X0, X8
	PXOR	X1, X9
	PXOR	X2, X10
	PXOR	X3, X11
	PXOR	X4, X12
	PXOR	X5, X13
	PXOR	X6, X14
	PXOR	X7, X15

	// compute number of remaining 128-byte blocks
	DECQ	CX
	SHRQ	$7, CX

aesloop:
	// scramble state
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	// scramble state, xor in a block
	MOVOU	(AX), X0
	MOVOU	16(AX), X1
	MOVOU	32(AX), X2
	MOVOU	48(AX), X3
	AESENC	X0, X8
	AESENC	X1, X9
	AESENC	X2, X10
	AESENC	X3, X11
	MOVOU	64(AX), X4
	MOVOU	80(AX), X5
	MOVOU	96(AX), X6
	MOVOU	112(AX), X7
	AESENC	X4, X12
	AESENC	X5, X13
	AESENC	X6, X14
	AESENC	X7, X15

	ADDQ	$128, AX
	DECQ	CX
	JNE	aesloop

	// 3 more scrambles to finish
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	PXOR	X12, X8
	PXOR	X13, X9
	PXOR	X14, X10
	PXOR	X15, X11
	PXOR	X10, X8
	PXOR	X11, X9
	PXOR	X9, X8
	MOVQ	X8, (DX)
	RET

// func aeshash32(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·aeshash32(SB),NOSPLIT,$0-24
	MOVQ	p+0(FP), AX	// ptr to data
	MOVQ	h+8(FP), X0	// seed
	PINSRD	$2, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+32(SB), X0
	MOVQ	X0, ret+16(FP)
	RET

// func aeshash64(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·aeshash64(SB),NOSPLIT,$0-24
	MOVQ	p+0(FP), AX	// ptr to data
	MOVQ	h+8(FP), X0	// seed
	PINSRQ	$1, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+32(SB), X0
	MOVQ	X0, ret+16(FP)
	RET

// simple mask to get rid of data in the high part of the register.
DATA masks<>+0x00(SB)/8, $0x0000000000000000
DATA masks<>+0x08(SB)/8, $0x0000000000000000
DATA masks<>+0x10(SB)/8, $0x00000000000000ff
DATA masks<>+0x18(SB)/8, $0x0000000000000000
DATA masks<>+0x20(SB)/8, $0x000000000000ffff
DATA masks<>+0x28(SB)/8, $0x0000000000000000
DATA masks<>+0x30(SB)/8, $0x0000000000ffffff
DATA masks<>+0x38(SB)/8, $0x0000000000000000
DATA masks<>+0x40(SB)/8, $0x00000000ffffffff
DATA masks<>+0x48(SB)/8, $0x0000000000000000
DATA masks<>+0x50(SB)/8, $0x000000ffffffffff
DATA masks<>+0x58(SB)/8, $0x0000000000000000
DATA masks<>+0x60(SB)/8, $0x0000ffffffffffff
DATA masks<>+0x68(SB)/8, $0x0000000000000000
DATA masks<>+0x70(SB)/8, $0x00ffffffffffffff
DATA masks<>+0x78(SB)/8, $0x0000000000000000
DATA masks<>+0x80(SB)/8, $0xffffffffffffffff
DATA masks<>+0x88(SB)/8, $0x0000000000000000
DATA masks<>+0x90(SB)/8, $0xffffffffffffffff
DATA masks<>+0x98(SB)/8, $0x00000000000000ff
DATA masks<>+0xa0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xa8(SB)/8, $0x000000000000ffff
DATA masks<>+0xb0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xb8(SB)/8, $0x0000000000ffffff
DATA masks<>+0xc0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xc8(SB)/8, $0x00000000ffffffff
DATA masks<>+0xd0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xd8(SB)/8, $0x000000ffffffffff
DATA masks<>+0xe0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xe8(SB)/8, $0x0000ffffffffffff
DATA masks<>+0xf0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xf8(SB)/8, $0x00ffffffffffffff
GLOBL masks<>(SB),RODATA,$256

// func checkASM() bool
TEXT ·checkASM(SB),NOSPLIT,$0-1
	// check that masks<>(SB) and shifts<>(SB) are aligned to 16-byte
	MOVQ	$masks<>(SB), AX
	MOVQ	$shifts<>(SB), BX
	ORQ	BX, AX
	TESTQ	$15, AX
	SETEQ	ret+0(FP)
	RET

// these are arguments to pshufb. They move data down from
// the high bytes of the register to the low bytes of the register.
// index is how many bytes to move.
DATA shifts<>+0x00(SB)/8, $0x0000000000000000
DATA shifts<>+0x08(SB)/8, $0x0000000000000000
DATA shifts<>+0x10(SB)/8, $0xffffffffffffff0f
DATA shifts<>+0x18(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x20(SB)/8, $0xffffffffffff0f0e
DATA shifts<>+0x28(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x30(SB)/8, $0xffffffffff0f0e0d
DATA shifts<>+0x38(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x40(SB)/8, $0xffffffff0f0e0d0c
DATA shifts<>+0x48(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x50(SB)/8, $0xffffff0f0e0d0c0b
DATA shifts<>+0x58(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x60(SB)/8, $0xffff0f0e0d0c0b0a
DATA shifts<>+0x68(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x70(SB)/8, $0xff0f0e0d0c0b0a09
DATA shifts<>+0x78(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x80(SB)/8, $0x0f0e0d0c0b0a0908
DATA shifts<>+0x88(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x90(SB)/8, $0x0e0d0c0b0a090807
DATA shifts<>+0x98(SB)/8, $0xffffffffffffff0f
DATA shifts<>+0xa0(SB)/8, $0x0d0c0b0a09080706
DATA shifts<>+0xa8(SB)/8, $0xffffffffffff0f0e
DATA shifts<>+0xb0(SB)/8, $0x0c0b0a0908070605
DATA shifts<>+0xb8(SB)/8, $0xffffffffff0f0e0d
DATA shifts<>+0xc0(SB)/8, $0x0b0a090807060504
DATA shifts<>+0xc8(SB)/8, $0xffffffff0f0e0d0c
DATA shifts<>+0xd0(SB)/8, $0x0a09080706050403
DATA shifts<>+0xd8(SB)/8, $0xffffff0f0e0d0c0b
DATA shifts<>+0xe0(SB)/8, $0x0908070605040302
DATA shifts<>+0xe8(SB)/8, $0xffff0f0e0d0c0b0a
DATA shifts<>+0xf0(SB)/8, $0x0807060504030201
DATA shifts<>+0xf8(SB)/8, $0xff0f0e0d0c0b0a09
GLOBL shifts<>(SB),RODATA,$256

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVL	$0, AX
	RET


// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$0
	get_tls(CX)
	MOVQ	g(CX), AX
	MOVQ	g_m(AX), AX
	MOVQ	m_curg(AX), AX
	MOVQ	(g_stack+stack_hi)(AX), AX
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT,$0-0
	BYTE	$0x90	// NOP
	CALL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	BYTE	$0x90	// NOP

// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-0
	PUSHQ	R15 // The access to global variables below implicitly uses R15, which is callee-save
	MOVQ	runtime·lastmoduledatap(SB), AX
	MOVQ	DI, moduledata_next(AX)
	MOVQ	DI, runtime·lastmoduledatap(SB)
	POPQ	R15
	RET

// gcWriteBarrier performs a heap pointer write and informs the GC.
//
// gcWriteBarrier does NOT follow the Go ABI. It takes two arguments:
// - DI is the destination of the write
// - AX is the value being written at DI
// It clobbers FLAGS. It does not clobber any general-purpose registers,
// but may clobber others (e.g., SSE registers).
TEXT runtime·gcWriteBarrier(SB),NOSPLIT,$120
	// Save the registers clobbered by the fast path. This is slightly
	// faster than having the caller spill these.
	MOVQ	R14, 104(SP)
	MOVQ	R13, 112(SP)
	// TODO: Consider passing g.m.p in as an argument so they can be shared
	// across a sequence of write barriers.
	get_tls(R13)
	MOVQ	g(R13), R13
	MOVQ	g_m(R13), R13
	MOVQ	m_p(R13), R13
	MOVQ	(p_wbBuf+wbBuf_next)(R13), R14
	// Increment wbBuf.next position.
	LEAQ	16(R14), R14
	MOVQ	R14, (p_wbBuf+wbBuf_next)(R13)
	CMPQ	R14, (p_wbBuf+wbBuf_end)(R13)
	// Record the write.
	MOVQ	AX, -16(R14)	// Record value
	// Note: This turns bad pointer writes into bad
	// pointer reads, which could be confusing. We could avoid
	// reading from obviously bad pointers, which would
	// take care of the vast majority of these. We could
	// patch this up in the signal handler, or use XCHG to
	// combine the read and the write.
	MOVQ	(DI), R13
	MOVQ	R13, -8(R14)	// Record *slot
	// Is the buffer full? (flags set in CMPQ above)
	JEQ	flush
ret:
	MOVQ	104(SP), R14
	MOVQ	112(SP), R13
	// Do the write.
	MOVQ	AX, (DI)
	RET

flush:
	// Save all general purpose registers since these could be
	// clobbered by wbBufFlush and were not saved by the caller.
	// It is possible for wbBufFlush to clobber other registers
	// (e.g., SSE registers), but the compiler takes care of saving
	// those in the caller if necessary. This strikes a balance
	// with registers that are likely to be used.
	//
	// We don't have type information for these, but all code under
	// here is NOSPLIT, so nothing will observe these.
	//
	// TODO: We could strike a different balance; e.g., saving X0
	// and not saving GP registers that are less likely to be used.
	MOVQ	DI, 0(SP)	// Also first argument to wbBufFlush
	MOVQ	AX, 8(SP)	// Also second argument to wbBufFlush
	MOVQ	BX, 16(SP)
	MOVQ	CX, 24(SP)
	MOVQ	DX, 32(SP)
	// DI already saved
	MOVQ	SI, 40(SP)
	MOVQ	BP, 48(SP)
	MOVQ	R8, 56(SP)
	MOVQ	R9, 64(SP)
	MOVQ	R10, 72(SP)
	MOVQ	R11, 80(SP)
	MOVQ	R12, 88(SP)
	// R13 already saved
	// R14 already saved
	MOVQ	R15, 96(SP)

	// This takes arguments DI and AX
	CALL	runtime·wbBufFlush(SB)

	MOVQ	0(SP), DI
	MOVQ	8(SP), AX
	MOVQ	16(SP), BX
	MOVQ	24(SP), CX
	MOVQ	32(SP), DX
	MOVQ	40(SP), SI
	MOVQ	48(SP), BP
	MOVQ	56(SP), R8
	MOVQ	64(SP), R9
	MOVQ	72(SP), R10
	MOVQ	80(SP), R11
	MOVQ	88(SP), R12
	MOVQ	96(SP), R15
	JMP	ret

DATA	debugCallFrameTooLarge<>+0x00(SB)/20, $"call frame too large"
GLOBL	debugCallFrameTooLarge<>(SB), RODATA, $20	// Size duplicated below

// debugCallV1 is the entry point for debugger-injected function
// calls on running goroutines. It informs the runtime that a
// debug call has been injected and creates a call frame for the
// debugger to fill in.
//
// To inject a function call, a debugger should:
// 1. Check that the goroutine is in state _Grunning and that
//    there are at least 256 bytes free on the stack.
// 2. Push the current PC on the stack (updating SP).
// 3. Write the desired argument frame size at SP-16 (using the SP
//    after step 2).
// 4. Save all machine registers (including flags and XMM reigsters)
//    so they can be restored later by the debugger.
// 5. Set the PC to debugCallV1 and resume execution.
//
// If the goroutine is in state _Grunnable, then it's not generally
// safe to inject a call because it may return out via other runtime
// operations. Instead, the debugger should unwind the stack to find
// the return to non-runtime code, add a temporary breakpoint there,
// and inject the call once that breakpoint is hit.
//
// If the goroutine is in any other state, it's not safe to inject a call.
//
// This function communicates back to the debugger by setting RAX and
// invoking INT3 to raise a breakpoint signal. See the comments in the
// implementation for the protocol the debugger is expected to
// follow. InjectDebugCall in the runtime tests demonstrates this protocol.
//
// The debugger must ensure that any pointers passed to the function
// obey escape analysis requirements. Specifically, it must not pass
// a stack pointer to an escaping argument. debugCallV1 cannot check
// this invariant.
TEXT runtime·debugCallV1(SB),NOSPLIT,$152-0
	// Save all registers that may contain pointers in GC register
	// map order (see ssa.registersAMD64). This makes it possible
	// to copy the stack while updating pointers currently held in
	// registers, and for the GC to find roots in registers.
	//
	// We can't do anything that might clobber any of these
	// registers before this.
	MOVQ	R15, r15-(14*8+8)(SP)
	MOVQ	R14, r14-(13*8+8)(SP)
	MOVQ	R13, r13-(12*8+8)(SP)
	MOVQ	R12, r12-(11*8+8)(SP)
	MOVQ	R11, r11-(10*8+8)(SP)
	MOVQ	R10, r10-(9*8+8)(SP)
	MOVQ	R9, r9-(8*8+8)(SP)
	MOVQ	R8, r8-(7*8+8)(SP)
	MOVQ	DI, di-(6*8+8)(SP)
	MOVQ	SI, si-(5*8+8)(SP)
	MOVQ	BP, bp-(4*8+8)(SP)
	MOVQ	BX, bx-(3*8+8)(SP)
	MOVQ	DX, dx-(2*8+8)(SP)
	// Save the frame size before we clobber it. Either of the last
	// saves could clobber this depending on whether there's a saved BP.
	MOVQ	frameSize-24(FP), DX	// aka -16(RSP) before prologue
	MOVQ	CX, cx-(1*8+8)(SP)
	MOVQ	AX, ax-(0*8+8)(SP)

	// Save the argument frame size.
	MOVQ	DX, frameSize-128(SP)

	// Perform a safe-point check.
	MOVQ	retpc-8(FP), AX	// Caller's PC
	MOVQ	AX, 0(SP)
	CALL	runtime·debugCallCheck(SB)
	MOVQ	8(SP), AX
	TESTQ	AX, AX
	JZ	good
	// The safety check failed. Put the reason string at the top
	// of the stack.
	MOVQ	AX, 0(SP)
	MOVQ	16(SP), AX
	MOVQ	AX, 8(SP)
	// Set AX to 8 and invoke INT3. The debugger should get the
	// reason a call can't be injected from the top of the stack
	// and resume execution.
	MOVQ	$8, AX
	BYTE	$0xcc
	JMP	restore

good:
	// Registers are saved and it's safe to make a call.
	// Open up a call frame, moving the stack if necessary.
	//
	// Once the frame is allocated, this will set AX to 0 and
	// invoke INT3. The debugger should write the argument
	// frame for the call at SP, push the trapping PC on the
	// stack, set the PC to the function to call, set RCX to point
	// to the closure (if a closure call), and resume execution.
	//
	// If the function returns, this will set AX to 1 and invoke
	// INT3. The debugger can then inspect any return value saved
	// on the stack at SP and resume execution again.
	//
	// If the function panics, this will set AX to 2 and invoke INT3.
	// The interface{} value of the panic will be at SP. The debugger
	// can inspect the panic value and resume execution again.
#define DEBUG_CALL_DISPATCH(NAME,MAXSIZE)	\
	CMPQ	AX, $MAXSIZE;			\
	JA	5(PC);				\
	MOVQ	$NAME(SB), AX;			\
	MOVQ	AX, 0(SP);			\
	CALL	runtime·debugCallWrap(SB);	\
	JMP	restore

	MOVQ	frameSize-128(SP), AX
	DEBUG_CALL_DISPATCH(debugCall32<>, 32)
	DEBUG_CALL_DISPATCH(debugCall64<>, 64)
	DEBUG_CALL_DISPATCH(debugCall128<>, 128)
	DEBUG_CALL_DISPATCH(debugCall256<>, 256)
	DEBUG_CALL_DISPATCH(debugCall512<>, 512)
	DEBUG_CALL_DISPATCH(debugCall1024<>, 1024)
	DEBUG_CALL_DISPATCH(debugCall2048<>, 2048)
	DEBUG_CALL_DISPATCH(debugCall4096<>, 4096)
	DEBUG_CALL_DISPATCH(debugCall8192<>, 8192)
	DEBUG_CALL_DISPATCH(debugCall16384<>, 16384)
	DEBUG_CALL_DISPATCH(debugCall32768<>, 32768)
	DEBUG_CALL_DISPATCH(debugCall65536<>, 65536)
	// The frame size is too large. Report the error.
	MOVQ	$debugCallFrameTooLarge<>(SB), AX
	MOVQ	AX, 0(SP)
	MOVQ	$20, 8(SP) // length of debugCallFrameTooLarge string
	MOVQ	$8, AX
	BYTE	$0xcc
	JMP	restore

restore:
	// Calls and failures resume here.
	//
	// Set AX to 16 and invoke INT3. The debugger should restore
	// all registers except RIP and RSP and resume execution.
	MOVQ	$16, AX
	BYTE	$0xcc
	// We must not modify flags after this point.

	// Restore pointer-containing registers, which may have been
	// modified from the debugger's copy by stack copying.
	MOVQ	ax-(0*8+8)(SP), AX
	MOVQ	cx-(1*8+8)(SP), CX
	MOVQ	dx-(2*8+8)(SP), DX
	MOVQ	bx-(3*8+8)(SP), BX
	MOVQ	bp-(4*8+8)(SP), BP
	MOVQ	si-(5*8+8)(SP), SI
	MOVQ	di-(6*8+8)(SP), DI
	MOVQ	r8-(7*8+8)(SP), R8
	MOVQ	r9-(8*8+8)(SP), R9
	MOVQ	r10-(9*8+8)(SP), R10
	MOVQ	r11-(10*8+8)(SP), R11
	MOVQ	r12-(11*8+8)(SP), R12
	MOVQ	r13-(12*8+8)(SP), R13
	MOVQ	r14-(13*8+8)(SP), R14
	MOVQ	r15-(14*8+8)(SP), R15

	RET

// runtime.debugCallCheck assumes that functions defined with the
// DEBUG_CALL_FN macro are safe points to inject calls.
#define DEBUG_CALL_FN(NAME,MAXSIZE)		\
TEXT NAME(SB),WRAPPER,$MAXSIZE-0;		\
	NO_LOCAL_POINTERS;			\
	MOVQ	$0, AX;				\
	BYTE	$0xcc;				\
	MOVQ	$1, AX;				\
	BYTE	$0xcc;				\
	RET
DEBUG_CALL_FN(debugCall32<>, 32)
DEBUG_CALL_FN(debugCall64<>, 64)
DEBUG_CALL_FN(debugCall128<>, 128)
DEBUG_CALL_FN(debugCall256<>, 256)
DEBUG_CALL_FN(debugCall512<>, 512)
DEBUG_CALL_FN(debugCall1024<>, 1024)
DEBUG_CALL_FN(debugCall2048<>, 2048)
DEBUG_CALL_FN(debugCall4096<>, 4096)
DEBUG_CALL_FN(debugCall8192<>, 8192)
DEBUG_CALL_FN(debugCall16384<>, 16384)
DEBUG_CALL_FN(debugCall32768<>, 32768)
DEBUG_CALL_FN(debugCall65536<>, 65536)

// func debugCallPanicked(val interface{})
TEXT runtime·debugCallPanicked(SB),NOSPLIT,$16-16
	// Copy the panic value to the top of stack.
	MOVQ	val_type+0(FP), AX
	MOVQ	AX, 0(SP)
	MOVQ	val_data+8(FP), AX
	MOVQ	AX, 8(SP)
	MOVQ	$2, AX
	BYTE	$0xcc
	RET

// Note: these functions use a special calling convention to save generated code space.
// Arguments are passed in registers, but the space for those arguments are allocated
// in the caller's stack frame. These stubs write the args into that stack space and
// then tail call to the corresponding runtime handler.
// The tail call makes these stubs disappear in backtraces.
TEXT runtime·panicIndex(SB),NOSPLIT,$0-16
	MOVQ	AX, x+0(FP)
	MOVQ	CX, y+8(FP)
	JMP	runtime·goPanicIndex(SB)
TEXT runtime·panicIndexU(SB),NOSPLIT,$0-16
	MOVQ	AX, x+0(FP)
	MOVQ	CX, y+8(FP)
	JMP	runtime·goPanicIndexU(SB)
TEXT runtime·panicSliceAlen(SB),NOSPLIT,$0-16
	MOVQ	CX, x+0(FP)
	MOVQ	DX, y+8(FP)
	JMP	runtime·goPanicSliceAlen(SB)
TEXT runtime·panicSliceAlenU(SB),NOSPLIT,$0-16
	MOVQ	CX, x+0(FP)
	MOVQ	DX, y+8(FP)
	JMP	runtime·goPanicSliceAlenU(SB)
TEXT runtime·panicSliceAcap(SB),NOSPLIT,$0-16
	MOVQ	CX, x+0(FP)
	MOVQ	DX, y+8(FP)
	JMP	runtime·goPanicSliceAcap(SB)
TEXT runtime·panicSliceAcapU(SB),NOSPLIT,$0-16
	MOVQ	CX, x+0(FP)
	MOVQ	DX, y+8(FP)
	JMP	runtime·goPanicSliceAcapU(SB)
TEXT runtime·panicSliceB(SB),NOSPLIT,$0-16
	MOVQ	AX, x+0(FP)
	MOVQ	CX, y+8(FP)
	JMP	runtime·goPanicSliceB(SB)
TEXT runtime·panicSliceBU(SB),NOSPLIT,$0-16
	MOVQ	AX, x+0(FP)
	MOVQ	CX, y+8(FP)
	JMP	runtime·goPanicSliceBU(SB)
TEXT runtime·panicSlice3Alen(SB),NOSPLIT,$0-16
	MOVQ	DX, x+0(FP)
	MOVQ	BX, y+8(FP)
	JMP	runtime·goPanicSlice3Alen(SB)
TEXT runtime·panicSlice3AlenU(SB),NOSPLIT,$0-16
	MOVQ	DX, x+0(FP)
	MOVQ	BX, y+8(FP)
	JMP	runtime·goPanicSlice3AlenU(SB)
TEXT runtime·panicSlice3Acap(SB),NOSPLIT,$0-16
	MOVQ	DX, x+0(FP)
	MOVQ	BX, y+8(FP)
	JMP	runtime·goPanicSlice3Acap(SB)
TEXT runtime·panicSlice3AcapU(SB),NOSPLIT,$0-16
	MOVQ	DX, x+0(FP)
	MOVQ	BX, y+8(FP)
	JMP	runtime·goPanicSlice3AcapU(SB)
TEXT runtime·panicSlice3B(SB),NOSPLIT,$0-16
	MOVQ	CX, x+0(FP)
	MOVQ	DX, y+8(FP)
	JMP	runtime·goPanicSlice3B(SB)
TEXT runtime·panicSlice3BU(SB),NOSPLIT,$0-16
	MOVQ	CX, x+0(FP)
	MOVQ	DX, y+8(FP)
	JMP	runtime·goPanicSlice3BU(SB)
TEXT runtime·panicSlice3C(SB),NOSPLIT,$0-16
	MOVQ	AX, x+0(FP)
	MOVQ	CX, y+8(FP)
	JMP	runtime·goPanicSlice3C(SB)
TEXT runtime·panicSlice3CU(SB),NOSPLIT,$0-16
	MOVQ	AX, x+0(FP)
	MOVQ	CX, y+8(FP)
	JMP	runtime·goPanicSlice3CU(SB)

#ifdef GOOS_android
// Use the free TLS_SLOT_APP slot #2 on Android Q.
// Earlier androids are set up in gcc_android.c.
DATA runtime·tls_g+0(SB)/8, $16
GLOBL runtime·tls_g+0(SB), NOPTR, $8
#endif
