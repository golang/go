// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

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

	TESTL	$(1<<26), DX // SSE2
	SETNE	runtime·support_sse2(SB)

	TESTL	$(1<<9), CX // SSSE3
	SETNE	runtime·support_ssse3(SB)

	TESTL	$(1<<19), CX // SSE4.1
	SETNE	runtime·support_sse41(SB)

	TESTL	$(1<<20), CX // SSE4.2
	SETNE	runtime·support_sse42(SB)

	TESTL	$(1<<23), CX // POPCNT
	SETNE	runtime·support_popcnt(SB)

	TESTL	$(1<<25), CX // AES
	SETNE	runtime·support_aes(SB)

	TESTL	$(1<<27), CX // OSXSAVE
	SETNE	runtime·support_osxsave(SB)

	// If OS support for XMM and YMM is not present
	// support_avx will be set back to false later.
	TESTL	$(1<<28), CX // AVX
	SETNE	runtime·support_avx(SB)

eax7:
	// Load EAX=7/ECX=0 cpuid flags
	CMPL	SI, $7
	JLT	osavx
	MOVL	$7, AX
	MOVL	$0, CX
	CPUID

	TESTL	$(1<<3), BX // BMI1
	SETNE	runtime·support_bmi1(SB)

	// If OS support for XMM and YMM is not present
	// support_avx2 will be set back to false later.
	TESTL	$(1<<5), BX
	SETNE	runtime·support_avx2(SB)

	TESTL	$(1<<8), BX // BMI2
	SETNE	runtime·support_bmi2(SB)

	TESTL	$(1<<9), BX // ERMS
	SETNE	runtime·support_erms(SB)

osavx:
	CMPB	runtime·support_osxsave(SB), $1
	JNE	noavx
	MOVL	$0, CX
	// For XGETBV, OSXSAVE bit is required and sufficient
	XGETBV
	ANDL	$6, AX
	CMPL	AX, $6 // Check for OS support of XMM and YMM registers.
	JE nocpuinfo
noavx:
	MOVB $0, runtime·support_avx(SB)
	MOVB $0, runtime·support_avx2(SB)

nocpuinfo:
	// if there is an _cgo_init, call it.
	MOVQ	_cgo_init(SB), AX
	TESTQ	AX, AX
	JZ	needtls
	// g0 already in DI
	MOVQ	DI, CX	// Win64 uses CX for first parameter
	MOVQ	$setg_gcc<>(SB), SI
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

	LEAQ	runtime·m0+m_tls(SB), DI
	CALL	runtime·settls(SB)

	// store through it, to make sure it works
	get_tls(BX)
	MOVQ	$0x123, g(BX)
	MOVQ	runtime·m0+m_tls(SB), AX
	CMPQ	AX, $0x123
	JEQ 2(PC)
	MOVL	AX, 0	// abort
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

	MOVL	$0xf1, 0xf1  // crash
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

// void gosave(Gobuf*)
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

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $16-8
	MOVQ	buf+0(FP), BX		// gobuf

	// If ctxt is not nil, invoke deletion barrier before overwriting.
	MOVQ	gobuf_ctxt(BX), AX
	TESTQ	AX, AX
	JZ	nilctxt
	LEAQ	gobuf_ctxt(BX), AX
	MOVQ	AX, 0(SP)
	MOVQ	$0, 8(SP)
	CALL	runtime·writebarrierptr_prewrite(SB)
	MOVQ	buf+0(FP), BX

nilctxt:
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

	MOVQ	m_gsignal(BX), DX	// DX = gsignal
	CMPQ	AX, DX
	JEQ	noswitch

	MOVQ	m_g0(BX), DX	// DX = g0
	CMPQ	AX, DX
	JEQ	noswitch

	MOVQ	m_curg(BX), R8
	CMPQ	AX, R8
	JEQ	switch
	
	// Bad: g is not gsignal, not g0, not curg. What is it?
	MOVQ	$runtime·badsystemstack(SB), AX
	CALL	AX

switch:
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
	// already on m stack, just call directly
	MOVQ	DI, DX
	MOVQ	0(DI), DI
	CALL	DI
	RET

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
	INT	$3

	// Cannot grow signal stack (m->gsignal).
	MOVQ	m_gsignal(BX), SI
	CMPQ	g(CX), SI
	JNE	3(PC)
	CALL	runtime·badmorestackgsignal(SB)
	INT	$3

	// Called from f.
	// Set m->morebuf to f's caller.
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
	// newstack will fill gobuf.ctxt.

	// Call newstack on m->g0's stack.
	MOVQ	m_g0(BX), BX
	MOVQ	BX, g(CX)
	MOVQ	(g_sched+gobuf_sp)(BX), SP
	PUSHQ	DX	// ctxt argument
	CALL	runtime·newstack(SB)
	MOVQ	$0, 0x1003	// crash if newstack returns
	POPQ	DX	// keep balance check happy
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

TEXT reflect·call(SB), NOSPLIT, $0-0
	JMP	·reflectcall(SB)

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

// void jmpdefer(fn, sp);
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

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize, uintptr ctxt)
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

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize, uintptr ctxt)
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

// void setg(G*); set g. for use by needm.
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

// check that SP is in range [g->stack.lo, g->stack.hi)
TEXT runtime·stackcheck(SB), NOSPLIT, $0-0
	get_tls(CX)
	MOVQ	g(CX), AX
	CMPQ	(g_stack+stack_hi)(AX), SP
	JHI	2(PC)
	INT	$3
	CMPQ	SP, (g_stack+stack_lo)(AX)
	JHI	2(PC)
	INT	$3
	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$8-16
	MOVQ	argp+0(FP),AX		// addr of first arg
	MOVQ	-8(AX),AX		// get calling pc
	MOVQ	AX, ret+8(FP)
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

// memhash_varlen(p unsafe.Pointer, h seed) uintptr
// redirects to memhash(p, h, size) using the size
// stored in the closure.
TEXT runtime·memhash_varlen(SB),NOSPLIT,$32-24
	GO_ARGS
	NO_LOCAL_POINTERS
	MOVQ	p+0(FP), AX
	MOVQ	h+8(FP), BX
	MOVQ	8(DX), CX
	MOVQ	AX, 0(SP)
	MOVQ	BX, 8(SP)
	MOVQ	CX, 16(SP)
	CALL	runtime·memhash(SB)
	MOVQ	24(SP), AX
	MOVQ	AX, ret+16(FP)
	RET

// hash function using AES hardware instructions
TEXT runtime·aeshash(SB),NOSPLIT,$0-32
	MOVQ	p+0(FP), AX	// ptr to data
	MOVQ	s+16(FP), CX	// size
	LEAQ	ret+24(FP), DX
	JMP	runtime·aeshashbody(SB)

TEXT runtime·aeshashstr(SB),NOSPLIT,$0-24
	MOVQ	p+0(FP), AX	// ptr to string struct
	MOVQ	8(AX), CX	// length of string
	MOVQ	(AX), AX	// string data
	LEAQ	ret+16(FP), DX
	JMP	runtime·aeshashbody(SB)

// AX: data
// CX: length
// DX: address to put return value
TEXT runtime·aeshashbody(SB),NOSPLIT,$0-0
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
	
TEXT runtime·aeshash32(SB),NOSPLIT,$0-24
	MOVQ	p+0(FP), AX	// ptr to data
	MOVQ	h+8(FP), X0	// seed
	PINSRD	$2, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+32(SB), X0
	MOVQ	X0, ret+16(FP)
	RET

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

// memequal(p, q unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-25
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	MOVQ	size+16(FP), BX
	LEAQ	ret+24(FP), AX
	JMP	runtime·memeqbody(SB)
eq:
	MOVB	$1, ret+24(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$0-17
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	MOVQ	8(DX), BX    // compiler stores size at offset 8 in the closure
	LEAQ	ret+16(FP), AX
	JMP	runtime·memeqbody(SB)
eq:
	MOVB	$1, ret+16(FP)
	RET

// eqstring tests whether two strings are equal.
// The compiler guarantees that strings passed
// to eqstring have equal length.
// See runtime_test.go:eqstring_generic for
// equivalent Go code.
TEXT runtime·eqstring(SB),NOSPLIT,$0-33
	MOVQ	s1_base+0(FP), SI
	MOVQ	s2_base+16(FP), DI
	CMPQ	SI, DI
	JEQ	eq
	MOVQ	s1_len+8(FP), BX
	LEAQ	ret+32(FP), AX
	JMP	runtime·memeqbody(SB)
eq:
	MOVB	$1, ret+32(FP)
	RET

// a in SI
// b in DI
// count in BX
// address of result byte in AX
TEXT runtime·memeqbody(SB),NOSPLIT,$0-0
	CMPQ	BX, $8
	JB	small
	CMPQ	BX, $64
	JB	bigloop
	CMPB    runtime·support_avx2(SB), $1
	JE	hugeloop_avx2
	
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
	MOVB	$0, (AX)
	RET

	// 64 bytes at a time using ymm registers
hugeloop_avx2:
	CMPQ	BX, $64
	JB	bigloop_avx2
	VMOVDQU	(SI), Y0
	VMOVDQU	(DI), Y1
	VMOVDQU	32(SI), Y2
	VMOVDQU	32(DI), Y3
	VPCMPEQB	Y1, Y0, Y4
	VPCMPEQB	Y2, Y3, Y5
	VPAND	Y4, Y5, Y6
	VPMOVMSKB Y6, DX
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$64, BX
	CMPL	DX, $0xffffffff
	JEQ	hugeloop_avx2
	VZEROUPPER
	MOVB	$0, (AX)
	RET

bigloop_avx2:
	VZEROUPPER

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
	MOVB	$0, (AX)
	RET

	// remaining 0-8 bytes
leftover:
	MOVQ	-8(SI)(BX*1), CX
	MOVQ	-8(DI)(BX*1), DX
	CMPQ	CX, DX
	SETEQ	(AX)
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
	// address ends in 11111xxx. Load up to bytes we want, move to correct position.
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
	SETEQ	(AX)
	RET

TEXT runtime·cmpstring(SB),NOSPLIT,$0-40
	MOVQ	s1_base+0(FP), SI
	MOVQ	s1_len+8(FP), BX
	MOVQ	s2_base+16(FP), DI
	MOVQ	s2_len+24(FP), DX
	LEAQ	ret+32(FP), R9
	JMP	runtime·cmpbody(SB)

TEXT bytes·Compare(SB),NOSPLIT,$0-56
	MOVQ	s1+0(FP), SI
	MOVQ	s1+8(FP), BX
	MOVQ	s2+24(FP), DI
	MOVQ	s2+32(FP), DX
	LEAQ	res+48(FP), R9
	JMP	runtime·cmpbody(SB)

// input:
//   SI = a
//   DI = b
//   BX = alen
//   DX = blen
//   R9 = address of output word (stores -1/0/1 here)
TEXT runtime·cmpbody(SB),NOSPLIT,$0-0
	CMPQ	SI, DI
	JEQ	allsame
	CMPQ	BX, DX
	MOVQ	DX, R8
	CMOVQLT	BX, R8 // R8 = min(alen, blen) = # of bytes to compare
	CMPQ	R8, $8
	JB	small

	CMPQ	R8, $63
	JBE	loop
	CMPB    runtime·support_avx2(SB), $1
	JEQ     big_loop_avx2
	JMP	big_loop
loop:
	CMPQ	R8, $16
	JBE	_0through16
	MOVOU	(SI), X0
	MOVOU	(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, AX
	XORQ	$0xffff, AX	// convert EQ to NE
	JNE	diff16	// branch if at least one byte is not equal
	ADDQ	$16, SI
	ADDQ	$16, DI
	SUBQ	$16, R8
	JMP	loop
	
diff64:
	ADDQ	$48, SI
	ADDQ	$48, DI
	JMP	diff16
diff48:
	ADDQ	$32, SI
	ADDQ	$32, DI
	JMP	diff16
diff32:
	ADDQ	$16, SI
	ADDQ	$16, DI
	// AX = bit mask of differences
diff16:
	BSFQ	AX, BX	// index of first byte that differs
	XORQ	AX, AX
	MOVB	(SI)(BX*1), CX
	CMPB	CX, (DI)(BX*1)
	SETHI	AX
	LEAQ	-1(AX*2), AX	// convert 1/0 to +1/-1
	MOVQ	AX, (R9)
	RET

	// 0 through 16 bytes left, alen>=8, blen>=8
_0through16:
	CMPQ	R8, $8
	JBE	_0through8
	MOVQ	(SI), AX
	MOVQ	(DI), CX
	CMPQ	AX, CX
	JNE	diff8
_0through8:
	MOVQ	-8(SI)(R8*1), AX
	MOVQ	-8(DI)(R8*1), CX
	CMPQ	AX, CX
	JEQ	allsame

	// AX and CX contain parts of a and b that differ.
diff8:
	BSWAPQ	AX	// reverse order of bytes
	BSWAPQ	CX
	XORQ	AX, CX
	BSRQ	CX, CX	// index of highest bit difference
	SHRQ	CX, AX	// move a's bit to bottom
	ANDQ	$1, AX	// mask bit
	LEAQ	-1(AX*2), AX // 1/0 => +1/-1
	MOVQ	AX, (R9)
	RET

	// 0-7 bytes in common
small:
	LEAQ	(R8*8), CX	// bytes left -> bits left
	NEGQ	CX		//  - bits lift (== 64 - bits left mod 64)
	JEQ	allsame

	// load bytes of a into high bytes of AX
	CMPB	SI, $0xf8
	JA	si_high
	MOVQ	(SI), SI
	JMP	si_finish
si_high:
	MOVQ	-8(SI)(R8*1), SI
	SHRQ	CX, SI
si_finish:
	SHLQ	CX, SI

	// load bytes of b in to high bytes of BX
	CMPB	DI, $0xf8
	JA	di_high
	MOVQ	(DI), DI
	JMP	di_finish
di_high:
	MOVQ	-8(DI)(R8*1), DI
	SHRQ	CX, DI
di_finish:
	SHLQ	CX, DI

	BSWAPQ	SI	// reverse order of bytes
	BSWAPQ	DI
	XORQ	SI, DI	// find bit differences
	JEQ	allsame
	BSRQ	DI, CX	// index of highest bit difference
	SHRQ	CX, SI	// move a's bit to bottom
	ANDQ	$1, SI	// mask bit
	LEAQ	-1(SI*2), AX // 1/0 => +1/-1
	MOVQ	AX, (R9)
	RET

allsame:
	XORQ	AX, AX
	XORQ	CX, CX
	CMPQ	BX, DX
	SETGT	AX	// 1 if alen > blen
	SETEQ	CX	// 1 if alen == blen
	LEAQ	-1(CX)(AX*2), AX	// 1,0,-1 result
	MOVQ	AX, (R9)
	RET

	// this works for >= 64 bytes of data.
big_loop:
	MOVOU	(SI), X0
	MOVOU	(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, AX
	XORQ	$0xffff, AX
	JNE	diff16

	MOVOU	16(SI), X0
	MOVOU	16(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, AX
	XORQ	$0xffff, AX
	JNE	diff32

	MOVOU	32(SI), X0
	MOVOU	32(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, AX
	XORQ	$0xffff, AX
	JNE	diff48

	MOVOU	48(SI), X0
	MOVOU	48(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, AX
	XORQ	$0xffff, AX
	JNE	diff64

	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$64, R8
	CMPQ	R8, $64
	JBE	loop
	JMP	big_loop

	// Compare 64-bytes per loop iteration.
	// Loop is unrolled and uses AVX2.
big_loop_avx2:
	VMOVDQU	(SI), Y2
	VMOVDQU	(DI), Y3
	VMOVDQU	32(SI), Y4
	VMOVDQU	32(DI), Y5
	VPCMPEQB Y2, Y3, Y0
	VPMOVMSKB Y0, AX
	XORL	$0xffffffff, AX
	JNE	diff32_avx2
	VPCMPEQB Y4, Y5, Y6
	VPMOVMSKB Y6, AX
	XORL	$0xffffffff, AX
	JNE	diff64_avx2

	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$64, R8
	CMPQ	R8, $64
	JB	big_loop_avx2_exit
	JMP	big_loop_avx2

	// Avoid AVX->SSE transition penalty and search first 32 bytes of 64 byte chunk.
diff32_avx2:
	VZEROUPPER
	JMP diff16

	// Same as diff32_avx2, but for last 32 bytes.
diff64_avx2:
	VZEROUPPER
	JMP diff48

	// For <64 bytes remainder jump to normal loop.
big_loop_avx2_exit:
	VZEROUPPER
	JMP loop

TEXT strings·indexShortStr(SB),NOSPLIT,$0-40
	MOVQ s+0(FP), DI
	// We want len in DX and AX, because PCMPESTRI implicitly consumes them
	MOVQ s_len+8(FP), DX
	MOVQ c+16(FP), BP
	MOVQ c_len+24(FP), AX
	MOVQ DI, R10
	LEAQ ret+32(FP), R11
	JMP  runtime·indexShortStr(SB)

TEXT bytes·indexShortStr(SB),NOSPLIT,$0-56
	MOVQ s+0(FP), DI
	MOVQ s_len+8(FP), DX
	MOVQ c+24(FP), BP
	MOVQ c_len+32(FP), AX
	MOVQ DI, R10
	LEAQ ret+48(FP), R11
	JMP  runtime·indexShortStr(SB)

// AX: length of string, that we are searching for
// DX: length of string, in which we are searching
// DI: pointer to string, in which we are searching
// BP: pointer to string, that we are searching for
// R11: address, where to put return value
TEXT runtime·indexShortStr(SB),NOSPLIT,$0
	CMPQ AX, DX
	JA fail
	CMPQ DX, $16
	JAE sse42
no_sse42:
	CMPQ AX, $2
	JA   _3_or_more
	MOVW (BP), BP
	LEAQ -1(DI)(DX*1), DX
loop2:
	MOVW (DI), SI
	CMPW SI,BP
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop2
	JMP fail
_3_or_more:
	CMPQ AX, $3
	JA   _4_or_more
	MOVW 1(BP), BX
	MOVW (BP), BP
	LEAQ -2(DI)(DX*1), DX
loop3:
	MOVW (DI), SI
	CMPW SI,BP
	JZ   partial_success3
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop3
	JMP fail
partial_success3:
	MOVW 1(DI), SI
	CMPW SI,BX
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop3
	JMP fail
_4_or_more:
	CMPQ AX, $4
	JA   _5_or_more
	MOVL (BP), BP
	LEAQ -3(DI)(DX*1), DX
loop4:
	MOVL (DI), SI
	CMPL SI,BP
	JZ   success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop4
	JMP fail
_5_or_more:
	CMPQ AX, $7
	JA   _8_or_more
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	MOVL -4(BP)(AX*1), BX
	MOVL (BP), BP
loop5to7:
	MOVL (DI), SI
	CMPL SI,BP
	JZ   partial_success5to7
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop5to7
	JMP fail
partial_success5to7:
	MOVL -4(AX)(DI*1), SI
	CMPL SI,BX
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop5to7
	JMP fail
_8_or_more:
	CMPQ AX, $8
	JA   _9_or_more
	MOVQ (BP), BP
	LEAQ -7(DI)(DX*1), DX
loop8:
	MOVQ (DI), SI
	CMPQ SI,BP
	JZ   success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop8
	JMP fail
_9_or_more:
	CMPQ AX, $15
	JA   _16_or_more
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	MOVQ -8(BP)(AX*1), BX
	MOVQ (BP), BP
loop9to15:
	MOVQ (DI), SI
	CMPQ SI,BP
	JZ   partial_success9to15
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop9to15
	JMP fail
partial_success9to15:
	MOVQ -8(AX)(DI*1), SI
	CMPQ SI,BX
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop9to15
	JMP fail
_16_or_more:
	CMPQ AX, $16
	JA   _17_or_more
	MOVOU (BP), X1
	LEAQ -15(DI)(DX*1), DX
loop16:
	MOVOU (DI), X2
	PCMPEQB X1, X2
	PMOVMSKB X2, SI
	CMPQ  SI, $0xffff
	JE   success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop16
	JMP fail
_17_or_more:
	CMPQ AX, $31
	JA   _32_or_more
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	MOVOU -16(BP)(AX*1), X0
	MOVOU (BP), X1
loop17to31:
	MOVOU (DI), X2
	PCMPEQB X1,X2
	PMOVMSKB X2, SI
	CMPQ  SI, $0xffff
	JE   partial_success17to31
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop17to31
	JMP fail
partial_success17to31:
	MOVOU -16(AX)(DI*1), X3
	PCMPEQB X0, X3
	PMOVMSKB X3, SI
	CMPQ  SI, $0xffff
	JE success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop17to31
	JMP fail
// We can get here only when AVX2 is enabled and cutoff for indexShortStr is set to 63
// So no need to check cpuid
_32_or_more:
	CMPQ AX, $32
	JA   _33_to_63
	VMOVDQU (BP), Y1
	LEAQ -31(DI)(DX*1), DX
loop32:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPMOVMSKB Y3, SI
	CMPL  SI, $0xffffffff
	JE   success_avx2
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop32
	JMP fail_avx2
_33_to_63:
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	VMOVDQU -32(BP)(AX*1), Y0
	VMOVDQU (BP), Y1
loop33to63:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPMOVMSKB Y3, SI
	CMPL  SI, $0xffffffff
	JE   partial_success33to63
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop33to63
	JMP fail_avx2
partial_success33to63:
	VMOVDQU -32(AX)(DI*1), Y3
	VPCMPEQB Y0, Y3, Y4
	VPMOVMSKB Y4, SI
	CMPL  SI, $0xffffffff
	JE success_avx2
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop33to63
fail_avx2:
	VZEROUPPER
fail:
	MOVQ $-1, (R11)
	RET
success_avx2:
	VZEROUPPER
	JMP success
sse42:
	CMPB runtime·support_sse42(SB), $1
	JNE no_sse42
	CMPQ AX, $12
	// PCMPESTRI is slower than normal compare,
	// so using it makes sense only if we advance 4+ bytes per compare
	// This value was determined experimentally and is the ~same
	// on Nehalem (first with SSE42) and Haswell.
	JAE _9_or_more
	LEAQ 16(BP), SI
	TESTW $0xff0, SI
	JEQ no_sse42
	MOVOU (BP), X1
	LEAQ -15(DI)(DX*1), SI
	MOVQ $16, R9
	SUBQ AX, R9 // We advance by 16-len(sep) each iteration, so precalculate it into R9
loop_sse42:
	// 0x0c means: unsigned byte compare (bits 0,1 are 00)
	// for equality (bits 2,3 are 11)
	// result is not masked or inverted (bits 4,5 are 00)
	// and corresponds to first matching byte (bit 6 is 0)
	PCMPESTRI $0x0c, (DI), X1
	// CX == 16 means no match,
	// CX > R9 means partial match at the end of the string,
	// otherwise sep is at offset CX from X1 start
	CMPQ CX, R9
	JBE sse42_success
	ADDQ R9, DI
	CMPQ DI, SI
	JB loop_sse42
	PCMPESTRI $0x0c, -1(SI), X1
	CMPQ CX, R9
	JA fail
	LEAQ -1(SI), DI
sse42_success:
	ADDQ CX, DI
success:
	SUBQ R10, DI
	MOVQ DI, (R11)
	RET


TEXT bytes·IndexByte(SB),NOSPLIT,$0-40
	MOVQ s+0(FP), SI
	MOVQ s_len+8(FP), BX
	MOVB c+24(FP), AL
	LEAQ ret+32(FP), R8
	JMP  runtime·indexbytebody(SB)

TEXT strings·IndexByte(SB),NOSPLIT,$0-32
	MOVQ s+0(FP), SI
	MOVQ s_len+8(FP), BX
	MOVB c+16(FP), AL
	LEAQ ret+24(FP), R8
	JMP  runtime·indexbytebody(SB)

// input:
//   SI: data
//   BX: data len
//   AL: byte sought
//   R8: address to put result
TEXT runtime·indexbytebody(SB),NOSPLIT,$0
	// Shuffle X0 around so that each byte contains
	// the character we're looking for.
	MOVD AX, X0
	PUNPCKLBW X0, X0
	PUNPCKLBW X0, X0
	PSHUFL $0, X0, X0
	
	CMPQ BX, $16
	JLT small

	MOVQ SI, DI

	CMPQ BX, $32
	JA avx2
sse:
	LEAQ	-16(SI)(BX*1), AX	// AX = address of last 16 bytes
	JMP	sseloopentry
	
sseloop:
	// Move the next 16-byte chunk of the data into X1.
	MOVOU	(DI), X1
	// Compare bytes in X0 to X1.
	PCMPEQB	X0, X1
	// Take the top bit of each byte in X1 and put the result in DX.
	PMOVMSKB X1, DX
	// Find first set bit, if any.
	BSFL	DX, DX
	JNZ	ssesuccess
	// Advance to next block.
	ADDQ	$16, DI
sseloopentry:
	CMPQ	DI, AX
	JB	sseloop

	// Search the last 16-byte chunk. This chunk may overlap with the
	// chunks we've already searched, but that's ok.
	MOVQ	AX, DI
	MOVOU	(AX), X1
	PCMPEQB	X0, X1
	PMOVMSKB X1, DX
	BSFL	DX, DX
	JNZ	ssesuccess

failure:
	MOVQ $-1, (R8)
	RET

// We've found a chunk containing the byte.
// The chunk was loaded from DI.
// The index of the matching byte in the chunk is DX.
// The start of the data is SI.
ssesuccess:
	SUBQ SI, DI	// Compute offset of chunk within data.
	ADDQ DX, DI	// Add offset of byte within chunk.
	MOVQ DI, (R8)
	RET

// handle for lengths < 16
small:
	TESTQ	BX, BX
	JEQ	failure

	// Check if we'll load across a page boundary.
	LEAQ	16(SI), AX
	TESTW	$0xff0, AX
	JEQ	endofpage

	MOVOU	(SI), X1 // Load data
	PCMPEQB	X0, X1	// Compare target byte with each byte in data.
	PMOVMSKB X1, DX	// Move result bits to integer register.
	BSFL	DX, DX	// Find first set bit.
	JZ	failure	// No set bit, failure.
	CMPL	DX, BX
	JAE	failure	// Match is past end of data.
	MOVQ	DX, (R8)
	RET

endofpage:
	MOVOU	-16(SI)(BX*1), X1	// Load data into the high end of X1.
	PCMPEQB	X0, X1	// Compare target byte with each byte in data.
	PMOVMSKB X1, DX	// Move result bits to integer register.
	MOVL	BX, CX
	SHLL	CX, DX
	SHRL	$16, DX	// Shift desired bits down to bottom of register.
	BSFL	DX, DX	// Find first set bit.
	JZ	failure	// No set bit, failure.
	MOVQ	DX, (R8)
	RET

avx2:
	CMPB   runtime·support_avx2(SB), $1
	JNE sse
	MOVD AX, X0
	LEAQ -32(SI)(BX*1), R11
	VPBROADCASTB  X0, Y1
avx2_loop:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPTEST Y3, Y3
	JNZ avx2success
	ADDQ $32, DI
	CMPQ DI, R11
	JLT avx2_loop
	MOVQ R11, DI
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPTEST Y3, Y3
	JNZ avx2success
	VZEROUPPER
	MOVQ $-1, (R8)
	RET

avx2success:
	VPMOVMSKB Y3, DX
	BSFL DX, DX
	SUBQ SI, DI
	ADDQ DI, DX
	MOVQ DX, (R8)
	VZEROUPPER
	RET

TEXT bytes·Equal(SB),NOSPLIT,$0-49
	MOVQ	a_len+8(FP), BX
	MOVQ	b_len+32(FP), CX
	CMPQ	BX, CX
	JNE	eqret
	MOVQ	a+0(FP), SI
	MOVQ	b+24(FP), DI
	LEAQ	ret+48(FP), AX
	JMP	runtime·memeqbody(SB)
eqret:
	MOVB	$0, ret+48(FP)
	RET


TEXT bytes·countByte(SB),NOSPLIT,$0-40
	MOVQ s+0(FP), SI
	MOVQ s_len+8(FP), BX
	MOVB c+24(FP), AL
	LEAQ ret+32(FP), R8
	JMP  runtime·countByte(SB)

TEXT strings·countByte(SB),NOSPLIT,$0-32
	MOVQ s+0(FP), SI
	MOVQ s_len+8(FP), BX
	MOVB c+16(FP), AL
	LEAQ ret+24(FP), R8
	JMP  runtime·countByte(SB)

// input:
//   SI: data
//   BX: data len
//   AL: byte sought
//   R8: address to put result
// This requires the POPCNT instruction
TEXT runtime·countByte(SB),NOSPLIT,$0
	// Shuffle X0 around so that each byte contains
	// the character we're looking for.
	MOVD AX, X0
	PUNPCKLBW X0, X0
	PUNPCKLBW X0, X0
	PSHUFL $0, X0, X0

	CMPQ BX, $16
	JLT small

	MOVQ $0, R12 // Accumulator

	MOVQ SI, DI

	CMPQ BX, $32
	JA avx2
sse:
	LEAQ	-16(SI)(BX*1), AX	// AX = address of last 16 bytes
	JMP	sseloopentry

sseloop:
	// Move the next 16-byte chunk of the data into X1.
	MOVOU	(DI), X1
	// Compare bytes in X0 to X1.
	PCMPEQB	X0, X1
	// Take the top bit of each byte in X1 and put the result in DX.
	PMOVMSKB X1, DX
	// Count number of matching bytes
	POPCNTL DX, DX
	// Accumulate into R12
	ADDQ DX, R12
	// Advance to next block.
	ADDQ	$16, DI
sseloopentry:
	CMPQ	DI, AX
	JBE	sseloop

	// Get the number of bytes to consider in the last 16 bytes
	ANDQ $15, BX
	JZ end

	// Create mask to ignore overlap between previous 16 byte block
	// and the next.
	MOVQ $16,CX
	SUBQ BX, CX
	MOVQ $0xFFFF, R10
	SARQ CL, R10
	SALQ CL, R10

	// Process the last 16-byte chunk. This chunk may overlap with the
	// chunks we've already searched so we need to mask part of it.
	MOVOU	(AX), X1
	PCMPEQB	X0, X1
	PMOVMSKB X1, DX
	// Apply mask
	ANDQ R10, DX
	POPCNTL DX, DX
	ADDQ DX, R12
end:
	MOVQ R12, (R8)
	RET

// handle for lengths < 16
small:
	TESTQ	BX, BX
	JEQ	endzero

	// Check if we'll load across a page boundary.
	LEAQ	16(SI), AX
	TESTW	$0xff0, AX
	JEQ	endofpage

	// We must ignore high bytes as they aren't part of our slice.
	// Create mask.
	MOVB BX, CX
	MOVQ $1, R10
	SALQ CL, R10
	SUBQ $1, R10

	// Load data
	MOVOU	(SI), X1
	// Compare target byte with each byte in data.
	PCMPEQB	X0, X1
	// Move result bits to integer register.
	PMOVMSKB X1, DX
	// Apply mask
	ANDQ R10, DX
	POPCNTL DX, DX
	// Directly return DX, we don't need to accumulate
	// since we have <16 bytes.
	MOVQ	DX, (R8)
	RET
endzero:
	MOVQ $0, (R8)
	RET

endofpage:
	// We must ignore low bytes as they aren't part of our slice.
	MOVQ $16,CX
	SUBQ BX, CX
	MOVQ $0xFFFF, R10
	SARQ CL, R10
	SALQ CL, R10

	// Load data into the high end of X1.
	MOVOU	-16(SI)(BX*1), X1
	// Compare target byte with each byte in data.
	PCMPEQB	X0, X1
	// Move result bits to integer register.
	PMOVMSKB X1, DX
	// Apply mask
	ANDQ R10, DX
	// Directly return DX, we don't need to accumulate
	// since we have <16 bytes.
	POPCNTL DX, DX
	MOVQ	DX, (R8)
	RET

avx2:
	CMPB   runtime·support_avx2(SB), $1
	JNE sse
	MOVD AX, X0
	LEAQ -32(SI)(BX*1), R11
	VPBROADCASTB  X0, Y1
avx2_loop:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPMOVMSKB Y3, DX
	POPCNTL DX, DX
	ADDQ DX, R12
	ADDQ $32, DI
	CMPQ DI, R11
	JLE avx2_loop

	// If last block is already processed,
	// skip to the end.
	CMPQ DI, R11
	JEQ endavx

	// Load address of the last 32 bytes.
	// There is an overlap with the previous block.
	MOVQ R11, DI
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPMOVMSKB Y3, DX
	// Exit AVX mode.
	VZEROUPPER

	// Create mask to ignore overlap between previous 32 byte block
	// and the next.
	ANDQ $31, BX
	MOVQ $32,CX
	SUBQ BX, CX
	MOVQ $0xFFFFFFFF, R10
	SARQ CL, R10
	SALQ CL, R10
	// Apply mask
	ANDQ R10, DX
	POPCNTL DX, DX
	ADDQ DX, R12
	MOVQ R12, (R8)
	RET
endavx:
	// Exit AVX mode.
	VZEROUPPER
	MOVQ R12, (R8)
	RET

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

TEXT runtime·prefetcht0(SB),NOSPLIT,$0-8
	MOVQ	addr+0(FP), AX
	PREFETCHT0	(AX)
	RET

TEXT runtime·prefetcht1(SB),NOSPLIT,$0-8
	MOVQ	addr+0(FP), AX
	PREFETCHT1	(AX)
	RET

TEXT runtime·prefetcht2(SB),NOSPLIT,$0-8
	MOVQ	addr+0(FP), AX
	PREFETCHT2	(AX)
	RET

TEXT runtime·prefetchnta(SB),NOSPLIT,$0-8
	MOVQ	addr+0(FP), AX
	PREFETCHNTA	(AX)
	RET

// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-0
	PUSHQ	R15 // The access to global variables below implicitly uses R15, which is callee-save
	MOVQ	runtime·lastmoduledatap(SB), AX
	MOVQ	DI, moduledata_next(AX)
	MOVQ	DI, runtime·lastmoduledatap(SB)
	POPQ	R15
	RET
