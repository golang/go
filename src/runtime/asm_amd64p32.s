// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

TEXT runtime·rt0_go(SB),NOSPLIT,$0
	// copy arguments forward on an even stack
	MOVL	argc+0(FP), AX
	MOVL	argv+4(FP), BX
	MOVL	SP, CX
	SUBL	$128, CX		// plenty of scratch
	ANDL	$~15, CX
	MOVL	CX, SP

	MOVL	AX, 16(SP)
	MOVL	BX, 24(SP)
	
	// create istack out of the given (operating system) stack.
	MOVL	$runtime·g0(SB), DI
	LEAL	(-64*1024+104)(SP), BX
	MOVL	BX, g_stackguard0(DI)
	MOVL	BX, g_stackguard1(DI)
	MOVL	BX, (g_stack+stack_lo)(DI)
	MOVL	SP, (g_stack+stack_hi)(DI)

	// find out information about the processor we're on
	MOVL	$0, AX
	CPUID
	CMPL	AX, $0
	JE	nocpuinfo

	CMPL	BX, $0x756E6547  // "Genu"
	JNE	notintel
	CMPL	DX, $0x49656E69  // "ineI"
	JNE	notintel
	CMPL	CX, $0x6C65746E  // "ntel"
	JNE	notintel
	MOVB	$1, runtime·isIntel(SB)
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
	// nacl does not support XGETBV to test
	// for XMM and YMM OS support.
#ifndef GOOS_nacl
	CMPB	runtime·support_osxsave(SB), $1
	JNE	noavx
	MOVL	$0, CX
	// For XGETBV, OSXSAVE bit is required and sufficient
	XGETBV
	ANDL	$6, AX
	CMPL	AX, $6 // Check for OS support of XMM and YMM registers.
	JE nocpuinfo
#endif
noavx:
	MOVB $0, runtime·support_avx(SB)
	MOVB $0, runtime·support_avx2(SB)

nocpuinfo:

needtls:
	LEAL	runtime·m0+m_tls(SB), DI
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
	LEAL	runtime·g0(SB), CX
	MOVL	CX, g(BX)
	LEAL	runtime·m0(SB), AX

	// save m->g0 = g0
	MOVL	CX, m_g0(AX)
	// save m0 to g0->m
	MOVL	AX, g_m(CX)

	CLD				// convention is D is always left cleared
	CALL	runtime·check(SB)

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVL	24(SP), AX		// copy argv
	MOVL	AX, 4(SP)
	CALL	runtime·args(SB)
	CALL	runtime·osinit(SB)
	CALL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVL	$runtime·mainPC(SB), AX	// entry
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)
	CALL	runtime·newproc(SB)

	// start this M
	CALL	runtime·mstart(SB)

	MOVL	$0xf1, 0xf1  // crash
	RET

DATA	runtime·mainPC+0(SB)/4,$runtime·main(SB)
GLOBL	runtime·mainPC(SB),RODATA,$4

TEXT runtime·breakpoint(SB),NOSPLIT,$0-0
	INT $3
	RET

TEXT runtime·asminit(SB),NOSPLIT,$0-0
	// No per-thread init.
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT, $0-4
	MOVL	buf+0(FP), AX	// gobuf
	LEAL	buf+0(FP), BX	// caller's SP
	MOVL	BX, gobuf_sp(AX)
	MOVL	0(SP), BX		// caller's PC
	MOVL	BX, gobuf_pc(AX)
	MOVQ	$0, gobuf_ret(AX)
	// Assert ctxt is zero. See func save.
	MOVL	gobuf_ctxt(AX), BX
	TESTL	BX, BX
	JZ	2(PC)
	CALL	runtime·badctxt(SB)
	get_tls(CX)
	MOVL	g(CX), BX
	MOVL	BX, gobuf_g(AX)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $8-4
	MOVL	buf+0(FP), BX		// gobuf

	// If ctxt is not nil, invoke deletion barrier before overwriting.
	MOVL	gobuf_ctxt(BX), DX
	TESTL	DX, DX
	JZ	nilctxt
	LEAL	gobuf_ctxt(BX), AX
	MOVL	AX, 0(SP)
	MOVL	$0, 4(SP)
	CALL	runtime·writebarrierptr_prewrite(SB)
	MOVL	buf+0(FP), BX

nilctxt:
	MOVL	gobuf_g(BX), DX
	MOVL	0(DX), CX		// make sure g != nil
	get_tls(CX)
	MOVL	DX, g(CX)
	MOVL	gobuf_sp(BX), SP	// restore SP
	MOVL	gobuf_ctxt(BX), DX
	MOVQ	gobuf_ret(BX), AX
	MOVL	$0, gobuf_sp(BX)	// clear to help garbage collector
	MOVQ	$0, gobuf_ret(BX)
	MOVL	$0, gobuf_ctxt(BX)
	MOVL	gobuf_pc(BX), BX
	JMP	BX

// func mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $0-4
	MOVL	fn+0(FP), DI
	
	get_tls(CX)
	MOVL	g(CX), AX	// save state in g->sched
	MOVL	0(SP), BX	// caller's PC
	MOVL	BX, (g_sched+gobuf_pc)(AX)
	LEAL	fn+0(FP), BX	// caller's SP
	MOVL	BX, (g_sched+gobuf_sp)(AX)
	MOVL	AX, (g_sched+gobuf_g)(AX)

	// switch to m->g0 & its stack, call fn
	MOVL	g(CX), BX
	MOVL	g_m(BX), BX
	MOVL	m_g0(BX), SI
	CMPL	SI, AX	// if g == m->g0 call badmcall
	JNE	3(PC)
	MOVL	$runtime·badmcall(SB), AX
	JMP	AX
	MOVL	SI, g(CX)	// g = m->g0
	MOVL	(g_sched+gobuf_sp)(SI), SP	// sp = m->g0->sched.sp
	PUSHQ	AX
	MOVL	DI, DX
	MOVL	0(DI), DI
	CALL	DI
	POPQ	AX
	MOVL	$runtime·badmcall2(SB), AX
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
TEXT runtime·systemstack(SB), NOSPLIT, $0-4
	MOVL	fn+0(FP), DI	// DI = fn
	get_tls(CX)
	MOVL	g(CX), AX	// AX = g
	MOVL	g_m(AX), BX	// BX = m

	MOVL	m_gsignal(BX), DX	// DX = gsignal
	CMPL	AX, DX
	JEQ	noswitch

	MOVL	m_g0(BX), DX	// DX = g0
	CMPL	AX, DX
	JEQ	noswitch

	MOVL	m_curg(BX), R8
	CMPL	AX, R8
	JEQ	switch
	
	// Not g0, not curg. Must be gsignal, but that's not allowed.
	// Hide call from linker nosplit analysis.
	MOVL	$runtime·badsystemstack(SB), AX
	CALL	AX

switch:
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVL	$runtime·systemstack_switch(SB), SI
	MOVL	SI, (g_sched+gobuf_pc)(AX)
	MOVL	SP, (g_sched+gobuf_sp)(AX)
	MOVL	AX, (g_sched+gobuf_g)(AX)

	// switch to g0
	MOVL	DX, g(CX)
	MOVL	(g_sched+gobuf_sp)(DX), SP

	// call target function
	MOVL	DI, DX
	MOVL	0(DI), DI
	CALL	DI

	// switch back to g
	get_tls(CX)
	MOVL	g(CX), AX
	MOVL	g_m(AX), BX
	MOVL	m_curg(BX), AX
	MOVL	AX, g(CX)
	MOVL	(g_sched+gobuf_sp)(AX), SP
	MOVL	$0, (g_sched+gobuf_sp)(AX)
	RET

noswitch:
	// already on m stack, just call directly
	MOVL	DI, DX
	MOVL	0(DI), DI
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
	get_tls(CX)
	MOVL	g(CX), BX
	MOVL	g_m(BX), BX

	// Cannot grow scheduler stack (m->g0).
	MOVL	m_g0(BX), SI
	CMPL	g(CX), SI
	JNE	3(PC)
	CALL	runtime·badmorestackg0(SB)
	MOVL	0, AX

	// Cannot grow signal stack (m->gsignal).
	MOVL	m_gsignal(BX), SI
	CMPL	g(CX), SI
	JNE	3(PC)
	CALL	runtime·badmorestackgsignal(SB)
	MOVL	0, AX

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVL	8(SP), AX	// f's caller's PC
	MOVL	AX, (m_morebuf+gobuf_pc)(BX)
	LEAL	16(SP), AX	// f's caller's SP
	MOVL	AX, (m_morebuf+gobuf_sp)(BX)
	get_tls(CX)
	MOVL	g(CX), SI
	MOVL	SI, (m_morebuf+gobuf_g)(BX)

	// Set g->sched to context in f.
	MOVL	0(SP), AX // f's PC
	MOVL	AX, (g_sched+gobuf_pc)(SI)
	MOVL	SI, (g_sched+gobuf_g)(SI)
	LEAL	8(SP), AX // f's SP
	MOVL	AX, (g_sched+gobuf_sp)(SI)
	// newstack will fill gobuf.ctxt.

	// Call newstack on m->g0's stack.
	MOVL	m_g0(BX), BX
	MOVL	BX, g(CX)
	MOVL	(g_sched+gobuf_sp)(BX), SP
	PUSHQ	DX	// ctxt argument
	CALL	runtime·newstack(SB)
	MOVL	$0, 0x1003	// crash if newstack returns
	POPQ	DX	// keep balance check happy
	RET

// morestack trampolines
TEXT runtime·morestack_noctxt(SB),NOSPLIT,$0
	MOVL	$0, DX
	JMP	runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMPL	CX, $MAXSIZE;		\
	JA	3(PC);			\
	MOVL	$NAME(SB), AX;		\
	JMP	AX
// Note: can't just "JMP NAME(SB)" - bad inlining results.

TEXT reflect·call(SB), NOSPLIT, $0-0
	JMP	·reflectcall(SB)

TEXT ·reflectcall(SB), NOSPLIT, $0-20
	MOVLQZX argsize+12(FP), CX
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
	MOVL	$runtime·badreflectcall(SB), AX
	JMP	AX

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-20;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVL	argptr+8(FP), SI;		\
	MOVL	argsize+12(FP), CX;		\
	MOVL	SP, DI;				\
	REP;MOVSB;				\
	/* call function */			\
	MOVL	f+4(FP), DX;			\
	MOVL	(DX), AX;			\
	CALL	AX;				\
	/* copy return values back */		\
	MOVL	argtype+0(FP), DX;		\
	MOVL	argptr+8(FP), DI;		\
	MOVL	argsize+12(FP), CX;		\
	MOVL	retoffset+16(FP), BX;		\
	MOVL	SP, SI;				\
	ADDL	BX, DI;				\
	ADDL	BX, SI;				\
	SUBL	BX, CX;				\
	CALL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $16-0
	MOVL	DX, 0(SP)
	MOVL	DI, 4(SP)
	MOVL	SI, 8(SP)
	MOVL	CX, 12(SP)
	CALL	runtime·reflectcallmove(SB)
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
TEXT runtime·jmpdefer(SB), NOSPLIT, $0-8
	MOVL	fv+0(FP), DX
	MOVL	argp+4(FP), BX
	LEAL	-8(BX), SP	// caller sp after CALL
	SUBL	$5, (SP)	// return to CALL again
	MOVL	0(DX), BX
	JMP	BX	// but first run the deferred function

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Not implemented.
TEXT runtime·asmcgocall(SB),NOSPLIT,$0-12
	MOVL	0, AX
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Not implemented.
TEXT runtime·cgocallback(SB),NOSPLIT,$0-16
	MOVL	0, AX
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize)
// Not implemented.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$0-16
	MOVL	0, AX
	RET

// void setg(G*); set g. for use by needm.
// Not implemented.
TEXT runtime·setg(SB), NOSPLIT, $0-4
	MOVL	0, AX
	RET

// check that SP is in range [g->stack.lo, g->stack.hi)
TEXT runtime·stackcheck(SB), NOSPLIT, $0-0
	get_tls(CX)
	MOVL	g(CX), AX
	CMPL	(g_stack+stack_hi)(AX), SP
	JHI	2(PC)
	MOVL	0, AX
	CMPL	SP, (g_stack+stack_lo)(AX)
	JHI	2(PC)
	MOVL	0, AX
	RET

TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT,$0-8
	MOVL	ptr+0(FP), DI
	MOVL	n+4(FP), CX
	MOVQ	CX, BX
	ANDQ	$3, BX
	SHRQ	$2, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSL
	MOVQ	BX, CX
	REP
	STOSB
	// Note: we zero only 4 bytes at a time so that the tail is at most
	// 3 bytes. That guarantees that we aren't zeroing pointers with STOSB.
	// See issue 13160.
	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$8-12
	MOVL	argp+0(FP),AX		// addr of first arg
	MOVL	-8(AX),AX		// get calling pc
	MOVL	AX, ret+8(FP)
	RET

// int64 runtime·cputicks(void)
TEXT runtime·cputicks(SB),NOSPLIT,$0-0
	RDTSC
	SHLQ	$32, DX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET

// memhash_varlen(p unsafe.Pointer, h seed) uintptr
// redirects to memhash(p, h, size) using the size
// stored in the closure.
TEXT runtime·memhash_varlen(SB),NOSPLIT,$24-12
	GO_ARGS
	NO_LOCAL_POINTERS
	MOVL	p+0(FP), AX
	MOVL	h+4(FP), BX
	MOVL	4(DX), CX
	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	MOVL	CX, 8(SP)
	CALL	runtime·memhash(SB)
	MOVL	16(SP), AX
	MOVL	AX, ret+8(FP)
	RET

// hash function using AES hardware instructions
// For now, our one amd64p32 system (NaCl) does not
// support using AES instructions, so have not bothered to
// write the implementations. Can copy and adjust the ones
// in asm_amd64.s when the time comes.

TEXT runtime·aeshash(SB),NOSPLIT,$0-20
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·aeshashstr(SB),NOSPLIT,$0-12
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·aeshash32(SB),NOSPLIT,$0-12
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·aeshash64(SB),NOSPLIT,$0-12
	MOVL	AX, ret+8(FP)
	RET

// memequal(p, q unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT,$0-17
	MOVL	a+0(FP), SI
	MOVL	b+4(FP), DI
	CMPL	SI, DI
	JEQ	eq
	MOVL	size+8(FP), BX
	CALL	runtime·memeqbody(SB)
	MOVB	AX, ret+16(FP)
	RET
eq:
	MOVB    $1, ret+16(FP)
	RET

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT,$0-9
	MOVL    a+0(FP), SI
	MOVL    b+4(FP), DI
	CMPL    SI, DI
	JEQ     eq
	MOVL    4(DX), BX    // compiler stores size at offset 4 in the closure
	CALL    runtime·memeqbody(SB)
	MOVB    AX, ret+8(FP)
	RET
eq:
	MOVB    $1, ret+8(FP)
	RET

// eqstring tests whether two strings are equal.
// The compiler guarantees that strings passed
// to eqstring have equal length.
// See runtime_test.go:eqstring_generic for
// equivalent Go code.
TEXT runtime·eqstring(SB),NOSPLIT,$0-17
	MOVL	s1_base+0(FP), SI
	MOVL	s2_base+8(FP), DI
	CMPL	SI, DI
	JEQ	same
	MOVL	s1_len+4(FP), BX
	CALL	runtime·memeqbody(SB)
	MOVB	AX, ret+16(FP)
	RET
same:
	MOVB	$1, ret+16(FP)
	RET

// a in SI
// b in DI
// count in BX
TEXT runtime·memeqbody(SB),NOSPLIT,$0-0
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
	ADDQ	BX, SI
	ADDQ	BX, DI
	MOVQ	-8(SI), CX
	MOVQ	-8(DI), DX
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
	// address ends in 11111xxx. Load up to bytes we want, move to correct position.
	MOVQ	BX, DX
	ADDQ	SI, DX
	MOVQ	-8(DX), SI
	SHRQ	CX, SI
si_finish:

	// same for DI.
	CMPB	DI, $0xf8
	JA	di_high
	MOVQ	(DI), DI
	JMP	di_finish
di_high:
	MOVQ	BX, DX
	ADDQ	DI, DX
	MOVQ	-8(DX), DI
	SHRQ	CX, DI
di_finish:

	SUBQ	SI, DI
	SHLQ	CX, DI
equal:
	SETEQ	AX
	RET

TEXT runtime·cmpstring(SB),NOSPLIT,$0-20
	MOVL	s1_base+0(FP), SI
	MOVL	s1_len+4(FP), BX
	MOVL	s2_base+8(FP), DI
	MOVL	s2_len+12(FP), DX
	CALL	runtime·cmpbody(SB)
	MOVL	AX, ret+16(FP)
	RET

TEXT bytes·Compare(SB),NOSPLIT,$0-28
	MOVL	s1+0(FP), SI
	MOVL	s1+4(FP), BX
	MOVL	s2+12(FP), DI
	MOVL	s2+16(FP), DX
	CALL	runtime·cmpbody(SB)
	MOVL	AX, res+24(FP)
	RET

// input:
//   SI = a
//   DI = b
//   BX = alen
//   DX = blen
// output:
//   AX = 1/0/-1
TEXT runtime·cmpbody(SB),NOSPLIT,$0-0
	CMPQ	SI, DI
	JEQ	allsame
	CMPQ	BX, DX
	MOVQ	DX, R8
	CMOVQLT	BX, R8 // R8 = min(alen, blen) = # of bytes to compare
	CMPQ	R8, $8
	JB	small

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
	
	// AX = bit mask of differences
diff16:
	BSFQ	AX, BX	// index of first byte that differs
	XORQ	AX, AX
	ADDQ	BX, SI
	MOVB	(SI), CX
	ADDQ	BX, DI
	CMPB	CX, (DI)
	SETHI	AX
	LEAQ	-1(AX*2), AX	// convert 1/0 to +1/-1
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
	ADDQ	R8, SI
	ADDQ	R8, DI
	MOVQ	-8(SI), AX
	MOVQ	-8(DI), CX
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
	ADDQ	R8, SI
	MOVQ	-8(SI), SI
	SHRQ	CX, SI
si_finish:
	SHLQ	CX, SI

	// load bytes of b in to high bytes of BX
	CMPB	DI, $0xf8
	JA	di_high
	MOVQ	(DI), DI
	JMP	di_finish
di_high:
	ADDQ	R8, DI
	MOVQ	-8(DI), DI
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
	RET

allsame:
	XORQ	AX, AX
	XORQ	CX, CX
	CMPQ	BX, DX
	SETGT	AX	// 1 if alen > blen
	SETEQ	CX	// 1 if alen == blen
	LEAQ	-1(CX)(AX*2), AX	// 1,0,-1 result
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-20
	MOVL s+0(FP), SI
	MOVL s_len+4(FP), BX
	MOVB c+12(FP), AL
	CALL runtime·indexbytebody(SB)
	MOVL AX, ret+16(FP)
	RET

TEXT strings·IndexByte(SB),NOSPLIT,$0-20
	MOVL s+0(FP), SI
	MOVL s_len+4(FP), BX
	MOVB c+8(FP), AL
	CALL runtime·indexbytebody(SB)
	MOVL AX, ret+16(FP)
	RET

// input:
//   SI: data
//   BX: data len
//   AL: byte sought
// output:
//   AX
TEXT runtime·indexbytebody(SB),NOSPLIT,$0
	MOVL SI, DI

	CMPL BX, $16
	JLT small

	// round up to first 16-byte boundary
	TESTL $15, SI
	JZ aligned
	MOVL SI, CX
	ANDL $~15, CX
	ADDL $16, CX

	// search the beginning
	SUBL SI, CX
	REPN; SCASB
	JZ success

// DI is 16-byte aligned; get ready to search using SSE instructions
aligned:
	// round down to last 16-byte boundary
	MOVL BX, R11
	ADDL SI, R11
	ANDL $~15, R11

	// shuffle X0 around so that each byte contains c
	MOVD AX, X0
	PUNPCKLBW X0, X0
	PUNPCKLBW X0, X0
	PSHUFL $0, X0, X0
	JMP condition

sse:
	// move the next 16-byte chunk of the buffer into X1
	MOVO (DI), X1
	// compare bytes in X0 to X1
	PCMPEQB X0, X1
	// take the top bit of each byte in X1 and put the result in DX
	PMOVMSKB X1, DX
	TESTL DX, DX
	JNZ ssesuccess
	ADDL $16, DI

condition:
	CMPL DI, R11
	JLT sse

	// search the end
	MOVL SI, CX
	ADDL BX, CX
	SUBL R11, CX
	// if CX == 0, the zero flag will be set and we'll end up
	// returning a false success
	JZ failure
	REPN; SCASB
	JZ success

failure:
	MOVL $-1, AX
	RET

// handle for lengths < 16
small:
	MOVL BX, CX
	REPN; SCASB
	JZ success
	MOVL $-1, AX
	RET

// we've found the chunk containing the byte
// now just figure out which specific byte it is
ssesuccess:
	// get the index of the least significant set bit
	BSFW DX, DX
	SUBL SI, DI
	ADDL DI, DX
	MOVL DX, AX
	RET

success:
	SUBL SI, DI
	SUBL $1, DI
	MOVL DI, AX
	RET

TEXT bytes·Equal(SB),NOSPLIT,$0-25
	MOVL	a_len+4(FP), BX
	MOVL	b_len+16(FP), CX
	XORL	AX, AX
	CMPL	BX, CX
	JNE	eqret
	MOVL	a+0(FP), SI
	MOVL	b+12(FP), DI
	CALL	runtime·memeqbody(SB)
eqret:
	MOVB	AX, ret+24(FP)
	RET

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVL	$0, AX
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT,$0-0
	BYTE	$0x90	// NOP
	CALL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	BYTE	$0x90	// NOP

TEXT runtime·prefetcht0(SB),NOSPLIT,$0-4
	MOVL	addr+0(FP), AX
	PREFETCHT0	(AX)
	RET

TEXT runtime·prefetcht1(SB),NOSPLIT,$0-4
	MOVL	addr+0(FP), AX
	PREFETCHT1	(AX)
	RET


TEXT runtime·prefetcht2(SB),NOSPLIT,$0-4
	MOVL	addr+0(FP), AX
	PREFETCHT2	(AX)
	RET

TEXT runtime·prefetchnta(SB),NOSPLIT,$0-4
	MOVL	addr+0(FP), AX
	PREFETCHNTA	(AX)
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVB	$1, ret+0(FP)
	RET
