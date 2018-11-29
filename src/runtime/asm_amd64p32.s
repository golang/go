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

nocpuinfo:
	LEAL	runtime·m0+m_tls(SB), DI
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

	CMPL	AX, m_gsignal(BX)
	JEQ	noswitch

	MOVL	m_g0(BX), DX	// DX = g0
	CMPL	AX, DX
	JEQ	noswitch

	CMPL	AX, m_curg(BX)
	JNE	bad

	// switch stacks
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
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVL	DI, DX
	MOVL	0(DI), DI
	JMP	DI

bad:
	// Not g0, not curg. Must be gsignal, but that's not allowed.
	// Hide call from linker nosplit analysis.
	MOVL	$runtime·badsystemstack(SB), AX
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
	MOVL	DX, (g_sched+gobuf_ctxt)(SI)

	// Call newstack on m->g0's stack.
	MOVL	m_g0(BX), BX
	MOVL	BX, g(CX)
	MOVL	(g_sched+gobuf_sp)(BX), SP
	CALL	runtime·newstack(SB)
	MOVL	$0, 0x1003	// crash if newstack returns
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

TEXT runtime·abort(SB),NOSPLIT,$0-0
	INT	$3
loop:
	JMP	loop

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

// int64 runtime·cputicks(void)
TEXT runtime·cputicks(SB),NOSPLIT,$0-0
	RDTSC
	SHLQ	$32, DX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
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

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVB	$1, ret+0(FP)
	RET

// gcWriteBarrier performs a heap pointer write and informs the GC.
//
// gcWriteBarrier does NOT follow the Go ABI. It takes two arguments:
// - DI is the destination of the write
// - AX is the value being written at DI
// It clobbers FLAGS and SI. It does not clobber any other general-purpose registers,
// but may clobber others (e.g., SSE registers).
TEXT runtime·gcWriteBarrier(SB),NOSPLIT,$88
	// Save the registers clobbered by the fast path. This is slightly
	// faster than having the caller spill these.
	MOVQ	R14, 72(SP)
	MOVQ	R13, 80(SP)
	// TODO: Consider passing g.m.p in as an argument so they can be shared
	// across a sequence of write barriers.
	get_tls(R13)
	MOVL	g(R13), R13
	MOVL	g_m(R13), R13
	MOVL	m_p(R13), R13
	MOVL	(p_wbBuf+wbBuf_next)(R13), R14
	// Increment wbBuf.next position.
	LEAL	8(R14), R14
	MOVL	R14, (p_wbBuf+wbBuf_next)(R13)
	CMPL	R14, (p_wbBuf+wbBuf_end)(R13)
	// Record the write.
	MOVL	AX, -8(R14)	// Record value
	MOVL	(DI), R13	// TODO: This turns bad writes into bad reads.
	MOVL	R13, -4(R14)	// Record *slot
	// Is the buffer full? (flags set in CMPL above)
	JEQ	flush
ret:
	MOVQ	72(SP), R14
	MOVQ	80(SP), R13
	// Do the write.
	MOVL	AX, (DI)
	RET			// Clobbers SI on NaCl

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
	MOVL	DI, 0(SP)	// Also first argument to wbBufFlush
	MOVL	AX, 4(SP)	// Also second argument to wbBufFlush
	MOVQ	BX, 8(SP)
	MOVQ	CX, 16(SP)
	MOVQ	DX, 24(SP)
	// DI already saved
	// SI is always clobbered on nacl
	// BP is reserved on nacl
	MOVQ	R8, 32(SP)
	MOVQ	R9, 40(SP)
	MOVQ	R10, 48(SP)
	MOVQ	R11, 56(SP)
	MOVQ	R12, 64(SP)
	// R13 already saved
	// R14 already saved
	// R15 is reserved on nacl

	// This takes arguments DI and AX
	CALL	runtime·wbBufFlush(SB)

	MOVL	0(SP), DI
	MOVL	4(SP), AX
	MOVQ	8(SP), BX
	MOVQ	16(SP), CX
	MOVQ	24(SP), DX
	MOVQ	32(SP), R8
	MOVQ	40(SP), R9
	MOVQ	48(SP), R10
	MOVQ	56(SP), R11
	MOVQ	64(SP), R12
	JMP	ret
