// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"
#include "funcdata.h"
#include "textflag.h"

TEXT runtime·rt0_go(SB),NOSPLIT,$0
	// copy arguments forward on an even stack
	MOVL	argc+0(FP), AX
	MOVL	argv+4(FP), BX
	SUBL	$128, SP		// plenty of scratch
	ANDL	$~15, SP
	MOVL	AX, 120(SP)		// save argc, argv away
	MOVL	BX, 124(SP)

	// set default stack bounds.
	// _cgo_init may update stackguard.
	MOVL	$runtime·g0(SB), BP
	LEAL	(-64*1024+104)(SP), BX
	MOVL	BX, g_stackguard0(BP)
	MOVL	BX, g_stackguard1(BP)
	MOVL	BX, (g_stack+stack_lo)(BP)
	MOVL	SP, (g_stack+stack_hi)(BP)
	
	// find out information about the processor we're on
	MOVL	$0, AX
	CPUID
	CMPL	AX, $0
	JE	nocpuinfo
	MOVL	$1, AX
	CPUID
	MOVL	CX, runtime·cpuid_ecx(SB)
	MOVL	DX, runtime·cpuid_edx(SB)
nocpuinfo:	

	// if there is an _cgo_init, call it to let it
	// initialize and to set up GS.  if not,
	// we set up GS ourselves.
	MOVL	_cgo_init(SB), AX
	TESTL	AX, AX
	JZ	needtls
	MOVL	$setg_gcc<>(SB), BX
	MOVL	BX, 4(SP)
	MOVL	BP, 0(SP)
	CALL	AX

	// update stackguard after _cgo_init
	MOVL	$runtime·g0(SB), CX
	MOVL	(g_stack+stack_lo)(CX), AX
	ADDL	$const_StackGuard, AX
	MOVL	AX, g_stackguard0(CX)
	MOVL	AX, g_stackguard1(CX)

	// skip runtime·ldt0setup(SB) and tls test after _cgo_init for non-windows
	CMPL runtime·iswindows(SB), $0
	JEQ ok
needtls:
	// skip runtime·ldt0setup(SB) and tls test on Plan 9 in all cases
	CMPL	runtime·isplan9(SB), $1
	JEQ	ok

	// set up %gs
	CALL	runtime·ldt0setup(SB)

	// store through it, to make sure it works
	get_tls(BX)
	MOVL	$0x123, g(BX)
	MOVL	runtime·tls0(SB), AX
	CMPL	AX, $0x123
	JEQ	ok
	MOVL	AX, 0	// abort
ok:
	// set up m and g "registers"
	get_tls(BX)
	LEAL	runtime·g0(SB), CX
	MOVL	CX, g(BX)
	LEAL	runtime·m0(SB), AX

	// save m->g0 = g0
	MOVL	CX, m_g0(AX)
	// save g0->m = m0
	MOVL	AX, g_m(CX)

	CALL	runtime·emptyfunc(SB)	// fault if stack check is wrong

	// convention is D is always cleared
	CLD

	CALL	runtime·check(SB)

	// saved argc, argv
	MOVL	120(SP), AX
	MOVL	AX, 0(SP)
	MOVL	124(SP), AX
	MOVL	AX, 4(SP)
	CALL	runtime·args(SB)
	CALL	runtime·osinit(SB)
	CALL	runtime·schedinit(SB)

	// create a new goroutine to start program
	PUSHL	$runtime·main·f(SB)	// entry
	PUSHL	$0	// arg size
	CALL	runtime·newproc(SB)
	POPL	AX
	POPL	AX

	// start this M
	CALL	runtime·mstart(SB)

	INT $3
	RET

DATA	runtime·main·f+0(SB)/4,$runtime·main(SB)
GLOBL	runtime·main·f(SB),RODATA,$4

TEXT runtime·breakpoint(SB),NOSPLIT,$0-0
	INT $3
	RET

TEXT runtime·asminit(SB),NOSPLIT,$0-0
	// Linux and MinGW start the FPU in extended double precision.
	// Other operating systems use double precision.
	// Change to double precision to match them,
	// and to match other hardware that only has double.
	PUSHL $0x27F
	FLDCW	0(SP)
	POPL AX
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT, $0-4
	MOVL	buf+0(FP), AX		// gobuf
	LEAL	buf+0(FP), BX		// caller's SP
	MOVL	BX, gobuf_sp(AX)
	MOVL	0(SP), BX		// caller's PC
	MOVL	BX, gobuf_pc(AX)
	MOVL	$0, gobuf_ret(AX)
	MOVL	$0, gobuf_ctxt(AX)
	get_tls(CX)
	MOVL	g(CX), BX
	MOVL	BX, gobuf_g(AX)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $0-4
	MOVL	buf+0(FP), BX		// gobuf
	MOVL	gobuf_g(BX), DX
	MOVL	0(DX), CX		// make sure g != nil
	get_tls(CX)
	MOVL	DX, g(CX)
	MOVL	gobuf_sp(BX), SP	// restore SP
	MOVL	gobuf_ret(BX), AX
	MOVL	gobuf_ctxt(BX), DX
	MOVL	$0, gobuf_sp(BX)	// clear to help garbage collector
	MOVL	$0, gobuf_ret(BX)
	MOVL	$0, gobuf_ctxt(BX)
	MOVL	gobuf_pc(BX), BX
	JMP	BX

// func mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
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
	PUSHL	AX
	MOVL	DI, DX
	MOVL	0(DI), DI
	CALL	DI
	POPL	AX
	MOVL	$runtime·badmcall2(SB), AX
	JMP	AX
	RET

// switchtoM is a dummy routine that onM leaves at the bottom
// of the G stack.  We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the M stack because the one at the top of
// the M stack terminates the stack walk (see topofstack()).
TEXT runtime·switchtoM(SB), NOSPLIT, $0-0
	RET

// func onM_signalok(fn func())
TEXT runtime·onM_signalok(SB), NOSPLIT, $0-4
	get_tls(CX)
	MOVL	g(CX), AX	// AX = g
	MOVL	g_m(AX), BX	// BX = m
	MOVL	m_gsignal(BX), DX	// DX = gsignal
	CMPL	AX, DX
	JEQ	ongsignal
	JMP	runtime·onM(SB)

ongsignal:
	MOVL	fn+0(FP), DI	// DI = fn
	MOVL	DI, DX
	MOVL	0(DI), DI
	CALL	DI
	RET

// func onM(fn func())
TEXT runtime·onM(SB), NOSPLIT, $0-4
	MOVL	fn+0(FP), DI	// DI = fn
	get_tls(CX)
	MOVL	g(CX), AX	// AX = g
	MOVL	g_m(AX), BX	// BX = m

	MOVL	m_g0(BX), DX	// DX = g0
	CMPL	AX, DX
	JEQ	onm

	MOVL	m_curg(BX), BP
	CMPL	AX, BP
	JEQ	oncurg
	
	// Not g0, not curg. Must be gsignal, but that's not allowed.
	// Hide call from linker nosplit analysis.
	MOVL	$runtime·badonm(SB), AX
	CALL	AX

oncurg:
	// save our state in g->sched.  Pretend to
	// be switchtoM if the G stack is scanned.
	MOVL	$runtime·switchtoM(SB), (g_sched+gobuf_pc)(AX)
	MOVL	SP, (g_sched+gobuf_sp)(AX)
	MOVL	AX, (g_sched+gobuf_g)(AX)

	// switch to g0
	MOVL	DX, g(CX)
	MOVL	(g_sched+gobuf_sp)(DX), BX
	// make it look like mstart called onM on g0, to stop traceback
	SUBL	$4, BX
	MOVL	$runtime·mstart(SB), DX
	MOVL	DX, 0(BX)
	MOVL	BX, SP

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

onm:
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
	// Cannot grow scheduler stack (m->g0).
	get_tls(CX)
	MOVL	g(CX), BX
	MOVL	g_m(BX), BX
	MOVL	m_g0(BX), SI
	CMPL	g(CX), SI
	JNE	2(PC)
	INT	$3

	// Cannot grow signal stack.
	MOVL	m_gsignal(BX), SI
	CMPL	g(CX), SI
	JNE	2(PC)
	INT	$3

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVL	4(SP), DI	// f's caller's PC
	MOVL	DI, (m_morebuf+gobuf_pc)(BX)
	LEAL	8(SP), CX	// f's caller's SP
	MOVL	CX, (m_morebuf+gobuf_sp)(BX)
	get_tls(CX)
	MOVL	g(CX), SI
	MOVL	SI, (m_morebuf+gobuf_g)(BX)

	// Set g->sched to context in f.
	MOVL	0(SP), AX	// f's PC
	MOVL	AX, (g_sched+gobuf_pc)(SI)
	MOVL	SI, (g_sched+gobuf_g)(SI)
	LEAL	4(SP), AX	// f's SP
	MOVL	AX, (g_sched+gobuf_sp)(SI)
	MOVL	DX, (g_sched+gobuf_ctxt)(SI)

	// Call newstack on m->g0's stack.
	MOVL	m_g0(BX), BP
	MOVL	BP, g(CX)
	MOVL	(g_sched+gobuf_sp)(BP), AX
	MOVL	-4(AX), BX	// fault if CALL would, before smashing SP
	MOVL	AX, SP
	CALL	runtime·newstack(SB)
	MOVL	$0, 0x1003	// crash if newstack returns
	RET

TEXT runtime·morestack_noctxt(SB),NOSPLIT,$0-0
	MOVL	$0, DX
	JMP runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMPL	CX, $MAXSIZE;		\
	JA	3(PC);			\
	MOVL	$NAME(SB), AX;		\
	JMP	AX
// Note: can't just "JMP NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT, $0-16
	MOVL	argsize+8(FP), CX
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
TEXT NAME(SB), WRAPPER, $MAXSIZE-16;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVL	argptr+4(FP), SI;		\
	MOVL	argsize+8(FP), CX;		\
	MOVL	SP, DI;				\
	REP;MOVSB;				\
	/* call function */			\
	MOVL	f+0(FP), DX;			\
	MOVL	(DX), AX; 			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	CALL	AX;				\
	/* copy return values back */		\
	MOVL	argptr+4(FP), DI;		\
	MOVL	argsize+8(FP), CX;		\
	MOVL	retoffset+12(FP), BX;		\
	MOVL	SP, SI;				\
	ADDL	BX, DI;				\
	ADDL	BX, SI;				\
	SUBL	BX, CX;				\
	REP;MOVSB;				\
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

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime·cas(SB), NOSPLIT, $0-13
	MOVL	ptr+0(FP), BX
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ 4(PC)
	MOVL	$0, AX
	MOVB	AX, ret+12(FP)
	RET
	MOVL	$1, AX
	MOVB	AX, ret+12(FP)
	RET

TEXT runtime·casuintptr(SB), NOSPLIT, $0-13
	JMP	runtime·cas(SB)

TEXT runtime·atomicloaduintptr(SB), NOSPLIT, $0-8
	JMP	runtime·atomicload(SB)

TEXT runtime·atomicloaduint(SB), NOSPLIT, $0-8
	JMP	runtime·atomicload(SB)

TEXT runtime·atomicstoreuintptr(SB), NOSPLIT, $0-8
	JMP	runtime·atomicstore(SB)

// bool runtime·cas64(uint64 *val, uint64 old, uint64 new)
// Atomically:
//	if(*val == *old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT runtime·cas64(SB), NOSPLIT, $0-21
	MOVL	ptr+0(FP), BP
	MOVL	old_lo+4(FP), AX
	MOVL	old_hi+8(FP), DX
	MOVL	new_lo+12(FP), BX
	MOVL	new_hi+16(FP), CX
	LOCK
	CMPXCHG8B	0(BP)
	JNZ	cas64_fail
	MOVL	$1, AX
	MOVB	AX, ret+20(FP)
	RET
cas64_fail:
	MOVL	$0, AX
	MOVB	AX, ret+20(FP)
	RET

// bool casp(void **p, void *old, void *new)
// Atomically:
//	if(*p == old){
//		*p = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime·casp(SB), NOSPLIT, $0-13
	MOVL	ptr+0(FP), BX
	MOVL	old+4(FP), AX
	MOVL	new+8(FP), CX
	LOCK
	CMPXCHGL	CX, 0(BX)
	JZ 4(PC)
	MOVL	$0, AX
	MOVB	AX, ret+12(FP)
	RET
	MOVL	$1, AX
	MOVB	AX, ret+12(FP)
	RET

// uint32 xadd(uint32 volatile *val, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT runtime·xadd(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	delta+4(FP), AX
	MOVL	AX, CX
	LOCK
	XADDL	AX, 0(BX)
	ADDL	CX, AX
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·xchg(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	new+4(FP), AX
	XCHGL	AX, 0(BX)
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·xchgp(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), BX
	MOVL	new+4(FP), AX
	XCHGL	AX, 0(BX)
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·xchguintptr(SB), NOSPLIT, $0-12
	JMP	runtime·xchg(SB)

TEXT runtime·procyield(SB),NOSPLIT,$0-0
	MOVL	cycles+0(FP), AX
again:
	PAUSE
	SUBL	$1, AX
	JNZ	again
	RET

TEXT runtime·atomicstorep(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

TEXT runtime·atomicstore(SB), NOSPLIT, $0-8
	MOVL	ptr+0(FP), BX
	MOVL	val+4(FP), AX
	XCHGL	AX, 0(BX)
	RET

// uint64 atomicload64(uint64 volatile* addr);
TEXT runtime·atomicload64(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), AX
	LEAL	ret_lo+4(FP), BX
	// MOVQ (%EAX), %MM0
	BYTE $0x0f; BYTE $0x6f; BYTE $0x00
	// MOVQ %MM0, 0(%EBX)
	BYTE $0x0f; BYTE $0x7f; BYTE $0x03
	// EMMS
	BYTE $0x0F; BYTE $0x77
	RET

// void runtime·atomicstore64(uint64 volatile* addr, uint64 v);
TEXT runtime·atomicstore64(SB), NOSPLIT, $0-12
	MOVL	ptr+0(FP), AX
	// MOVQ and EMMS were introduced on the Pentium MMX.
	// MOVQ 0x8(%ESP), %MM0
	BYTE $0x0f; BYTE $0x6f; BYTE $0x44; BYTE $0x24; BYTE $0x08
	// MOVQ %MM0, (%EAX)
	BYTE $0x0f; BYTE $0x7f; BYTE $0x00 
	// EMMS
	BYTE $0x0F; BYTE $0x77
	// This is essentially a no-op, but it provides required memory fencing.
	// It can be replaced with MFENCE, but MFENCE was introduced only on the Pentium4 (SSE2).
	MOVL	$0, AX
	LOCK
	XADDL	AX, (SP)
	RET

// void	runtime·atomicor8(byte volatile*, byte);
TEXT runtime·atomicor8(SB), NOSPLIT, $0-5
	MOVL	ptr+0(FP), AX
	MOVB	val+4(FP), BX
	LOCK
	ORB	BX, (AX)
	RET

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. pop the caller
// 2. sub 5 bytes from the callers return
// 3. jmp to the argument
TEXT runtime·jmpdefer(SB), NOSPLIT, $0-8
	MOVL	fv+0(FP), DX	// fn
	MOVL	argp+4(FP), BX	// caller sp
	LEAL	-4(BX), SP	// caller sp after CALL
	SUBL	$5, (SP)	// return to CALL again
	MOVL	0(DX), BX
	JMP	BX	// but first run the deferred function

// Save state of caller into g->sched.
TEXT gosave<>(SB),NOSPLIT,$0
	PUSHL	AX
	PUSHL	BX
	get_tls(BX)
	MOVL	g(BX), BX
	LEAL	arg+0(FP), AX
	MOVL	AX, (g_sched+gobuf_sp)(BX)
	MOVL	-4(AX), AX
	MOVL	AX, (g_sched+gobuf_pc)(BX)
	MOVL	$0, (g_sched+gobuf_ret)(BX)
	MOVL	$0, (g_sched+gobuf_ctxt)(BX)
	POPL	BX
	POPL	AX
	RET

// asmcgocall(void(*fn)(void*), void *arg)
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.c for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-8
	MOVL	fn+0(FP), AX
	MOVL	arg+4(FP), BX
	CALL	asmcgocall<>(SB)
	RET

TEXT ·asmcgocall_errno(SB),NOSPLIT,$0-12
	MOVL	fn+0(FP), AX
	MOVL	arg+4(FP), BX
	CALL	asmcgocall<>(SB)
	MOVL	AX, ret+8(FP)
	RET

TEXT asmcgocall<>(SB),NOSPLIT,$0-0
	// fn in AX, arg in BX
	MOVL	SP, DX

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	get_tls(CX)
	MOVL	g(CX), BP
	MOVL	g_m(BP), BP
	MOVL	m_g0(BP), SI
	MOVL	g(CX), DI
	CMPL	SI, DI
	JEQ	4(PC)
	CALL	gosave<>(SB)
	MOVL	SI, g(CX)
	MOVL	(g_sched+gobuf_sp)(SI), SP

	// Now on a scheduling stack (a pthread-created stack).
	SUBL	$32, SP
	ANDL	$~15, SP	// alignment, perhaps unnecessary
	MOVL	DI, 8(SP)	// save g
	MOVL	(g_stack+stack_hi)(DI), DI
	SUBL	DX, DI
	MOVL	DI, 4(SP)	// save depth in stack (can't just save SP, as stack might be copied during a callback)
	MOVL	BX, 0(SP)	// first argument in x86-32 ABI
	CALL	AX

	// Restore registers, g, stack pointer.
	get_tls(CX)
	MOVL	8(SP), DI
	MOVL	(g_stack+stack_hi)(DI), SI
	SUBL	4(SP), SI
	MOVL	DI, g(CX)
	MOVL	SI, SP
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$12-12
	LEAL	fn+0(FP), AX
	MOVL	AX, 0(SP)
	MOVL	frame+4(FP), AX
	MOVL	AX, 4(SP)
	MOVL	framesize+8(FP), AX
	MOVL	AX, 8(SP)
	MOVL	$runtime·cgocallback_gofunc(SB), AX
	CALL	AX
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize)
// See cgocall.c for more details.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$12-12
	NO_LOCAL_POINTERS

	// If g is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call through AX.
	get_tls(CX)
#ifdef GOOS_windows
	MOVL	$0, BP
	CMPL	CX, $0
	JEQ	2(PC) // TODO
#endif
	MOVL	g(CX), BP
	CMPL	BP, $0
	JEQ	needm
	MOVL	g_m(BP), BP
	MOVL	BP, DX // saved copy of oldm
	JMP	havem
needm:
	MOVL	$0, 0(SP)
	MOVL	$runtime·needm(SB), AX
	CALL	AX
	MOVL	0(SP), DX
	get_tls(CX)
	MOVL	g(CX), BP
	MOVL	g_m(BP), BP

	// Set m->sched.sp = SP, so that if a panic happens
	// during the function we are about to execute, it will
	// have a valid SP to run on the g0 stack.
	// The next few lines (after the havem label)
	// will save this SP onto the stack and then write
	// the same SP back to m->sched.sp. That seems redundant,
	// but if an unrecovered panic happens, unwindm will
	// restore the g->sched.sp from the stack location
	// and then onM will try to use it. If we don't set it here,
	// that restored SP will be uninitialized (typically 0) and
	// will not be usable.
	MOVL	m_g0(BP), SI
	MOVL	SP, (g_sched+gobuf_sp)(SI)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 0(SP).
	MOVL	m_g0(BP), SI
	MOVL	(g_sched+gobuf_sp)(SI), AX
	MOVL	AX, 0(SP)
	MOVL	SP, (g_sched+gobuf_sp)(SI)

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
	// In the new goroutine, 0(SP) holds the saved oldm (DX) register.
	// 4(SP) and 8(SP) are unused.
	MOVL	m_curg(BP), SI
	MOVL	SI, g(CX)
	MOVL	(g_sched+gobuf_sp)(SI), DI // prepare stack as DI
	MOVL	(g_sched+gobuf_pc)(SI), BP
	MOVL	BP, -4(DI)
	LEAL	-(4+12)(DI), SP
	MOVL	DX, 0(SP)
	CALL	runtime·cgocallbackg(SB)
	MOVL	0(SP), DX

	// Restore g->sched (== m->curg->sched) from saved values.
	get_tls(CX)
	MOVL	g(CX), SI
	MOVL	12(SP), BP
	MOVL	BP, (g_sched+gobuf_pc)(SI)
	LEAL	(12+4)(SP), DI
	MOVL	DI, (g_sched+gobuf_sp)(SI)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVL	g(CX), BP
	MOVL	g_m(BP), BP
	MOVL	m_g0(BP), SI
	MOVL	SI, g(CX)
	MOVL	(g_sched+gobuf_sp)(SI), SP
	MOVL	0(SP), AX
	MOVL	AX, (g_sched+gobuf_sp)(SI)
	
	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	CMPL	DX, $0
	JNE 3(PC)
	MOVL	$runtime·dropm(SB), AX
	CALL	AX

	// Done!
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-4
	MOVL	gg+0(FP), BX
#ifdef GOOS_windows
	CMPL	BX, $0
	JNE	settls
	MOVL	$0, 0x14(FS)
	RET
settls:
	MOVL	g_m(BX), AX
	LEAL	m_tls(AX), AX
	MOVL	AX, 0x14(FS)
#endif
	get_tls(CX)
	MOVL	BX, g(CX)
	RET

// void setg_gcc(G*); set g. for use by gcc
TEXT setg_gcc<>(SB), NOSPLIT, $0
	get_tls(AX)
	MOVL	gg+0(FP), DX
	MOVL	DX, g(AX)
	RET

// check that SP is in range [g->stack.lo, g->stack.hi)
TEXT runtime·stackcheck(SB), NOSPLIT, $0-0
	get_tls(CX)
	MOVL	g(CX), AX
	CMPL	(g_stack+stack_hi)(AX), SP
	JHI	2(PC)
	INT	$3
	CMPL	SP, (g_stack+stack_lo)(AX)
	JHI	2(PC)
	INT	$3
	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$0-8
	MOVL	argp+0(FP),AX		// addr of first arg
	MOVL	-4(AX),AX		// get calling pc
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·gogetcallerpc(SB),NOSPLIT,$0-8
	MOVL	p+0(FP),AX		// addr of first arg
	MOVL	-4(AX),AX		// get calling pc
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·setcallerpc(SB),NOSPLIT,$0-8
	MOVL	argp+0(FP),AX		// addr of first arg
	MOVL	pc+4(FP), BX
	MOVL	BX, -4(AX)		// set calling pc
	RET

TEXT runtime·getcallersp(SB), NOSPLIT, $0-8
	MOVL	argp+0(FP), AX
	MOVL	AX, ret+4(FP)
	RET

// func gogetcallersp(p unsafe.Pointer) uintptr
TEXT runtime·gogetcallersp(SB),NOSPLIT,$0-8
	MOVL	p+0(FP),AX		// addr of first arg
	MOVL	AX, ret+4(FP)
	RET

// int64 runtime·cputicks(void), so really
// void runtime·cputicks(int64 *ticks)
TEXT runtime·cputicks(SB),NOSPLIT,$0-8
	RDTSC
	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET

TEXT runtime·ldt0setup(SB),NOSPLIT,$16-0
	// set up ldt 7 to point at tls0
	// ldt 1 would be fine on Linux, but on OS X, 7 is as low as we can go.
	// the entry number is just a hint.  setldt will set up GS with what it used.
	MOVL	$7, 0(SP)
	LEAL	runtime·tls0(SB), AX
	MOVL	AX, 4(SP)
	MOVL	$32, 8(SP)	// sizeof(tls array)
	CALL	runtime·setldt(SB)
	RET

TEXT runtime·emptyfunc(SB),0,$0-0
	RET

TEXT runtime·abort(SB),NOSPLIT,$0-0
	INT $0x3

// hash function using AES hardware instructions
TEXT runtime·aeshash(SB),NOSPLIT,$0-16
	MOVL	p+0(FP), AX	// ptr to data
	MOVL	s+4(FP), CX	// size
	JMP	runtime·aeshashbody(SB)

TEXT runtime·aeshashstr(SB),NOSPLIT,$0-16
	MOVL	p+0(FP), AX	// ptr to string object
	// s+4(FP) is ignored, it is always sizeof(String)
	MOVL	4(AX), CX	// length of string
	MOVL	(AX), AX	// string data
	JMP	runtime·aeshashbody(SB)

// AX: data
// CX: length
TEXT runtime·aeshashbody(SB),NOSPLIT,$0-16
	MOVL	h+8(FP), X0	// seed to low 32 bits of xmm0
	PINSRD	$1, CX, X0	// size to next 32 bits of xmm0
	MOVO	runtime·aeskeysched+0(SB), X2
	MOVO	runtime·aeskeysched+16(SB), X3
	CMPL	CX, $16
	JB	aessmall
aesloop:
	CMPL	CX, $16
	JBE	aesloopend
	MOVOU	(AX), X1
	AESENC	X2, X0
	AESENC	X1, X0
	SUBL	$16, CX
	ADDL	$16, AX
	JMP	aesloop
// 1-16 bytes remaining
aesloopend:
	// This load may overlap with the previous load above.
	// We'll hash some bytes twice, but that's ok.
	MOVOU	-16(AX)(CX*1), X1
	JMP	partial
// 0-15 bytes
aessmall:
	TESTL	CX, CX
	JE	finalize	// 0 bytes

	CMPB	AX, $0xf0
	JA	highpartial

	// 16 bytes loaded at this address won't cross
	// a page boundary, so we can load it directly.
	MOVOU	(AX), X1
	ADDL	CX, CX
	PAND	masks<>(SB)(CX*8), X1
	JMP	partial
highpartial:
	// address ends in 1111xxxx.  Might be up against
	// a page boundary, so load ending at last byte.
	// Then shift bytes down using pshufb.
	MOVOU	-16(AX)(CX*1), X1
	ADDL	CX, CX
	PSHUFB	shifts<>(SB)(CX*8), X1
partial:
	// incorporate partial block into hash
	AESENC	X3, X0
	AESENC	X1, X0
finalize:	
	// finalize hash
	AESENC	X2, X0
	AESENC	X3, X0
	AESENC	X2, X0
	MOVL	X0, ret+12(FP)
	RET

TEXT runtime·aeshash32(SB),NOSPLIT,$0-16
	MOVL	p+0(FP), AX	// ptr to data
	// s+4(FP) is ignored, it is always sizeof(int32)
	MOVL	h+8(FP), X0	// seed
	PINSRD	$1, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+0(SB), X0
	MOVL	X0, ret+12(FP)
	RET

TEXT runtime·aeshash64(SB),NOSPLIT,$0-16
	MOVL	p+0(FP), AX	// ptr to data
	// s+4(FP) is ignored, it is always sizeof(int64)
	MOVQ	(AX), X0	// data
	PINSRD	$2, h+8(FP), X0	// seed
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+0(SB), X0
	MOVL	X0, ret+12(FP)
	RET

// simple mask to get rid of data in the high part of the register.
DATA masks<>+0x00(SB)/4, $0x00000000
DATA masks<>+0x04(SB)/4, $0x00000000
DATA masks<>+0x08(SB)/4, $0x00000000
DATA masks<>+0x0c(SB)/4, $0x00000000
	
DATA masks<>+0x10(SB)/4, $0x000000ff
DATA masks<>+0x14(SB)/4, $0x00000000
DATA masks<>+0x18(SB)/4, $0x00000000
DATA masks<>+0x1c(SB)/4, $0x00000000
	
DATA masks<>+0x20(SB)/4, $0x0000ffff
DATA masks<>+0x24(SB)/4, $0x00000000
DATA masks<>+0x28(SB)/4, $0x00000000
DATA masks<>+0x2c(SB)/4, $0x00000000
	
DATA masks<>+0x30(SB)/4, $0x00ffffff
DATA masks<>+0x34(SB)/4, $0x00000000
DATA masks<>+0x38(SB)/4, $0x00000000
DATA masks<>+0x3c(SB)/4, $0x00000000
	
DATA masks<>+0x40(SB)/4, $0xffffffff
DATA masks<>+0x44(SB)/4, $0x00000000
DATA masks<>+0x48(SB)/4, $0x00000000
DATA masks<>+0x4c(SB)/4, $0x00000000
	
DATA masks<>+0x50(SB)/4, $0xffffffff
DATA masks<>+0x54(SB)/4, $0x000000ff
DATA masks<>+0x58(SB)/4, $0x00000000
DATA masks<>+0x5c(SB)/4, $0x00000000
	
DATA masks<>+0x60(SB)/4, $0xffffffff
DATA masks<>+0x64(SB)/4, $0x0000ffff
DATA masks<>+0x68(SB)/4, $0x00000000
DATA masks<>+0x6c(SB)/4, $0x00000000
	
DATA masks<>+0x70(SB)/4, $0xffffffff
DATA masks<>+0x74(SB)/4, $0x00ffffff
DATA masks<>+0x78(SB)/4, $0x00000000
DATA masks<>+0x7c(SB)/4, $0x00000000
	
DATA masks<>+0x80(SB)/4, $0xffffffff
DATA masks<>+0x84(SB)/4, $0xffffffff
DATA masks<>+0x88(SB)/4, $0x00000000
DATA masks<>+0x8c(SB)/4, $0x00000000
	
DATA masks<>+0x90(SB)/4, $0xffffffff
DATA masks<>+0x94(SB)/4, $0xffffffff
DATA masks<>+0x98(SB)/4, $0x000000ff
DATA masks<>+0x9c(SB)/4, $0x00000000
	
DATA masks<>+0xa0(SB)/4, $0xffffffff
DATA masks<>+0xa4(SB)/4, $0xffffffff
DATA masks<>+0xa8(SB)/4, $0x0000ffff
DATA masks<>+0xac(SB)/4, $0x00000000
	
DATA masks<>+0xb0(SB)/4, $0xffffffff
DATA masks<>+0xb4(SB)/4, $0xffffffff
DATA masks<>+0xb8(SB)/4, $0x00ffffff
DATA masks<>+0xbc(SB)/4, $0x00000000
	
DATA masks<>+0xc0(SB)/4, $0xffffffff
DATA masks<>+0xc4(SB)/4, $0xffffffff
DATA masks<>+0xc8(SB)/4, $0xffffffff
DATA masks<>+0xcc(SB)/4, $0x00000000
	
DATA masks<>+0xd0(SB)/4, $0xffffffff
DATA masks<>+0xd4(SB)/4, $0xffffffff
DATA masks<>+0xd8(SB)/4, $0xffffffff
DATA masks<>+0xdc(SB)/4, $0x000000ff
	
DATA masks<>+0xe0(SB)/4, $0xffffffff
DATA masks<>+0xe4(SB)/4, $0xffffffff
DATA masks<>+0xe8(SB)/4, $0xffffffff
DATA masks<>+0xec(SB)/4, $0x0000ffff
	
DATA masks<>+0xf0(SB)/4, $0xffffffff
DATA masks<>+0xf4(SB)/4, $0xffffffff
DATA masks<>+0xf8(SB)/4, $0xffffffff
DATA masks<>+0xfc(SB)/4, $0x00ffffff

GLOBL masks<>(SB),RODATA,$256

// these are arguments to pshufb.  They move data down from
// the high bytes of the register to the low bytes of the register.
// index is how many bytes to move.
DATA shifts<>+0x00(SB)/4, $0x00000000
DATA shifts<>+0x04(SB)/4, $0x00000000
DATA shifts<>+0x08(SB)/4, $0x00000000
DATA shifts<>+0x0c(SB)/4, $0x00000000
	
DATA shifts<>+0x10(SB)/4, $0xffffff0f
DATA shifts<>+0x14(SB)/4, $0xffffffff
DATA shifts<>+0x18(SB)/4, $0xffffffff
DATA shifts<>+0x1c(SB)/4, $0xffffffff
	
DATA shifts<>+0x20(SB)/4, $0xffff0f0e
DATA shifts<>+0x24(SB)/4, $0xffffffff
DATA shifts<>+0x28(SB)/4, $0xffffffff
DATA shifts<>+0x2c(SB)/4, $0xffffffff
	
DATA shifts<>+0x30(SB)/4, $0xff0f0e0d
DATA shifts<>+0x34(SB)/4, $0xffffffff
DATA shifts<>+0x38(SB)/4, $0xffffffff
DATA shifts<>+0x3c(SB)/4, $0xffffffff
	
DATA shifts<>+0x40(SB)/4, $0x0f0e0d0c
DATA shifts<>+0x44(SB)/4, $0xffffffff
DATA shifts<>+0x48(SB)/4, $0xffffffff
DATA shifts<>+0x4c(SB)/4, $0xffffffff
	
DATA shifts<>+0x50(SB)/4, $0x0e0d0c0b
DATA shifts<>+0x54(SB)/4, $0xffffff0f
DATA shifts<>+0x58(SB)/4, $0xffffffff
DATA shifts<>+0x5c(SB)/4, $0xffffffff
	
DATA shifts<>+0x60(SB)/4, $0x0d0c0b0a
DATA shifts<>+0x64(SB)/4, $0xffff0f0e
DATA shifts<>+0x68(SB)/4, $0xffffffff
DATA shifts<>+0x6c(SB)/4, $0xffffffff
	
DATA shifts<>+0x70(SB)/4, $0x0c0b0a09
DATA shifts<>+0x74(SB)/4, $0xff0f0e0d
DATA shifts<>+0x78(SB)/4, $0xffffffff
DATA shifts<>+0x7c(SB)/4, $0xffffffff
	
DATA shifts<>+0x80(SB)/4, $0x0b0a0908
DATA shifts<>+0x84(SB)/4, $0x0f0e0d0c
DATA shifts<>+0x88(SB)/4, $0xffffffff
DATA shifts<>+0x8c(SB)/4, $0xffffffff
	
DATA shifts<>+0x90(SB)/4, $0x0a090807
DATA shifts<>+0x94(SB)/4, $0x0e0d0c0b
DATA shifts<>+0x98(SB)/4, $0xffffff0f
DATA shifts<>+0x9c(SB)/4, $0xffffffff
	
DATA shifts<>+0xa0(SB)/4, $0x09080706
DATA shifts<>+0xa4(SB)/4, $0x0d0c0b0a
DATA shifts<>+0xa8(SB)/4, $0xffff0f0e
DATA shifts<>+0xac(SB)/4, $0xffffffff
	
DATA shifts<>+0xb0(SB)/4, $0x08070605
DATA shifts<>+0xb4(SB)/4, $0x0c0b0a09
DATA shifts<>+0xb8(SB)/4, $0xff0f0e0d
DATA shifts<>+0xbc(SB)/4, $0xffffffff
	
DATA shifts<>+0xc0(SB)/4, $0x07060504
DATA shifts<>+0xc4(SB)/4, $0x0b0a0908
DATA shifts<>+0xc8(SB)/4, $0x0f0e0d0c
DATA shifts<>+0xcc(SB)/4, $0xffffffff
	
DATA shifts<>+0xd0(SB)/4, $0x06050403
DATA shifts<>+0xd4(SB)/4, $0x0a090807
DATA shifts<>+0xd8(SB)/4, $0x0e0d0c0b
DATA shifts<>+0xdc(SB)/4, $0xffffff0f
	
DATA shifts<>+0xe0(SB)/4, $0x05040302
DATA shifts<>+0xe4(SB)/4, $0x09080706
DATA shifts<>+0xe8(SB)/4, $0x0d0c0b0a
DATA shifts<>+0xec(SB)/4, $0xffff0f0e
	
DATA shifts<>+0xf0(SB)/4, $0x04030201
DATA shifts<>+0xf4(SB)/4, $0x08070605
DATA shifts<>+0xf8(SB)/4, $0x0c0b0a09
DATA shifts<>+0xfc(SB)/4, $0xff0f0e0d

GLOBL shifts<>(SB),RODATA,$256

TEXT runtime·memeq(SB),NOSPLIT,$0-13
	MOVL	a+0(FP), SI
	MOVL	b+4(FP), DI
	MOVL	size+8(FP), BX
	CALL	runtime·memeqbody(SB)
	MOVB	AX, ret+12(FP)
	RET

// eqstring tests whether two strings are equal.
// See runtime_test.go:eqstring_generic for
// equivalent Go code.
TEXT runtime·eqstring(SB),NOSPLIT,$0-17
	MOVL	s1len+4(FP), AX
	MOVL	s2len+12(FP), BX
	CMPL	AX, BX
	JNE	different
	MOVL	s1str+0(FP), SI
	MOVL	s2str+8(FP), DI
	CMPL	SI, DI
	JEQ	same
	CALL	runtime·memeqbody(SB)
	MOVB	AX, v+16(FP)
	RET
same:
	MOVB	$1, v+16(FP)
	RET
different:
	MOVB	$0, v+16(FP)
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

// a in SI
// b in DI
// count in BX
TEXT runtime·memeqbody(SB),NOSPLIT,$0-0
	XORL	AX, AX

	CMPL	BX, $4
	JB	small

	// 64 bytes at a time using xmm registers
hugeloop:
	CMPL	BX, $64
	JB	bigloop
	TESTL	$0x4000000, runtime·cpuid_edx(SB) // check for sse2
	JE	bigloop
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
	ADDL	$64, SI
	ADDL	$64, DI
	SUBL	$64, BX
	CMPL	DX, $0xffff
	JEQ	hugeloop
	RET

	// 4 bytes at a time using 32-bit register
bigloop:
	CMPL	BX, $4
	JBE	leftover
	MOVL	(SI), CX
	MOVL	(DI), DX
	ADDL	$4, SI
	ADDL	$4, DI
	SUBL	$4, BX
	CMPL	CX, DX
	JEQ	bigloop
	RET

	// remaining 0-4 bytes
leftover:
	MOVL	-4(SI)(BX*1), CX
	MOVL	-4(DI)(BX*1), DX
	CMPL	CX, DX
	SETEQ	AX
	RET

small:
	CMPL	BX, $0
	JEQ	equal

	LEAL	0(BX*8), CX
	NEGL	CX

	MOVL	SI, DX
	CMPB	DX, $0xfc
	JA	si_high

	// load at SI won't cross a page boundary.
	MOVL	(SI), SI
	JMP	si_finish
si_high:
	// address ends in 111111xx.  Load up to bytes we want, move to correct position.
	MOVL	-4(SI)(BX*1), SI
	SHRL	CX, SI
si_finish:

	// same for DI.
	MOVL	DI, DX
	CMPB	DX, $0xfc
	JA	di_high
	MOVL	(DI), DI
	JMP	di_finish
di_high:
	MOVL	-4(DI)(BX*1), DI
	SHRL	CX, DI
di_finish:

	SUBL	SI, DI
	SHLL	CX, DI
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

TEXT runtime·cmpbytes(SB),NOSPLIT,$0-28
	MOVL	s1+0(FP), SI
	MOVL	s1+4(FP), BX
	MOVL	s2+12(FP), DI
	MOVL	s2+16(FP), DX
	CALL	runtime·cmpbody(SB)
	MOVL	AX, ret+24(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0
	MOVL	s+0(FP), SI
	MOVL	s_len+4(FP), CX
	MOVB	c+12(FP), AL
	MOVL	SI, DI
	CLD; REPN; SCASB
	JZ 3(PC)
	MOVL	$-1, ret+16(FP)
	RET
	SUBL	SI, DI
	SUBL	$1, DI
	MOVL	DI, ret+16(FP)
	RET

TEXT strings·IndexByte(SB),NOSPLIT,$0
	MOVL	s+0(FP), SI
	MOVL	s_len+4(FP), CX
	MOVB	c+8(FP), AL
	MOVL	SI, DI
	CLD; REPN; SCASB
	JZ 3(PC)
	MOVL	$-1, ret+12(FP)
	RET
	SUBL	SI, DI
	SUBL	$1, DI
	MOVL	DI, ret+12(FP)
	RET

// input:
//   SI = a
//   DI = b
//   BX = alen
//   DX = blen
// output:
//   AX = 1/0/-1
TEXT runtime·cmpbody(SB),NOSPLIT,$0-0
	CMPL	SI, DI
	JEQ	cmp_allsame
	CMPL	BX, DX
	MOVL	DX, BP
	CMOVLLT	BX, BP // BP = min(alen, blen)
	CMPL	BP, $4
	JB	cmp_small
	TESTL	$0x4000000, runtime·cpuid_edx(SB) // check for sse2
	JE	cmp_mediumloop
cmp_largeloop:
	CMPL	BP, $16
	JB	cmp_mediumloop
	MOVOU	(SI), X0
	MOVOU	(DI), X1
	PCMPEQB X0, X1
	PMOVMSKB X1, AX
	XORL	$0xffff, AX	// convert EQ to NE
	JNE	cmp_diff16	// branch if at least one byte is not equal
	ADDL	$16, SI
	ADDL	$16, DI
	SUBL	$16, BP
	JMP	cmp_largeloop

cmp_diff16:
	BSFL	AX, BX	// index of first byte that differs
	XORL	AX, AX
	MOVB	(SI)(BX*1), CX
	CMPB	CX, (DI)(BX*1)
	SETHI	AX
	LEAL	-1(AX*2), AX	// convert 1/0 to +1/-1
	RET

cmp_mediumloop:
	CMPL	BP, $4
	JBE	cmp_0through4
	MOVL	(SI), AX
	MOVL	(DI), CX
	CMPL	AX, CX
	JNE	cmp_diff4
	ADDL	$4, SI
	ADDL	$4, DI
	SUBL	$4, BP
	JMP	cmp_mediumloop

cmp_0through4:
	MOVL	-4(SI)(BP*1), AX
	MOVL	-4(DI)(BP*1), CX
	CMPL	AX, CX
	JEQ	cmp_allsame

cmp_diff4:
	BSWAPL	AX	// reverse order of bytes
	BSWAPL	CX
	XORL	AX, CX	// find bit differences
	BSRL	CX, CX	// index of highest bit difference
	SHRL	CX, AX	// move a's bit to bottom
	ANDL	$1, AX	// mask bit
	LEAL	-1(AX*2), AX // 1/0 => +1/-1
	RET

	// 0-3 bytes in common
cmp_small:
	LEAL	(BP*8), CX
	NEGL	CX
	JEQ	cmp_allsame

	// load si
	CMPB	SI, $0xfc
	JA	cmp_si_high
	MOVL	(SI), SI
	JMP	cmp_si_finish
cmp_si_high:
	MOVL	-4(SI)(BP*1), SI
	SHRL	CX, SI
cmp_si_finish:
	SHLL	CX, SI

	// same for di
	CMPB	DI, $0xfc
	JA	cmp_di_high
	MOVL	(DI), DI
	JMP	cmp_di_finish
cmp_di_high:
	MOVL	-4(DI)(BP*1), DI
	SHRL	CX, DI
cmp_di_finish:
	SHLL	CX, DI

	BSWAPL	SI	// reverse order of bytes
	BSWAPL	DI
	XORL	SI, DI	// find bit differences
	JEQ	cmp_allsame
	BSRL	DI, CX	// index of highest bit difference
	SHRL	CX, SI	// move a's bit to bottom
	ANDL	$1, SI	// mask bit
	LEAL	-1(SI*2), AX // 1/0 => +1/-1
	RET

	// all the bytes in common are the same, so we just need
	// to compare the lengths.
cmp_allsame:
	XORL	AX, AX
	XORL	CX, CX
	CMPL	BX, DX
	SETGT	AX	// 1 if alen > blen
	SETEQ	CX	// 1 if alen == blen
	LEAL	-1(CX)(AX*2), AX	// 1,0,-1 result
	RET

// A Duff's device for zeroing memory.
// The compiler jumps to computed addresses within
// this routine to zero chunks of memory.  Do not
// change this code without also changing the code
// in ../../cmd/8g/ggen.c:clearfat.
// AX: zero
// DI: ptr to memory to be zeroed
// DI is updated as a side effect.
TEXT runtime·duffzero(SB), NOSPLIT, $0-0
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	STOSL
	RET

// A Duff's device for copying memory.
// The compiler jumps to computed addresses within
// this routine to copy chunks of memory.  Source
// and destination must not overlap.  Do not
// change this code without also changing the code
// in ../../cmd/6g/cgen.c:sgen.
// SI: ptr to source memory
// DI: ptr to destination memory
// SI and DI are updated as a side effect.

// NOTE: this is equivalent to a sequence of MOVSL but
// for some reason MOVSL is really slow.
TEXT runtime·duffcopy(SB), NOSPLIT, $0-0
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	MOVL	(SI),CX
	ADDL	$4,SI
	MOVL	CX,(DI)
	ADDL	$4,DI
	
	RET

TEXT runtime·fastrand1(SB), NOSPLIT, $0-4
	get_tls(CX)
	MOVL	g(CX), AX
	MOVL	g_m(AX), AX
	MOVL	m_fastrand(AX), DX
	ADDL	DX, DX
	MOVL	DX, BX
	XORL	$0x88888eef, DX
	CMOVLMI	BX, DX
	MOVL	DX, m_fastrand(AX)
	MOVL	DX, ret+0(FP)
	RET

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVL	$0, AX
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$0
	get_tls(CX)
	MOVL	g(CX), AX
	MOVL	g_m(AX), AX
	MOVL	m_curg(AX), AX
	MOVL	(g_stack+stack_hi)(AX), AX
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT,$0-0
	BYTE	$0x90	// NOP
	CALL	runtime·goexit1(SB)	// does not return
