// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"
#include "asm_ppc64x.h"

#ifdef GOOS_aix
#define cgoCalleeStackSize 48
#else
#define cgoCalleeStackSize 32
#endif

TEXT runtime·rt0_go(SB),NOSPLIT,$0
	// R1 = stack; R3 = argc; R4 = argv; R13 = C TLS base pointer

	// initialize essential registers
	BL	runtime·reginit(SB)

	SUB	$(FIXED_FRAME+16), R1
	MOVD	R2, 24(R1)		// stash the TOC pointer away again now we've created a new frame
	MOVW	R3, FIXED_FRAME+0(R1)	// argc
	MOVD	R4, FIXED_FRAME+8(R1)	// argv

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVD	$runtime·g0(SB), g
	BL	runtime·save_g(SB)
	MOVD	$(-64*1024), R31
	ADD	R31, R1, R3
	MOVD	R3, g_stackguard0(g)
	MOVD	R3, g_stackguard1(g)
	MOVD	R3, (g_stack+stack_lo)(g)
	MOVD	R1, (g_stack+stack_hi)(g)

	// if there is a _cgo_init, call it using the gcc ABI.
	MOVD	_cgo_init(SB), R12
	CMP	R0, R12
	BEQ	nocgo
#ifdef GOARCH_ppc64
	// ppc64 use elf ABI v1. we must get the real entry address from
	// first slot of the function descriptor before call.
	MOVD	8(R12), R2
	MOVD	(R12), R12
#endif
	MOVD	R12, CTR		// r12 = "global function entry point"
	MOVD	R13, R5			// arg 2: TLS base pointer
	MOVD	$setg_gcc<>(SB), R4 	// arg 1: setg
	MOVD	g, R3			// arg 0: G
	// C functions expect 32 (48 for AIX) bytes of space on caller
	// stack frame and a 16-byte aligned R1
	MOVD	R1, R14			// save current stack
	SUB	$cgoCalleeStackSize, R1	// reserve the callee area
	RLDCR	$0, R1, $~15, R1	// 16-byte align
	BL	(CTR)			// may clobber R0, R3-R12
	MOVD	R14, R1			// restore stack
#ifndef GOOS_aix
	MOVD	24(R1), R2
#endif
	XOR	R0, R0			// fix R0

nocgo:
	// update stackguard after _cgo_init
	MOVD	(g_stack+stack_lo)(g), R3
	ADD	$const__StackGuard, R3
	MOVD	R3, g_stackguard0(g)
	MOVD	R3, g_stackguard1(g)

	// set the per-goroutine and per-mach "registers"
	MOVD	$runtime·m0(SB), R3

	// save m->g0 = g0
	MOVD	g, m_g0(R3)
	// save m0 to g0->m
	MOVD	R3, g_m(g)

	BL	runtime·check(SB)

	// args are already prepared
	BL	runtime·args(SB)
	BL	runtime·osinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVD	$runtime·mainPC(SB), R3		// entry
	MOVDU	R3, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	BL	runtime·newproc(SB)
	ADD	$(16+FIXED_FRAME), R1

	// start this M
	BL	runtime·mstart(SB)

	MOVD	R0, 0(R0)
	RET

DATA	runtime·mainPC+0(SB)/8,$runtime·main(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R0, 0(R0) // TODO: TD
	RET

TEXT runtime·asminit(SB),NOSPLIT|NOFRAME,$0-0
	RET

// Any changes must be reflected to runtime/cgo/gcc_aix_ppc64.S:.crosscall_ppc64
TEXT _cgo_reginit(SB),NOSPLIT|NOFRAME,$0-0
	// crosscall_ppc64 and crosscall2 need to reginit, but can't
	// get at the 'runtime.reginit' symbol.
	BR	runtime·reginit(SB)

TEXT runtime·reginit(SB),NOSPLIT|NOFRAME,$0-0
	// set R0 to zero, it's expected by the toolchain
	XOR R0, R0
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT|NOFRAME, $0-8
	MOVD	buf+0(FP), R3
	MOVD	R1, gobuf_sp(R3)
	MOVD	LR, R31
	MOVD	R31, gobuf_pc(R3)
	MOVD	g, gobuf_g(R3)
	MOVD	R0, gobuf_lr(R3)
	MOVD	R0, gobuf_ret(R3)
	// Assert ctxt is zero. See func save.
	MOVD	gobuf_ctxt(R3), R3
	CMP	R0, R3
	BEQ	2(PC)
	BL	runtime·badctxt(SB)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $16-8
	MOVD	buf+0(FP), R5
	MOVD	gobuf_g(R5), g	// make sure g is not nil
	BL	runtime·save_g(SB)

	MOVD	0(g), R4
	MOVD	gobuf_sp(R5), R1
	MOVD	gobuf_lr(R5), R31
#ifndef GOOS_aix
	MOVD	24(R1), R2	// restore R2
#endif
	MOVD	R31, LR
	MOVD	gobuf_ret(R5), R3
	MOVD	gobuf_ctxt(R5), R11
	MOVD	R0, gobuf_sp(R5)
	MOVD	R0, gobuf_ret(R5)
	MOVD	R0, gobuf_lr(R5)
	MOVD	R0, gobuf_ctxt(R5)
	CMP	R0, R0 // set condition codes for == test, needed by stack split
	MOVD	gobuf_pc(R5), R12
	MOVD	R12, CTR
	BR	(CTR)

// void mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT|NOFRAME, $0-8
	// Save caller state in g->sched
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	LR, R31
	MOVD	R31, (g_sched+gobuf_pc)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	MOVD	g, (g_sched+gobuf_g)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVD	g, R3
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	CMP	g, R3
	BNE	2(PC)
	BR	runtime·badmcall(SB)
	MOVD	fn+0(FP), R11			// context
	MOVD	0(R11), R12			// code pointer
	MOVD	R12, CTR
	MOVD	(g_sched+gobuf_sp)(g), R1	// sp = m->g0->sched.sp
	MOVDU	R3, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	BL	(CTR)
	MOVD	24(R1), R2
	BR	runtime·badmcall2(SB)

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack. We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	// We have several undefs here so that 16 bytes past
	// $runtime·systemstack_switch lies within them whether or not the
        // instructions that derive r2 from r12 are there.
	UNDEF
	UNDEF
	UNDEF
	BL	(LR)	// make sure this function is not leaf
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	MOVD	fn+0(FP), R3	// R3 = fn
	MOVD	R3, R11		// context
	MOVD	g_m(g), R4	// R4 = m

	MOVD	m_gsignal(R4), R5	// R5 = gsignal
	CMP	g, R5
	BEQ	noswitch

	MOVD	m_g0(R4), R5	// R5 = g0
	CMP	g, R5
	BEQ	noswitch

	MOVD	m_curg(R4), R6
	CMP	g, R6
	BEQ	switch

	// Bad: g is not gsignal, not g0, not curg. What is it?
	// Hide call from linker nosplit analysis.
	MOVD	$runtime·badsystemstack(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
	BL	runtime·abort(SB)

switch:
	// save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVD	$runtime·systemstack_switch(SB), R6
	ADD     $16, R6 // get past prologue (including r2-setting instructions when they're there)
	MOVD	R6, (g_sched+gobuf_pc)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	MOVD	g, (g_sched+gobuf_g)(g)

	// switch to g0
	MOVD	R5, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R3
	// make it look like mstart called systemstack on g0, to stop traceback
	SUB	$FIXED_FRAME, R3
	MOVD	$runtime·mstart(SB), R4
	MOVD	R4, 0(R3)
	MOVD	R3, R1

	// call target function
	MOVD	0(R11), R12	// code pointer
	MOVD	R12, CTR
	BL	(CTR)

	// restore TOC pointer. It seems unlikely that we will use systemstack
	// to call a function defined in another module, but the results of
	// doing so would be so confusing that it's worth doing this.
	MOVD	g_m(g), R3
	MOVD	m_curg(R3), g
	MOVD	(g_sched+gobuf_sp)(g), R3
#ifndef GOOS_aix
	MOVD	24(R3), R2
#endif
	// switch back to g
	MOVD	g_m(g), R3
	MOVD	m_curg(R3), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1
	MOVD	R0, (g_sched+gobuf_sp)(g)
	RET

noswitch:
	// already on m stack, just call directly
	// On other arches we do a tail call here, but it appears to be
	// impossible to tail call a function pointer in shared mode on
	// ppc64 because the caller is responsible for restoring the TOC.
	MOVD	0(R11), R12	// code pointer
	MOVD	R12, CTR
	BL	(CTR)
#ifndef GOOS_aix
	MOVD	24(R1), R2
#endif
	RET

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// Caller has already loaded:
// R3: framesize, R4: argsize, R5: LR
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT|NOFRAME,$0-0
	// Cannot grow scheduler stack (m->g0).
	MOVD	g_m(g), R7
	MOVD	m_g0(R7), R8
	CMP	g, R8
	BNE	3(PC)
	BL	runtime·badmorestackg0(SB)
	BL	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVD	m_gsignal(R7), R8
	CMP	g, R8
	BNE	3(PC)
	BL	runtime·badmorestackgsignal(SB)
	BL	runtime·abort(SB)

	// Called from f.
	// Set g->sched to context in f.
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	LR, R8
	MOVD	R8, (g_sched+gobuf_pc)(g)
	MOVD	R5, (g_sched+gobuf_lr)(g)
	MOVD	R11, (g_sched+gobuf_ctxt)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVD	R5, (m_morebuf+gobuf_pc)(R7)	// f's caller's PC
	MOVD	R1, (m_morebuf+gobuf_sp)(R7)	// f's caller's SP
	MOVD	g, (m_morebuf+gobuf_g)(R7)

	// Call newstack on m->g0's stack.
	MOVD	m_g0(R7), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1
	MOVDU   R0, -(FIXED_FRAME+0)(R1)	// create a call frame on g0
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R0, R11
	BR	runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVD	$MAXSIZE, R31;		\
	CMP	R3, R31;		\
	BGT	4(PC);			\
	MOVD	$NAME(SB), R12;		\
	MOVD	R12, CTR;		\
	BR	(CTR)
// Note: can't just "BR NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT|NOFRAME, $0-32
	MOVWZ argsize+24(FP), R3
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
	MOVD	$runtime·badreflectcall(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-24;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVD	arg+16(FP), R3;			\
	MOVWZ	argsize+24(FP), R4;			\
	MOVD    R1, R5;				\
	CMP	R4, $8;				\
	BLT	tailsetup;			\
	/* copy 8 at a time if possible */	\
	ADD	$(FIXED_FRAME-8), R5;			\
	SUB	$8, R3;				\
top: \
	MOVDU	8(R3), R7;			\
	MOVDU	R7, 8(R5);			\
	SUB	$8, R4;				\
	CMP	R4, $8;				\
	BGE	top;				\
	/* handle remaining bytes */	\
	CMP	$0, R4;			\
	BEQ	callfn;			\
	ADD	$7, R3;			\
	ADD	$7, R5;			\
	BR	tail;			\
tailsetup: \
	CMP	$0, R4;			\
	BEQ	callfn;			\
	ADD     $(FIXED_FRAME-1), R5;	\
	SUB     $1, R3;			\
tail: \
	MOVBU	1(R3), R6;		\
	MOVBU	R6, 1(R5);		\
	SUB	$1, R4;			\
	CMP	$0, R4;			\
	BGT	tail;			\
callfn: \
	/* call function */			\
	MOVD	f+8(FP), R11;			\
#ifdef GOOS_aix				\
	/* AIX won't trigger a SIGSEGV if R11 = nil */	\
	/* So it manually triggers it */	\
	CMP	R0, R11				\
	BNE	2(PC)				\
	MOVD	R0, 0(R0)			\
#endif						\
	MOVD	(R11), R12;			\
	MOVD	R12, CTR;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	BL	(CTR);				\
#ifndef GOOS_aix				\
	MOVD	24(R1), R2;			\
#endif						\
	/* copy return values back */		\
	MOVD	argtype+0(FP), R7;		\
	MOVD	arg+16(FP), R3;			\
	MOVWZ	n+24(FP), R4;			\
	MOVWZ	retoffset+28(FP), R6;		\
	ADD	$FIXED_FRAME, R1, R5;		\
	ADD	R6, R5; 			\
	ADD	R6, R3;				\
	SUB	R6, R4;				\
	BL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $32-0
	MOVD	R7, FIXED_FRAME+0(R1)
	MOVD	R3, FIXED_FRAME+8(R1)
	MOVD	R5, FIXED_FRAME+16(R1)
	MOVD	R4, FIXED_FRAME+24(R1)
	BL	runtime·reflectcallmove(SB)
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

TEXT runtime·procyield(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	cycles+0(FP), R7
	// POWER does not have a pause/yield instruction equivalent.
	// Instead, we can lower the program priority by setting the
	// Program Priority Register prior to the wait loop and set it
	// back to default afterwards. On Linux, the default priority is
	// medium-low. For details, see page 837 of the ISA 3.0.
	OR	R1, R1, R1	// Set PPR priority to low
again:
	SUB	$1, R7
	CMP	$0, R7
	BNE	again
	OR	R6, R6, R6	// Set PPR priority back to medium-low
	RET

// void jmpdefer(fv, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 8 bytes to get back to either nop or toc reload before deferreturn
// 3. BR to fn
// When dynamically linking Go, it is not sufficient to rewind to the BL
// deferreturn -- we might be jumping between modules and so we need to reset
// the TOC pointer in r2. To do this, codegen inserts MOVD 24(R1), R2 *before*
// the BL deferreturn and jmpdefer rewinds to that.
TEXT runtime·jmpdefer(SB), NOSPLIT|NOFRAME, $0-16
	MOVD	0(R1), R31
	SUB     $8, R31
	MOVD	R31, LR

	MOVD	fv+0(FP), R11
	MOVD	argp+8(FP), R1
	SUB	$FIXED_FRAME, R1
#ifdef GOOS_aix
	// AIX won't trigger a SIGSEGV if R11 = nil
	// So it manually triggers it
	CMP	R0, R11
	BNE	2(PC)
	MOVD	R0, 0(R0)
#endif
	MOVD	0(R11), R12
	MOVD	R12, CTR
	BR	(CTR)

// Save state of caller into g->sched. Smashes R31.
TEXT gosave<>(SB),NOSPLIT|NOFRAME,$0
	MOVD	LR, R31
	MOVD	R31, (g_sched+gobuf_pc)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	MOVD	R0, (g_sched+gobuf_ret)(g)
	// Assert ctxt is zero. See func save.
	MOVD	(g_sched+gobuf_ctxt)(g), R31
	CMP	R0, R31
	BEQ	2(PC)
	BL	runtime·badctxt(SB)
	RET

#ifdef GOOS_aix
#define asmcgocallSaveOffset cgoCalleeStackSize + 8
#else
#define asmcgocallSaveOffset cgoCalleeStackSize
#endif

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	MOVD	fn+0(FP), R3
	MOVD	arg+8(FP), R4

	MOVD	R1, R7		// save original stack pointer
	MOVD	g, R5

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	// Moreover, if it's called inside the signal handler, it must not switch
	// to g0 as it can be in use by another syscall.
	MOVD	g_m(g), R8
	MOVD	m_gsignal(R8), R6
	CMP	R6, g
	BEQ	g0
	MOVD	m_g0(R8), R6
	CMP	R6, g
	BEQ	g0
	BL	gosave<>(SB)
	MOVD	R6, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1

	// Now on a scheduling stack (a pthread-created stack).
g0:
#ifdef GOOS_aix
	// Create a fake LR to improve backtrace.
	MOVD	$runtime·asmcgocall(SB), R6
	MOVD	R6, 16(R1)
	// AIX also save one argument on the stack.
	SUB $8, R1
#endif
	// Save room for two of our pointers, plus the callee
	// save area that lives on the caller stack.
	SUB	$(asmcgocallSaveOffset+16), R1
	RLDCR	$0, R1, $~15, R1	// 16-byte alignment for gcc ABI
	MOVD	R5, (asmcgocallSaveOffset+8)(R1)// save old g on stack
	MOVD	(g_stack+stack_hi)(R5), R5
	SUB	R7, R5
	MOVD	R5, asmcgocallSaveOffset(R1)    // save depth in old g stack (can't just save SP, as stack might be copied during a callback)
#ifdef GOOS_aix
	MOVD	R7, 0(R1)	// Save frame pointer to allow manual backtrace with gdb
#else
	MOVD	R0, 0(R1)	// clear back chain pointer (TODO can we give it real back trace information?)
#endif
	// This is a "global call", so put the global entry point in r12
	MOVD	R3, R12

#ifdef GOARCH_ppc64
	// ppc64 use elf ABI v1. we must get the real entry address from
	// first slot of the function descriptor before call.
	// Same for AIX.
	MOVD	8(R12), R2
	MOVD	(R12), R12
#endif
	MOVD	R12, CTR
	MOVD	R4, R3		// arg in r3
	BL	(CTR)
	// C code can clobber R0, so set it back to 0. F27-F31 are
	// callee save, so we don't need to recover those.
	XOR	R0, R0
	// Restore g, stack pointer, toc pointer.
	// R3 is errno, so don't touch it
	MOVD	(asmcgocallSaveOffset+8)(R1), g
	MOVD	(g_stack+stack_hi)(g), R5
	MOVD	asmcgocallSaveOffset(R1), R6
	SUB	R6, R5
#ifndef GOOS_aix
	MOVD	24(R5), R2
#endif
	MOVD	R5, R1
	BL	runtime·save_g(SB)

	MOVW	R3, ret+16(FP)
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize, uintptr ctxt)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$32-32
	MOVD	$fn+0(FP), R3
	MOVD	R3, FIXED_FRAME+0(R1)
	MOVD	frame+8(FP), R3
	MOVD	R3, FIXED_FRAME+8(R1)
	MOVD	framesize+16(FP), R3
	MOVD	R3, FIXED_FRAME+16(R1)
	MOVD	ctxt+24(FP), R3
	MOVD	R3, FIXED_FRAME+24(R1)
	MOVD	$runtime·cgocallback_gofunc(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize, uintptr ctxt)
// See cgocall.go for more details.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$16-32
	NO_LOCAL_POINTERS

	// Load m and g from thread-local storage.
	MOVBZ	runtime·iscgo(SB), R3
	CMP	R3, $0
	BEQ	nocgo
	BL	runtime·load_g(SB)
nocgo:

	// If g is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	CMP	g, $0
	BEQ	needm

	MOVD	g_m(g), R8
	MOVD	R8, savedm-8(SP)
	BR	havem

needm:
	MOVD	g, savedm-8(SP) // g is zero, so is m.
	MOVD	$runtime·needm(SB), R12
	MOVD	R12, CTR
	BL	(CTR)

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
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), R3
	MOVD	R1, (g_sched+gobuf_sp)(R3)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 8(R1) aka savedsp-16(SP).
	MOVD	m_g0(R8), R3
	MOVD	(g_sched+gobuf_sp)(R3), R4
	MOVD	R4, savedsp-16(SP)
	MOVD	R1, (g_sched+gobuf_sp)(R3)

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
	MOVD	m_curg(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R4 // prepare stack as R4
	MOVD	(g_sched+gobuf_pc)(g), R5
	MOVD	R5, -(FIXED_FRAME+16)(R4)
	MOVD	ctxt+24(FP), R3
	MOVD	R3, -16(R4)
	MOVD	$-(FIXED_FRAME+16)(R4), R1
	BL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVD	0(R1), R5
	MOVD	R5, (g_sched+gobuf_pc)(g)
	MOVD	$(FIXED_FRAME+16)(R1), R4
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1
	MOVD	savedsp-16(SP), R4
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	MOVD	savedm-8(SP), R6
	CMP	R6, $0
	BNE	droppedm
	MOVD	$runtime·dropm(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
droppedm:

	// Done!
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-8
	MOVD	gg+0(FP), g
	// This only happens if iscgo, so jump straight to save_g
	BL	runtime·save_g(SB)
	RET

#ifdef GOARCH_ppc64
#ifdef GOOS_aix
DATA    setg_gcc<>+0(SB)/8, $_setg_gcc<>(SB)
DATA    setg_gcc<>+8(SB)/8, $TOC(SB)
DATA    setg_gcc<>+16(SB)/8, $0
GLOBL   setg_gcc<>(SB), NOPTR, $24
#else
TEXT setg_gcc<>(SB),NOSPLIT|NOFRAME,$0-0
	DWORD	$_setg_gcc<>(SB)
	DWORD	$0
	DWORD	$0
#endif
#endif

// void setg_gcc(G*); set g in C TLS.
// Must obey the gcc calling convention.
#ifdef GOARCH_ppc64le
TEXT setg_gcc<>(SB),NOSPLIT|NOFRAME,$0-0
#else
TEXT _setg_gcc<>(SB),NOSPLIT|NOFRAME,$0-0
#endif
	// The standard prologue clobbers R31, which is callee-save in
	// the C ABI, so we have to use $-8-0 and save LR ourselves.
	MOVD	LR, R4
	// Also save g and R31, since they're callee-save in C ABI
	MOVD	R31, R5
	MOVD	g, R6

	MOVD	R3, g
	BL	runtime·save_g(SB)

	MOVD	R6, g
	MOVD	R5, R31
	MOVD	R4, LR
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R0
	UNDEF

#define	TBR	268

// int64 runtime·cputicks(void)
TEXT runtime·cputicks(SB),NOSPLIT,$0-8
	MOVD	SPR(TBR), R3
	MOVD	R3, ret+0(FP)
	RET

// AES hashing not implemented for ppc64
TEXT runtime·aeshash(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R1
TEXT runtime·aeshash32(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R1
TEXT runtime·aeshash64(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R1
TEXT runtime·aeshashstr(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R1

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVW	$0, R3
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
#ifdef GOOS_aix
// On AIX, _cgo_topofstack is defined in runtime/cgo, because it must
// be a longcall in order to prevent trampolines from ld.
TEXT __cgo_topofstack(SB),NOSPLIT|NOFRAME,$0
#else
TEXT _cgo_topofstack(SB),NOSPLIT|NOFRAME,$0
#endif
	// g (R30) and R31 are callee-save in the C ABI, so save them
	MOVD	g, R4
	MOVD	R31, R5
	MOVD	LR, R6

	BL	runtime·load_g(SB)	// clobbers g (R30), R31
	MOVD	g_m(g), R3
	MOVD	m_curg(R3), R3
	MOVD	(g_stack+stack_hi)(R3), R3

	MOVD	R4, g
	MOVD	R5, R31
	MOVD	R6, LR
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
//
// When dynamically linking Go, it can be returned to from a function
// implemented in a different module and so needs to reload the TOC pointer
// from the stack (although this function declares that it does not set up x-a
// frame, newproc1 does in fact allocate one for goexit and saves the TOC
// pointer in the correct place).
// goexit+_PCQuantum is halfway through the usual global entry point prologue
// that derives r2 from r12 which is a bit silly, but not harmful.
TEXT runtime·goexit(SB),NOSPLIT|NOFRAME|TOPFRAME,$0-0
	MOVD	24(R1), R2
	BL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	MOVD	R0, R0	// NOP

// prepGoExitFrame saves the current TOC pointer (i.e. the TOC pointer for the
// module containing runtime) to the frame that goexit will execute in when
// the goroutine exits. It's implemented in assembly mainly because that's the
// easiest way to get access to R2.
TEXT runtime·prepGoExitFrame(SB),NOSPLIT,$0-8
      MOVD    sp+0(FP), R3
      MOVD    R2, 24(R3)
      RET

TEXT runtime·addmoduledata(SB),NOSPLIT|NOFRAME,$0-0
	ADD	$-8, R1
	MOVD	R31, 0(R1)
	MOVD	runtime·lastmoduledatap(SB), R4
	MOVD	R3, moduledata_next(R4)
	MOVD	R3, runtime·lastmoduledatap(SB)
	MOVD	0(R1), R31
	ADD	$8, R1
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R3
	MOVB	R3, ret+0(FP)
	RET

// gcWriteBarrier performs a heap pointer write and informs the GC.
//
// gcWriteBarrier does NOT follow the Go ABI. It takes two arguments:
// - R20 is the destination of the write
// - R21 is the value being written at R20.
// It clobbers condition codes.
// It does not clobber R0 through R15,
// but may clobber any other register, *including* R31.
TEXT runtime·gcWriteBarrier(SB),NOSPLIT,$112
	// The standard prologue clobbers R31.
	// We use R16 and R17 as scratch registers.
	MOVD	g_m(g), R16
	MOVD	m_p(R16), R16
	MOVD	(p_wbBuf+wbBuf_next)(R16), R17
	// Increment wbBuf.next position.
	ADD	$16, R17
	MOVD	R17, (p_wbBuf+wbBuf_next)(R16)
	MOVD	(p_wbBuf+wbBuf_end)(R16), R16
	CMP	R16, R17
	// Record the write.
	MOVD	R21, -16(R17)	// Record value
	MOVD	(R20), R16	// TODO: This turns bad writes into bad reads.
	MOVD	R16, -8(R17)	// Record *slot
	// Is the buffer full? (flags set in CMP above)
	BEQ	flush
ret:
	// Do the write.
	MOVD	R21, (R20)
	RET

flush:
	// Save registers R0 through R15 since these were not saved by the caller.
	// We don't save all registers on ppc64 because it takes too much space.
	MOVD	R20, (FIXED_FRAME+0)(R1)	// Also first argument to wbBufFlush
	MOVD	R21, (FIXED_FRAME+8)(R1)	// Also second argument to wbBufFlush
	// R0 is always 0, so no need to spill.
	// R1 is SP.
	// R2 is SB.
	MOVD	R3, (FIXED_FRAME+16)(R1)
	MOVD	R4, (FIXED_FRAME+24)(R1)
	MOVD	R5, (FIXED_FRAME+32)(R1)
	MOVD	R6, (FIXED_FRAME+40)(R1)
	MOVD	R7, (FIXED_FRAME+48)(R1)
	MOVD	R8, (FIXED_FRAME+56)(R1)
	MOVD	R9, (FIXED_FRAME+64)(R1)
	MOVD	R10, (FIXED_FRAME+72)(R1)
	MOVD	R11, (FIXED_FRAME+80)(R1)
	MOVD	R12, (FIXED_FRAME+88)(R1)
	// R13 is REGTLS
	MOVD	R14, (FIXED_FRAME+96)(R1)
	MOVD	R15, (FIXED_FRAME+104)(R1)

	// This takes arguments R20 and R21.
	CALL	runtime·wbBufFlush(SB)

	MOVD	(FIXED_FRAME+0)(R1), R20
	MOVD	(FIXED_FRAME+8)(R1), R21
	MOVD	(FIXED_FRAME+16)(R1), R3
	MOVD	(FIXED_FRAME+24)(R1), R4
	MOVD	(FIXED_FRAME+32)(R1), R5
	MOVD	(FIXED_FRAME+40)(R1), R6
	MOVD	(FIXED_FRAME+48)(R1), R7
	MOVD	(FIXED_FRAME+56)(R1), R8
	MOVD	(FIXED_FRAME+64)(R1), R9
	MOVD	(FIXED_FRAME+72)(R1), R10
	MOVD	(FIXED_FRAME+80)(R1), R11
	MOVD	(FIXED_FRAME+88)(R1), R12
	MOVD	(FIXED_FRAME+96)(R1), R14
	MOVD	(FIXED_FRAME+104)(R1), R15
	JMP	ret

// Note: these functions use a special calling convention to save generated code space.
// Arguments are passed in registers, but the space for those arguments are allocated
// in the caller's stack frame. These stubs write the args into that stack space and
// then tail call to the corresponding runtime handler.
// The tail call makes these stubs disappear in backtraces.
TEXT runtime·panicIndex(SB),NOSPLIT,$0-16
	MOVD	R3, x+0(FP)
	MOVD	R4, y+8(FP)
	JMP	runtime·goPanicIndex(SB)
TEXT runtime·panicIndexU(SB),NOSPLIT,$0-16
	MOVD	R3, x+0(FP)
	MOVD	R4, y+8(FP)
	JMP	runtime·goPanicIndexU(SB)
TEXT runtime·panicSliceAlen(SB),NOSPLIT,$0-16
	MOVD	R4, x+0(FP)
	MOVD	R5, y+8(FP)
	JMP	runtime·goPanicSliceAlen(SB)
TEXT runtime·panicSliceAlenU(SB),NOSPLIT,$0-16
	MOVD	R4, x+0(FP)
	MOVD	R5, y+8(FP)
	JMP	runtime·goPanicSliceAlenU(SB)
TEXT runtime·panicSliceAcap(SB),NOSPLIT,$0-16
	MOVD	R4, x+0(FP)
	MOVD	R5, y+8(FP)
	JMP	runtime·goPanicSliceAcap(SB)
TEXT runtime·panicSliceAcapU(SB),NOSPLIT,$0-16
	MOVD	R4, x+0(FP)
	MOVD	R5, y+8(FP)
	JMP	runtime·goPanicSliceAcapU(SB)
TEXT runtime·panicSliceB(SB),NOSPLIT,$0-16
	MOVD	R3, x+0(FP)
	MOVD	R4, y+8(FP)
	JMP	runtime·goPanicSliceB(SB)
TEXT runtime·panicSliceBU(SB),NOSPLIT,$0-16
	MOVD	R3, x+0(FP)
	MOVD	R4, y+8(FP)
	JMP	runtime·goPanicSliceBU(SB)
TEXT runtime·panicSlice3Alen(SB),NOSPLIT,$0-16
	MOVD	R5, x+0(FP)
	MOVD	R6, y+8(FP)
	JMP	runtime·goPanicSlice3Alen(SB)
TEXT runtime·panicSlice3AlenU(SB),NOSPLIT,$0-16
	MOVD	R5, x+0(FP)
	MOVD	R6, y+8(FP)
	JMP	runtime·goPanicSlice3AlenU(SB)
TEXT runtime·panicSlice3Acap(SB),NOSPLIT,$0-16
	MOVD	R5, x+0(FP)
	MOVD	R6, y+8(FP)
	JMP	runtime·goPanicSlice3Acap(SB)
TEXT runtime·panicSlice3AcapU(SB),NOSPLIT,$0-16
	MOVD	R5, x+0(FP)
	MOVD	R6, y+8(FP)
	JMP	runtime·goPanicSlice3AcapU(SB)
TEXT runtime·panicSlice3B(SB),NOSPLIT,$0-16
	MOVD	R4, x+0(FP)
	MOVD	R5, y+8(FP)
	JMP	runtime·goPanicSlice3B(SB)
TEXT runtime·panicSlice3BU(SB),NOSPLIT,$0-16
	MOVD	R4, x+0(FP)
	MOVD	R5, y+8(FP)
	JMP	runtime·goPanicSlice3BU(SB)
TEXT runtime·panicSlice3C(SB),NOSPLIT,$0-16
	MOVD	R3, x+0(FP)
	MOVD	R4, y+8(FP)
	JMP	runtime·goPanicSlice3C(SB)
TEXT runtime·panicSlice3CU(SB),NOSPLIT,$0-16
	MOVD	R3, x+0(FP)
	MOVD	R4, y+8(FP)
	JMP	runtime·goPanicSlice3CU(SB)
