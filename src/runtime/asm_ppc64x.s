// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"
#include "asm_ppc64x.h"
#include "cgo/abi_ppc64x.h"

// This is called using the host ABI. argc and argv arguments
// should be in R3 and R4 respectively.
TEXT _rt0_ppc64x_lib(SB),NOSPLIT|NOFRAME,$0
	// Start with standard C stack frame layout and linkage, allocate
	// 16 bytes of argument space, save callee-save regs, and set R0 to $0.
	// Allocate an extra 16 bytes to account for the larger fixed frame size
	// of aix/elfv1 (48 vs 32) to ensure 16 bytes of parameter save space.
	STACK_AND_SAVE_HOST_TO_GO_ABI(32)
	// The above will not preserve R2 (TOC). Save it in case Go is
	// compiled without a TOC pointer (e.g -buildmode=default).
	MOVD	R2, 24(R1)

	MOVD	R3, _rt0_ppc64x_lib_argc<>(SB)
	MOVD	R4, _rt0_ppc64x_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·reginit(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	$runtime·libpreinit(SB), R12
	MOVD	R12, CTR
	BL	(CTR)

#ifdef GOOS_aix
	// See runtime/cgo/gcc_aix_ppc64.c
	MOVBZ	runtime·isarchive(SB), R3	// Check buildmode = c-archive
	CMP		$0, R3
	BEQ		done
#endif

	// Create a new thread to do the runtime initialization and return.
	// _cgo_sys_thread_create is a C function.
	MOVD	_cgo_sys_thread_create(SB), R12
	CMP	$0, R12
	BEQ	nocgo
	MOVD	$_rt0_ppc64x_lib_go(SB), R3
	MOVD	$0, R4
#ifdef GO_PPC64X_HAS_FUNCDESC
	// Load the real entry address from the first slot of the function descriptor.
	MOVD	8(R12), R2
	MOVD	(R12), R12
#endif
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	24(R1), R2 // Restore the old frame, and R2.
	BR	done

nocgo:
	MOVD	$0x800000, R12                     // stacksize = 8192KB
	MOVD	R12, 8+FIXED_FRAME(R1)
	MOVD	$_rt0_ppc64x_lib_go(SB), R12
	MOVD	R12, 16+FIXED_FRAME(R1)
	MOVD	$runtime·newosproc0(SB),R12
	MOVD	R12, CTR
	BL	(CTR)

done:
	UNSTACK_AND_RESTORE_GO_TO_HOST_ABI(32)
	RET

#ifdef GO_PPC64X_HAS_FUNCDESC
DEFINE_PPC64X_FUNCDESC(_rt0_ppc64x_lib_go, __rt0_ppc64x_lib_go)
TEXT __rt0_ppc64x_lib_go(SB),NOSPLIT,$0
#else
TEXT _rt0_ppc64x_lib_go(SB),NOSPLIT,$0
#endif
	MOVD	_rt0_ppc64x_lib_argc<>(SB), R3
	MOVD	_rt0_ppc64x_lib_argv<>(SB), R4
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

DATA _rt0_ppc64x_lib_argc<>(SB)/8, $0
GLOBL _rt0_ppc64x_lib_argc<>(SB),NOPTR, $8
DATA _rt0_ppc64x_lib_argv<>(SB)/8, $0
GLOBL _rt0_ppc64x_lib_argv<>(SB),NOPTR, $8


#ifdef GOOS_aix
#define cgoCalleeStackSize 48
#else
#define cgoCalleeStackSize 32
#endif

TEXT runtime·rt0_go(SB),NOSPLIT|TOPFRAME,$0
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

	// If there is a _cgo_init, call it using the gcc ABI.
	MOVD	_cgo_init(SB), R12
	CMP	R12, $0
	BEQ	nocgo

#ifdef GO_PPC64X_HAS_FUNCDESC
	// Load the real entry address from the first slot of the function descriptor.
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
	ADD	$const_stackGuard, R3
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
	BL	runtime·newproc(SB)
	ADD	$(8+FIXED_FRAME), R1

	// start this M
	BL	runtime·mstart(SB)
	// Prevent dead-code elimination of debugCallV2 and debugPinnerV1, which are
	// intended to be called by debuggers.
#ifdef GOARCH_ppc64le
	MOVD	$runtime·debugPinnerV1<ABIInternal>(SB), R31
	MOVD	$runtime·debugCallV2<ABIInternal>(SB), R31
#endif
	MOVD	R0, 0(R0)
	RET

DATA	runtime·mainPC+0(SB)/8,$runtime·main<ABIInternal>(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT|NOFRAME,$0-0
	TW	$31, R0, R0
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

TEXT runtime·mstart(SB),NOSPLIT|TOPFRAME,$0
	BL	runtime·mstart0(SB)
	RET // not reached

/*
 *  go-routine
 */

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT|NOFRAME, $0-8
	MOVD	buf+0(FP), R5
	MOVD	gobuf_g(R5), R6
	MOVD	0(R6), R4	// make sure g != nil
	BR	gogo<>(SB)

TEXT gogo<>(SB), NOSPLIT|NOFRAME, $0
	MOVD	R6, g
	BL	runtime·save_g(SB)

	MOVD	gobuf_sp(R5), R1
	MOVD	gobuf_lr(R5), R31
#ifndef GOOS_aix
	MOVD	24(R1), R2	// restore R2
#endif
	MOVD	R31, LR
	MOVD	gobuf_ctxt(R5), R11
	MOVD	R0, gobuf_sp(R5)
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
TEXT runtime·mcall<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-8
	// Save caller state in g->sched
	// R11 should be safe across save_g??
	MOVD	R3, R11
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	LR, R31
	MOVD	R31, (g_sched+gobuf_pc)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVD	g, R3
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	CMP	g, R3
	BNE	2(PC)
	BR	runtime·badmcall(SB)
	MOVD	0(R11), R12			// code pointer
	MOVD	R12, CTR
	MOVD	(g_sched+gobuf_sp)(g), R1	// sp = m->g0->sched.sp
	// Don't need to do anything special for regabiargs here
	// R3 is g; stack is set anyway
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
	BL	gosave_systemstack_switch<>(SB)

	// switch to g0
	MOVD	R5, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1

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

// func switchToCrashStack0(fn func())
TEXT runtime·switchToCrashStack0<ABIInternal>(SB), NOSPLIT, $0-8
	MOVD	R3, R11				// context register
	MOVD	g_m(g), R3			// curm

	// set g to gcrash
	MOVD	$runtime·gcrash(SB), g	// g = &gcrash
	CALL	runtime·save_g(SB)	// clobbers R31
	MOVD	R3, g_m(g)			// g.m = curm
	MOVD	g, m_g0(R3)			// curm.g0 = g

	// switch to crashstack
	MOVD	(g_stack+stack_hi)(g), R3
	SUB	$(4*8), R3
	MOVD	R3, R1

	// call target function
	MOVD	0(R11), R12			// code pointer
	MOVD	R12, CTR
	BL	(CTR)

	// should never return
	CALL	runtime·abort(SB)
	UNDEF

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
	// Called from f.
	// Set g->sched to context in f.
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	LR, R8
	MOVD	R8, (g_sched+gobuf_pc)(g)
	MOVD	R5, (g_sched+gobuf_lr)(g)
	MOVD	R11, (g_sched+gobuf_ctxt)(g)

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
	// Force SPWRITE. This function doesn't actually write SP,
	// but it is called with a special calling convention where
	// the caller doesn't save LR on stack but passes it as a
	// register (R5), and the unwinder currently doesn't understand.
	// Make it SPWRITE to stop unwinding. (See issue 54332)
	// Use OR R0, R1 instead of MOVD R1, R1 as the MOVD instruction
	// has a special affect on Power8,9,10 by lowering the thread 
	// priority and causing a slowdown in execution time

	OR	R0, R1
	MOVD	R0, R11
	BR	runtime·morestack(SB)

// reflectcall: call a function with the given argument list
// func call(stackArgsType *_type, f *FuncVal, stackArgs *byte, stackArgsSize, stackRetOffset, frameSize uint32, regArgs *abi.RegArgs).
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

TEXT ·reflectcall(SB), NOSPLIT|NOFRAME, $0-48
	MOVWZ	frameSize+32(FP), R3
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
	MOVD	$runtime·badreflectcall(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-48;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVD	stackArgs+16(FP), R3;			\
	MOVWZ	stackArgsSize+24(FP), R4;			\
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
	CMP	R11, $0				\
	BNE	2(PC)				\
	MOVD	R0, 0(R0)			\
#endif						\
	MOVD    regArgs+40(FP), R20;    \
	BL      runtime·unspillArgs(SB);        \
	MOVD	(R11), R12;			\
	MOVD	R12, CTR;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	BL	(CTR);				\
#ifndef GOOS_aix				\
	MOVD	24(R1), R2;			\
#endif						\
	/* copy return values back */		\
	MOVD	regArgs+40(FP), R20;		\
	BL	runtime·spillArgs(SB);			\
	MOVD	stackArgsType+0(FP), R7;		\
	MOVD	stackArgs+16(FP), R3;			\
	MOVWZ	stackArgsSize+24(FP), R4;			\
	MOVWZ	stackRetOffset+28(FP), R6;		\
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
TEXT callRet<>(SB), NOSPLIT, $40-0
	NO_LOCAL_POINTERS
	MOVD	R7, FIXED_FRAME+0(R1)
	MOVD	R3, FIXED_FRAME+8(R1)
	MOVD	R5, FIXED_FRAME+16(R1)
	MOVD	R4, FIXED_FRAME+24(R1)
	MOVD	R20, FIXED_FRAME+32(R1)
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

TEXT runtime·procyieldAsm(SB),NOSPLIT|NOFRAME,$0-4
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

// Save state of caller into g->sched,
// but using fake PC from systemstack_switch.
// Must only be called from functions with no locals ($0)
// or else unwinding from systemstack_switch is incorrect.
// Smashes R31.
TEXT gosave_systemstack_switch<>(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·systemstack_switch(SB), R31
	ADD     $16, R31 // get past prologue (including r2-setting instructions when they're there)
	MOVD	R31, (g_sched+gobuf_pc)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	// Assert ctxt is zero. See func save.
	MOVD	(g_sched+gobuf_ctxt)(g), R31
	CMP	R31, $0
	BEQ	2(PC)
	BL	runtime·abort(SB)
	RET

#ifdef GOOS_aix
#define asmcgocallSaveOffset cgoCalleeStackSize + 8
#else
#define asmcgocallSaveOffset cgoCalleeStackSize
#endif

// func asmcgocall_no_g(fn, arg unsafe.Pointer)
// Call fn(arg) aligned appropriately for the gcc ABI.
// Called on a system stack, and there may be no g yet (during needm).
TEXT ·asmcgocall_no_g(SB),NOSPLIT,$0-16
	MOVD	fn+0(FP), R3
	MOVD	arg+8(FP), R4

	MOVD	R1, R15
	SUB	$(asmcgocallSaveOffset+8), R1
	RLDCR	$0, R1, $~15, R1	// 16-byte alignment for gcc ABI
	MOVD	R15, asmcgocallSaveOffset(R1)

	MOVD	R0, 0(R1)	// clear back chain pointer (TODO can we give it real back trace information?)

	// This is a "global call", so put the global entry point in r12
	MOVD	R3, R12

#ifdef GO_PPC64X_HAS_FUNCDESC
	// Load the real entry address from the first slot of the function descriptor.
	MOVD	8(R12), R2
	MOVD	(R12), R12
#endif
	MOVD	R12, CTR
	MOVD	R4, R3		// arg in r3
	BL	(CTR)

	// C code can clobber R0, so set it back to 0. F27-F31 are
	// callee save, so we don't need to recover those.
	XOR	R0, R0

	MOVD	asmcgocallSaveOffset(R1), R1	// Restore stack pointer.
#ifndef GOOS_aix
	MOVD	24(R1), R2
#endif

	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall<ABIInternal>(SB),NOSPLIT,$0-20
	// R3 = fn
	// R4 = arg

	MOVD	R1, R7		// save original stack pointer
	CMP	$0, g
	BEQ	nosave
	MOVD	g, R5

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already. Or we might already
	// be on the m->gsignal stack.
	MOVD	g_m(g), R8
	MOVD	m_gsignal(R8), R6
	CMP	R6, g
	BEQ	nosave
	MOVD	m_g0(R8), R6
	CMP	R6, g
	BEQ	nosave

	BL	gosave_systemstack_switch<>(SB)
	MOVD	R6, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1

	// Now on a scheduling stack (a pthread-created stack).
#ifdef GOOS_aix
	// Create a fake LR to improve backtrace.
	MOVD	$runtime·asmcgocall(SB), R6
	MOVD	R6, 16(R1)
	// AIX also saves one argument on the stack.
	SUB	$8, R1
#endif
	// Save room for two of our pointers, plus the callee
	// save area that lives on the caller stack.
	// Do arithmetics in R10 to hide from the assembler
	// counting it as SP delta, which is irrelevant as we are
	// on the system stack.
	SUB	$(asmcgocallSaveOffset+16), R1, R10
	RLDCR	$0, R10, $~15, R1	// 16-byte alignment for gcc ABI
	MOVD	R5, (asmcgocallSaveOffset+8)(R1)	// save old g on stack
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

#ifdef GO_PPC64X_HAS_FUNCDESC
	// Load the real entry address from the first slot of the function descriptor.
	MOVD	8(R12), R2
	MOVD	(R12), R12
#endif
	MOVD	R12, CTR
	MOVD	R4, R3		// arg in r3
	BL	(CTR)

	// Reinitialise zero value register.
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

	// ret = R3
	RET

nosave:
	// Running on a system stack, perhaps even without a g.
	// Having no g can happen during thread creation or thread teardown.
	// This code is like the above sequence but without saving/restoring g
	// and without worrying about the stack moving out from under us
	// (because we're on a system stack, not a goroutine stack).
	// The above code could be used directly if already on a system stack,
	// but then the only path through this code would be a rare case.
	// Using this code for all "already on system stack" calls exercises it more,
	// which should help keep it correct.

	SUB	$(asmcgocallSaveOffset+8), R1, R10
	RLDCR	$0, R10, $~15, R1		// 16-byte alignment for gcc ABI
	MOVD	R7, asmcgocallSaveOffset(R1)	// Save original stack pointer.

	MOVD	R3, R12		// fn
#ifdef GO_PPC64X_HAS_FUNCDESC
	// Load the real entry address from the first slot of the function descriptor.
	MOVD	8(R12), R2
	MOVD	(R12), R12
#endif
	MOVD	R12, CTR
	MOVD	R4, R3		// arg
	BL	(CTR)

	// Reinitialise zero value register.
	XOR	R0, R0

	MOVD	asmcgocallSaveOffset(R1), R1	// Restore stack pointer.
#ifndef GOOS_aix
	MOVD	24(R1), R2
#endif
	// ret = R3
	RET

// func cgocallback(fn, frame unsafe.Pointer, ctxt uintptr)
// See cgocall.go for more details.
TEXT ·cgocallback(SB),NOSPLIT,$24-24
	NO_LOCAL_POINTERS

	// Skip cgocallbackg, just dropm when fn is nil, and frame is the saved g.
	// It is used to dropm while thread is exiting.
	MOVD	fn+0(FP), R5
	CMP	R5, $0
	BNE	loadg
	// Restore the g from frame.
	MOVD	frame+8(FP), g
	BR	dropm

loadg:
	// Load m and g from thread-local storage.
#ifndef GOOS_openbsd
	MOVBZ	runtime·iscgo(SB), R3
	CMP	R3, $0
	BEQ	nocgo
#endif
	BL	runtime·load_g(SB)
nocgo:

	// If g is nil, Go did not create the current thread,
	// or if this thread never called into Go on pthread platforms.
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
	MOVD	$runtime·needAndBindM(SB), R12
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
	MOVD	R4, savedsp-24(SP)      // must match frame size
	MOVD	R1, (g_sched+gobuf_sp)(R3)

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
	MOVD	m_curg(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R4 // prepare stack as R4
	MOVD	(g_sched+gobuf_pc)(g), R5
	MOVD	R5, -(24+FIXED_FRAME)(R4)       // "saved LR"; must match frame size
	// Gather our arguments into registers.
	MOVD	fn+0(FP), R5
	MOVD	frame+8(FP), R6
	MOVD	ctxt+16(FP), R7
	MOVD	$-(24+FIXED_FRAME)(R4), R1      // switch stack; must match frame size
	MOVD    R5, FIXED_FRAME+0(R1)
	MOVD    R6, FIXED_FRAME+8(R1)
	MOVD    R7, FIXED_FRAME+16(R1)

	MOVD	$runtime·cgocallbackg(SB), R12
	MOVD	R12, CTR
	CALL	(CTR) // indirect call to bypass nosplit check. We're on a different stack now.

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVD	0(R1), R5
	MOVD	R5, (g_sched+gobuf_pc)(g)
	MOVD	$(24+FIXED_FRAME)(R1), R4       // must match frame size
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R1
	MOVD	savedsp-24(SP), R4      // must match frame size
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m,
	// 1. for the duration of the call on non-pthread platforms,
	// 2. or the duration of the C thread alive on pthread platforms.
	// If the m on entry wasn't nil,
	// 1. the thread might be a Go thread,
	// 2. or it wasn't the first call from a C thread on pthread platforms,
	//    since then we skip dropm to reuse the m in the first call.
	MOVD	savedm-8(SP), R6
	CMP	R6, $0
	BNE	droppedm

	// Skip dropm to reuse it in the next call, when a pthread key has been created.
	MOVD	_cgo_pthread_key_created(SB), R6
	// It means cgo is disabled when _cgo_pthread_key_created is a nil pointer, need dropm.
	CMP	R6, $0
	BEQ	dropm
	MOVD	(R6), R6
	CMP	R6, $0
	BNE	droppedm

dropm:
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

#ifdef GO_PPC64X_HAS_FUNCDESC
DEFINE_PPC64X_FUNCDESC(setg_gcc<>, _setg_gcc<>)
TEXT _setg_gcc<>(SB),NOSPLIT|NOFRAME,$0-0
#else
TEXT setg_gcc<>(SB),NOSPLIT|NOFRAME,$0-0
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

// spillArgs stores return values from registers to a *internal/abi.RegArgs in R20.
TEXT runtime·spillArgs(SB),NOSPLIT,$0-0
	MOVD    R3, 0(R20)
	MOVD    R4, 8(R20)
	MOVD    R5, 16(R20)
	MOVD    R6, 24(R20)
	MOVD    R7, 32(R20)
	MOVD    R8, 40(R20)
	MOVD    R9, 48(R20)
	MOVD    R10, 56(R20)
	MOVD	R14, 64(R20)
	MOVD	R15, 72(R20)
	MOVD	R16, 80(R20)
	MOVD	R17, 88(R20)
	FMOVD	F1, 96(R20)
	FMOVD	F2, 104(R20)
	FMOVD   F3, 112(R20)
	FMOVD   F4, 120(R20)
	FMOVD   F5, 128(R20)
	FMOVD   F6, 136(R20)
	FMOVD   F7, 144(R20)
	FMOVD   F8, 152(R20)
	FMOVD   F9, 160(R20)
	FMOVD   F10, 168(R20)
	FMOVD   F11, 176(R20)
	FMOVD   F12, 184(R20)
	RET

// unspillArgs loads args into registers from a *internal/abi.RegArgs in R20.
TEXT runtime·unspillArgs(SB),NOSPLIT,$0-0
	MOVD    0(R20), R3
	MOVD    8(R20), R4
	MOVD    16(R20), R5
	MOVD    24(R20), R6
	MOVD    32(R20), R7
	MOVD    40(R20), R8
	MOVD    48(R20), R9
	MOVD    56(R20), R10
	MOVD    64(R20), R14
	MOVD    72(R20), R15
	MOVD    80(R20), R16
	MOVD    88(R20), R17
	FMOVD   96(R20), F1
	FMOVD   104(R20), F2
	FMOVD   112(R20), F3
	FMOVD   120(R20), F4
	FMOVD   128(R20), F5
	FMOVD   136(R20), F6
	FMOVD   144(R20), F7
	FMOVD   152(R20), F8
	FMOVD   160(R20), F9
	FMOVD	168(R20), F10
	FMOVD	176(R20), F11
	FMOVD	184(R20), F12
	RET

// AES hashing not implemented for ppc64
TEXT runtime·memhash<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-32
	JMP	runtime·memhashFallback<ABIInternal>(SB)
TEXT runtime·strhash<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·strhashFallback<ABIInternal>(SB)
TEXT runtime·memhash32<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash32Fallback<ABIInternal>(SB)
TEXT runtime·memhash64<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	JMP	runtime·memhash64Fallback<ABIInternal>(SB)

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

// gcWriteBarrier informs the GC about heap pointer writes.
//
// gcWriteBarrier does NOT follow the Go ABI. It accepts the
// number of bytes of buffer needed in R29, and returns a pointer
// to the buffer space in R29.
// It clobbers condition codes.
// It does not clobber R0 through R17 (except special registers),
// but may clobber any other register, *including* R31.
TEXT gcWriteBarrier<>(SB),NOSPLIT,$120
	// The standard prologue clobbers R31.
	// We use R18, R19, and R31 as scratch registers.
retry:
	MOVD	g_m(g), R18
	MOVD	m_p(R18), R18
	MOVD	(p_wbBuf+wbBuf_next)(R18), R19
	MOVD	(p_wbBuf+wbBuf_end)(R18), R31
	// Increment wbBuf.next position.
	ADD	R29, R19
	// Is the buffer full?
	CMPU	R31, R19
	BLT	flush
	// Commit to the larger buffer.
	MOVD	R19, (p_wbBuf+wbBuf_next)(R18)
	// Make return value (the original next position)
	SUB	R29, R19, R29
	RET

flush:
	// Save registers R0 through R15 since these were not saved by the caller.
	// We don't save all registers on ppc64 because it takes too much space.
	MOVD	R20, (FIXED_FRAME+0)(R1)
	MOVD	R21, (FIXED_FRAME+8)(R1)
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
	// R11, R12 may be clobbered by external-linker-inserted trampoline
	// R13 is REGTLS
	MOVD	R14, (FIXED_FRAME+80)(R1)
	MOVD	R15, (FIXED_FRAME+88)(R1)
	MOVD	R16, (FIXED_FRAME+96)(R1)
	MOVD	R17, (FIXED_FRAME+104)(R1)
	MOVD	R29, (FIXED_FRAME+112)(R1)

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
	MOVD	(FIXED_FRAME+80)(R1), R14
	MOVD	(FIXED_FRAME+88)(R1), R15
	MOVD	(FIXED_FRAME+96)(R1), R16
	MOVD	(FIXED_FRAME+104)(R1), R17
	MOVD	(FIXED_FRAME+112)(R1), R29
	JMP	retry

TEXT runtime·gcWriteBarrier1<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$8, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier2<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$16, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier3<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$24, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier4<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$32, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier5<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$40, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier6<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$48, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier7<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$56, R29
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier8<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$64, R29
	JMP	gcWriteBarrier<>(SB)

DATA	debugCallFrameTooLarge<>+0x00(SB)/20, $"call frame too large"
GLOBL	debugCallFrameTooLarge<>(SB), RODATA, $20	// Size duplicated below

// debugCallV2 is the entry point for debugger-injected function
// calls on running goroutines. It informs the runtime that a
// debug call has been injected and creates a call frame for the
// debugger to fill in.
//
// To inject a function call, a debugger should:
// 1. Check that the goroutine is in state _Grunning and that
//    there are at least 320 bytes free on the stack.
// 2. Set SP as SP-32.
// 3. Store the current LR in (SP) (using the SP after step 2).
// 4. Store the current PC in the LR register.
// 5. Write the desired argument frame size at SP-32
// 6. Save all machine registers (including flags and floating point registers)
//    so they can be restored later by the debugger.
// 7. Set the PC to debugCallV2 and resume execution.
//
// If the goroutine is in state _Grunnable, then it's not generally
// safe to inject a call because it may return out via other runtime
// operations. Instead, the debugger should unwind the stack to find
// the return to non-runtime code, add a temporary breakpoint there,
// and inject the call once that breakpoint is hit.
//
// If the goroutine is in any other state, it's not safe to inject a call.
//
// This function communicates back to the debugger by setting R20 and
// invoking TW to raise a breakpoint signal. Note that the signal PC of
// the signal triggered by the TW instruction is the PC where the signal
// is trapped, not the next PC, so to resume execution, the debugger needs
// to set the signal PC to PC+4. See the comments in the implementation for
// the protocol the debugger is expected to follow. InjectDebugCall in the
// runtime tests demonstrates this protocol.
// The debugger must ensure that any pointers passed to the function
// obey escape analysis requirements. Specifically, it must not pass
// a stack pointer to an escaping argument. debugCallV2 cannot check
// this invariant.
//
// This is ABIInternal because Go code injects its PC directly into new
// goroutine stacks.
#ifdef GOARCH_ppc64le
TEXT runtime·debugCallV2<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-0
	// save scratch register R31 first
	MOVD	R31, -184(R1)
	MOVD	0(R1), R31
	// save caller LR
	MOVD	R31, -304(R1)
	MOVD	-32(R1), R31
	// save argument frame size
	MOVD	R31, -192(R1)
	MOVD	LR, R31
	MOVD	R31, -320(R1)
	ADD	$-320, R1
	// save all registers that can contain pointers
	// and the CR register
	MOVW	CR, R31
	MOVD	R31, 8(R1)
	MOVD	R2, 24(R1)
	MOVD	R3, 56(R1)
	MOVD	R4, 64(R1)
	MOVD	R5, 72(R1)
	MOVD	R6, 80(R1)
	MOVD	R7, 88(R1)
	MOVD	R8, 96(R1)
	MOVD	R9, 104(R1)
	MOVD	R10, 112(R1)
	MOVD	R11, 120(R1)
	MOVD	R12, 144(R1)
	MOVD	R13, 152(R1)
	MOVD	R14, 160(R1)
	MOVD	R15, 168(R1)
	MOVD	R16, 176(R1)
	MOVD	R17, 184(R1)
	MOVD	R18, 192(R1)
	MOVD	R19, 200(R1)
	MOVD	R20, 208(R1)
	MOVD	R21, 216(R1)
	MOVD	R22, 224(R1)
	MOVD	R23, 232(R1)
	MOVD	R24, 240(R1)
	MOVD	R25, 248(R1)
	MOVD	R26, 256(R1)
	MOVD	R27, 264(R1)
	MOVD	R28, 272(R1)
	MOVD	R29, 280(R1)
	MOVD	g, 288(R1)
	MOVD	LR, R31
	MOVD	R31, 32(R1)
	CALL	runtime·debugCallCheck(SB)
	MOVD	40(R1), R22
	XOR	R0, R0
	CMP	R22, $0
	BEQ	good
	MOVD	48(R1), R22
	MOVD	$8, R20
	TW	$31, R0, R0

	BR	restore

good:
#define DEBUG_CALL_DISPATCH(NAME,MAXSIZE)	\
	MOVD	$MAXSIZE, R23;			\
	CMP	R26, R23;			\
	BGT	5(PC);				\
	MOVD	$NAME(SB), R26;			\
	MOVD	R26, 32(R1);			\
	CALL	runtime·debugCallWrap(SB);	\
	BR	restore

	// the argument frame size
	MOVD	128(R1), R26

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
	MOVD	$debugCallFrameTooLarge<>(SB), R22
	MOVD	R22, 32(R1)
	MOVD	$20, R22
	// length of debugCallFrameTooLarge string
	MOVD	R22, 40(R1)
	MOVD	$8, R20
	TW	$31, R0, R0
	BR	restore
restore:
	MOVD	$16, R20
	TW	$31, R0, R0
	// restore all registers that can contain
	// pointers including CR
	MOVD	8(R1), R31
	MOVW	R31, CR
	MOVD	24(R1), R2
	MOVD	56(R1), R3
	MOVD	64(R1), R4
	MOVD	72(R1), R5
	MOVD	80(R1), R6
	MOVD	88(R1), R7
	MOVD	96(R1), R8
	MOVD	104(R1), R9
	MOVD	112(R1), R10
	MOVD	120(R1), R11
	MOVD	144(R1), R12
	MOVD	152(R1), R13
	MOVD	160(R1), R14
	MOVD	168(R1), R15
	MOVD	176(R1), R16
	MOVD	184(R1), R17
	MOVD	192(R1), R18
	MOVD	200(R1), R19
	MOVD	208(R1), R20
	MOVD	216(R1), R21
	MOVD	224(R1), R22
	MOVD	232(R1), R23
	MOVD	240(R1), R24
	MOVD	248(R1), R25
	MOVD	256(R1), R26
	MOVD	264(R1), R27
	MOVD	272(R1), R28
	MOVD	280(R1), R29
	MOVD	288(R1), g
	MOVD	16(R1), R31
	// restore old LR
	MOVD	R31, LR
	// restore caller PC
	MOVD	0(R1), CTR
	MOVD	136(R1), R31
	// Add 32 bytes more to compensate for SP change in saveSigContext
	ADD	$352, R1
	JMP	(CTR)
#endif
#define DEBUG_CALL_FN(NAME,MAXSIZE)	\
TEXT NAME(SB),WRAPPER,$MAXSIZE-0;	\
	NO_LOCAL_POINTERS;		\
	MOVD	$0, R20;		\
	TW	$31, R0, R0		\
	MOVD	$1, R20;		\
	TW	$31, R0, R0		\
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

#ifdef GOARCH_ppc64le
// func debugCallPanicked(val interface{})
TEXT runtime·debugCallPanicked(SB),NOSPLIT,$32-16
	// Copy the panic value to the top of stack at SP+32.
	MOVD	val_type+0(FP), R31
	MOVD	R31, 32(R1)
	MOVD	val_data+8(FP), R31
	MOVD	R31, 40(R1)
	MOVD	$2, R20
	TW	$31, R0, R0
	RET
#endif

TEXT runtime·panicBounds<ABIInternal>(SB),NOSPLIT,$88-0
	// Note: frame size is 16 bytes larger than necessary
	// in order to pacify vet. Vet doesn't understand ppc64
	// layout properly.
	NO_LOCAL_POINTERS
	// Save all 7 int registers that could have an index in them.
	// They may be pointers, but if so they are dead.
	// Skip R0 aka ZERO, R1 aka SP, R2 aka SB
	MOVD	R3, 48(R1)
	MOVD	R4, 56(R1)
	MOVD	R5, 64(R1)
	MOVD	R6, 72(R1)
	MOVD	R7, 80(R1)
	MOVD	R8, 88(R1)
	MOVD	R9, 96(R1)
	// Note: we only save 7 registers to keep under nosplit stack limit
	// Also, R11 is clobbered in dynamic linking situations

	MOVD	LR, R3		// PC immediately after call to panicBounds
	ADD	$48, R1, R4	// pointer to save area
	CALL	runtime·panicBounds64<ABIInternal>(SB)
	RET

// These functions are used when internal linking cgo with external
// objects compiled with the -Os on gcc. They reduce prologue/epilogue
// size by deferring preservation of callee-save registers to a shared
// function. These are defined in PPC64 ELFv2 2.3.3 (but also present
// in ELFv1)
//
// These appear unused, but the linker will redirect calls to functions
// like _savegpr0_14 or _restgpr1_14 to runtime.elf_savegpr0 or
// runtime.elf_restgpr1 with an appropriate offset based on the number
// register operations required when linking external objects which
// make these calls. For GPR/FPR saves, the minimum register value is
// 14, for VR it is 20.
//
// These are only used when linking such cgo code internally. Note, R12
// and R0 may be used in different ways than regular ELF compliant
// functions.
TEXT runtime·elf_savegpr0(SB),NOSPLIT|NOFRAME,$0
	// R0 holds the LR of the caller's caller, R1 holds save location
	MOVD	R14, -144(R1)
	MOVD	R15, -136(R1)
	MOVD	R16, -128(R1)
	MOVD	R17, -120(R1)
	MOVD	R18, -112(R1)
	MOVD	R19, -104(R1)
	MOVD	R20, -96(R1)
	MOVD	R21, -88(R1)
	MOVD	R22, -80(R1)
	MOVD	R23, -72(R1)
	MOVD	R24, -64(R1)
	MOVD	R25, -56(R1)
	MOVD	R26, -48(R1)
	MOVD	R27, -40(R1)
	MOVD	R28, -32(R1)
	MOVD	R29, -24(R1)
	MOVD	g, -16(R1)
	MOVD	R31, -8(R1)
	MOVD	R0, 16(R1)
	RET
TEXT runtime·elf_restgpr0(SB),NOSPLIT|NOFRAME,$0
	// R1 holds save location. This returns to the LR saved on stack (bypassing the caller)
	MOVD	-144(R1), R14
	MOVD	-136(R1), R15
	MOVD	-128(R1), R16
	MOVD	-120(R1), R17
	MOVD	-112(R1), R18
	MOVD	-104(R1), R19
	MOVD	-96(R1), R20
	MOVD	-88(R1), R21
	MOVD	-80(R1), R22
	MOVD	-72(R1), R23
	MOVD	-64(R1), R24
	MOVD	-56(R1), R25
	MOVD	-48(R1), R26
	MOVD	-40(R1), R27
	MOVD	-32(R1), R28
	MOVD	-24(R1), R29
	MOVD	-16(R1), g
	MOVD	-8(R1), R31
	MOVD	16(R1), R0	// Load and return to saved LR
	MOVD	R0, LR
	RET
TEXT runtime·elf_savegpr1(SB),NOSPLIT|NOFRAME,$0
	// R12 holds the save location
	MOVD	R14, -144(R12)
	MOVD	R15, -136(R12)
	MOVD	R16, -128(R12)
	MOVD	R17, -120(R12)
	MOVD	R18, -112(R12)
	MOVD	R19, -104(R12)
	MOVD	R20, -96(R12)
	MOVD	R21, -88(R12)
	MOVD	R22, -80(R12)
	MOVD	R23, -72(R12)
	MOVD	R24, -64(R12)
	MOVD	R25, -56(R12)
	MOVD	R26, -48(R12)
	MOVD	R27, -40(R12)
	MOVD	R28, -32(R12)
	MOVD	R29, -24(R12)
	MOVD	g, -16(R12)
	MOVD	R31, -8(R12)
	RET
TEXT runtime·elf_restgpr1(SB),NOSPLIT|NOFRAME,$0
	// R12 holds the save location
	MOVD	-144(R12), R14
	MOVD	-136(R12), R15
	MOVD	-128(R12), R16
	MOVD	-120(R12), R17
	MOVD	-112(R12), R18
	MOVD	-104(R12), R19
	MOVD	-96(R12), R20
	MOVD	-88(R12), R21
	MOVD	-80(R12), R22
	MOVD	-72(R12), R23
	MOVD	-64(R12), R24
	MOVD	-56(R12), R25
	MOVD	-48(R12), R26
	MOVD	-40(R12), R27
	MOVD	-32(R12), R28
	MOVD	-24(R12), R29
	MOVD	-16(R12), g
	MOVD	-8(R12), R31
	RET
TEXT runtime·elf_savefpr(SB),NOSPLIT|NOFRAME,$0
	// R0 holds the LR of the caller's caller, R1 holds save location
	FMOVD	F14, -144(R1)
	FMOVD	F15, -136(R1)
	FMOVD	F16, -128(R1)
	FMOVD	F17, -120(R1)
	FMOVD	F18, -112(R1)
	FMOVD	F19, -104(R1)
	FMOVD	F20, -96(R1)
	FMOVD	F21, -88(R1)
	FMOVD	F22, -80(R1)
	FMOVD	F23, -72(R1)
	FMOVD	F24, -64(R1)
	FMOVD	F25, -56(R1)
	FMOVD	F26, -48(R1)
	FMOVD	F27, -40(R1)
	FMOVD	F28, -32(R1)
	FMOVD	F29, -24(R1)
	FMOVD	F30, -16(R1)
	FMOVD	F31, -8(R1)
	MOVD	R0, 16(R1)
	RET
TEXT runtime·elf_restfpr(SB),NOSPLIT|NOFRAME,$0
	// R1 holds save location. This returns to the LR saved on stack (bypassing the caller)
	FMOVD	-144(R1), F14
	FMOVD	-136(R1), F15
	FMOVD	-128(R1), F16
	FMOVD	-120(R1), F17
	FMOVD	-112(R1), F18
	FMOVD	-104(R1), F19
	FMOVD	-96(R1), F20
	FMOVD	-88(R1), F21
	FMOVD	-80(R1), F22
	FMOVD	-72(R1), F23
	FMOVD	-64(R1), F24
	FMOVD	-56(R1), F25
	FMOVD	-48(R1), F26
	FMOVD	-40(R1), F27
	FMOVD	-32(R1), F28
	FMOVD	-24(R1), F29
	FMOVD	-16(R1), F30
	FMOVD	-8(R1), F31
	MOVD	16(R1), R0	// Load and return to saved LR
	MOVD	R0, LR
	RET
TEXT runtime·elf_savevr(SB),NOSPLIT|NOFRAME,$0
	// R0 holds the save location, R12 is clobbered
	MOVD	$-192, R12
	STVX	V20, (R0+R12)
	MOVD	$-176, R12
	STVX	V21, (R0+R12)
	MOVD	$-160, R12
	STVX	V22, (R0+R12)
	MOVD	$-144, R12
	STVX	V23, (R0+R12)
	MOVD	$-128, R12
	STVX	V24, (R0+R12)
	MOVD	$-112, R12
	STVX	V25, (R0+R12)
	MOVD	$-96, R12
	STVX	V26, (R0+R12)
	MOVD	$-80, R12
	STVX	V27, (R0+R12)
	MOVD	$-64, R12
	STVX	V28, (R0+R12)
	MOVD	$-48, R12
	STVX	V29, (R0+R12)
	MOVD	$-32, R12
	STVX	V30, (R0+R12)
	MOVD	$-16, R12
	STVX	V31, (R0+R12)
	RET
TEXT runtime·elf_restvr(SB),NOSPLIT|NOFRAME,$0
	// R0 holds the save location, R12 is clobbered
	MOVD	$-192, R12
	LVX	(R0+R12), V20
	MOVD	$-176, R12
	LVX	(R0+R12), V21
	MOVD	$-160, R12
	LVX	(R0+R12), V22
	MOVD	$-144, R12
	LVX	(R0+R12), V23
	MOVD	$-128, R12
	LVX	(R0+R12), V24
	MOVD	$-112, R12
	LVX	(R0+R12), V25
	MOVD	$-96, R12
	LVX	(R0+R12), V26
	MOVD	$-80, R12
	LVX	(R0+R12), V27
	MOVD	$-64, R12
	LVX	(R0+R12), V28
	MOVD	$-48, R12
	LVX	(R0+R12), V29
	MOVD	$-32, R12
	LVX	(R0+R12), V30
	MOVD	$-16, R12
	LVX	(R0+R12), V31
	RET
