// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"

// Indicate the status of vector facility
// -1: 	init value
// 0:	vector not installed
// 1:	vector installed and enabled
// 2:	vector installed but not enabled

DATA runtime·vectorfacility+0x00(SB)/4, $-1
GLOBL runtime·vectorfacility(SB), NOPTR, $4

TEXT runtime·checkvectorfacility(SB),NOSPLIT,$32-0
	MOVD    $2, R0
	MOVD	R1, tmp-32(SP)
	MOVD    $x-24(SP), R1
	XC	$24, 0(R1), 0(R1)
//      STFLE   0(R1)
	WORD    $0xB2B01000
	MOVBZ   z-8(SP), R1
	AND     $0x40, R1
	BNE     vectorinstalled
	MOVB    $0, runtime·vectorfacility(SB) //Vector not installed
	MOVD	tmp-32(SP), R1
	MOVD    $0, R0
	RET
vectorinstalled:
	// check if the vector instruction has been enabled
	VLEIB   $0, $0xF, V16
	VLGVB   $0, V16, R0
	CMPBEQ  R0, $0xF, vectorenabled
	MOVB    $2, runtime·vectorfacility(SB) //Vector installed but not enabled
	MOVD    tmp-32(SP), R1
	MOVD    $0, R0
	RET
vectorenabled:
	MOVB    $1, runtime·vectorfacility(SB) //Vector installed and enabled
	MOVD    tmp-32(SP), R1
	MOVD    $0, R0
	RET

TEXT runtime·rt0_go(SB),NOSPLIT,$0
	// R2 = argc; R3 = argv; R11 = temp; R13 = g; R15 = stack pointer
	// C TLS base pointer in AR0:AR1

	// initialize essential registers
	XOR	R0, R0

	SUB	$24, R15
	MOVW	R2, 8(R15) // argc
	MOVD	R3, 16(R15) // argv

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVD	$runtime·g0(SB), g
	MOVD	R15, R11
	SUB	$(64*1024), R11
	MOVD	R11, g_stackguard0(g)
	MOVD	R11, g_stackguard1(g)
	MOVD	R11, (g_stack+stack_lo)(g)
	MOVD	R15, (g_stack+stack_hi)(g)

	// if there is a _cgo_init, call it using the gcc ABI.
	MOVD	_cgo_init(SB), R11
	CMPBEQ	R11, $0, nocgo
	MOVW	AR0, R4			// (AR0 << 32 | AR1) is the TLS base pointer; MOVD is translated to EAR
	SLD	$32, R4, R4
	MOVW	AR1, R4			// arg 2: TLS base pointer
	MOVD	$setg_gcc<>(SB), R3 	// arg 1: setg
	MOVD	g, R2			// arg 0: G
	// C functions expect 160 bytes of space on caller stack frame
	// and an 8-byte aligned stack pointer
	MOVD	R15, R9			// save current stack (R9 is preserved in the Linux ABI)
	SUB	$160, R15		// reserve 160 bytes
	MOVD    $~7, R6
	AND 	R6, R15			// 8-byte align
	BL	R11			// this call clobbers volatile registers according to Linux ABI (R0-R5, R14)
	MOVD	R9, R15			// restore stack
	XOR	R0, R0			// zero R0

nocgo:
	// update stackguard after _cgo_init
	MOVD	(g_stack+stack_lo)(g), R2
	ADD	$const__StackGuard, R2
	MOVD	R2, g_stackguard0(g)
	MOVD	R2, g_stackguard1(g)

	// set the per-goroutine and per-mach "registers"
	MOVD	$runtime·m0(SB), R2

	// save m->g0 = g0
	MOVD	g, m_g0(R2)
	// save m0 to g0->m
	MOVD	R2, g_m(g)

	BL	runtime·check(SB)

	// argc/argv are already prepared on stack
	BL	runtime·args(SB)
	BL	runtime·osinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVD	$runtime·mainPC(SB), R2		// entry
	SUB     $24, R15
	MOVD 	R2, 16(R15)
	MOVD 	R0, 8(R15)
	MOVD 	R0, 0(R15)
	BL	runtime·newproc(SB)
	ADD	$24, R15

	// start this M
	BL	runtime·mstart(SB)

	MOVD	R0, 1(R0)
	RET

DATA	runtime·mainPC+0(SB)/8,$runtime·main(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	R0, 2(R0)
	RET

TEXT runtime·asminit(SB),NOSPLIT|NOFRAME,$0-0
	RET

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT, $-8-8
	MOVD	buf+0(FP), R3
	MOVD	R15, gobuf_sp(R3)
	MOVD	LR, gobuf_pc(R3)
	MOVD	g, gobuf_g(R3)
	MOVD	$0, gobuf_lr(R3)
	MOVD	$0, gobuf_ret(R3)
	MOVD	$0, gobuf_ctxt(R3)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $-8-8
	MOVD	buf+0(FP), R5
	MOVD	gobuf_g(R5), g	// make sure g is not nil
	BL	runtime·save_g(SB)

	MOVD	0(g), R4
	MOVD	gobuf_sp(R5), R15
	MOVD	gobuf_lr(R5), LR
	MOVD	gobuf_ret(R5), R3
	MOVD	gobuf_ctxt(R5), R12
	MOVD	$0, gobuf_sp(R5)
	MOVD	$0, gobuf_ret(R5)
	MOVD	$0, gobuf_lr(R5)
	MOVD	$0, gobuf_ctxt(R5)
	CMP	R0, R0 // set condition codes for == test, needed by stack split
	MOVD	gobuf_pc(R5), R6
	BR	(R6)

// void mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $-8-8
	// Save caller state in g->sched
	MOVD	R15, (g_sched+gobuf_sp)(g)
	MOVD	LR, (g_sched+gobuf_pc)(g)
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
	MOVD	fn+0(FP), R12			// context
	MOVD	0(R12), R4			// code pointer
	MOVD	(g_sched+gobuf_sp)(g), R15	// sp = m->g0->sched.sp
	SUB	$16, R15
	MOVD	R3, 8(R15)
	MOVD	$0, 0(R15)
	BL	(R4)
	BR	runtime·badmcall2(SB)

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack.  We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	UNDEF
	BL	(LR)	// make sure this function is not leaf
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	MOVD	fn+0(FP), R3	// R3 = fn
	MOVD	R3, R12		// context
	MOVD	g_m(g), R4	// R4 = m

	MOVD	m_gsignal(R4), R5	// R5 = gsignal
	CMPBEQ	g, R5, noswitch

	MOVD	m_g0(R4), R5	// R5 = g0
	CMPBEQ	g, R5, noswitch

	MOVD	m_curg(R4), R6
	CMPBEQ	g, R6, switch

	// Bad: g is not gsignal, not g0, not curg. What is it?
	// Hide call from linker nosplit analysis.
	MOVD	$runtime·badsystemstack(SB), R3
	BL	(R3)

switch:
	// save our state in g->sched.  Pretend to
	// be systemstack_switch if the G stack is scanned.
	MOVD	$runtime·systemstack_switch(SB), R6
	ADD	$16, R6	// get past prologue
	MOVD	R6, (g_sched+gobuf_pc)(g)
	MOVD	R15, (g_sched+gobuf_sp)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	MOVD	g, (g_sched+gobuf_g)(g)

	// switch to g0
	MOVD	R5, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R3
	// make it look like mstart called systemstack on g0, to stop traceback
	SUB	$8, R3
	MOVD	$runtime·mstart(SB), R4
	MOVD	R4, 0(R3)
	MOVD	R3, R15

	// call target function
	MOVD	0(R12), R3	// code pointer
	BL	(R3)

	// switch back to g
	MOVD	g_m(g), R3
	MOVD	m_curg(R3), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R15
	MOVD	$0, (g_sched+gobuf_sp)(g)
	RET

noswitch:
	// already on m stack, just call directly
	MOVD	0(R12), R3	// code pointer
	BL	(R3)
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
	CMPBNE	g, R8, 2(PC)
	BL	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVD	m_gsignal(R7), R8
	CMP	g, R8
	BNE	2(PC)
	BL	runtime·abort(SB)

	// Called from f.
	// Set g->sched to context in f.
	MOVD	R12, (g_sched+gobuf_ctxt)(g)
	MOVD	R15, (g_sched+gobuf_sp)(g)
	MOVD	LR, R8
	MOVD	R8, (g_sched+gobuf_pc)(g)
	MOVD	R5, (g_sched+gobuf_lr)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVD	R5, (m_morebuf+gobuf_pc)(R7)	// f's caller's PC
	MOVD	R15, (m_morebuf+gobuf_sp)(R7)	// f's caller's SP
	MOVD	g, (m_morebuf+gobuf_g)(R7)

	// Call newstack on m->g0's stack.
	MOVD	m_g0(R7), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R15
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$0, R12
	BR	runtime·morestack(SB)

TEXT runtime·stackBarrier(SB),NOSPLIT,$0
	// We came here via a RET to an overwritten LR.
	// R3 may be live. Other registers are available.

	// Get the original return PC, g.stkbar[g.stkbarPos].savedLRVal.
	MOVD	(g_stkbar+slice_array)(g), R4
	MOVD	g_stkbarPos(g), R5
	MOVD	$stkbar__size, R6
	MULLD	R5, R6
	ADD	R4, R6
	MOVD	stkbar_savedLRVal(R6), R6
	// Record that this stack barrier was hit.
	ADD	$1, R5
	MOVD	R5, g_stkbarPos(g)
	// Jump to the original return PC.
	BR	(R6)

// reflectcall: call a function with the given argument list
// func call(argtype *_type, f *FuncVal, arg *byte, argsize, retoffset uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVD	$MAXSIZE, R4;		\
	CMP	R3, R4;		\
	BGT	3(PC);			\
	MOVD	$NAME(SB), R5;	\
	BR	(R5)
// Note: can't just "BR NAME(SB)" - bad inlining results.

TEXT reflect·call(SB), NOSPLIT, $0-0
	BR	·reflectcall(SB)

TEXT ·reflectcall(SB), NOSPLIT, $-8-32
	MOVWZ argsize+24(FP), R3
	// NOTE(rsc): No call16, because CALLFN needs four words
	// of argument space to invoke callwritebarrier.
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
	MOVD	$runtime·badreflectcall(SB), R5
	BR	(R5)

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-24;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVD	arg+16(FP), R3;			\
	MOVWZ	argsize+24(FP), R4;			\
	MOVD	R15, R5;				\
	ADD	$(8-1), R5;			\
	SUB	$1, R3;				\
	ADD	R5, R4;				\
	CMP	R5, R4;				\
	BEQ	6(PC);				\
	ADD	$1, R3;				\
	ADD	$1, R5;				\
	MOVBZ	0(R3), R6;			\
	MOVBZ	R6, 0(R5);			\
	BR	-6(PC);				\
	/* call function */			\
	MOVD	f+8(FP), R12;			\
	MOVD	(R12), R8;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	BL	(R8);				\
	/* copy return values back */		\
	MOVD	arg+16(FP), R3;			\
	MOVWZ	n+24(FP), R4;			\
	MOVWZ	retoffset+28(FP), R6;		\
	MOVD	R15, R5;				\
	ADD	R6, R5; 			\
	ADD	R6, R3;				\
	SUB	R6, R4;				\
	ADD	$(8-1), R5;			\
	SUB	$1, R3;				\
	ADD	R5, R4;				\
loop:						\
	CMP	R5, R4;				\
	BEQ	end;				\
	ADD	$1, R5;				\
	ADD	$1, R3;				\
	MOVBZ	0(R5), R6;			\
	MOVBZ	R6, 0(R3);			\
	BR	loop;				\
end:						\
	/* execute write barrier updates */	\
	MOVD	argtype+0(FP), R7;		\
	MOVD	arg+16(FP), R3;			\
	MOVWZ	n+24(FP), R4;			\
	MOVWZ	retoffset+28(FP), R6;		\
	MOVD	R7, 8(R15);			\
	MOVD	R3, 16(R15);			\
	MOVD	R4, 24(R15);			\
	MOVD	R6, 32(R15);			\
	BL	runtime·callwritebarrier(SB);	\
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
	RET

// void jmpdefer(fv, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 6 bytes to get back to BL deferreturn (size of BRASL instruction)
// 3. BR to fn
TEXT runtime·jmpdefer(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	0(R15), R1
	SUB	$6, R1, LR

	MOVD	fv+0(FP), R12
	MOVD	argp+8(FP), R15
	SUB	$8, R15
	MOVD	0(R12), R3
	BR	(R3)

// Save state of caller into g->sched. Smashes R31.
TEXT gosave<>(SB),NOSPLIT|NOFRAME,$0
	MOVD	LR, (g_sched+gobuf_pc)(g)
	MOVD	R15, (g_sched+gobuf_sp)(g)
	MOVD	$0, (g_sched+gobuf_lr)(g)
	MOVD	$0, (g_sched+gobuf_ret)(g)
	MOVD	$0, (g_sched+gobuf_ctxt)(g)
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	// R2 = argc; R3 = argv; R11 = temp; R13 = g; R15 = stack pointer
	// C TLS base pointer in AR0:AR1
	MOVD	fn+0(FP), R3
	MOVD	arg+8(FP), R4

	MOVD	R15, R2		// save original stack pointer
	MOVD	g, R5

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	MOVD	g_m(g), R6
	MOVD	m_g0(R6), R6
	CMPBEQ	R6, g, g0
	BL	gosave<>(SB)
	MOVD	R6, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R15

	// Now on a scheduling stack (a pthread-created stack).
g0:
	// Save room for two of our pointers, plus 160 bytes of callee
	// save area that lives on the caller stack.
	SUB	$176, R15
	MOVD	$~7, R6
	AND	R6, R15                 // 8-byte alignment for gcc ABI
	MOVD	R5, 168(R15)             // save old g on stack
	MOVD	(g_stack+stack_hi)(R5), R5
	SUB	R2, R5
	MOVD	R5, 160(R15)             // save depth in old g stack (can't just save SP, as stack might be copied during a callback)
	MOVD	R0, 0(R15)              // clear back chain pointer (TODO can we give it real back trace information?)
	MOVD	R4, R2                  // arg in R2
	BL	R3                      // can clobber: R0-R5, R14, F0-F3, F5, F7-F15

	XOR	R0, R0                  // set R0 back to 0.
	// Restore g, stack pointer.
	MOVD	168(R15), g
	BL	runtime·save_g(SB)
	MOVD	(g_stack+stack_hi)(g), R5
	MOVD	160(R15), R6
	SUB	R6, R5
	MOVD	R5, R15

	MOVW	R2, ret+16(FP)
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize, uintptr ctxt)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$32-32
	MOVD	$fn+0(FP), R3
	MOVD	R3, 8(R15)
	MOVD	frame+8(FP), R3
	MOVD	R3, 16(R15)
	MOVD	framesize+16(FP), R3
	MOVD	R3, 24(R15)
	MOVD	ctxt+24(FP), R3
	MOVD	R3, 32(R15)
	MOVD	$runtime·cgocallback_gofunc(SB), R3
	BL	(R3)
	RET

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize, uintptr ctxt)
// See cgocall.go for more details.
TEXT ·cgocallback_gofunc(SB),NOSPLIT,$16-32
	NO_LOCAL_POINTERS

	// Load m and g from thread-local storage.
	MOVB	runtime·iscgo(SB), R3
	CMPBEQ	R3, $0, nocgo
	BL	runtime·load_g(SB)

nocgo:
	// If g is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	CMPBEQ	g, $0, needm

	MOVD	g_m(g), R8
	MOVD	R8, savedm-8(SP)
	BR	havem

needm:
	MOVD	g, savedm-8(SP) // g is zero, so is m.
	MOVD	$runtime·needm(SB), R3
	BL	(R3)

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
	MOVD	R15, (g_sched+gobuf_sp)(R3)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 8(R1) aka savedsp-16(SP).
	MOVD	m_g0(R8), R3
	MOVD	(g_sched+gobuf_sp)(R3), R4
	MOVD	R4, savedsp-16(SP)
	MOVD	R15, (g_sched+gobuf_sp)(R3)

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
	MOVD	R5, -24(R4)
	MOVD	ctxt+24(FP), R5
	MOVD	R5, -16(R4)
	MOVD	$-24(R4), R15
	BL	runtime·cgocallbackg(SB)

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVD	0(R15), R5
	MOVD	R5, (g_sched+gobuf_pc)(g)
	MOVD	$24(R15), R4
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R15
	MOVD	savedsp-16(SP), R4
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m
	// for the duration of the call. Since the call is over, return it with dropm.
	MOVD	savedm-8(SP), R6
	CMPBNE	R6, $0, droppedm
	MOVD	$runtime·dropm(SB), R3
	BL	(R3)
droppedm:

	// Done!
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-8
	MOVD	gg+0(FP), g
	// This only happens if iscgo, so jump straight to save_g
	BL	runtime·save_g(SB)
	RET

// void setg_gcc(G*); set g in C TLS.
// Must obey the gcc calling convention.
TEXT setg_gcc<>(SB),NOSPLIT|NOFRAME,$0-0
	// The standard prologue clobbers LR (R14), which is callee-save in
	// the C ABI, so we have to use NOFRAME and save LR ourselves.
	MOVD	LR, R1
	// Also save g, R10, and R11 since they're callee-save in C ABI
	MOVD	R10, R3
	MOVD	g, R4
	MOVD	R11, R5

	MOVD	R2, g
	BL	runtime·save_g(SB)

	MOVD	R5, R11
	MOVD	R4, g
	MOVD	R3, R10
	MOVD	R1, LR
	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$8-16
	MOVD	16(R15), R3		// LR saved by caller
	MOVD	runtime·stackBarrierPC(SB), R4
	CMPBNE	R3, R4, nobar
	// Get original return PC.
	BL	runtime·nextBarrierPC(SB)
	MOVD	8(R15), R3
nobar:
	MOVD	R3, ret+8(FP)
	RET

TEXT runtime·setcallerpc(SB),NOSPLIT,$8-16
	MOVD	pc+8(FP), R3
	MOVD	16(R15), R4
	MOVD	runtime·stackBarrierPC(SB), R5
	CMPBEQ	R4, R5, setbar
	MOVD	R3, 16(R15)		// set LR in caller
	RET
setbar:
	// Set the stack barrier return PC.
	MOVD	R3, 8(R15)
	BL	runtime·setNextBarrierPC(SB)
	RET

TEXT runtime·getcallersp(SB),NOSPLIT,$0-16
	MOVD	argp+0(FP), R3
	SUB	$8, R3
	MOVD	R3, ret+8(FP)
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R0
	UNDEF

// int64 runtime·cputicks(void)
TEXT runtime·cputicks(SB),NOSPLIT,$0-8
	// The TOD clock on s390 counts from the year 1900 in ~250ps intervals.
	// This means that since about 1972 the msb has been set, making the
	// result of a call to STORE CLOCK (stck) a negative number.
	// We clear the msb to make it positive.
	STCK	ret+0(FP)      // serialises before and after call
	MOVD	ret+0(FP), R3  // R3 will wrap to 0 in the year 2043
	SLD	$1, R3
	SRD	$1, R3
	MOVD	R3, ret+0(FP)
	RET

// memhash_varlen(p unsafe.Pointer, h seed) uintptr
// redirects to memhash(p, h, size) using the size
// stored in the closure.
TEXT runtime·memhash_varlen(SB),NOSPLIT,$40-24
	GO_ARGS
	NO_LOCAL_POINTERS
	MOVD	p+0(FP), R3
	MOVD	h+8(FP), R4
	MOVD	8(R12), R5
	MOVD	R3, 8(R15)
	MOVD	R4, 16(R15)
	MOVD	R5, 24(R15)
	BL	runtime·memhash(SB)
	MOVD	32(R15), R3
	MOVD	R3, ret+16(FP)
	RET

// AES hashing not implemented for s390x
TEXT runtime·aeshash(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R15
TEXT runtime·aeshash32(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R15
TEXT runtime·aeshash64(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R15
TEXT runtime·aeshashstr(SB),NOSPLIT|NOFRAME,$0-0
	MOVW	(R0), R15

// memequal(p, q unsafe.Pointer, size uintptr) bool
TEXT runtime·memequal(SB),NOSPLIT|NOFRAME,$0-25
	MOVD	p+0(FP), R3
	MOVD	q+8(FP), R5
	MOVD	size+16(FP), R6
	LA	ret+24(FP), R7
	BR	runtime·memeqbody(SB)

// memequal_varlen(a, b unsafe.Pointer) bool
TEXT runtime·memequal_varlen(SB),NOSPLIT|NOFRAME,$0-17
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R5
	MOVD	8(R12), R6    // compiler stores size at offset 8 in the closure
	LA	ret+16(FP), R7
	BR	runtime·memeqbody(SB)

// eqstring tests whether two strings are equal.
// The compiler guarantees that strings passed
// to eqstring have equal length.
// See runtime_test.go:eqstring_generic for
// equivalent Go code.
TEXT runtime·eqstring(SB),NOSPLIT|NOFRAME,$0-33
	MOVD	s1str+0(FP), R3
	MOVD	s1len+8(FP), R6
	MOVD	s2str+16(FP), R5
	LA	ret+32(FP), R7
	BR	runtime·memeqbody(SB)

TEXT bytes·Equal(SB),NOSPLIT|NOFRAME,$0-49
	MOVD	a_len+8(FP), R2
	MOVD	b_len+32(FP), R6
	MOVD	a+0(FP), R3
	MOVD	b+24(FP), R5
	LA	ret+48(FP), R7
	CMPBNE	R2, R6, notequal
	BR	runtime·memeqbody(SB)
notequal:
	MOVB	$0, ret+48(FP)
	RET

// input:
//   R3 = a
//   R5 = b
//   R6 = len
//   R7 = address of output byte (stores 0 or 1 here)
//   a and b have the same length
TEXT runtime·memeqbody(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R3, R5, equal
loop:
	CMPBEQ	R6, $0, equal
	CMPBLT	R6, $32, tiny
	CMP	R6, $256
	BLT	tail
	CLC	$256, 0(R3), 0(R5)
	BNE	notequal
	SUB	$256, R6
	LA	256(R3), R3
	LA	256(R5), R5
	BR	loop
tail:
	SUB	$1, R6, R8
	EXRL	$runtime·memeqbodyclc(SB), R8
	BEQ	equal
notequal:
	MOVB	$0, 0(R7)
	RET
equal:
	MOVB	$1, 0(R7)
	RET
tiny:
	MOVD	$0, R2
	CMPBLT	R6, $16, lt16
	MOVD	0(R3), R8
	MOVD	0(R5), R9
	CMPBNE	R8, R9, notequal
	MOVD	8(R3), R8
	MOVD	8(R5), R9
	CMPBNE	R8, R9, notequal
	LA	16(R2), R2
	SUB	$16, R6
lt16:
	CMPBLT	R6, $8, lt8
	MOVD	0(R3)(R2*1), R8
	MOVD	0(R5)(R2*1), R9
	CMPBNE	R8, R9, notequal
	LA	8(R2), R2
	SUB	$8, R6
lt8:
	CMPBLT	R6, $4, lt4
	MOVWZ	0(R3)(R2*1), R8
	MOVWZ	0(R5)(R2*1), R9
	CMPBNE	R8, R9, notequal
	LA	4(R2), R2
	SUB	$4, R6
lt4:
#define CHECK(n) \
	CMPBEQ	R6, $n, equal \
	MOVB	n(R3)(R2*1), R8 \
	MOVB	n(R5)(R2*1), R9 \
	CMPBNE	R8, R9, notequal
	CHECK(0)
	CHECK(1)
	CHECK(2)
	CHECK(3)
	BR	equal

TEXT runtime·memeqbodyclc(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R3), 0(R5)
	RET

TEXT runtime·fastrand1(SB), NOSPLIT, $0-4
	MOVD	g_m(g), R4
	MOVWZ	m_fastrand(R4), R3
	ADD	R3, R3
	CMPW	R3, $0
	BGE	2(PC)
	XOR	$0x88888eef, R3
	MOVW	R3, m_fastrand(R4)
	MOVW	R3, ret+0(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0-40
	MOVD	s+0(FP), R3     // s => R3
	MOVD	s_len+8(FP), R4 // s_len => R4
	MOVBZ	c+24(FP), R5    // c => R5
	MOVD	$ret+32(FP), R2 // &ret => R9
	BR	runtime·indexbytebody(SB)

TEXT strings·IndexByte(SB),NOSPLIT,$0-32
	MOVD	s+0(FP), R3     // s => R3
	MOVD	s_len+8(FP), R4 // s_len => R4
	MOVBZ	c+16(FP), R5    // c => R5
	MOVD	$ret+24(FP), R2 // &ret => R9
	BR	runtime·indexbytebody(SB)

// input:
// R3: s
// R4: s_len
// R5: c -- byte sought
// R2: &ret -- address to put index into
TEXT runtime·indexbytebody(SB),NOSPLIT,$0
	CMPBEQ	R4, $0, notfound
	MOVD	R3, R6          // store base for later
	ADD	R3, R4, R8      // the address after the end of the string
	//if the length is small, use loop; otherwise, use vector or srst search
	CMPBGE	R4, $16, large

residual:
	CMPBEQ	R3, R8, notfound
	MOVBZ	0(R3), R7
	LA	1(R3), R3
	CMPBNE	R7, R5, residual

found:
	SUB	R6, R3
	SUB	$1, R3
	MOVD	R3, 0(R2)
	RET

notfound:
	MOVD	$-1, 0(R2)
	RET

large:
	MOVB	runtime·vectorfacility(SB), R1
	CMPBEQ	R1, $-1, checkvector	// vectorfacility = -1, vector not checked yet
vectorchecked:
	CMPBEQ	R1, $1, vectorimpl      // vectorfacility = 1, vector supported

srstimpl:                       // vectorfacility != 1, not support or enable vector
	MOVBZ	R5, R0          // c needs to be in R0, leave until last minute as currently R0 is expected to be 0
srstloop:
	WORD	$0xB25E0083     // srst %r8, %r3 (search the range [R3, R8))
	BVS	srstloop        // interrupted - continue
	BGT	notfoundr0
foundr0:
	XOR	R0, R0          // reset R0
	SUB	R6, R8          // remove base
	MOVD	R8, 0(R2)
	RET
notfoundr0:
	XOR	R0, R0          // reset R0
	MOVD	$-1, 0(R2)
	RET

vectorimpl:
	//if the address is not 16byte aligned, use loop for the header
	AND	$15, R3, R8
	CMPBGT	R8, $0, notaligned

aligned:
	ADD	R6, R4, R8
	AND	$-16, R8, R7
	// replicate c across V17
	VLVGB	$0, R5, V19
	VREPB	$0, V19, V17

vectorloop:
	CMPBGE	R3, R7, residual
	VL	0(R3), V16    // load string to be searched into V16
	ADD	$16, R3
	VFEEBS	V16, V17, V18 // search V17 in V16 and set conditional code accordingly
	BVS	vectorloop

	// when vector search found c in the string
	VLGVB	$7, V18, R7   // load 7th element of V18 containing index into R7
	SUB	$16, R3
	SUB	R6, R3
	ADD	R3, R7
	MOVD	R7, 0(R2)
	RET

notaligned:
	AND	$-16, R3, R8
	ADD     $16, R8
notalignedloop:
	CMPBEQ	R3, R8, aligned
	MOVBZ	0(R3), R7
	LA	1(R3), R3
	CMPBNE	R7, R5, notalignedloop
	BR	found

checkvector:
	CALL	runtime·checkvectorfacility(SB)
	MOVB    runtime·vectorfacility(SB), R1
	BR	vectorchecked

TEXT runtime·return0(SB), NOSPLIT, $0
	MOVW	$0, R3
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT|NOFRAME,$0
	// g (R13), R10, R11 and LR (R14) are callee-save in the C ABI, so save them
	MOVD	g, R1
	MOVD	R10, R3
	MOVD	LR, R4
	MOVD	R11, R5

	BL	runtime·load_g(SB)	// clobbers g (R13), R10, R11
	MOVD	g_m(g), R2
	MOVD	m_curg(R2), R2
	MOVD	(g_stack+stack_hi)(R2), R2

	MOVD	R1, g
	MOVD	R3, R10
	MOVD	R4, LR
	MOVD	R5, R11
	RET

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT|NOFRAME,$0-0
	BYTE $0x07; BYTE $0x00; // 2-byte nop
	BL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	BYTE $0x07; BYTE $0x00; // 2-byte nop

TEXT runtime·prefetcht0(SB),NOSPLIT,$0-8
	RET

TEXT runtime·prefetcht1(SB),NOSPLIT,$0-8
	RET

TEXT runtime·prefetcht2(SB),NOSPLIT,$0-8
	RET

TEXT runtime·prefetchnta(SB),NOSPLIT,$0-8
	RET

TEXT runtime·sigreturn(SB),NOSPLIT,$0-8
	RET

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	SYNC
	RET

TEXT runtime·cmpstring(SB),NOSPLIT|NOFRAME,$0-40
	MOVD	s1_base+0(FP), R3
	MOVD	s1_len+8(FP), R4
	MOVD	s2_base+16(FP), R5
	MOVD	s2_len+24(FP), R6
	LA	ret+32(FP), R7
	BR	runtime·cmpbody(SB)

TEXT bytes·Compare(SB),NOSPLIT|NOFRAME,$0-56
	MOVD	s1+0(FP), R3
	MOVD	s1+8(FP), R4
	MOVD	s2+24(FP), R5
	MOVD	s2+32(FP), R6
	LA	res+48(FP), R7
	BR	runtime·cmpbody(SB)

// input:
//   R3 = a
//   R4 = alen
//   R5 = b
//   R6 = blen
//   R7 = address of output word (stores -1/0/1 here)
TEXT runtime·cmpbody(SB),NOSPLIT|NOFRAME,$0-0
	CMPBEQ	R3, R5, cmplengths
	MOVD	R4, R8
	CMPBLE	R4, R6, amin
	MOVD	R6, R8
amin:
	CMPBEQ	R8, $0, cmplengths
	CMP	R8, $256
	BLE	tail
loop:
	CLC	$256, 0(R3), 0(R5)
	BGT	gt
	BLT	lt
	SUB	$256, R8
	CMP	R8, $256
	BGT	loop
tail:
	SUB	$1, R8
	EXRL	$runtime·cmpbodyclc(SB), R8
	BGT	gt
	BLT	lt
cmplengths:
	CMP	R4, R6
	BEQ	eq
	BLT	lt
gt:
	MOVD	$1, 0(R7)
	RET
lt:
	MOVD	$-1, 0(R7)
	RET
eq:
	MOVD	$0, 0(R7)
	RET

TEXT runtime·cmpbodyclc(SB),NOSPLIT|NOFRAME,$0-0
	CLC	$1, 0(R3), 0(R5)
	RET

// This is called from .init_array and follows the platform, not Go, ABI.
// We are overly conservative. We could only save the registers we use.
// However, since this function is only called once per loaded module
// performance is unimportant.
TEXT runtime·addmoduledata(SB),NOSPLIT|NOFRAME,$0-0
	// Save R6-R15, F0, F2, F4 and F6 in the
	// register save area of the calling function
	STMG	R6, R15, 48(R15)
	FMOVD	F0, 128(R15)
	FMOVD	F2, 136(R15)
	FMOVD	F4, 144(R15)
	FMOVD	F6, 152(R15)

	// append the argument (passed in R2, as per the ELF ABI) to the
	// moduledata linked list.
	MOVD	runtime·lastmoduledatap(SB), R1
	MOVD	R2, moduledata_next(R1)
	MOVD	R2, runtime·lastmoduledatap(SB)

	// Restore R6-R15, F0, F2, F4 and F6
	LMG	48(R15), R6, R15
	FMOVD	F0, 128(R15)
	FMOVD	F2, 136(R15)
	FMOVD	F4, 144(R15)
	FMOVD	F6, 152(R15)
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVB	$1, ret+0(FP)
	RET
