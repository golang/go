// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"
#include "funcdata.h"
#include "../../cmd/ld/textflag.h"

// using frame size $-4 means do not save LR on stack.
TEXT _rt0_go(SB),NOSPLIT,$-4
	MOVW	$0xcafebabe, R12

	// copy arguments forward on an even stack
	// use R13 instead of SP to avoid linker rewriting the offsets
	MOVW	0(R13), R0		// argc
	MOVW	4(R13), R1		// argv
	SUB	$64, R13		// plenty of scratch
	AND	$~7, R13
	MOVW	R0, 60(R13)		// save argc, argv away
	MOVW	R1, 64(R13)

	// set up m and g registers
	// g is R10, m is R9
	MOVW	$runtime·g0(SB), g
	MOVW	$runtime·m0(SB), m

	// save m->g0 = g0
	MOVW	g, m_g0(m)

	// create istack out of the OS stack
	MOVW	$(-8192+104)(R13), R0
	MOVW	R0, g_stackguard(g)	// (w 104b guard)
	MOVW	R0, g_stackguard0(g)
	MOVW	R13, g_stackbase(g)
	BL	runtime·emptyfunc(SB)	// fault if stack check is wrong

	// if there is an _cgo_init, call it.
	MOVW	_cgo_init(SB), R4
	CMP	$0, R4
	B.EQ	nocgo
	BL		runtime·save_gm(SB);
	MOVW	g, R0 // first argument of _cgo_init is g
	MOVW	$setmg_gcc<>(SB), R1 // second argument is address of save_gm
	BL		(R4) // will clobber R0-R3

nocgo:
	// update stackguard after _cgo_init
	MOVW	g_stackguard0(g), R0
	MOVW	R0, g_stackguard(g)

	BL	runtime·checkgoarm(SB)
	BL	runtime·check(SB)

	// saved argc, argv
	MOVW	60(R13), R0
	MOVW	R0, 4(R13)
	MOVW	64(R13), R1
	MOVW	R1, 8(R13)
	BL	runtime·args(SB)
	BL	runtime·osinit(SB)
	BL	runtime·hashinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVW	$runtime·main·f(SB), R0
	MOVW.W	R0, -4(R13)
	MOVW	$8, R0
	MOVW.W	R0, -4(R13)
	MOVW	$0, R0
	MOVW.W	R0, -4(R13)	// push $0 as guard
	ARGSIZE(12)
	BL	runtime·newproc(SB)
	ARGSIZE(-1)
	MOVW	$12(R13), R13	// pop args and LR

	// start this M
	BL	runtime·mstart(SB)

	MOVW	$1234, R0
	MOVW	$1000, R1
	MOVW	R0, (R1)	// fail hard

DATA	runtime·main·f+0(SB)/4,$runtime·main(SB)
GLOBL	runtime·main·f(SB),RODATA,$4

TEXT runtime·breakpoint(SB),NOSPLIT,$0-0
	// gdb won't skip this breakpoint instruction automatically,
	// so you must manually "set $pc+=4" to skip it and continue.
	WORD	$0xe1200071	// BKPT 0x0001
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
TEXT runtime·gosave(SB), NOSPLIT, $-4-4
	MOVW	0(FP), R0		// gobuf
	MOVW	SP, gobuf_sp(R0)
	MOVW	LR, gobuf_pc(R0)
	MOVW	g, gobuf_g(R0)
	MOVW	$0, R11
	MOVW	R11, gobuf_lr(R0)
	MOVW	R11, gobuf_ret(R0)
	MOVW	R11, gobuf_ctxt(R0)
	RET

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $-4-4
	MOVW	0(FP), R1		// gobuf
	MOVW	gobuf_g(R1), g
	MOVW	0(g), R2		// make sure g != nil
	MOVB	runtime·iscgo(SB), R2
	CMP 	$0, R2 // if in Cgo, we have to save g and m
	BL.NE	runtime·save_gm(SB) // this call will clobber R0
	MOVW	gobuf_sp(R1), SP	// restore SP
	MOVW	gobuf_lr(R1), LR
	MOVW	gobuf_ret(R1), R0
	MOVW	gobuf_ctxt(R1), R7
	MOVW	$0, R11
	MOVW	R11, gobuf_sp(R1)	// clear to help garbage collector
	MOVW	R11, gobuf_ret(R1)
	MOVW	R11, gobuf_lr(R1)
	MOVW	R11, gobuf_ctxt(R1)
	CMP	R11, R11 // set condition codes for == test, needed by stack split
	MOVW	gobuf_pc(R1), PC

// void mcall(void (*fn)(G*))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $-4-4
	MOVW	fn+0(FP), R0

	// Save caller state in g->sched.
	MOVW	SP, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	$0, R11
	MOVW	R11, (g_sched+gobuf_lr)(g)
	MOVW	g, (g_sched+gobuf_g)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVW	g, R1
	MOVW	m_g0(m), g
	CMP	g, R1
	B.NE	2(PC)
	B	runtime·badmcall(SB)
	MOVW	(g_sched+gobuf_sp)(g), SP
	SUB	$8, SP
	MOVW	R1, 4(SP)
	BL	(R0)
	B	runtime·badmcall2(SB)
	RET

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// R1 frame size
// R2 arg size
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
	MOVW	m_g0(m), R4
	CMP	g, R4
	BL.EQ	runtime·abort(SB)

	MOVW	R1, m_moreframesize(m)
	MOVW	R2, m_moreargsize(m)

	// Called from f.
	// Set g->sched to context in f.
	MOVW	R7, (g_sched+gobuf_ctxt)(g)
	MOVW	SP, (g_sched+gobuf_sp)(g)
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	R3, (g_sched+gobuf_lr)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVW	R3, (m_morebuf+gobuf_pc)(m)	// f's caller's PC
	MOVW	SP, (m_morebuf+gobuf_sp)(m)	// f's caller's SP
	MOVW	$4(SP), R3			// f's argument pointer
	MOVW	R3, m_moreargp(m)	
	MOVW	g, (m_morebuf+gobuf_g)(m)

	// Call newstack on m->g0's stack.
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), SP
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	RET

TEXT runtime·morestack_noctxt(SB),NOSPLIT,$-4-0
	MOVW	$0, R7
	B runtime·morestack(SB)

// Called from panic.  Mimics morestack,
// reuses stack growth code to create a frame
// with the desired args running the desired function.
//
// func call(fn *byte, arg *byte, argsize uint32).
TEXT runtime·newstackcall(SB), NOSPLIT, $-4-12
	// Save our caller's state as the PC and SP to
	// restore when returning from f.
	MOVW	LR, (m_morebuf+gobuf_pc)(m)	// our caller's PC
	MOVW	SP, (m_morebuf+gobuf_sp)(m)	// our caller's SP
	MOVW	g,  (m_morebuf+gobuf_g)(m)

	// Save our own state as the PC and SP to restore
	// if this goroutine needs to be restarted.
	MOVW	$runtime·newstackcall(SB), R11
	MOVW	R11, (g_sched+gobuf_pc)(g)
	MOVW	LR, (g_sched+gobuf_lr)(g)
	MOVW	SP, (g_sched+gobuf_sp)(g)

	// Set up morestack arguments to call f on a new stack.
	// We set f's frame size to 1, as a hint to newstack
	// that this is a call from runtime·newstackcall.
	// If it turns out that f needs a larger frame than
	// the default stack, f's usual stack growth prolog will
	// allocate a new segment (and recopy the arguments).
	MOVW	4(SP), R0			// fn
	MOVW	8(SP), R1			// arg frame
	MOVW	12(SP), R2			// arg size

	MOVW	R0, m_cret(m)			// f's PC
	MOVW	R1, m_moreargp(m)		// f's argument pointer
	MOVW	R2, m_moreargsize(m)		// f's argument size
	MOVW	$1, R3
	MOVW	R3, m_moreframesize(m)		// f's frame size

	// Call newstack on m->g0's stack.
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), SP
	B	runtime·newstack(SB)

// reflect·call: call a function with the given argument list
// func call(f *FuncVal, arg *byte, argsize uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMP	$MAXSIZE, R0;		\
	B.HI	3(PC);			\
	MOVW	$runtime·NAME(SB), R1;	\
	B	(R1)

TEXT reflect·call(SB), NOSPLIT, $-4-16
	MOVW	argsize+8(FP), R0
	DISPATCH(call16, 16)
	DISPATCH(call32, 32)
	DISPATCH(call64, 64)
	DISPATCH(call128, 128)
	DISPATCH(call256, 256)
	DISPATCH(call512, 512)
	DISPATCH(call1024, 1024)
	DISPATCH(call2048, 2048)
	DISPATCH(call4096, 4096)
	DISPATCH(call8192, 8192)
	DISPATCH(call16384, 16384)
	DISPATCH(call32768, 32768)
	DISPATCH(call65536, 65536)
	DISPATCH(call131072, 131072)
	DISPATCH(call262144, 262144)
	DISPATCH(call524288, 524288)
	DISPATCH(call1048576, 1048576)
	DISPATCH(call2097152, 2097152)
	DISPATCH(call4194304, 4194304)
	DISPATCH(call8388608, 8388608)
	DISPATCH(call16777216, 16777216)
	DISPATCH(call33554432, 33554432)
	DISPATCH(call67108864, 67108864)
	DISPATCH(call134217728, 134217728)
	DISPATCH(call268435456, 268435456)
	DISPATCH(call536870912, 536870912)
	DISPATCH(call1073741824, 1073741824)
	MOVW	$runtime·badreflectcall(SB), R1
	B	(R1)

#define CALLFN(NAME,MAXSIZE)			\
TEXT runtime·NAME(SB), WRAPPER, $MAXSIZE-16;	\
	/* copy arguments to stack */		\
	MOVW	argptr+4(FP), R0;		\
	MOVW	argsize+8(FP), R2;		\
	ADD	$4, SP, R1;			\
	CMP	$0, R2;				\
	B.EQ	5(PC);				\
	MOVBU.P	1(R0), R5;			\
	MOVBU.P R5, 1(R1);			\
	SUB	$1, R2, R2;			\
	B	-5(PC);				\
	/* call function */			\
	MOVW	f+0(FP), R7;			\
	MOVW	(R7), R0;			\
	BL	(R0);				\
	/* copy return values back */		\
	MOVW	argptr+4(FP), R0;		\
	MOVW	argsize+8(FP), R2;		\
	MOVW	retoffset+12(FP), R3;		\
	ADD	$4, SP, R1;			\
	ADD	R3, R1;				\
	ADD	R3, R0;				\
	SUB	R3, R2;				\
	CMP	$0, R2;				\
	RET.EQ	;				\
	MOVBU.P	1(R1), R5;			\
	MOVBU.P R5, 1(R0);			\
	SUB	$1, R2, R2;			\
	B	-5(PC)				\

CALLFN(call16, 16)
CALLFN(call32, 32)
CALLFN(call64, 64)
CALLFN(call128, 128)
CALLFN(call256, 256)
CALLFN(call512, 512)
CALLFN(call1024, 1024)
CALLFN(call2048, 2048)
CALLFN(call4096, 4096)
CALLFN(call8192, 8192)
CALLFN(call16384, 16384)
CALLFN(call32768, 32768)
CALLFN(call65536, 65536)
CALLFN(call131072, 131072)
CALLFN(call262144, 262144)
CALLFN(call524288, 524288)
CALLFN(call1048576, 1048576)
CALLFN(call2097152, 2097152)
CALLFN(call4194304, 4194304)
CALLFN(call8388608, 8388608)
CALLFN(call16777216, 16777216)
CALLFN(call33554432, 33554432)
CALLFN(call67108864, 67108864)
CALLFN(call134217728, 134217728)
CALLFN(call268435456, 268435456)
CALLFN(call536870912, 536870912)
CALLFN(call1073741824, 1073741824)

// Return point when leaving stack.
// using frame size $-4 means do not save LR on stack.
//
// Lessstack can appear in stack traces for the same reason
// as morestack; in that context, it has 0 arguments.
TEXT runtime·lessstack(SB), NOSPLIT, $-4-0
	// Save return value in m->cret
	MOVW	R0, m_cret(m)

	// Call oldstack on m->g0's stack.
	MOVW	m_g0(m), g
	MOVW	(g_sched+gobuf_sp)(g), SP
	BL	runtime·oldstack(SB)

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 4 bytes to get back to BL deferreturn
// 3. B to fn
TEXT runtime·jmpdefer(SB), NOSPLIT, $0-8
	MOVW	0(SP), LR
	MOVW	$-4(LR), LR	// BL deferreturn
	MOVW	fn+0(FP), R7
	MOVW	argp+4(FP), SP
	MOVW	$-4(SP), SP	// SP is 4 below argp, due to saved LR
	MOVW	0(R7), R1
	B	(R1)

// Save state of caller into g->sched. Smashes R11.
TEXT gosave<>(SB),NOSPLIT,$0
	MOVW	LR, (g_sched+gobuf_pc)(g)
	MOVW	R13, (g_sched+gobuf_sp)(g)
	MOVW	$0, R11
	MOVW	R11, (g_sched+gobuf_lr)(g)
	MOVW	R11, (g_sched+gobuf_ret)(g)
	MOVW	R11, (g_sched+gobuf_ctxt)(g)
	RET

// asmcgocall(void(*fn)(void*), void *arg)
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.c for more details.
TEXT	runtime·asmcgocall(SB),NOSPLIT,$0-8
	MOVW	fn+0(FP), R1
	MOVW	arg+4(FP), R0
	MOVW	R13, R2
	MOVW	g, R5

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already.
	MOVW	m_g0(m), R3
	CMP	R3, g
	BEQ	4(PC)
	BL	gosave<>(SB)
	MOVW	R3, g
	MOVW	(g_sched+gobuf_sp)(g), R13

	// Now on a scheduling stack (a pthread-created stack).
	SUB	$24, R13
	BIC	$0x7, R13	// alignment for gcc ABI
	MOVW	R5, 20(R13) // save old g
	MOVW	R2, 16(R13)	// save old SP
	// R0 already contains the first argument
	BL	(R1)

	// Restore registers, g, stack pointer.
	MOVW	20(R13), g
	MOVW	16(R13), R13
	RET

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$12-12
	MOVW	$fn+0(FP), R0
	MOVW	R0, 4(R13)
	MOVW	frame+4(FP), R0
	MOVW	R0, 8(R13)
	MOVW	framesize+8(FP), R0
	MOVW	R0, 12(R13)
	MOVW	$runtime·cgocallback_gofunc(SB), R0
	BL	(R0)
	RET

// cgocallback_gofunc(void (*fn)(void*), void *frame, uintptr framesize)
// See cgocall.c for more details.
TEXT	runtime·cgocallback_gofunc(SB),NOSPLIT,$8-12
	// Load m and g from thread-local storage.
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BL.NE	runtime·load_gm(SB)

	// If m is nil, Go did not create the current thread.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	MOVW	m, savedm-4(SP)
	CMP	$0, m
	B.NE	havem
	MOVW	$runtime·needm(SB), R0
	BL	(R0)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 4(R13) aka savedsp-8(SP).
	MOVW	m_g0(m), R3
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
	// In the new goroutine, -8(SP) and -4(SP) are unused.
	MOVW	fn+4(FP), R0
	MOVW	frame+8(FP), R1
	MOVW	framesize+12(FP), R2
	MOVW	m_curg(m), g
	MOVW	(g_sched+gobuf_sp)(g), R4 // prepare stack as R4
	MOVW	(g_sched+gobuf_pc)(g), R5
	MOVW	R5, -12(R4)
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
	MOVW	m_g0(m), g
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

// void setmg(M*, G*); set m and g. for use by needm.
TEXT runtime·setmg(SB), NOSPLIT, $0-8
	MOVW	mm+0(FP), m
	MOVW	gg+4(FP), g

	// Save m and g to thread-local storage.
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BL.NE	runtime·save_gm(SB)

	RET

TEXT runtime·getcallerpc(SB),NOSPLIT,$-4-4
	MOVW	0(SP), R0
	RET

TEXT runtime·setcallerpc(SB),NOSPLIT,$-4-8
	MOVW	x+4(FP), R0
	MOVW	R0, 0(SP)
	RET

TEXT runtime·getcallersp(SB),NOSPLIT,$-4-4
	MOVW	0(FP), R0
	MOVW	$-4(R0), R0
	RET

TEXT runtime·emptyfunc(SB),0,$0-0
	RET

TEXT runtime·abort(SB),NOSPLIT,$-4-0
	MOVW	$0, R0
	MOVW	(R0), R1

// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
//
// To implement runtime·cas in sys_$GOOS_arm.s
// using the native instructions, use:
//
//	TEXT runtime·cas(SB),NOSPLIT,$0
//		B	runtime·armcas(SB)
//
TEXT runtime·armcas(SB),NOSPLIT,$0-12
	MOVW	valptr+0(FP), R1
	MOVW	old+4(FP), R2
	MOVW	new+8(FP), R3
casl:
	LDREX	(R1), R0
	CMP	R0, R2
	BNE	casfail
	STREX	R3, (R1), R0
	CMP	$0, R0
	BNE	casl
	MOVW	$1, R0
	RET
casfail:
	MOVW	$0, R0
	RET

TEXT runtime·stackguard(SB),NOSPLIT,$0-8
	MOVW	R13, R1
	MOVW	g_stackguard(g), R2
	MOVW	R1, sp+0(FP)
	MOVW	R2, limit+4(FP)
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

TEXT runtime·memeq(SB),NOSPLIT,$-4-12
	MOVW	a+0(FP), R1
	MOVW	b+4(FP), R2
	MOVW	n+8(FP), R3
	ADD	R1, R3, R6
	MOVW	$1, R0
_next:
	CMP	R1, R6
	RET.EQ
	MOVBU.P	1(R1), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	BEQ	_next

	MOVW	$0, R0
	RET

// We have to resort to TLS variable to save g(R10) and
// m(R9). One reason is that external code might trigger
// SIGSEGV, and our runtime.sigtramp don't even know we
// are in external code, and will continue to use R10/R9,
// this might as well result in another SIGSEGV.
// Note: all three functions will clobber R0, and the last
// two can be called from 5c ABI code.

// save_gm saves the g and m registers into pthread-provided
// thread-local memory, so that we can call externally compiled
// ARM code that will overwrite those registers.
// NOTE: runtime.gogo assumes that R1 is preserved by this function.
TEXT runtime·save_gm(SB),NOSPLIT,$0
	MRC		15, 0, R0, C13, C0, 3 // fetch TLS base pointer
	// $runtime.tlsgm(SB) is a special linker symbol.
	// It is the offset from the TLS base pointer to our
	// thread-local storage for g and m.
	MOVW	$runtime·tlsgm(SB), R11
	ADD	R11, R0
	MOVW	g, 0(R0)
	MOVW	m, 4(R0)
	RET

// load_gm loads the g and m registers from pthread-provided
// thread-local memory, for use after calling externally compiled
// ARM code that overwrote those registers.
TEXT runtime·load_gm(SB),NOSPLIT,$0
	MRC		15, 0, R0, C13, C0, 3 // fetch TLS base pointer
	// $runtime.tlsgm(SB) is a special linker symbol.
	// It is the offset from the TLS base pointer to our
	// thread-local storage for g and m.
	MOVW	$runtime·tlsgm(SB), R11
	ADD	R11, R0
	MOVW	0(R0), g
	MOVW	4(R0), m
	RET

// void setmg_gcc(M*, G*); set m and g called from gcc.
TEXT setmg_gcc<>(SB),NOSPLIT,$0
	MOVW	R0, m
	MOVW	R1, g
	B		runtime·save_gm(SB)

// TODO: share code with memeq?
TEXT bytes·Equal(SB),NOSPLIT,$0
	MOVW	a_len+4(FP), R1
	MOVW	b_len+16(FP), R3
	
	CMP	R1, R3		// unequal lengths are not equal
	B.NE	_notequal

	MOVW	a+0(FP), R0
	MOVW	b+12(FP), R2
	ADD	R0, R1		// end

_byteseq_next:
	CMP	R0, R1
	B.EQ	_equal		// reached the end
	MOVBU.P	1(R0), R4
	MOVBU.P	1(R2), R5
	CMP	R4, R5
	B.EQ	_byteseq_next

_notequal:
	MOVW	$0, R0
	MOVBU	R0, ret+24(FP)
	RET

_equal:
	MOVW	$1, R0
	MOVBU	R0, ret+24(FP)
	RET

TEXT bytes·IndexByte(SB),NOSPLIT,$0
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

TEXT strings·IndexByte(SB),NOSPLIT,$0
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
