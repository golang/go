// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build power64 power64le

#include "zasm_GOOS_GOARCH.h"
#include "funcdata.h"
#include "../../cmd/ld/textflag.h"

TEXT _rt0_go(SB),NOSPLIT,$0
	// initialize essential registers
	BL	runtime·reginit(SB)

	SUB	$24, R1
	MOVW	R3, 8(R1) // argc
	MOVD	R4, 16(R1) // argv

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVD	$runtime·g0(SB), g
	MOVD	$(-64*1024), R31
	ADD	R31, R1, R3
	MOVD	R3, g_stackguard(g)
	MOVD	R3, g_stackguard0(g)
	MOVD	R1, g_stackbase(g)

	// TODO: if there is a _cgo_init, call it.
	// TODO: add TLS

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
	BL	runtime·hashinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVD	$runtime·main·f(SB), R3		// entry
	MOVDU	R3, -8(R1)
	MOVDU	R0, -8(R1)
	MOVDU	R0, -8(R1)
	ARGSIZE(24)
	BL	runtime·newproc(SB)
	ARGSIZE(-1)
	ADD	$24, R1

	// start this M
	BL	runtime·mstart(SB)

	MOVD	R0, 1(R0)
	RETURN

DATA	runtime·main·f+0(SB)/8,$runtime·main(SB)
GLOBL	runtime·main·f(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT,$-8-0
	MOVD	R0, 2(R0) // TODO: TD
	RETURN

TEXT runtime·asminit(SB),NOSPLIT,$-8-0
	RETURN

TEXT runtime·reginit(SB),NOSPLIT,$-8-0
	// set R0 to zero, it's expected by the toolchain
	XOR R0, R0
	// initialize essential FP registers
	FMOVD	$4503601774854144.0, F27
	FMOVD	$0.5, F29
	FSUB	F29, F29, F28
	FADD	F29, F29, F30
	FADD	F30, F30, F31
	RETURN

/*
 *  go-routine
 */

// void gosave(Gobuf*)
// save state in Gobuf; setjmp
TEXT runtime·gosave(SB), NOSPLIT, $-8-8
	MOVD	gobuf+0(FP), R3
	MOVD	R1, gobuf_sp(R3)
	MOVD	LR, R31
	MOVD	R31, gobuf_pc(R3)
	MOVD	g, gobuf_g(R3)
	MOVD	R0, gobuf_lr(R3)
	MOVD	R0, gobuf_ret(R3)
	MOVD	R0, gobuf_ctxt(R3)
	RETURN

// void gogo(Gobuf*)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $-8-8
	MOVD	gobuf+0(FP), R5
	MOVD	gobuf_g(R5), g	// make sure g is not nil
	MOVD	0(g), R4
	MOVD	gobuf_sp(R5), R1
	MOVD	gobuf_lr(R5), R31
	MOVD	R31, LR
	MOVD	gobuf_ret(R5), R3
	MOVD	gobuf_ctxt(R5), R11
	MOVD	R0, gobuf_sp(R5)
	MOVD	R0, gobuf_ret(R5)
	MOVD	R0, gobuf_lr(R5)
	MOVD	R0, gobuf_ctxt(R5)
	CMP	R0, R0 // set condition codes for == test, needed by stack split
	MOVD	gobuf_pc(R5), R31
	MOVD	R31, CTR
	BR	(CTR)

// void mcall(void (*fn)(G*))
// Switch to m->g0's stack, call fn(g).
// Fn must never return.  It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall(SB), NOSPLIT, $-8-8
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
	CMP	g, R3
	BNE	2(PC)
	BR	runtime·badmcall(SB)
	MOVD	fn+0(FP), R4
	MOVD	R4, CTR
	MOVD	(g_sched+gobuf_sp)(g), R1
	MOVDU	R3, -8(R1)
	MOVDU	R0, -8(R1)
	BL	(CTR)
	BR	runtime·badmcall2(SB)

// switchtoM is a dummy routine that onM leaves at the bottom
// of the G stack.  We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the M stack because the one at the top of
// the M stack terminates the stack walk (see topofstack()).
TEXT runtime·switchtoM(SB), NOSPLIT, $0-8
	UNDEF
	BL	(LR)	// make sure this function is not leaf
	RETURN

// void onM(void (*fn)())
// calls fn() on the M stack.
// switches to the M stack if not already on it, and
// switches back when fn() returns.
TEXT runtime·onM(SB), NOSPLIT, $0-8
	MOVD	fn+0(FP), R3	// R3 = fn
	MOVD	R3, CTR
	MOVD	g_m(g), R4	// R4 = m
	MOVD	m_g0(R4), R5	// R5 = g0
	CMP	g, R5
	BEQ	onm

	// save our state in g->sched.  Pretend to
	// be switchtoM if the G stack is scanned.
	MOVD	$runtime·switchtoM(SB), R6
	ADD	$8, R6	// get past prologue
	MOVD	R6, (g_sched+gobuf_pc)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	MOVD	g, (g_sched+gobuf_g)(g)

	// switch to g0
	MOVD	R5, g
	MOVD	(g_sched+gobuf_sp)(g), R1

	// call target function
	ARGSIZE(0)
	BL	(CTR)

	// switch back to g
	MOVD	g_m(g), R3
	MOVD	m_curg(R3), g
	MOVD	(g_sched+gobuf_sp)(g), R1
	MOVD	R0, (g_sched+gobuf_sp)(g)
	RETURN

onm:
	// already on m stack, just call directly
	BL	(CTR)
	RETURN

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
TEXT runtime·morestack(SB),NOSPLIT,$-8-0
	// Cannot grow scheduler stack (m->g0).
	MOVD	g_m(g), R7
	MOVD	m_g0(R7), R8
	CMP	g, R8
	BNE	2(PC)
	BL	runtime·abort(SB)

	MOVW	R3, m_moreframesize(R7)
	MOVW	R4, m_moreargsize(R7)

	// Called from f.
	// Set g->sched to context in f.
	MOVD	R11, (g_sched+gobuf_ctxt)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	LR, R8
	MOVD	R8, (g_sched+gobuf_pc)(g)
	MOVD	R5, (g_sched+gobuf_lr)(g)

	// Called from f.
	// Set m->morebuf to f's caller.
	MOVD	R5, (m_morebuf+gobuf_pc)(R7)	// f's caller's PC
	MOVD	R1, (m_morebuf+gobuf_sp)(R7)	// f's caller's SP
	MOVD	$8(R1), R8			// f's argument pointer
	MOVD	R8, m_moreargp(R7)	
	MOVD	g, (m_morebuf+gobuf_g)(R7)

	// Call newstack on m->g0's stack.
	MOVD	m_g0(R7), g
	MOVD	(g_sched+gobuf_sp)(g), R1
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT,$-8-0
	MOVD	R0, R11
	BR	runtime·morestack(SB)

// Called from panic.  Mimics morestack,
// reuses stack growth code to create a frame
// with the desired args running the desired function.
//
// func call(fn *byte, arg *byte, argsize uint32).
TEXT runtime·newstackcall(SB), NOSPLIT, $-8-20
	// Save our caller's state as the PC and SP to restore when
	// returning from f.
	MOVD	g_m(g), R5
	MOVD	LR, R31
	MOVD	R31, (m_morebuf+gobuf_pc)(R5)	// our caller's PC
	MOVD	R1, (m_morebuf+gobuf_sp)(R5)	// our caller's SP
	MOVD	g, (m_morebuf+gobuf_g)(R5)

	// Save our own state as the PC and SP to restore if this
	// goroutine needs to be restarted.
	MOVD	$runtime·newstackcall(SB), R7
	MOVD	R7, (g_sched+gobuf_pc)(g)
	MOVD	LR, R31
	MOVD	R31, (g_sched+gobuf_lr)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)

	// Set up morestack arguments to call f on a new stack.
	// We set f's frame size to 1, as a hint to newstack that
	// this is a call from runtime.newstackcall.
	// If it turns out that f needs a larger frame than the
	// default stack, f's usual stack growth prolog will
	// allocate a new segment (and recopy the arguments).
	MOVD	fn+0(FP), R7
	MOVD	args+8(FP), R8
	MOVW	n+16(FP), R9

	MOVD	R7, m_cret(R5)
	MOVD	R8, m_moreargp(R5)
	MOVW	R9, m_moreargsize(R5)
	MOVD	$1, R10
	MOVW	R10, m_moreframesize(R5)

	// call newstack on m->g0's stack
	MOVD	m_g0(R5), g
	MOVD	(g_sched+gobuf_sp)(g), R1
	BR	runtime·newstack(SB)

// reflect·call: call a function with the given argument list
// func call(f *FuncVal, arg *byte, argsize uint32).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVD	$MAXSIZE, R31;		\
	CMP	R3, R31;		\
	BGT	4(PC);			\
	MOVD	$runtime·NAME(SB), R31;	\
	MOVD	R31, CTR;		\
	BR	(CTR)

// Note: can't just "BR runtime·NAME(SB)" - bad inlining results.
TEXT reflect·call(SB), NOSPLIT, $-8-24
	MOVW argsize+16(FP), R3
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
	MOVD	$runtime·badreflectcall(SB), R31
	MOVD	R31, CTR
	BR	(CTR)

// Argument map for the callXX frames.  Each has one
// stack map (for the single call) with 3 arguments.
DATA gcargs_reflectcall<>+0x00(SB)/4, $1  // 1 stackmap
DATA gcargs_reflectcall<>+0x04(SB)/4, $6  // 3 args
DATA gcargs_reflectcall<>+0x08(SB)/4, $(const_BitsPointer+(const_BitsPointer<<2)+(const_BitsScalar<<4))
GLOBL gcargs_reflectcall<>(SB),RODATA,$12

// callXX frames have no locals
DATA gclocals_reflectcall<>+0x00(SB)/4, $1  // 1 stackmap
DATA gclocals_reflectcall<>+0x04(SB)/4, $0  // 0 locals
GLOBL gclocals_reflectcall<>(SB),RODATA,$8

#define CALLFN(NAME,MAXSIZE)			\
TEXT runtime·NAME(SB), WRAPPER, $MAXSIZE-24;	\
	FUNCDATA $FUNCDATA_ArgsPointerMaps,gcargs_reflectcall<>(SB);	\
	FUNCDATA $FUNCDATA_LocalsPointerMaps,gclocals_reflectcall<>(SB);\
	/* copy arguments to stack */		\
	MOVD	argptr+8(FP), R3;		\
	MOVW	argsize+16(FP), R4;		\
	MOVD	R1, R5;				\
	ADD	$(8-1), R5;			\
	SUB	$1, R3;				\
	ADD	R5, R4;				\
	CMP	R5, R4;				\
	BEQ	4(PC);				\
	MOVBZU	1(R3), R6;			\
	MOVBZU	R6, 1(R5);			\
	BR	-4(PC);				\
	/* call function */			\
	MOVD	f+0(FP), R11;			\
	MOVD	(R11), R31;			\
	MOVD	R31, CTR;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	BL	(CTR);				\
	/* copy return values back */		\
	MOVD	argptr+8(FP), R3;		\
	MOVW	argsize+16(FP), R4;		\
	MOVW	retoffset+20(FP), R6;		\
	MOVD	R1, R5;				\
	ADD	R6, R5; 			\
	ADD	R6, R3;				\
	SUB	R6, R4;				\
	ADD	$(8-1), R5;			\
	SUB	$1, R3;				\
	ADD	R5, R4;				\
	CMP	R5, R4;				\
	BEQ	4(PC);				\
	MOVBZU	1(R5), R6;			\
	MOVBZU	R6, 1(R3);			\
	BR	-4(PC);				\
	RETURN

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
//
// Lessstack can appear in stack traces for the same reason
// as morestack; in that context, it has 0 arguments.
TEXT runtime·lessstack(SB), NOSPLIT, $-8-0
	// Save return value in m->cret
	MOVD	g_m(g), R5
	MOVD	R3, m_cret(R5)

	// Call oldstack on m->g0's stack.
	MOVD	m_g0(R5), g
	MOVD	(g_sched+gobuf_sp)(g), R1
	BL	runtime·oldstack(SB)

// bool cas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime·cas(SB), NOSPLIT, $0-16
	MOVD	p+0(FP), R3
	MOVW	old+8(FP), R4
	MOVW	new+12(FP), R5
	SYNC
	LWAR	(R3), R6
	CMPW	R6, R4
	BNE	7(PC)
	STWCCC	R5, (R3)
	BNE	-5(PC)
	MOVD	$1, R3
	SYNC
	ISYNC
	RETURN
	MOVD	$0, R3
	BR	-4(PC)

// bool	runtime·cas64(uint64 *val, uint64 old, uint64 new)
// Atomically:
//	if(*val == *old){
//		*val = new;
//		return 1;
//	} else {
//		return 0;
//	}
TEXT runtime·cas64(SB), NOSPLIT, $0-24
	MOVD	p+0(FP), R3
	MOVD	old+8(FP), R4
	MOVD	new+16(FP), R5
	SYNC
	LDAR	(R3), R6
	CMP	R6, R4
	BNE	7(PC)
	STDCCC	R5, (R3)
	BNE	-5(PC)
	MOVD	$1, R3
	SYNC
	ISYNC
	RETURN
	MOVD	$0, R3
	BR	-4(PC)

// bool casp(void **val, void *old, void *new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	} else
//		return 0;
TEXT runtime·casp(SB), NOSPLIT, $0-24
	BR runtime·cas64(SB)

// uint32 xadd(uint32 volatile *val, int32 delta)
// Atomically:
//	*val += delta;
//	return *val;
TEXT runtime·xadd(SB), NOSPLIT, $0-12
	MOVD	p+0(FP), R4
	MOVW	delta+8(FP), R5
	SYNC
	LWAR	(R4), R3
	ADD	R5, R3
	STWCCC	R3, (R4)
	BNE	-4(PC)
	SYNC
	ISYNC
	MOVW	R3, R3
	RETURN

TEXT runtime·xadd64(SB), NOSPLIT, $0-16
	MOVD	p+0(FP), R4
	MOVD	delta+8(FP), R5
	SYNC
	LDAR	(R4), R3
	ADD	R5, R3
	STDCCC	R3, (R4)
	BNE	-4(PC)
	SYNC
	ISYNC
	RETURN

TEXT runtime·xchg(SB), NOSPLIT, $0-12
	MOVD	p+0(FP), R4
	MOVW	new+8(FP), R5
	SYNC
	LWAR	(R4), R3
	STWCCC	R5, (R4)
	BNE	-3(PC)
	SYNC
	ISYNC
	RETURN

TEXT runtime·xchg64(SB), NOSPLIT, $0-16
	MOVD	p+0(FP), R4
	MOVD	new+8(FP), R5
	SYNC
	LDAR	(R4), R3
	STDCCC	R5, (R4)
	BNE	-3(PC)
	SYNC
	ISYNC
	RETURN

TEXT runtime·xchgp(SB), NOSPLIT, $0-16
	BR	runtime·xchg64(SB)

TEXT runtime·procyield(SB),NOSPLIT,$0-0
	MOVD	R0, 17(R0)

TEXT runtime·atomicstorep(SB), NOSPLIT, $0-16
	BR	runtime·atomicstore64(SB)

TEXT runtime·atomicstore(SB), NOSPLIT, $0-12
	MOVD	0(FP), R3
	MOVW	8(FP), R4
	SYNC
	MOVW	R4, 0(R3)
	RETURN

TEXT runtime·atomicstore64(SB), NOSPLIT, $0-16
	MOVD	0(FP), R3
	MOVD	8(FP), R4
	SYNC
	MOVD	R4, 0(R3)
	RETURN

// void jmpdefer(fn, sp);
// called from deferreturn.
// 1. grab stored LR for caller
// 2. sub 4 bytes to get back to BL deferreturn
// 3. BR to fn
TEXT runtime·jmpdefer(SB), NOSPLIT, $-8-16
	MOVD	0(R1), R31
	SUB	$4, R31
	MOVD	R31, LR

	MOVD	fn+0(FP), R11
	MOVD	argp+8(FP), R1
	SUB	$8, R1
	MOVD	0(R11), R3
	MOVD	R3, CTR
	BR	(CTR)

// Save state of caller into g->sched. Smashes R31.
TEXT gosave<>(SB),NOSPLIT,$-8
	MOVD	LR, R31
	MOVD	R31, (g_sched+gobuf_pc)(g)
	MOVD	R1, (g_sched+gobuf_sp)(g)
	MOVD	R0, (g_sched+gobuf_lr)(g)
	MOVD	R0, (g_sched+gobuf_ret)(g)
	MOVD	R0, (g_sched+gobuf_ctxt)(g)
	RETURN

// asmcgocall(void(*fn)(void*), void *arg)
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.c for more details.
TEXT runtime·asmcgocall(SB),NOSPLIT,$0-16
	MOVD	R0, 21(R0)

// cgocallback(void (*fn)(void*), void *frame, uintptr framesize)
// Turn the fn into a Go func (by taking its address) and call
// cgocallback_gofunc.
TEXT runtime·cgocallback(SB),NOSPLIT,$24-24
	MOVD	R0, 22(R0)

// cgocallback_gofunc(FuncVal*, void *frame, uintptr framesize)
// See cgocall.c for more details.
TEXT runtime·cgocallback_gofunc(SB),NOSPLIT,$8-24
	MOVD	R0, 23(R0)

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-16
	MOVD	R0, 24(R0)

// void setg_gcc(G*); set g called from gcc.
TEXT setg_gcc<>(SB),NOSPLIT,$0
	MOVD	R0, 25(R0)

TEXT runtime·getcallerpc(SB),NOSPLIT,$-8-8
	MOVD	0(R1), R3
	RETURN

TEXT runtime·gogetcallerpc(SB),NOSPLIT,$-8-16
	MOVD	0(R1), R3
	MOVD	R3,ret+8(FP)
	RETURN

TEXT runtime·setcallerpc(SB),NOSPLIT,$-8-16
	MOVD	x+8(FP),R3		// addr of first arg
	MOVD	R3, 0(R1)		// set calling pc
	RETURN

TEXT runtime·getcallersp(SB),NOSPLIT,$0-8
	MOVD	sp+0(FP), R3
	SUB	$8, R3
	RETURN

TEXT runtime·abort(SB),NOSPLIT,$-4-0
	MOVW	(R0), R0
	UNDEF

#define	TBRL	268
#define	TBRU	269		/* Time base Upper/Lower */

// int64 runtime·cputicks(void)
TEXT runtime·cputicks(SB),NOSPLIT,$0-0
	MOVW	SPR(TBRU), R4
	MOVW	SPR(TBRL), R3
	MOVW	SPR(TBRU), R5
	CMPW	R4, R5
	BNE	-4(PC)
	SLD	$32, R5
	OR	R5, R3
	RETURN

TEXT runtime·stackguard(SB),NOSPLIT,$0-16
	MOVD	R1, R3
	MOVD	R3, sp+0(FP)
	MOVD	g_stackguard(g), R3
	MOVD	R3, limit+8(FP)
	RETURN

GLOBL runtime·tls0(SB), $64

// AES hashing not implemented for Power
TEXT runtime·aeshash(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1
TEXT runtime·aeshash32(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1
TEXT runtime·aeshash64(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1
TEXT runtime·aeshashstr(SB),NOSPLIT,$-8-0
	MOVW	(R0), R1

TEXT runtime·memeq(SB),NOSPLIT,$-8-24
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R4
	MOVD	count+16(FP), R5
	SUB	$1, R3
	SUB	$1, R4
	ADD	R3, R5, R8
_next:
	CMP	R3, R8
	BNE	3(PC)
	MOVD	$1, R3
	RETURN
	MOVBZU	1(R3), R6
	MOVBZU	1(R4), R7
	CMP	R6, R7
	BEQ	_next

	MOVD	$0, R3
	RETURN

TEXT runtime·gomemeq(SB),NOSPLIT,$0-25
	MOVD	a+0(FP), R3
	MOVD	b+8(FP), R4
	MOVD	count+16(FP), R5
	SUB	$1, R3
	SUB	$1, R4
	ADD	R3, R5, R8
_next2:
	CMP	R3, R8
	BNE	4(PC)
	MOVD	$1, R3
	MOVB	R3, ret+24(FP)
	RETURN
	MOVBZU	1(R3), R6
	MOVBZU	1(R4), R7
	CMP	R6, R7
	BEQ	_next2

	MOVB	R0, ret+24(FP)
	RETURN

// eqstring tests whether two strings are equal.
// See runtime_test.go:eqstring_generic for
// equivlaent Go code.
TEXT runtime·eqstring(SB),NOSPLIT,$0-33
	MOVD	s1len+8(FP), R4
	MOVD	s2len+24(FP), R5
	CMP	R4, R5
	BNE	str_noteq

	MOVD	s1str+0(FP), R3
	MOVD	s2str+16(FP), R4
	SUB	$1, R3
	SUB	$1, R4
	ADD	R3, R5, R8
eq_next:
	CMP	R3, R8
	BNE	4(PC)
	MOVD	$1, R3
	MOVB	R3, ret+32(FP)
	RETURN
	MOVBZU	1(R3), R6
	MOVBZU	1(R4), R7
	CMP	R6, R7
	BEQ	eq_next
str_noteq:
	MOVB	R0, ret+32(FP)
	RETURN

// TODO: share code with memeq?
TEXT bytes·Equal(SB),NOSPLIT,$0-49
	MOVD	a_len+8(FP), R3
	MOVD	b_len+32(FP), R4

	CMP	R3, R4		// unequal lengths are not equal
	BNE	_notequal

	MOVD	a+0(FP), R5
	MOVD	b+24(FP), R6
	SUB	$1, R5
	SUB	$1, R6
	ADD	R5, R3		// end-1

_byteseq_next:
	CMP	R5, R3
	BEQ	_equal		// reached the end
	MOVBZU	1(R5), R4
	MOVBZU	1(R6), R7
	CMP	R4, R7
	BEQ	_byteseq_next

_notequal:
	MOVBZ	R0, ret+48(FP)
	RETURN

_equal:
	MOVD	$1, R3
	MOVBZ	R3, ret+48(FP)
	RETURN

TEXT bytes·IndexByte(SB),NOSPLIT,$0-40
	MOVD	s+0(FP), R3
	MOVD	s_len+8(FP), R4
	MOVBZ	c+24(FP), R5	// byte to find
	MOVD	R3, R6		// store base for later
	SUB	$1, R3
	ADD	R3, R4		// end-1

_index_loop:
	CMP	R3, R4
	BEQ	_index_notfound
	MOVBZU	1(R3), R7
	CMP	R7, R5
	BNE	_index_loop

	SUB	R6, R3		// remove base
	MOVD	R3, ret+32(FP)
	RETURN

_index_notfound:
	MOVD	$-1, R3
	MOVD	R3, ret+32(FP)
	RETURN

TEXT strings·IndexByte(SB),NOSPLIT,$0
	MOVD	p+0(FP), R3
	MOVD	b_len+8(FP), R4
	MOVBZ	c+16(FP), R5	// byte to find
	MOVD	R3, R6		// store base for later
	SUB	$1, R3
	ADD	R3, R4		// end-1

_index2_loop:
	CMP	R3, R4
	BEQ	_index2_notfound
	MOVBZU	1(R3), R7
	CMP	R7, R5
	BNE	_index2_loop

	SUB	R6, R3		// remove base
	MOVD	R3, ret+24(FP)
	RETURN

_index2_notfound:
	MOVD	$-1, R3
	MOVD	R3, ret+24(FP)
	RETURN


TEXT runtime·timenow(SB), NOSPLIT, $0-0
	BR	time·now(SB)

// A Duff's device for zeroing memory.
// The compiler jumps to computed addresses within
// this routine to zero chunks of memory.  Do not
// change this code without also changing the code
// in ../../cmd/9g/ggen.c:/^clearfat.
// R0: always zero
// R3 (aka REGRT1): ptr to memory to be zeroed - 8
// R3 is updated as a side effect.
TEXT runtime·duffzero(SB), NOSPLIT, $-8-0
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	MOVDU	R0, 8(R3)
	RETURN

TEXT runtime·fastrand2(SB), NOSPLIT, $0-4
	MOVD	g_m(g), R4
	MOVD	m_fastrand(R4), R3
	ADD	R3, R3
	CMP	R3, $0
	BGE	2(PC)
	XOR	$0x88888eef, R3
	MOVD	R3, m_fastrand(R4)
	MOVD	R3, ret+0(FP)
	RETURN

// The gohash and goeq trampolines are necessary while we have
// both Go and C calls to alg functions.  Once we move all call
// sites to Go, we can redo the hash/eq functions to use the
// Go calling convention and remove these.

// convert call to:
//   func (alg unsafe.Pointer, p unsafe.Pointer, size uintpr, seed uintptr) uintptr
// to:
//   func (hash *uintptr, size uintptr, p unsafe.Pointer)
TEXT runtime·gohash(SB), NOSPLIT, $24-40
	FUNCDATA $FUNCDATA_ArgsPointerMaps,gcargs_gohash<>(SB)
	FUNCDATA $FUNCDATA_LocalsPointerMaps,gclocals_gohash<>(SB)
	MOVD	a+0(FP), R3
	MOVD	alg_hash(R3), R3
	MOVD	R3, CTR
	MOVD	p+8(FP), R4
	MOVD	size+16(FP), R5
	MOVD	seed+24(FP), R6
	MOVD	R6, ret+32(FP)
	MOVD	$ret+32(FP), R7
	MOVD	R7, 8(R1)
	MOVD	R5, 16(R1)
	MOVD	R4, 24(R1)
	PCDATA  $PCDATA_StackMapIndex, $0
	BL	(CTR)
	RETURN

DATA gcargs_gohash<>+0x00(SB)/4, $1  // 1 stackmap
DATA gcargs_gohash<>+0x04(SB)/4, $10  // 5 args
DATA gcargs_gohash<>+0x08(SB)/4, $(const_BitsPointer+(const_BitsPointer<<2))
GLOBL gcargs_gohash<>(SB),RODATA,$12

DATA gclocals_gohash<>+0x00(SB)/4, $1  // 1 stackmap
DATA gclocals_gohash<>+0x04(SB)/4, $0  // 0 locals
GLOBL gclocals_gohash<>(SB),RODATA,$8

// convert call to:
//   func (alg unsafe.Pointer, p, q unsafe.Pointer, size uintptr) bool
// to:
//   func (eq *bool, size uintptr, p, q unsafe.Pointer)
TEXT runtime·goeq(SB), NOSPLIT, $32-33
	FUNCDATA $FUNCDATA_ArgsPointerMaps,gcargs_goeq<>(SB)
	FUNCDATA $FUNCDATA_LocalsPointerMaps,gclocals_goeq<>(SB)
	MOVD	alg+0(FP), R3
	MOVD	alg_equal(R3), R3
	MOVD	R3, CTR
	MOVD	p+8(FP), R4
	MOVD	q+16(FP), R5
	MOVD	size+24(FP), R6
	MOVD	$ret+32(FP), R7
	MOVD	R7, 8(R1)
	MOVD	R6, 16(R1)
	MOVD	R5, 24(R1)
	MOVD	R4, 32(R1)
	PCDATA  $PCDATA_StackMapIndex, $0
	BL	(CTR)
	RETURN

DATA gcargs_goeq<>+0x00(SB)/4, $1  // 1 stackmap
DATA gcargs_goeq<>+0x04(SB)/4, $10  // 5 args
DATA gcargs_goeq<>+0x08(SB)/4, $(const_BitsPointer+(const_BitsPointer<<2)+(const_BitsPointer<<4))
GLOBL gcargs_goeq<>(SB),RODATA,$12

DATA gclocals_goeq<>+0x00(SB)/4, $1  // 1 stackmap
DATA gclocals_goeq<>+0x04(SB)/4, $0  // 0 locals
GLOBL gclocals_goeq<>(SB),RODATA,$8
