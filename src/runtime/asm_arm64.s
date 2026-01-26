// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "tls_arm64.h"
#include "funcdata.h"
#include "textflag.h"
#include "cgo/abi_arm64.h"

// _rt0_arm64 is common startup code for most arm64 systems when using
// internal linking. This is the entry point for the program from the
// kernel for an ordinary -buildmode=exe program. The stack holds the
// number of arguments and the C-style argv.
TEXT _rt0_arm64(SB),NOSPLIT,$0
	MOVD	0(RSP), R0	// argc
	ADD	$8, RSP, R1	// argv
	JMP	runtime·rt0_go(SB)

// main is common startup code for most amd64 systems when using
// external linking. The C startup code will call the symbol "main"
// passing argc and argv in the usual C ABI registers R0 and R1.
TEXT main(SB),NOSPLIT,$0
	JMP	runtime·rt0_go(SB)

// _rt0_arm64_lib is common startup code for most arm64 systems when
// using -buildmode=c-archive or -buildmode=c-shared. The linker will
// arrange to invoke this function as a global constructor (for
// c-archive) or when the shared library is loaded (for c-shared).
// We expect argc and argv to be passed in the usual C ABI registers
// R0 and R1.
TEXT _rt0_arm64_lib(SB),NOSPLIT,$184
	// Preserve callee-save registers.
	SAVE_R19_TO_R28(24)
	SAVE_F8_TO_F15(104)

	// Initialize g as null in case of using g later e.g. sigaction in cgo_sigaction.go
	MOVD	ZR, g

	MOVD	R0, _rt0_arm64_lib_argc<>(SB)
	MOVD	R1, _rt0_arm64_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·libpreinit(SB), R4
	BL	(R4)

	// Create a new thread to do the runtime initialization and return.
	MOVD	_cgo_sys_thread_create(SB), R4
	CBZ	R4, nocgo
	MOVD	$_rt0_arm64_lib_go(SB), R0
	MOVD	$0, R1
	SUB	$16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL	(R4)
	ADD	$16, RSP
	B	restore

nocgo:
	MOVD	$0x800000, R0                     // stacksize = 8192KB
	MOVD	$_rt0_arm64_lib_go(SB), R1
	MOVD	R0, 8(RSP)
	MOVD	R1, 16(RSP)
	MOVD	$runtime·newosproc0(SB),R4
	BL	(R4)

restore:
	// Restore callee-save registers.
	RESTORE_R19_TO_R28(24)
	RESTORE_F8_TO_F15(104)
	RET

TEXT _rt0_arm64_lib_go(SB),NOSPLIT,$0
	MOVD	_rt0_arm64_lib_argc<>(SB), R0
	MOVD	_rt0_arm64_lib_argv<>(SB), R1
	MOVD	$runtime·rt0_go(SB),R4
	B	(R4)

DATA _rt0_arm64_lib_argc<>(SB)/8, $0
GLOBL _rt0_arm64_lib_argc<>(SB),NOPTR, $8
DATA _rt0_arm64_lib_argv<>(SB)/8, $0
GLOBL _rt0_arm64_lib_argv<>(SB),NOPTR, $8

#ifdef GOARM64_LSE
DATA no_lse_msg<>+0x00(SB)/64, $"This program can only run on ARM64 processors with LSE support.\n"
GLOBL no_lse_msg<>(SB), RODATA, $64
#endif

// We know for sure that Linux and FreeBSD allow to read instruction set
// attribute registers (while some others OSes, like OpenBSD and Darwin,
// are not). Let's be conservative and allow code reading such registers
// only when we sure this won't lead to sigill.
#ifdef GOOS_linux
#define ISA_REGS_READABLE
#endif
#ifdef GOOS_freebsd
#define ISA_REGS_READABLE
#endif

#ifdef GOARM64_LSE
#ifdef ISA_REGS_READABLE
#define CHECK_GOARM64_LSE
#endif
#endif

TEXT runtime·rt0_go(SB),NOSPLIT|TOPFRAME,$0
	// SP = stack; R0 = argc; R1 = argv

	SUB	$32, RSP
	MOVW	R0, 8(RSP) // argc
	MOVD	R1, 16(RSP) // argv

	// This is typically the entry point for Go programs.
	// Call stack unwinding must not proceed past this frame.
	// Set the frame pointer register to 0 so that frame pointer-based unwinders
	// (which don't use debug info for performance reasons)
	// won't attempt to unwind past this function.
	// See go.dev/issue/63630
	MOVD	$0, R29

#ifdef TLS_darwin
	// Initialize TLS.
	MOVD	ZR, g // clear g, make sure it's not junk.
	SUB	$32, RSP
	MRS_TPIDR_R0
	AND	$~7, R0
	MOVD	R0, 16(RSP)             // arg2: TLS base
	MOVD	$runtime·tls_g(SB), R2
	MOVD	R2, 8(RSP)              // arg1: &tlsg
	BL	·tlsinit(SB)
	ADD	$32, RSP
#endif

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVD	$runtime·g0(SB), g
	MOVD	RSP, R7
	MOVD	$(-64*1024)(R7), R0
	MOVD	R0, g_stackguard0(g)
	MOVD	R0, g_stackguard1(g)
	MOVD	R0, (g_stack+stack_lo)(g)
	MOVD	R7, (g_stack+stack_hi)(g)

	// if there is a _cgo_init, call it using the gcc ABI.
	MOVD	_cgo_init(SB), R12
	CBZ	R12, nocgo

#ifdef GOOS_android
	MRS_TPIDR_R0			// load TLS base pointer
	MOVD	R0, R3			// arg 3: TLS base pointer
	MOVD	$runtime·tls_g(SB), R2 	// arg 2: &tls_g
#else
	MOVD	$0, R2		        // arg 2: not used when using platform's TLS
#endif
	MOVD	$setg_gcc<>(SB), R1	// arg 1: setg
	MOVD	g, R0			// arg 0: G
	SUB	$16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL	(R12)
	ADD	$16, RSP

nocgo:
	BL	runtime·save_g(SB)
	// update stackguard after _cgo_init
	MOVD	(g_stack+stack_lo)(g), R0
	ADD	$const_stackGuard, R0
	MOVD	R0, g_stackguard0(g)
	MOVD	R0, g_stackguard1(g)

	// set the per-goroutine and per-mach "registers"
	MOVD	$runtime·m0(SB), R0

	// save m->g0 = g0
	MOVD	g, m_g0(R0)
	// save m0 to g0->m
	MOVD	R0, g_m(g)

	BL	runtime·check(SB)

#ifdef GOOS_windows
	BL	runtime·wintls(SB)
#endif

	// Check that CPU we use for execution supports instructions targeted during compile-time.
#ifdef CHECK_GOARM64_LSE
	// Read the ID_AA64ISAR0_EL1 register
	MRS	ID_AA64ISAR0_EL1, R0

	// Extract the LSE field (bits [23:20])
	LSR	$20, R0, R0
	AND	$0xf, R0, R0

	// LSE support is indicated by a non-zero value
	CBZ	R0, no_lse
#endif

	MOVW	8(RSP), R0	// copy argc
	MOVW	R0, -8(RSP)
	MOVD	16(RSP), R0		// copy argv
	MOVD	R0, 0(RSP)
	BL	runtime·args(SB)
	BL	runtime·osinit(SB)
	BL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVD	$runtime·mainPC(SB), R0		// entry
	SUB	$16, RSP
	MOVD	R0, 8(RSP) // arg
	MOVD	$0, 0(RSP) // dummy LR
	BL	runtime·newproc(SB)
	ADD	$16, RSP

	// start this M
	BL	runtime·mstart(SB)
	UNDEF

#ifdef CHECK_GOARM64_LSE
no_lse:
	MOVD	$1, R0 // stderr
	MOVD	R0, 8(RSP)
	MOVD	$no_lse_msg<>(SB), R1 // message address
	MOVD	R1, 16(RSP)
	MOVD	$64, R2 // message length
	MOVD	R2, 24(RSP)
	CALL	runtime·write(SB)
	CALL	runtime·exit(SB)
	CALL	runtime·abort(SB)
	RET
#endif

	// Prevent dead-code elimination of debugCallV2 and debugPinnerV1, which are
	// intended to be called by debuggers.
	MOVD	$runtime·debugPinnerV1<ABIInternal>(SB), R0
	MOVD	$runtime·debugCallV2<ABIInternal>(SB), R0

	MOVD	$0, R0
	MOVD	R0, (R0)	// boom
	UNDEF

DATA	runtime·mainPC+0(SB)/8,$runtime·main<ABIInternal>(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

// Windows ARM64 needs an immediate 0xf000 argument.
// See go.dev/issues/53837.
#define BREAK	\
#ifdef GOOS_windows	\
	BRK	$0xf000 	\
#else 				\
	BRK 			\
#endif 				\


TEXT runtime·breakpoint(SB),NOSPLIT|NOFRAME,$0-0
	BREAK
	RET

TEXT runtime·asminit(SB),NOSPLIT|NOFRAME,$0-0
	RET

TEXT runtime·mstart(SB),NOSPLIT|TOPFRAME,$0
	// This is the root frame of new Go-created OS threads.
	// Call stack unwinding must not proceed past this frame.
	// Set the frame pointer register to 0 so that frame pointer-based unwinders
	// (which don't use debug info for performance reasons)
	// won't attempt to unwind past this function.
	// See go.dev/issue/63630
	MOVD	$0, R29
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
	B	gogo<>(SB)

TEXT gogo<>(SB), NOSPLIT|NOFRAME, $0
	MOVD	R6, g
	BL	runtime·save_g(SB)

	MOVD	gobuf_sp(R5), R0
	MOVD	R0, RSP
	MOVD	gobuf_bp(R5), R29
	MOVD	gobuf_lr(R5), LR
	MOVD	gobuf_ctxt(R5), R26
	MOVD	$0, gobuf_sp(R5)
	MOVD	$0, gobuf_bp(R5)
	MOVD	$0, gobuf_lr(R5)
	MOVD	$0, gobuf_ctxt(R5)
	CMP	ZR, ZR // set condition codes for == test, needed by stack split
	MOVD	gobuf_pc(R5), R6
	B	(R6)

// void mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-8
#ifdef GOEXPERIMENT_runtimesecret
	MOVW	g_secret(g), R26
	CBZ 	R26, nosecret
	// Use R26 as a secondary link register
	// We purposefully don't erase it in secretEraseRegistersMcall
	MOVD	LR, R26
	BL 	runtime·secretEraseRegistersMcall(SB)
	MOVD	R26, LR

nosecret:
#endif
	MOVD	R0, R26				// context

	// Save caller state in g->sched
	MOVD	RSP, R0
	MOVD	R0, (g_sched+gobuf_sp)(g)
	MOVD	R29, (g_sched+gobuf_bp)(g)
	MOVD	LR, (g_sched+gobuf_pc)(g)
	MOVD	$0, (g_sched+gobuf_lr)(g)

	// Switch to m->g0 & its stack, call fn.
	MOVD	g, R3
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	CMP	g, R3
	BNE	2(PC)
	B	runtime·badmcall(SB)

	MOVD	(g_sched+gobuf_sp)(g), R0
	MOVD	R0, RSP	// sp = m->g0->sched.sp
	MOVD	$0, R29				// clear frame pointer, as caller may execute on another M
	MOVD	R3, R0				// arg = g
	MOVD	$0, -16(RSP)			// dummy LR
	SUB	$16, RSP
	MOVD	0(R26), R4			// code pointer
	BL	(R4)
	B	runtime·badmcall2(SB)

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack. We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	UNDEF
	BL	(LR)	// make sure this function is not leaf
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
#ifdef GOEXPERIMENT_runtimesecret
	MOVW	g_secret(g), R3
	CBZ		R3, nosecret
	BL 		·secretEraseRegisters(SB)

nosecret:
#endif
	MOVD	fn+0(FP), R3	// R3 = fn
	MOVD	R3, R26		// context
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
	MOVD	$runtime·badsystemstack(SB), R3
	BL	(R3)
	B	runtime·abort(SB)

switch:
	// Switch stacks.
	// The original frame pointer is stored in R29,
	// which is useful for stack unwinding.
	// Save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	BL	gosave_systemstack_switch<>(SB)

	// switch to g0
	MOVD	R5, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R3
	MOVD	R3, RSP

	// call target function
	MOVD	0(R26), R3	// code pointer
	BL	(R3)

	// switch back to g
	MOVD	g_m(g), R3
	MOVD	m_curg(R3), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R0
	MOVD	R0, RSP
	MOVD	(g_sched+gobuf_bp)(g), R29
	MOVD	$0, (g_sched+gobuf_sp)(g)
	MOVD	$0, (g_sched+gobuf_bp)(g)
	RET

noswitch:
	// already on m stack, just call directly
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVD	0(R26), R3	// code pointer
	MOVD.P	16(RSP), R30	// restore LR
	SUB	$8, RSP, R29	// restore FP
	B	(R3)

// func switchToCrashStack0(fn func())
TEXT runtime·switchToCrashStack0<ABIInternal>(SB), NOSPLIT, $0-8
	MOVD	R0, R26    // context register
	MOVD	g_m(g), R1 // curm

	// set g to gcrash
	MOVD	$runtime·gcrash(SB), g // g = &gcrash
	BL	runtime·save_g(SB)         // clobbers R0
	MOVD	R1, g_m(g)             // g.m = curm
	MOVD	g, m_g0(R1)            // curm.g0 = g

	// switch to crashstack
	MOVD	(g_stack+stack_hi)(g), R1
	SUB	$(4*8), R1
	MOVD	R1, RSP

	// call target function
	MOVD	0(R26), R0
	CALL	(R0)

	// should never return
	CALL	runtime·abort(SB)
	UNDEF

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
// Caller has already loaded:
// R3 prolog's LR (R30)
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT|NOFRAME,$0-0
	// Cannot grow scheduler stack (m->g0).
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), R4

	// Called from f.
	// Set g->sched to context in f
	MOVD	RSP, R0
	MOVD	R0, (g_sched+gobuf_sp)(g)
	MOVD	R29, (g_sched+gobuf_bp)(g)
	MOVD	LR, (g_sched+gobuf_pc)(g)
	MOVD	R3, (g_sched+gobuf_lr)(g)
	MOVD	R26, (g_sched+gobuf_ctxt)(g)

	CMP	g, R4
	BNE	3(PC)
	BL	runtime·badmorestackg0(SB)
	B	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVD	m_gsignal(R8), R4
	CMP	g, R4
	BNE	3(PC)
	BL	runtime·badmorestackgsignal(SB)
	B	runtime·abort(SB)

	// Called from f.
	// Set m->morebuf to f's callers.
	MOVD	R3, (m_morebuf+gobuf_pc)(R8)	// f's caller's PC
	MOVD	RSP, R0
	MOVD	R0, (m_morebuf+gobuf_sp)(R8)	// f's caller's RSP
	MOVD	g, (m_morebuf+gobuf_g)(R8)

	// If in secret mode, erase registers on transition
	// from G stack to M stack,
#ifdef GOEXPERIMENT_runtimesecret
	MOVW	g_secret(g), R4
	CBZ 	R4, nosecret
	BL	·secretEraseRegisters(SB)
	MOVD	g_m(g), R8
nosecret:
#endif

	// Call newstack on m->g0's stack.
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R0
	MOVD	R0, RSP
	MOVD	$0, R29		// clear frame pointer, as caller may execute on another M
	MOVD.W	$0, -16(RSP)	// create a call frame on g0 (saved LR; keep 16-aligned)
	BL	runtime·newstack(SB)

	// Not reached, but make sure the return PC from the call to newstack
	// is still in this function, and not the beginning of the next.
	UNDEF

TEXT runtime·morestack_noctxt(SB),NOSPLIT|NOFRAME,$0-0
	// Force SPWRITE. This function doesn't actually write SP,
	// but it is called with a special calling convention where
	// the caller doesn't save LR on stack but passes it as a
	// register (R3), and the unwinder currently doesn't understand.
	// Make it SPWRITE to stop unwinding. (See issue 54332)
	MOVD	RSP, RSP

	MOVW	$0, R26
	B runtime·morestack(SB)

// spillArgs stores return values from registers to a *internal/abi.RegArgs in R20.
TEXT ·spillArgs(SB),NOSPLIT,$0-0
	STP	(R0, R1), (0*8)(R20)
	STP	(R2, R3), (2*8)(R20)
	STP	(R4, R5), (4*8)(R20)
	STP	(R6, R7), (6*8)(R20)
	STP	(R8, R9), (8*8)(R20)
	STP	(R10, R11), (10*8)(R20)
	STP	(R12, R13), (12*8)(R20)
	STP	(R14, R15), (14*8)(R20)
	FSTPD	(F0, F1), (16*8)(R20)
	FSTPD	(F2, F3), (18*8)(R20)
	FSTPD	(F4, F5), (20*8)(R20)
	FSTPD	(F6, F7), (22*8)(R20)
	FSTPD	(F8, F9), (24*8)(R20)
	FSTPD	(F10, F11), (26*8)(R20)
	FSTPD	(F12, F13), (28*8)(R20)
	FSTPD	(F14, F15), (30*8)(R20)
	RET

// unspillArgs loads args into registers from a *internal/abi.RegArgs in R20.
TEXT ·unspillArgs(SB),NOSPLIT,$0-0
	LDP	(0*8)(R20), (R0, R1)
	LDP	(2*8)(R20), (R2, R3)
	LDP	(4*8)(R20), (R4, R5)
	LDP	(6*8)(R20), (R6, R7)
	LDP	(8*8)(R20), (R8, R9)
	LDP	(10*8)(R20), (R10, R11)
	LDP	(12*8)(R20), (R12, R13)
	LDP	(14*8)(R20), (R14, R15)
	FLDPD	(16*8)(R20), (F0, F1)
	FLDPD	(18*8)(R20), (F2, F3)
	FLDPD	(20*8)(R20), (F4, F5)
	FLDPD	(22*8)(R20), (F6, F7)
	FLDPD	(24*8)(R20), (F8, F9)
	FLDPD	(26*8)(R20), (F10, F11)
	FLDPD	(28*8)(R20), (F12, F13)
	FLDPD	(30*8)(R20), (F14, F15)
	RET

// reflectcall: call a function with the given argument list
// func call(stackArgsType *_type, f *FuncVal, stackArgs *byte, stackArgsSize, stackRetOffset, frameSize uint32, regArgs *abi.RegArgs).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	MOVD	$MAXSIZE, R27;		\
	CMP	R27, R16;		\
	BGT	3(PC);			\
	MOVD	$NAME(SB), R27;	\
	B	(R27)
// Note: can't just "B NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT|NOFRAME, $0-48
	MOVWU	frameSize+32(FP), R16
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
	MOVD	$runtime·badreflectcall(SB), R0
	B	(R0)

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-48;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVD	stackArgs+16(FP), R3;			\
	MOVWU	stackArgsSize+24(FP), R4;		\
	ADD	$8, RSP, R5;			\
	BIC	$0xf, R4, R6;			\
	CBZ	R6, 6(PC);			\
	/* if R6=(argsize&~15) != 0 */		\
	ADD	R6, R5, R6;			\
	/* copy 16 bytes a time */		\
	LDP.P	16(R3), (R7, R8);		\
	STP.P	(R7, R8), 16(R5);		\
	CMP	R5, R6;				\
	BNE	-3(PC);				\
	AND	$0xf, R4, R6;			\
	CBZ	R6, 6(PC);			\
	/* if R6=(argsize&15) != 0 */		\
	ADD	R6, R5, R6;			\
	/* copy 1 byte a time for the rest */	\
	MOVBU.P	1(R3), R7;			\
	MOVBU.P	R7, 1(R5);			\
	CMP	R5, R6;				\
	BNE	-3(PC);				\
	/* set up argument registers */		\
	MOVD	regArgs+40(FP), R20;		\
	CALL	·unspillArgs(SB);		\
	/* call function */			\
	MOVD	f+8(FP), R26;			\
	MOVD	(R26), R20;			\
	PCDATA	$PCDATA_StackMapIndex, $0;	\
	BL	(R20);				\
	/* copy return values back */		\
	MOVD	regArgs+40(FP), R20;		\
	CALL	·spillArgs(SB);		\
	MOVD	stackArgsType+0(FP), R7;		\
	MOVD	stackArgs+16(FP), R3;			\
	MOVWU	stackArgsSize+24(FP), R4;			\
	MOVWU	stackRetOffset+28(FP), R6;		\
	ADD	$8, RSP, R5;			\
	ADD	R6, R5; 			\
	ADD	R6, R3;				\
	SUB	R6, R4;				\
	BL	callRet<>(SB);			\
	RET

// callRet copies return values back at the end of call*. This is a
// separate function so it can allocate stack space for the arguments
// to reflectcallmove. It does not follow the Go ABI; it expects its
// arguments in registers.
TEXT callRet<>(SB), NOSPLIT, $48-0
	NO_LOCAL_POINTERS
	STP	(R7, R3), 8(RSP)
	STP	(R5, R4), 24(RSP)
	MOVD	R20, 40(RSP)
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

// func memhash32(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·memhash32<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	MOVB	runtime·useAeshash(SB), R10
	CBZ	R10, noaes
	MOVD	$runtime·aeskeysched+0(SB), R3

	VEOR	V0.B16, V0.B16, V0.B16
	VLD1	(R3), [V2.B16]
	VLD1	(R0), V0.S[1]
	VMOV	R1, V0.S[0]

	AESE	V2.B16, V0.B16
	AESMC	V0.B16, V0.B16
	AESE	V2.B16, V0.B16
	AESMC	V0.B16, V0.B16
	AESE	V2.B16, V0.B16

	VMOV	V0.D[0], R0
	RET
noaes:
	B	runtime·memhash32Fallback<ABIInternal>(SB)

// func memhash64(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·memhash64<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	MOVB	runtime·useAeshash(SB), R10
	CBZ	R10, noaes
	MOVD	$runtime·aeskeysched+0(SB), R3

	VEOR	V0.B16, V0.B16, V0.B16
	VLD1	(R3), [V2.B16]
	VLD1	(R0), V0.D[1]
	VMOV	R1, V0.D[0]

	AESE	V2.B16, V0.B16
	AESMC	V0.B16, V0.B16
	AESE	V2.B16, V0.B16
	AESMC	V0.B16, V0.B16
	AESE	V2.B16, V0.B16

	VMOV	V0.D[0], R0
	RET
noaes:
	B	runtime·memhash64Fallback<ABIInternal>(SB)

// func memhash(p unsafe.Pointer, h, size uintptr) uintptr
TEXT runtime·memhash<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-32
	MOVB	runtime·useAeshash(SB), R10
	CBZ	R10, noaes
	B	aeshashbody<>(SB)
noaes:
	B	runtime·memhashFallback<ABIInternal>(SB)

// func strhash(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·strhash<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-24
	MOVB	runtime·useAeshash(SB), R10
	CBZ	R10, noaes
	LDP	(R0), (R0, R2)	// string data / length
	B	aeshashbody<>(SB)
noaes:
	B	runtime·strhashFallback<ABIInternal>(SB)

// R0: data
// R1: seed data
// R2: length
// At return, R0 = return value
TEXT aeshashbody<>(SB),NOSPLIT|NOFRAME,$0
	VEOR	V30.B16, V30.B16, V30.B16
	VMOV	R1, V30.D[0]
	VMOV	R2, V30.D[1] // load length into seed

	MOVD	$runtime·aeskeysched+0(SB), R4
	VLD1.P	16(R4), [V0.B16]
	AESE	V30.B16, V0.B16
	AESMC	V0.B16, V0.B16
	CMP	$16, R2
	BLO	aes0to15
	BEQ	aes16
	CMP	$32, R2
	BLS	aes17to32
	CMP	$64, R2
	BLS	aes33to64
	CMP	$128, R2
	BLS	aes65to128
	B	aes129plus

aes0to15:
	CBZ	R2, aes0
	VEOR	V2.B16, V2.B16, V2.B16
	TBZ	$3, R2, less_than_8
	VLD1.P	8(R0), V2.D[0]

less_than_8:
	TBZ	$2, R2, less_than_4
	VLD1.P	4(R0), V2.S[2]

less_than_4:
	TBZ	$1, R2, less_than_2
	VLD1.P	2(R0), V2.H[6]

less_than_2:
	TBZ	$0, R2, done
	VLD1	(R0), V2.B[14]
done:
	AESE	V0.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V0.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V0.B16, V2.B16
	AESMC	V2.B16, V2.B16

	VMOV	V2.D[0], R0
	RET

aes0:
	VMOV	V0.D[0], R0
	RET

aes16:
	VLD1	(R0), [V2.B16]
	B	done

aes17to32:
	// make second seed
	VLD1	(R4), [V1.B16]
	AESE	V30.B16, V1.B16
	AESMC	V1.B16, V1.B16
	SUB	$16, R2, R10
	VLD1.P	(R0)(R10), [V2.B16]
	VLD1	(R0), [V3.B16]

	AESE	V0.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V1.B16, V3.B16
	AESMC	V3.B16, V3.B16

	AESE	V0.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V1.B16, V3.B16
	AESMC	V3.B16, V3.B16

	AESE	V0.B16, V2.B16
	AESE	V1.B16, V3.B16

	VEOR	V3.B16, V2.B16, V2.B16

	VMOV	V2.D[0], R0
	RET

aes33to64:
	VLD1	(R4), [V1.B16, V2.B16, V3.B16]
	AESE	V30.B16, V1.B16
	AESMC	V1.B16, V1.B16
	AESE	V30.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V30.B16, V3.B16
	AESMC	V3.B16, V3.B16
	SUB	$32, R2, R10

	VLD1.P	(R0)(R10), [V4.B16, V5.B16]
	VLD1	(R0), [V6.B16, V7.B16]

	AESE	V0.B16, V4.B16
	AESMC	V4.B16, V4.B16
	AESE	V1.B16, V5.B16
	AESMC	V5.B16, V5.B16
	AESE	V2.B16, V6.B16
	AESMC	V6.B16, V6.B16
	AESE	V3.B16, V7.B16
	AESMC	V7.B16, V7.B16

	AESE	V0.B16, V4.B16
	AESMC	V4.B16, V4.B16
	AESE	V1.B16, V5.B16
	AESMC	V5.B16, V5.B16
	AESE	V2.B16, V6.B16
	AESMC	V6.B16, V6.B16
	AESE	V3.B16, V7.B16
	AESMC	V7.B16, V7.B16

	AESE	V0.B16, V4.B16
	AESE	V1.B16, V5.B16
	AESE	V2.B16, V6.B16
	AESE	V3.B16, V7.B16

	VEOR	V6.B16, V4.B16, V4.B16
	VEOR	V7.B16, V5.B16, V5.B16
	VEOR	V5.B16, V4.B16, V4.B16

	VMOV	V4.D[0], R0
	RET

aes65to128:
	VLD1.P	64(R4), [V1.B16, V2.B16, V3.B16, V4.B16]
	VLD1	(R4), [V5.B16, V6.B16, V7.B16]
	AESE	V30.B16, V1.B16
	AESMC	V1.B16, V1.B16
	AESE	V30.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V30.B16, V3.B16
	AESMC	V3.B16, V3.B16
	AESE	V30.B16, V4.B16
	AESMC	V4.B16, V4.B16
	AESE	V30.B16, V5.B16
	AESMC	V5.B16, V5.B16
	AESE	V30.B16, V6.B16
	AESMC	V6.B16, V6.B16
	AESE	V30.B16, V7.B16
	AESMC	V7.B16, V7.B16

	SUB	$64, R2, R10
	VLD1.P	(R0)(R10), [V8.B16, V9.B16, V10.B16, V11.B16]
	VLD1	(R0), [V12.B16, V13.B16, V14.B16, V15.B16]
	AESE	V0.B16,	 V8.B16
	AESMC	V8.B16,  V8.B16
	AESE	V1.B16,	 V9.B16
	AESMC	V9.B16,  V9.B16
	AESE	V2.B16, V10.B16
	AESMC	V10.B16,  V10.B16
	AESE	V3.B16, V11.B16
	AESMC	V11.B16,  V11.B16
	AESE	V4.B16, V12.B16
	AESMC	V12.B16,  V12.B16
	AESE	V5.B16, V13.B16
	AESMC	V13.B16,  V13.B16
	AESE	V6.B16, V14.B16
	AESMC	V14.B16,  V14.B16
	AESE	V7.B16, V15.B16
	AESMC	V15.B16,  V15.B16

	AESE	V0.B16,	 V8.B16
	AESMC	V8.B16,  V8.B16
	AESE	V1.B16,	 V9.B16
	AESMC	V9.B16,  V9.B16
	AESE	V2.B16, V10.B16
	AESMC	V10.B16,  V10.B16
	AESE	V3.B16, V11.B16
	AESMC	V11.B16,  V11.B16
	AESE	V4.B16, V12.B16
	AESMC	V12.B16,  V12.B16
	AESE	V5.B16, V13.B16
	AESMC	V13.B16,  V13.B16
	AESE	V6.B16, V14.B16
	AESMC	V14.B16,  V14.B16
	AESE	V7.B16, V15.B16
	AESMC	V15.B16,  V15.B16

	AESE	V0.B16,	 V8.B16
	AESE	V1.B16,	 V9.B16
	AESE	V2.B16, V10.B16
	AESE	V3.B16, V11.B16
	AESE	V4.B16, V12.B16
	AESE	V5.B16, V13.B16
	AESE	V6.B16, V14.B16
	AESE	V7.B16, V15.B16

	VEOR	V12.B16, V8.B16, V8.B16
	VEOR	V13.B16, V9.B16, V9.B16
	VEOR	V14.B16, V10.B16, V10.B16
	VEOR	V15.B16, V11.B16, V11.B16
	VEOR	V10.B16, V8.B16, V8.B16
	VEOR	V11.B16, V9.B16, V9.B16
	VEOR	V9.B16, V8.B16, V8.B16

	VMOV	V8.D[0], R0
	RET

aes129plus:
	PRFM (R0), PLDL1KEEP
	VLD1.P	64(R4), [V1.B16, V2.B16, V3.B16, V4.B16]
	VLD1	(R4), [V5.B16, V6.B16, V7.B16]
	AESE	V30.B16, V1.B16
	AESMC	V1.B16, V1.B16
	AESE	V30.B16, V2.B16
	AESMC	V2.B16, V2.B16
	AESE	V30.B16, V3.B16
	AESMC	V3.B16, V3.B16
	AESE	V30.B16, V4.B16
	AESMC	V4.B16, V4.B16
	AESE	V30.B16, V5.B16
	AESMC	V5.B16, V5.B16
	AESE	V30.B16, V6.B16
	AESMC	V6.B16, V6.B16
	AESE	V30.B16, V7.B16
	AESMC	V7.B16, V7.B16
	ADD	R0, R2, R10
	SUB	$128, R10, R10
	VLD1.P	64(R10), [V8.B16, V9.B16, V10.B16, V11.B16]
	VLD1	(R10), [V12.B16, V13.B16, V14.B16, V15.B16]
	SUB	$1, R2, R2
	LSR	$7, R2, R2

aesloop:
	AESE	V8.B16,	 V0.B16
	AESMC	V0.B16,  V0.B16
	AESE	V9.B16,	 V1.B16
	AESMC	V1.B16,  V1.B16
	AESE	V10.B16, V2.B16
	AESMC	V2.B16,  V2.B16
	AESE	V11.B16, V3.B16
	AESMC	V3.B16,  V3.B16
	AESE	V12.B16, V4.B16
	AESMC	V4.B16,  V4.B16
	AESE	V13.B16, V5.B16
	AESMC	V5.B16,  V5.B16
	AESE	V14.B16, V6.B16
	AESMC	V6.B16,  V6.B16
	AESE	V15.B16, V7.B16
	AESMC	V7.B16,  V7.B16

	VLD1.P	64(R0), [V8.B16, V9.B16, V10.B16, V11.B16]
	AESE	V8.B16,	 V0.B16
	AESMC	V0.B16,  V0.B16
	AESE	V9.B16,	 V1.B16
	AESMC	V1.B16,  V1.B16
	AESE	V10.B16, V2.B16
	AESMC	V2.B16,  V2.B16
	AESE	V11.B16, V3.B16
	AESMC	V3.B16,  V3.B16

	VLD1.P	64(R0), [V12.B16, V13.B16, V14.B16, V15.B16]
	AESE	V12.B16, V4.B16
	AESMC	V4.B16,  V4.B16
	AESE	V13.B16, V5.B16
	AESMC	V5.B16,  V5.B16
	AESE	V14.B16, V6.B16
	AESMC	V6.B16,  V6.B16
	AESE	V15.B16, V7.B16
	AESMC	V7.B16,  V7.B16
	SUB	$1, R2, R2
	CBNZ	R2, aesloop

	AESE	V8.B16,	 V0.B16
	AESMC	V0.B16,  V0.B16
	AESE	V9.B16,	 V1.B16
	AESMC	V1.B16,  V1.B16
	AESE	V10.B16, V2.B16
	AESMC	V2.B16,  V2.B16
	AESE	V11.B16, V3.B16
	AESMC	V3.B16,  V3.B16
	AESE	V12.B16, V4.B16
	AESMC	V4.B16,  V4.B16
	AESE	V13.B16, V5.B16
	AESMC	V5.B16,  V5.B16
	AESE	V14.B16, V6.B16
	AESMC	V6.B16,  V6.B16
	AESE	V15.B16, V7.B16
	AESMC	V7.B16,  V7.B16

	AESE	V8.B16,	 V0.B16
	AESMC	V0.B16,  V0.B16
	AESE	V9.B16,	 V1.B16
	AESMC	V1.B16,  V1.B16
	AESE	V10.B16, V2.B16
	AESMC	V2.B16,  V2.B16
	AESE	V11.B16, V3.B16
	AESMC	V3.B16,  V3.B16
	AESE	V12.B16, V4.B16
	AESMC	V4.B16,  V4.B16
	AESE	V13.B16, V5.B16
	AESMC	V5.B16,  V5.B16
	AESE	V14.B16, V6.B16
	AESMC	V6.B16,  V6.B16
	AESE	V15.B16, V7.B16
	AESMC	V7.B16,  V7.B16

	AESE	V8.B16,	 V0.B16
	AESE	V9.B16,	 V1.B16
	AESE	V10.B16, V2.B16
	AESE	V11.B16, V3.B16
	AESE	V12.B16, V4.B16
	AESE	V13.B16, V5.B16
	AESE	V14.B16, V6.B16
	AESE	V15.B16, V7.B16

	VEOR	V0.B16, V1.B16, V0.B16
	VEOR	V2.B16, V3.B16, V2.B16
	VEOR	V4.B16, V5.B16, V4.B16
	VEOR	V6.B16, V7.B16, V6.B16
	VEOR	V0.B16, V2.B16, V0.B16
	VEOR	V4.B16, V6.B16, V4.B16
	VEOR	V4.B16, V0.B16, V0.B16

	VMOV	V0.D[0], R0
	RET

// The Arm architecture provides a user space accessible counter-timer which
// is incremented at a fixed but machine-specific rate. Software can (spin)
// wait until the counter-timer reaches some desired value.
//
// Armv8.7-A introduced the WFET (FEAT_WFxT) instruction, which allows the
// processor to enter a low power state for a set time, or until an event is
// received.
//
// However, WFET is not used here because it is only available on newer hardware,
// and we aim to maintain compatibility with older Armv8-A platforms that do not
// support this feature.
//
// As a fallback, we can instead use the ISB instruction to decrease processor
// activity and thus power consumption between checks of the counter-timer.
// Note that we do not depend on the latency of the ISB instruction which is
// implementation specific. Actual delay comes from comparing against a fresh
// read of the counter-timer value.
//
// Read more in this Arm blog post:
// https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/multi-threaded-applications-arm

TEXT runtime·procyieldAsm(SB),NOSPLIT,$0-0
	MOVWU	cycles+0(FP), R0
	CBZ	 R0, done
	//Prevent speculation of subsequent counter/timer reads and memory accesses.
	ISB     $15
	// If the delay is very short, just return.
	// Hardcode 18ns as the first ISB delay.
	CMP     $18, R0
	BLS     done
	// Adjust for overhead of initial ISB.
	SUB     $18, R0, R0
	// Convert the delay from nanoseconds to counter/timer ticks.
	// Read the counter/timer frequency.
	// delay_ticks = (delay * CNTFRQ_EL0) / 1e9
	// With the below simplifications and adjustments,
	// we are usually within 2% of the correct value:
	// delay_ticks = (delay + delay / 16) * CNTFRQ_EL0 >> 30
	MRS     CNTFRQ_EL0, R1
	ADD     R0>>4, R0, R0
	MUL     R1, R0, R0
	LSR     $30, R0, R0
	CBZ     R0, done
	// start = current counter/timer value
	MRS     CNTVCT_EL0, R2
delay:
	// Delay using ISB for all ticks.
	ISB     $15
	// Substract and compare to handle counter roll-over.
	// counter_read() - start < delay_ticks
	MRS     CNTVCT_EL0, R1
	SUB     R2, R1, R1
	CMP     R0, R1
	BCC     delay
done:
	RET

// Save state of caller into g->sched,
// but using fake PC from systemstack_switch.
// Must only be called from functions with no locals ($0)
// or else unwinding from systemstack_switch is incorrect.
// Smashes R0.
TEXT gosave_systemstack_switch<>(SB),NOSPLIT|NOFRAME,$0
	MOVD	$runtime·systemstack_switch(SB), R0
	ADD	$8, R0	// get past prologue
	MOVD	R0, (g_sched+gobuf_pc)(g)
	MOVD	RSP, R0
	MOVD	R0, (g_sched+gobuf_sp)(g)
	MOVD	R29, (g_sched+gobuf_bp)(g)
	MOVD	$0, (g_sched+gobuf_lr)(g)
	// Assert ctxt is zero. See func save.
	MOVD	(g_sched+gobuf_ctxt)(g), R0
	CBZ	R0, 2(PC)
	CALL	runtime·abort(SB)
	RET

// func asmcgocall_no_g(fn, arg unsafe.Pointer)
// Call fn(arg) aligned appropriately for the gcc ABI.
// Called on a system stack, and there may be no g yet (during needm).
TEXT ·asmcgocall_no_g(SB),NOSPLIT,$0-16
	MOVD	fn+0(FP), R1
	MOVD	arg+8(FP), R0
	SUB	$16, RSP	// skip over saved frame pointer below RSP
	BL	(R1)
	ADD	$16, RSP	// skip over saved frame pointer below RSP
	RET

// func asmcgocall(fn, arg unsafe.Pointer) int32
// Call fn(arg) on the scheduler stack,
// aligned appropriately for the gcc ABI.
// See cgocall.go for more details.
TEXT ·asmcgocall(SB),NOSPLIT,$0-20
	CBZ	g, nosave

	// Figure out if we need to switch to m->g0 stack.
	// We get called to create new OS threads too, and those
	// come in on the m->g0 stack already. Or we might already
	// be on the m->gsignal stack.
	MOVD	g_m(g), R8
	MOVD	m_gsignal(R8), R3
	CMP	R3, g
	BEQ	nosave
	MOVD	m_g0(R8), R3
	CMP	R3, g
	BEQ	nosave

	// running on a user stack. Figure out if we're running
	// secret code and clear our registers if so.
#ifdef GOEXPERIMENT_runtimesecret
	MOVW 	g_secret(g), R5
	CBZ		R5, nosecret
	BL 	·secretEraseRegisters(SB)
	// restore g0 back into R3
	MOVD	g_m(g), R3
	MOVD	m_g0(R3), R3

nosecret:
#endif
	MOVD	fn+0(FP), R1
	MOVD	arg+8(FP), R0
	MOVD	RSP, R2
	MOVD	g, R4

	// Switch to system stack.
	MOVD	R0, R9	// gosave_systemstack_switch<> and save_g might clobber R0
	BL	gosave_systemstack_switch<>(SB)
	MOVD	R3, g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R0
	MOVD	R0, RSP
	MOVD	(g_sched+gobuf_bp)(g), R29
	MOVD	R9, R0

	// Now on a scheduling stack (a pthread-created stack).
	// Save room for two of our pointers /*, plus 32 bytes of callee
	// save area that lives on the caller stack. */
	MOVD	RSP, R13
	SUB	$16, R13
	MOVD	R13, RSP
	MOVD	R4, 0(RSP)	// save old g on stack
	MOVD	(g_stack+stack_hi)(R4), R4
	SUB	R2, R4
	MOVD	R4, 8(RSP)	// save depth in old g stack (can't just save SP, as stack might be copied during a callback)
	BL	(R1)
	MOVD	R0, R9

	// Restore g, stack pointer. R0 is errno, so don't touch it
	MOVD	0(RSP), g
	BL	runtime·save_g(SB)
	MOVD	(g_stack+stack_hi)(g), R5
	MOVD	8(RSP), R6
	SUB	R6, R5
	MOVD	R9, R0
	MOVD	R5, RSP

	MOVW	R0, ret+16(FP)
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
	MOVD	fn+0(FP), R1
	MOVD	arg+8(FP), R0
	MOVD	RSP, R2
	MOVD 	R2, R13
	SUB	$16, R13
	MOVD	R13, RSP
	MOVD	$0, R4
	MOVD	R4, 0(RSP)	// Where above code stores g, in case someone looks during debugging.
	MOVD	R2, 8(RSP)	// Save original stack pointer.
	BL	(R1)
	// Restore stack pointer.
	MOVD	8(RSP), R2
	MOVD	R2, RSP
	MOVD	R0, ret+16(FP)
	RET

// cgocallback(fn, frame unsafe.Pointer, ctxt uintptr)
// See cgocall.go for more details.
TEXT ·cgocallback(SB),NOSPLIT,$24-24
	NO_LOCAL_POINTERS

	// Skip cgocallbackg, just dropm when fn is nil, and frame is the saved g.
	// It is used to dropm while thread is exiting.
	MOVD	fn+0(FP), R1
	CBNZ	R1, loadg
	// Restore the g from frame.
	MOVD	frame+8(FP), g
	B	dropm

loadg:
	// Load g from thread-local storage.
	BL	runtime·load_g(SB)

	// If g is nil, Go did not create the current thread,
	// or if this thread never called into Go on pthread platforms.
	// Call needm to obtain one for temporary use.
	// In this case, we're running on the thread stack, so there's
	// lots of space, but the linker doesn't know. Hide the call from
	// the linker analysis by using an indirect call.
	CBZ	g, needm

	MOVD	g_m(g), R8
	MOVD	R8, savedm-8(SP)
	B	havem

needm:
	MOVD	g, savedm-8(SP) // g is zero, so is m.
	MOVD	$runtime·needAndBindM(SB), R0
	BL	(R0)

	// Set m->g0->sched.sp = SP, so that if a panic happens
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
	MOVD	RSP, R0
	MOVD	R0, (g_sched+gobuf_sp)(R3)
	MOVD	R29, (g_sched+gobuf_bp)(R3)

havem:
	// Now there's a valid m, and we're running on its m->g0.
	// Save current m->g0->sched.sp on stack and then set it to SP.
	// Save current sp in m->g0->sched.sp in preparation for
	// switch back to m->curg stack.
	// NOTE: unwindm knows that the saved g->sched.sp is at 16(RSP) aka savedsp-16(SP).
	// Beware that the frame size is actually 32+16.
	MOVD	m_g0(R8), R3
	MOVD	(g_sched+gobuf_sp)(R3), R4
	MOVD	R4, savedsp-16(SP)
	MOVD	RSP, R0
	MOVD	R0, (g_sched+gobuf_sp)(R3)

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
	MOVD	R5, -48(R4)
	MOVD	(g_sched+gobuf_bp)(g), R5
	MOVD	R5, -56(R4)
	// Gather our arguments into registers.
	MOVD	fn+0(FP), R1
	MOVD	frame+8(FP), R2
	MOVD	ctxt+16(FP), R3
	MOVD	$-48(R4), R0 // maintain 16-byte SP alignment
	MOVD	R0, RSP	// switch stack
	MOVD	R1, 8(RSP)
	MOVD	R2, 16(RSP)
	MOVD	R3, 24(RSP)
	MOVD	$runtime·cgocallbackg(SB), R0
	CALL	(R0) // indirect call to bypass nosplit check. We're on a different stack now.

	// Restore g->sched (== m->curg->sched) from saved values.
	MOVD	0(RSP), R5
	MOVD	R5, (g_sched+gobuf_pc)(g)
	MOVD	RSP, R4
	ADD	$48, R4, R4
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// Switch back to m->g0's stack and restore m->g0->sched.sp.
	// (Unlike m->curg, the g0 goroutine never uses sched.pc,
	// so we do not have to restore it.)
	MOVD	g_m(g), R8
	MOVD	m_g0(R8), g
	BL	runtime·save_g(SB)
	MOVD	(g_sched+gobuf_sp)(g), R0
	MOVD	R0, RSP
	MOVD	savedsp-16(SP), R4
	MOVD	R4, (g_sched+gobuf_sp)(g)

	// If the m on entry was nil, we called needm above to borrow an m,
	// 1. for the duration of the call on non-pthread platforms,
	// 2. or the duration of the C thread alive on pthread platforms.
	// If the m on entry wasn't nil,
	// 1. the thread might be a Go thread,
	// 2. or it wasn't the first call from a C thread on pthread platforms,
	//    since then we skip dropm to reuse the m in the first call.
	MOVD	savedm-8(SP), R6
	CBNZ	R6, droppedm

	// Skip dropm to reuse it in the next call, when a pthread key has been created.
	MOVD	_cgo_pthread_key_created(SB), R6
	// It means cgo is disabled when _cgo_pthread_key_created is a nil pointer, need dropm.
	CBZ	R6, dropm
	MOVD	(R6), R6
	CBNZ	R6, droppedm

dropm:
	MOVD	$runtime·dropm(SB), R0
	BL	(R0)
droppedm:

	// Done!
	RET

// Called from cgo wrappers, this function returns g->m->curg.stack.hi.
// Must obey the gcc calling convention.
TEXT _cgo_topofstack(SB),NOSPLIT,$24
	// g (R28) and REGTMP (R27)  might be clobbered by load_g. They
	// are callee-save in the gcc calling convention, so save them.
	MOVD	R27, savedR27-8(SP)
	MOVD	g, saveG-16(SP)

	BL	runtime·load_g(SB)
	MOVD	g_m(g), R0
	MOVD	m_curg(R0), R0
	MOVD	(g_stack+stack_hi)(R0), R0

	MOVD	saveG-16(SP), g
	MOVD	savedR28-8(SP), R27
	RET

// void setg(G*); set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-8
	MOVD	gg+0(FP), g
	// This only happens if iscgo, so jump straight to save_g
	BL	runtime·save_g(SB)
	RET

// void setg_gcc(G*); set g called from gcc
TEXT setg_gcc<>(SB),NOSPLIT,$8
	MOVD	R0, g
	MOVD	R27, savedR27-8(SP)
	BL	runtime·save_g(SB)
	MOVD	savedR27-8(SP), R27
	RET

TEXT runtime·emptyfunc(SB),0,$0-0
	RET

TEXT runtime·abort(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	ZR, R0
	MOVD	(R0), R0
	UNDEF

// The top-most function running on a goroutine
// returns to goexit+PCQuantum.
TEXT runtime·goexit(SB),NOSPLIT|NOFRAME|TOPFRAME,$0-0
	MOVD	R0, R0	// NOP
	BL	runtime·goexit1(SB)	// does not return

// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-0
	SUB	$0x10, RSP
	MOVD	R27, 8(RSP) // The access to global variables below implicitly uses R27, which is callee-save
	MOVD	runtime·lastmoduledatap(SB), R1
	MOVD	R0, moduledata_next(R1)
	MOVD	R0, runtime·lastmoduledatap(SB)
	MOVD	8(RSP), R27
	ADD	$0x10, RSP
	RET

TEXT ·checkASM(SB),NOSPLIT,$0-1
	MOVW	$1, R3
	MOVB	R3, ret+0(FP)
	RET

// gcWriteBarrier informs the GC about heap pointer writes.
//
// gcWriteBarrier does NOT follow the Go ABI. It accepts the
// number of bytes of buffer needed in R25, and returns a pointer
// to the buffer space in R25.
// It clobbers condition codes.
// It does not clobber any general-purpose registers except R27,
// but may clobber others (e.g., floating point registers)
// The act of CALLing gcWriteBarrier will clobber R30 (LR).
TEXT gcWriteBarrier<>(SB),NOSPLIT,$200
	// Save the registers clobbered by the fast path.
	STP	(R0, R1), 184(RSP)
retry:
	MOVD	g_m(g), R0
	MOVD	m_p(R0), R0
	MOVD	(p_wbBuf+wbBuf_next)(R0), R1
	MOVD	(p_wbBuf+wbBuf_end)(R0), R27
	// Increment wbBuf.next position.
	ADD	R25, R1
	// Is the buffer full?
	CMP	R27, R1
	BHI	flush
	// Commit to the larger buffer.
	MOVD	R1, (p_wbBuf+wbBuf_next)(R0)
	// Make return value (the original next position)
	SUB	R25, R1, R25
	// Restore registers.
	LDP	184(RSP), (R0, R1)
	RET

flush:
	// Save all general purpose registers since these could be
	// clobbered by wbBufFlush and were not saved by the caller.
	// R0 and R1 already saved
	STP	(R2, R3), 1*8(RSP)
	STP	(R4, R5), 3*8(RSP)
	STP	(R6, R7), 5*8(RSP)
	STP	(R8, R9), 7*8(RSP)
	STP	(R10, R11), 9*8(RSP)
	STP	(R12, R13), 11*8(RSP)
	STP	(R14, R15), 13*8(RSP)
	// R16, R17 may be clobbered by linker trampoline
	// R18 is unused.
	STP	(R19, R20), 15*8(RSP)
	STP	(R21, R22), 17*8(RSP)
	STP	(R23, R24), 19*8(RSP)
	STP	(R25, R26), 21*8(RSP)
	// R27 is temp register.
	// R28 is g.
	// R29 is frame pointer (unused).
	// R30 is LR, which was saved by the prologue.
	// R31 is SP.

	CALL	runtime·wbBufFlush(SB)
	LDP	1*8(RSP), (R2, R3)
	LDP	3*8(RSP), (R4, R5)
	LDP	5*8(RSP), (R6, R7)
	LDP	7*8(RSP), (R8, R9)
	LDP	9*8(RSP), (R10, R11)
	LDP	11*8(RSP), (R12, R13)
	LDP	13*8(RSP), (R14, R15)
	LDP	15*8(RSP), (R19, R20)
	LDP	17*8(RSP), (R21, R22)
	LDP	19*8(RSP), (R23, R24)
	LDP	21*8(RSP), (R25, R26)
	JMP	retry

TEXT runtime·gcWriteBarrier1<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$8, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier2<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$16, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier3<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$24, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier4<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$32, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier5<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$40, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier6<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$48, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier7<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$56, R25
	JMP	gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier8<ABIInternal>(SB),NOSPLIT,$0
	MOVD	$64, R25
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
//    there are at least 288 bytes free on the stack.
// 2. Set SP as SP-16.
// 3. Store the current LR in (SP) (using the SP after step 2).
// 4. Store the current PC in the LR register.
// 5. Write the desired argument frame size at SP-16
// 6. Save all machine registers (including flags and fpsimd registers)
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
// invoking BRK to raise a breakpoint signal. Note that the signal PC of
// the signal triggered by the BRK instruction is the PC where the signal
// is trapped, not the next PC, so to resume execution, the debugger needs
// to set the signal PC to PC+4. See the comments in the implementation for
// the protocol the debugger is expected to follow. InjectDebugCall in the
// runtime tests demonstrates this protocol.
//
// The debugger must ensure that any pointers passed to the function
// obey escape analysis requirements. Specifically, it must not pass
// a stack pointer to an escaping argument. debugCallV2 cannot check
// this invariant.
//
// This is ABIInternal because Go code injects its PC directly into new
// goroutine stacks.
TEXT runtime·debugCallV2<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-0
	STP	(R29, R30), -280(RSP)
	SUB	$272, RSP, RSP
	SUB	$8, RSP, R29
	// Save all registers that may contain pointers so they can be
	// conservatively scanned.
	//
	// We can't do anything that might clobber any of these
	// registers before this.
	STP	(R27, g), (30*8)(RSP)
	STP	(R25, R26), (28*8)(RSP)
	STP	(R23, R24), (26*8)(RSP)
	STP	(R21, R22), (24*8)(RSP)
	STP	(R19, R20), (22*8)(RSP)
	STP	(R16, R17), (20*8)(RSP)
	STP	(R14, R15), (18*8)(RSP)
	STP	(R12, R13), (16*8)(RSP)
	STP	(R10, R11), (14*8)(RSP)
	STP	(R8, R9), (12*8)(RSP)
	STP	(R6, R7), (10*8)(RSP)
	STP	(R4, R5), (8*8)(RSP)
	STP	(R2, R3), (6*8)(RSP)
	STP	(R0, R1), (4*8)(RSP)

	// Perform a safe-point check.
	MOVD	R30, 8(RSP) // Caller's PC
	CALL	runtime·debugCallCheck(SB)
	MOVD	16(RSP), R0
	CBZ	R0, good

	// The safety check failed. Put the reason string at the top
	// of the stack.
	MOVD	R0, 8(RSP)
	MOVD	24(RSP), R0
	MOVD	R0, 16(RSP)

	// Set R20 to 8 and invoke BRK. The debugger should get the
	// reason a call can't be injected from SP+8 and resume execution.
	MOVD	$8, R20
	BREAK
	JMP	restore

good:
	// Registers are saved and it's safe to make a call.
	// Open up a call frame, moving the stack if necessary.
	//
	// Once the frame is allocated, this will set R20 to 0 and
	// invoke BRK. The debugger should write the argument
	// frame for the call at SP+8, set up argument registers,
	// set the LR as the signal PC + 4, set the PC to the function
	// to call, set R26 to point to the closure (if a closure call),
	// and resume execution.
	//
	// If the function returns, this will set R20 to 1 and invoke
	// BRK. The debugger can then inspect any return value saved
	// on the stack at SP+8 and in registers. To resume execution,
	// the debugger should restore the LR from (SP).
	//
	// If the function panics, this will set R20 to 2 and invoke BRK.
	// The interface{} value of the panic will be at SP+8. The debugger
	// can inspect the panic value and resume execution again.
#define DEBUG_CALL_DISPATCH(NAME,MAXSIZE)	\
	CMP	$MAXSIZE, R0;			\
	BGT	5(PC);				\
	MOVD	$NAME(SB), R0;			\
	MOVD	R0, 8(RSP);			\
	CALL	runtime·debugCallWrap(SB);	\
	JMP	restore

	MOVD	256(RSP), R0 // the argument frame size
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
	MOVD	$debugCallFrameTooLarge<>(SB), R0
	MOVD	R0, 8(RSP)
	MOVD	$20, R0
	MOVD	R0, 16(RSP) // length of debugCallFrameTooLarge string
	MOVD	$8, R20
	BREAK
	JMP	restore

restore:
	// Calls and failures resume here.
	//
	// Set R20 to 16 and invoke BRK. The debugger should restore
	// all registers except for PC and RSP and resume execution.
	MOVD	$16, R20
	BREAK
	// We must not modify flags after this point.

	// Restore pointer-containing registers, which may have been
	// modified from the debugger's copy by stack copying.
	LDP	(30*8)(RSP), (R27, g)
	LDP	(28*8)(RSP), (R25, R26)
	LDP	(26*8)(RSP), (R23, R24)
	LDP	(24*8)(RSP), (R21, R22)
	LDP	(22*8)(RSP), (R19, R20)
	LDP	(20*8)(RSP), (R16, R17)
	LDP	(18*8)(RSP), (R14, R15)
	LDP	(16*8)(RSP), (R12, R13)
	LDP	(14*8)(RSP), (R10, R11)
	LDP	(12*8)(RSP), (R8, R9)
	LDP	(10*8)(RSP), (R6, R7)
	LDP	(8*8)(RSP), (R4, R5)
	LDP	(6*8)(RSP), (R2, R3)
	LDP	(4*8)(RSP), (R0, R1)

	LDP	-8(RSP), (R29, R27)
	ADD	$288, RSP, RSP // Add 16 more bytes, see saveSigContext
	MOVD	-16(RSP), R30 // restore old lr
	JMP	(R27)

// runtime.debugCallCheck assumes that functions defined with the
// DEBUG_CALL_FN macro are safe points to inject calls.
#define DEBUG_CALL_FN(NAME,MAXSIZE)		\
TEXT NAME(SB),WRAPPER,$MAXSIZE-0;		\
	NO_LOCAL_POINTERS;		\
	MOVD	$0, R20;		\
	BREAK;		\
	MOVD	$1, R20;		\
	BREAK;		\
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
	// Copy the panic value to the top of stack at SP+8.
	MOVD	val_type+0(FP), R0
	MOVD	R0, 8(RSP)
	MOVD	val_data+8(FP), R0
	MOVD	R0, 16(RSP)
	MOVD	$2, R20
	BREAK
	RET

TEXT runtime·panicBounds<ABIInternal>(SB),NOSPLIT,$144-0
	NO_LOCAL_POINTERS
	// Save all 16 int registers that could have an index in them.
	// They may be pointers, but if they are they are dead.
	STP	(R0, R1), 24(RSP)
	STP	(R2, R3), 40(RSP)
	STP	(R4, R5), 56(RSP)
	STP	(R6, R7), 72(RSP)
	STP	(R8, R9), 88(RSP)
	STP	(R10, R11), 104(RSP)
	STP	(R12, R13), 120(RSP)
	STP	(R14, R15), 136(RSP)
	MOVD	LR, R0		// PC immediately after call to panicBounds
	ADD	$24, RSP, R1	// pointer to save area
	CALL	runtime·panicBounds64<ABIInternal>(SB)
	RET

TEXT ·getfp<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVD R29, R0
	RET
