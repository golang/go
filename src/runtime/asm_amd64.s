// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "funcdata.h"
#include "textflag.h"
#include "cgo/abi_amd64.h"

// _rt0_amd64 is common startup code for most amd64 systems when using
// internal linking. This is the entry point for the program from the
// kernel for an ordinary -buildmode=exe program. The stack holds the
// number of arguments and the C-style argv.
TEXT _rt0_amd64(SB),NOSPLIT,$-8
	MOVQ	0(SP), DI	// argc
	LEAQ	8(SP), SI	// argv
	JMP	runtime·rt0_go(SB)

// main is common startup code for most amd64 systems when using
// external linking. The C startup code will call the symbol "main"
// passing argc and argv in the usual C ABI registers DI and SI.
TEXT main(SB),NOSPLIT,$-8
	JMP	runtime·rt0_go(SB)

// _rt0_amd64_lib is common startup code for most amd64 systems when
// using -buildmode=c-archive or -buildmode=c-shared. The linker will
// arrange to invoke this function as a global constructor (for
// c-archive) or when the shared library is loaded (for c-shared).
// We expect argc and argv to be passed in the usual C ABI registers
// DI and SI.
TEXT _rt0_amd64_lib(SB),NOSPLIT|NOFRAME,$0
	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	MOVQ	DI, _rt0_amd64_lib_argc<>(SB)
	MOVQ	SI, _rt0_amd64_lib_argv<>(SB)

	// Synchronous initialization.
	CALL	runtime·libpreinit(SB)

	// Create a new thread to finish Go runtime initialization.
	MOVQ	_cgo_sys_thread_create(SB), AX
	TESTQ	AX, AX
	JZ	nocgo

	// We're calling back to C.
	// Align stack per ELF ABI requirements.
	MOVQ	SP, BX  // Callee-save in C ABI
	ANDQ	$~15, SP
	MOVQ	$_rt0_amd64_lib_go(SB), DI
	MOVQ	$0, SI
	CALL	AX
	MOVQ	BX, SP
	JMP	restore

nocgo:
	ADJSP	$16
	MOVQ	$0x800000, 0(SP)		// stacksize
	MOVQ	$_rt0_amd64_lib_go(SB), AX
	MOVQ	AX, 8(SP)			// fn
	CALL	runtime·newosproc0(SB)
	ADJSP	$-16

restore:
	POP_REGS_HOST_TO_ABI0()
	RET

// _rt0_amd64_lib_go initializes the Go runtime.
// This is started in a separate thread by _rt0_amd64_lib.
TEXT _rt0_amd64_lib_go(SB),NOSPLIT,$0
	MOVQ	_rt0_amd64_lib_argc<>(SB), DI
	MOVQ	_rt0_amd64_lib_argv<>(SB), SI
	JMP	runtime·rt0_go(SB)

DATA _rt0_amd64_lib_argc<>(SB)/8, $0
GLOBL _rt0_amd64_lib_argc<>(SB),NOPTR, $8
DATA _rt0_amd64_lib_argv<>(SB)/8, $0
GLOBL _rt0_amd64_lib_argv<>(SB),NOPTR, $8

#ifdef GOAMD64_v2
DATA bad_cpu_msg<>+0x00(SB)/84, $"This program can only be run on AMD64 processors with v2 microarchitecture support.\n"
#endif

#ifdef GOAMD64_v3
DATA bad_cpu_msg<>+0x00(SB)/84, $"This program can only be run on AMD64 processors with v3 microarchitecture support.\n"
#endif

#ifdef GOAMD64_v4
DATA bad_cpu_msg<>+0x00(SB)/84, $"This program can only be run on AMD64 processors with v4 microarchitecture support.\n"
#endif

GLOBL bad_cpu_msg<>(SB), RODATA, $84

// Define a list of AMD64 microarchitecture level features
// https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels

                     // SSE3     SSSE3    CMPXCHNG16 SSE4.1    SSE4.2    POPCNT
#define V2_FEATURES_CX (1 << 0 | 1 << 9 | 1 << 13  | 1 << 19 | 1 << 20 | 1 << 23)
                         // LAHF/SAHF
#define V2_EXT_FEATURES_CX (1 << 0)
                                      // FMA       MOVBE     OSXSAVE   AVX       F16C
#define V3_FEATURES_CX (V2_FEATURES_CX | 1 << 12 | 1 << 22 | 1 << 27 | 1 << 28 | 1 << 29)
                                              // ABM (FOR LZNCT)
#define V3_EXT_FEATURES_CX (V2_EXT_FEATURES_CX | 1 << 5)
                         // BMI1     AVX2     BMI2
#define V3_EXT_FEATURES_BX (1 << 3 | 1 << 5 | 1 << 8)
                       // XMM      YMM
#define V3_OS_SUPPORT_AX (1 << 1 | 1 << 2)

#define V4_FEATURES_CX V3_FEATURES_CX

#define V4_EXT_FEATURES_CX V3_EXT_FEATURES_CX
                                              // AVX512F   AVX512DQ  AVX512CD  AVX512BW  AVX512VL
#define V4_EXT_FEATURES_BX (V3_EXT_FEATURES_BX | 1 << 16 | 1 << 17 | 1 << 28 | 1 << 30 | 1 << 31)
                                          // OPMASK   ZMM
#define V4_OS_SUPPORT_AX (V3_OS_SUPPORT_AX | 1 << 5 | (1 << 6 | 1 << 7))

#ifdef GOAMD64_v2
#define NEED_MAX_CPUID 0x80000001
#define NEED_FEATURES_CX V2_FEATURES_CX
#define NEED_EXT_FEATURES_CX V2_EXT_FEATURES_CX
#endif

#ifdef GOAMD64_v3
#define NEED_MAX_CPUID 0x80000001
#define NEED_FEATURES_CX V3_FEATURES_CX
#define NEED_EXT_FEATURES_CX V3_EXT_FEATURES_CX
#define NEED_EXT_FEATURES_BX V3_EXT_FEATURES_BX
#define NEED_OS_SUPPORT_AX V3_OS_SUPPORT_AX
#endif

#ifdef GOAMD64_v4
#define NEED_MAX_CPUID 0x80000001
#define NEED_FEATURES_CX V4_FEATURES_CX
#define NEED_EXT_FEATURES_CX V4_EXT_FEATURES_CX
#define NEED_EXT_FEATURES_BX V4_EXT_FEATURES_BX

// Darwin requires a different approach to check AVX512 support, see CL 285572.
#ifdef GOOS_darwin
#define NEED_OS_SUPPORT_AX V3_OS_SUPPORT_AX
// These values are from:
// https://github.com/apple/darwin-xnu/blob/xnu-4570.1.46/osfmk/i386/cpu_capabilities.h
#define commpage64_base_address         0x00007fffffe00000
#define commpage64_cpu_capabilities64   (commpage64_base_address+0x010)
#define commpage64_version              (commpage64_base_address+0x01E)
#define AVX512F                         0x0000004000000000
#define AVX512CD                        0x0000008000000000
#define AVX512DQ                        0x0000010000000000
#define AVX512BW                        0x0000020000000000
#define AVX512VL                        0x0000100000000000
#define NEED_DARWIN_SUPPORT             (AVX512F | AVX512DQ | AVX512CD | AVX512BW | AVX512VL)
#else
#define NEED_OS_SUPPORT_AX V4_OS_SUPPORT_AX
#endif

#endif

TEXT runtime·rt0_go(SB),NOSPLIT|NOFRAME|TOPFRAME,$0
	// copy arguments forward on an even stack
	MOVQ	DI, AX		// argc
	MOVQ	SI, BX		// argv
	SUBQ	$(5*8), SP		// 3args 2auto
	ANDQ	$~15, SP
	MOVQ	AX, 24(SP)
	MOVQ	BX, 32(SP)

	// create istack out of the given (operating system) stack.
	// _cgo_init may update stackguard.
	MOVQ	$runtime·g0(SB), DI
	LEAQ	(-64*1024)(SP), BX
	MOVQ	BX, g_stackguard0(DI)
	MOVQ	BX, g_stackguard1(DI)
	MOVQ	BX, (g_stack+stack_lo)(DI)
	MOVQ	SP, (g_stack+stack_hi)(DI)

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
	// if there is an _cgo_init, call it.
	MOVQ	_cgo_init(SB), AX
	TESTQ	AX, AX
	JZ	needtls
	// arg 1: g0, already in DI
	MOVQ	$setg_gcc<>(SB), SI // arg 2: setg_gcc
	MOVQ	$0, DX	// arg 3, 4: not used when using platform's TLS
	MOVQ	$0, CX
#ifdef GOOS_android
	MOVQ	$runtime·tls_g(SB), DX 	// arg 3: &tls_g
	// arg 4: TLS base, stored in slot 0 (Android's TLS_SLOT_SELF).
	// Compensate for tls_g (+16).
	MOVQ	-16(TLS), CX
#endif
#ifdef GOOS_windows
	MOVQ	$runtime·tls_g(SB), DX 	// arg 3: &tls_g
	// Adjust for the Win64 calling convention.
	MOVQ	CX, R9 // arg 4
	MOVQ	DX, R8 // arg 3
	MOVQ	SI, DX // arg 2
	MOVQ	DI, CX // arg 1
#endif
	CALL	AX

	// update stackguard after _cgo_init
	MOVQ	$runtime·g0(SB), CX
	MOVQ	(g_stack+stack_lo)(CX), AX
	ADDQ	$const_stackGuard, AX
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
#ifdef GOOS_illumos
	// skip TLS setup on illumos
	JMP ok
#endif
#ifdef GOOS_darwin
	// skip TLS setup on Darwin
	JMP ok
#endif
#ifdef GOOS_openbsd
	// skip TLS setup on OpenBSD
	JMP ok
#endif

#ifdef GOOS_windows
	CALL	runtime·wintls(SB)
#endif

	LEAQ	runtime·m0+m_tls(SB), DI
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
	LEAQ	runtime·g0(SB), CX
	MOVQ	CX, g(BX)
	LEAQ	runtime·m0(SB), AX

	// save m->g0 = g0
	MOVQ	CX, m_g0(AX)
	// save m0 to g0->m
	MOVQ	AX, g_m(CX)

	CLD				// convention is D is always left cleared

	// Check GOAMD64 requirements
	// We need to do this after setting up TLS, so that
	// we can report an error if there is a failure. See issue 49586.
#ifdef NEED_FEATURES_CX
	MOVL	$0, AX
	CPUID
	CMPL	AX, $0
	JE	bad_cpu
	MOVL	$1, AX
	CPUID
	ANDL	$NEED_FEATURES_CX, CX
	CMPL	CX, $NEED_FEATURES_CX
	JNE	bad_cpu
#endif

#ifdef NEED_MAX_CPUID
	MOVL	$0x80000000, AX
	CPUID
	CMPL	AX, $NEED_MAX_CPUID
	JL	bad_cpu
#endif

#ifdef NEED_EXT_FEATURES_BX
	MOVL	$7, AX
	MOVL	$0, CX
	CPUID
	ANDL	$NEED_EXT_FEATURES_BX, BX
	CMPL	BX, $NEED_EXT_FEATURES_BX
	JNE	bad_cpu
#endif

#ifdef NEED_EXT_FEATURES_CX
	MOVL	$0x80000001, AX
	CPUID
	ANDL	$NEED_EXT_FEATURES_CX, CX
	CMPL	CX, $NEED_EXT_FEATURES_CX
	JNE	bad_cpu
#endif

#ifdef NEED_OS_SUPPORT_AX
	XORL    CX, CX
	XGETBV
	ANDL	$NEED_OS_SUPPORT_AX, AX
	CMPL	AX, $NEED_OS_SUPPORT_AX
	JNE	bad_cpu
#endif

#ifdef NEED_DARWIN_SUPPORT
	MOVQ	$commpage64_version, BX
	CMPW	(BX), $13  // cpu_capabilities64 undefined in versions < 13
	JL	bad_cpu
	MOVQ	$commpage64_cpu_capabilities64, BX
	MOVQ	(BX), BX
	MOVQ	$NEED_DARWIN_SUPPORT, CX
	ANDQ	CX, BX
	CMPQ	BX, CX
	JNE	bad_cpu
#endif

	CALL	runtime·check(SB)

	MOVL	24(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	32(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	runtime·args(SB)
	CALL	runtime·osinit(SB)
	CALL	runtime·schedinit(SB)

	// create a new goroutine to start program
	MOVQ	$runtime·mainPC(SB), AX		// entry
	PUSHQ	AX
	CALL	runtime·newproc(SB)
	POPQ	AX

	// start this M
	CALL	runtime·mstart(SB)

	CALL	runtime·abort(SB)	// mstart should never return
	RET

bad_cpu: // show that the program requires a certain microarchitecture level.
	MOVQ	$2, 0(SP)
	MOVQ	$bad_cpu_msg<>(SB), AX
	MOVQ	AX, 8(SP)
	MOVQ	$84, 16(SP)
	CALL	runtime·write(SB)
	MOVQ	$1, 0(SP)
	CALL	runtime·exit(SB)
	CALL	runtime·abort(SB)
	RET

	// Prevent dead-code elimination of debugCallV2 and debugPinnerV1, which are
	// intended to be called by debuggers.
	MOVQ	$runtime·debugPinnerV1<ABIInternal>(SB), AX
	MOVQ	$runtime·debugCallV2<ABIInternal>(SB), AX
	RET

// mainPC is a function value for runtime.main, to be passed to newproc.
// The reference to runtime.main is made via ABIInternal, since the
// actual function (not the ABI0 wrapper) is needed by newproc.
DATA	runtime·mainPC+0(SB)/8,$runtime·main<ABIInternal>(SB)
GLOBL	runtime·mainPC(SB),RODATA,$8

TEXT runtime·breakpoint(SB),NOSPLIT,$0-0
	BYTE	$0xcc
	RET

TEXT runtime·asminit(SB),NOSPLIT,$0-0
	// No per-thread init.
	RET

TEXT runtime·mstart(SB),NOSPLIT|TOPFRAME|NOFRAME,$0
	CALL	runtime·mstart0(SB)
	RET // not reached

/*
 *  go-routine
 */

// func gogo(buf *gobuf)
// restore state from Gobuf; longjmp
TEXT runtime·gogo(SB), NOSPLIT, $0-8
	MOVQ	buf+0(FP), BX		// gobuf
	MOVQ	gobuf_g(BX), DX
	MOVQ	0(DX), CX		// make sure g != nil
	JMP	gogo<>(SB)

TEXT gogo<>(SB), NOSPLIT, $0
	get_tls(CX)
	MOVQ	DX, g(CX)
	MOVQ	DX, R14		// set the g register
	MOVQ	gobuf_sp(BX), SP	// restore SP
	MOVQ	gobuf_ctxt(BX), DX
	MOVQ	gobuf_bp(BX), BP
	MOVQ	$0, gobuf_sp(BX)	// clear to help garbage collector
	MOVQ	$0, gobuf_ctxt(BX)
	MOVQ	$0, gobuf_bp(BX)
	MOVQ	gobuf_pc(BX), BX
	JMP	BX

// func mcall(fn func(*g))
// Switch to m->g0's stack, call fn(g).
// Fn must never return. It should gogo(&g->sched)
// to keep running g.
TEXT runtime·mcall<ABIInternal>(SB), NOSPLIT, $0-8
	MOVQ	AX, DX	// DX = fn

	// Save state in g->sched. The caller's SP and PC are restored by gogo to
	// resume execution in the caller's frame (implicit return). The caller's BP
	// is also restored to support frame pointer unwinding.
	MOVQ	SP, BX	// hide (SP) reads from vet
	MOVQ	8(BX), BX	// caller's PC
	MOVQ	BX, (g_sched+gobuf_pc)(R14)
	LEAQ	fn+0(FP), BX	// caller's SP
	MOVQ	BX, (g_sched+gobuf_sp)(R14)
	// Get the caller's frame pointer by dereferencing BP. Storing BP as it is
	// can cause a frame pointer cycle, see CL 476235.
	MOVQ	(BP), BX // caller's BP
	MOVQ	BX, (g_sched+gobuf_bp)(R14)

	// switch to m->g0 & its stack, call fn
	MOVQ	g_m(R14), BX
	MOVQ	m_g0(BX), SI	// SI = g.m.g0
	CMPQ	SI, R14	// if g == m->g0 call badmcall
	JNE	goodm
	JMP	runtime·badmcall(SB)
goodm:
	MOVQ	R14, AX		// AX (and arg 0) = g
	MOVQ	SI, R14		// g = g.m.g0
	get_tls(CX)		// Set G in TLS
	MOVQ	R14, g(CX)
	MOVQ	(g_sched+gobuf_sp)(R14), SP	// sp = g0.sched.sp
	MOVQ	$0, BP	// clear frame pointer, as caller may execute on another M
	PUSHQ	AX	// open up space for fn's arg spill slot
	MOVQ	0(DX), R12
	CALL	R12		// fn(g)
	// The Windows native stack unwinder incorrectly classifies the next instruction
	// as part of the function epilogue, producing a wrong call stack.
	// Add a NOP to work around this issue. See go.dev/issue/67007.
	BYTE	$0x90
	POPQ	AX
	JMP	runtime·badmcall2(SB)
	RET

// systemstack_switch is a dummy routine that systemstack leaves at the bottom
// of the G stack. We need to distinguish the routine that
// lives at the bottom of the G stack from the one that lives
// at the top of the system stack because the one at the top of
// the system stack terminates the stack walk (see topofstack()).
// The frame layout needs to match systemstack
// so that it can pretend to be systemstack_switch.
TEXT runtime·systemstack_switch(SB), NOSPLIT, $0-0
	UNDEF
	// Make sure this function is not leaf,
	// so the frame is saved.
	CALL	runtime·abort(SB)
	RET

// func systemstack(fn func())
TEXT runtime·systemstack(SB), NOSPLIT, $0-8
	MOVQ	fn+0(FP), DI	// DI = fn
	get_tls(CX)
	MOVQ	g(CX), AX	// AX = g
	MOVQ	g_m(AX), BX	// BX = m

	CMPQ	AX, m_gsignal(BX)
	JEQ	noswitch

	MOVQ	m_g0(BX), DX	// DX = g0
	CMPQ	AX, DX
	JEQ	noswitch

	CMPQ	AX, m_curg(BX)
	JNE	bad

	// Switch stacks.
	// The original frame pointer is stored in BP,
	// which is useful for stack unwinding.
	// Save our state in g->sched. Pretend to
	// be systemstack_switch if the G stack is scanned.
	CALL	gosave_systemstack_switch<>(SB)

	// switch to g0
	MOVQ	DX, g(CX)
	MOVQ	DX, R14 // set the g register
	MOVQ	(g_sched+gobuf_sp)(DX), SP

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
	MOVQ	(g_sched+gobuf_bp)(AX), BP
	MOVQ	$0, (g_sched+gobuf_sp)(AX)
	MOVQ	$0, (g_sched+gobuf_bp)(AX)
	RET

noswitch:
	// already on m stack; tail call the function
	// Using a tail call here cleans up tracebacks since we won't stop
	// at an intermediate systemstack.
	MOVQ	DI, DX
	MOVQ	0(DI), DI
	// The function epilogue is not called on a tail call.
	// Pop BP from the stack to simulate it.
	POPQ	BP
	JMP	DI

bad:
	// Bad: g is not gsignal, not g0, not curg. What is it?
	MOVQ	$runtime·badsystemstack(SB), AX
	CALL	AX
	INT	$3

// func switchToCrashStack0(fn func())
TEXT runtime·switchToCrashStack0<ABIInternal>(SB), NOSPLIT, $0-8
	MOVQ	g_m(R14), BX // curm

	// set g to gcrash
	LEAQ	runtime·gcrash(SB), R14 // g = &gcrash
	MOVQ	BX, g_m(R14)            // g.m = curm
	MOVQ	R14, m_g0(BX)           // curm.g0 = g
	get_tls(CX)
	MOVQ	R14, g(CX)

	// switch to crashstack
	MOVQ	(g_stack+stack_hi)(R14), BX
	SUBQ	$(4*8), BX
	MOVQ	BX, SP

	// call target function
	MOVQ	AX, DX
	MOVQ	0(AX), AX
	CALL	AX

	// should never return
	CALL	runtime·abort(SB)
	UNDEF

/*
 * support for morestack
 */

// Called during function prolog when more stack is needed.
//
// The traceback routines see morestack on a g0 as being
// the top of a stack (for example, morestack calling newstack
// calling the scheduler calling newm calling gc), so we must
// record an argument size. For that purpose, it has no arguments.
TEXT runtime·morestack(SB),NOSPLIT|NOFRAME,$0-0
	// Cannot grow scheduler stack (m->g0).
	get_tls(CX)
	MOVQ	g(CX), DI     // DI = g
	MOVQ	g_m(DI), BX   // BX = m

	// Set g->sched to context in f.
	MOVQ	0(SP), AX // f's PC
	MOVQ	AX, (g_sched+gobuf_pc)(DI)
	LEAQ	8(SP), AX // f's SP
	MOVQ	AX, (g_sched+gobuf_sp)(DI)
	MOVQ	BP, (g_sched+gobuf_bp)(DI)
	MOVQ	DX, (g_sched+gobuf_ctxt)(DI)

	MOVQ	m_g0(BX), SI  // SI = m.g0
	CMPQ	DI, SI
	JNE	3(PC)
	CALL	runtime·badmorestackg0(SB)
	CALL	runtime·abort(SB)

	// Cannot grow signal stack (m->gsignal).
	MOVQ	m_gsignal(BX), SI
	CMPQ	DI, SI
	JNE	3(PC)
	CALL	runtime·badmorestackgsignal(SB)
	CALL	runtime·abort(SB)

	// Called from f.
	// Set m->morebuf to f's caller.
	NOP	SP	// tell vet SP changed - stop checking offsets
	MOVQ	8(SP), AX	// f's caller's PC
	MOVQ	AX, (m_morebuf+gobuf_pc)(BX)
	LEAQ	16(SP), AX	// f's caller's SP
	MOVQ	AX, (m_morebuf+gobuf_sp)(BX)
	MOVQ	DI, (m_morebuf+gobuf_g)(BX)

	// Call newstack on m->g0's stack.
	MOVQ	m_g0(BX), BX
	MOVQ	BX, g(CX)
	MOVQ	(g_sched+gobuf_sp)(BX), SP
	MOVQ	(g_sched+gobuf_bp)(BX), BP
	CALL	runtime·newstack(SB)
	CALL	runtime·abort(SB)	// crash if newstack returns
	RET

// morestack but not preserving ctxt.
TEXT runtime·morestack_noctxt(SB),NOSPLIT,$0
	MOVL	$0, DX
	JMP	runtime·morestack(SB)

// spillArgs stores return values from registers to a *internal/abi.RegArgs in R12.
TEXT ·spillArgs(SB),NOSPLIT,$0-0
	MOVQ AX, 0(R12)
	MOVQ BX, 8(R12)
	MOVQ CX, 16(R12)
	MOVQ DI, 24(R12)
	MOVQ SI, 32(R12)
	MOVQ R8, 40(R12)
	MOVQ R9, 48(R12)
	MOVQ R10, 56(R12)
	MOVQ R11, 64(R12)
	MOVQ X0, 72(R12)
	MOVQ X1, 80(R12)
	MOVQ X2, 88(R12)
	MOVQ X3, 96(R12)
	MOVQ X4, 104(R12)
	MOVQ X5, 112(R12)
	MOVQ X6, 120(R12)
	MOVQ X7, 128(R12)
	MOVQ X8, 136(R12)
	MOVQ X9, 144(R12)
	MOVQ X10, 152(R12)
	MOVQ X11, 160(R12)
	MOVQ X12, 168(R12)
	MOVQ X13, 176(R12)
	MOVQ X14, 184(R12)
	RET

// unspillArgs loads args into registers from a *internal/abi.RegArgs in R12.
TEXT ·unspillArgs(SB),NOSPLIT,$0-0
	MOVQ 0(R12), AX
	MOVQ 8(R12), BX
	MOVQ 16(R12), CX
	MOVQ 24(R12), DI
	MOVQ 32(R12), SI
	MOVQ 40(R12), R8
	MOVQ 48(R12), R9
	MOVQ 56(R12), R10
	MOVQ 64(R12), R11
	MOVQ 72(R12), X0
	MOVQ 80(R12), X1
	MOVQ 88(R12), X2
	MOVQ 96(R12), X3
	MOVQ 104(R12), X4
	MOVQ 112(R12), X5
	MOVQ 120(R12), X6
	MOVQ 128(R12), X7
	MOVQ 136(R12), X8
	MOVQ 144(R12), X9
	MOVQ 152(R12), X10
	MOVQ 160(R12), X11
	MOVQ 168(R12), X12
	MOVQ 176(R12), X13
	MOVQ 184(R12), X14
	RET

// reflectcall: call a function with the given argument list
// func call(stackArgsType *_type, f *FuncVal, stackArgs *byte, stackArgsSize, stackRetOffset, frameSize uint32, regArgs *abi.RegArgs).
// we don't have variable-sized frames, so we use a small number
// of constant-sized-frame functions to encode a few bits of size in the pc.
// Caution: ugly multiline assembly macros in your future!

#define DISPATCH(NAME,MAXSIZE)		\
	CMPQ	CX, $MAXSIZE;		\
	JA	3(PC);			\
	MOVQ	$NAME(SB), AX;		\
	JMP	AX
// Note: can't just "JMP NAME(SB)" - bad inlining results.

TEXT ·reflectcall(SB), NOSPLIT, $0-48
	MOVLQZX frameSize+32(FP), CX
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
	MOVQ	$runtime·badreflectcall(SB), AX
	JMP	AX

#define CALLFN(NAME,MAXSIZE)			\
TEXT NAME(SB), WRAPPER, $MAXSIZE-48;		\
	NO_LOCAL_POINTERS;			\
	/* copy arguments to stack */		\
	MOVQ	stackArgs+16(FP), SI;		\
	MOVLQZX stackArgsSize+24(FP), CX;		\
	MOVQ	SP, DI;				\
	REP;MOVSB;				\
	/* set up argument registers */		\
	MOVQ    regArgs+40(FP), R12;		\
	CALL    ·unspillArgs(SB);		\
	/* call function */			\
	MOVQ	f+8(FP), DX;			\
	PCDATA  $PCDATA_StackMapIndex, $0;	\
	MOVQ	(DX), R12;			\
	CALL	R12;				\
	/* copy register return values back */		\
	MOVQ    regArgs+40(FP), R12;		\
	CALL    ·spillArgs(SB);		\
	MOVLQZX	stackArgsSize+24(FP), CX;		\
	MOVLQZX	stackRetOffset+28(FP), BX;		\
	MOVQ	stackArgs+16(FP), DI;		\
	MOVQ	stackArgsType+0(FP), DX;		\
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
TEXT callRet<>(SB), NOSPLIT, $40-0
	NO_LOCAL_POINTERS
	MOVQ	DX, 0(SP)
	MOVQ	DI, 8(SP)
	MOVQ	SI, 16(SP)
	MOVQ	CX, 24(SP)
	MOVQ	R12, 32(SP)
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


TEXT ·publicationBarrier<ABIInternal>(SB),NOSPLIT,$0-0
	// Stores are already ordered on x86, so this is just a
	// compile barrier.
	RET

// Save state of caller into g->sched,
// but using fake PC from systemstack_switch.
// Must only be called from functions with frame pointer
// and without locals ($0) or else unwinding from
// systemstack_switch is incorrect.
// Smashes R9.
TEXT gosave_systemstack_switch<>(SB),NOSPLIT|NOFRAME,$0
	// Take systemstack_switch PC and add 8 bytes to skip
	// the prologue. The final location does not matter
	// as long as we are between the prologue and the epilogue.
	MOVQ	$runtime·systemstack_switch+8(SB), R9
	MOVQ	R9, (g_sched+gobuf_pc)(R14)
	LEAQ	8(SP), R9
	MOVQ	R9, (g_sched+gobuf_sp)(R14)
	MOVQ	BP, (g_sched+gobuf_bp)(R14)
	// Assert ctxt is zero. See func save.
	MOVQ	(g_sched+gobuf_ctxt)(R14), R9
	TESTQ	R9, R9
	JZ	2(PC)
	CALL	runtime·abort(SB)
	RET

// func asmcgocall_no_g(fn, arg unsafe.Pointer)
// Call fn(arg) aligned appropriately for the gcc ABI.
// Called on a system stack, and there may be no g yet (during needm).
TEXT ·asmcgocall_no_g(SB),NOSPLIT,$32-16
	MOVQ	fn+0(FP), AX
	MOVQ	arg+8(FP), BX
	MOVQ	SP, DX
	ANDQ	$~15, SP	// alignment
	MOVQ	DX, 8(SP)
	MOVQ	BX, DI		// DI = first argument in AMD64 ABI
	MOVQ	BX, CX		// CX = first argument in Win64
	CALL	AX
	MOVQ	8(SP), DX
	MOVQ	DX, SP
	RET

// asmcgocall_landingpad calls AX with BX as argument.
// Must be called on the system stack.
TEXT ·asmcgocall_landingpad(SB),NOSPLIT,$0-0
#ifdef GOOS_windows
	// Make sure we have enough room for 4 stack-backed fast-call
	// registers as per Windows amd64 calling convention.
	ADJSP	$32
	// On Windows, asmcgocall_landingpad acts as landing pad for exceptions
	// thrown in the cgo call. Exceptions that reach this function will be
	// handled by runtime.sehtramp thanks to the SEH metadata added
	// by the compiler.
	// Note that runtime.sehtramp can't be attached directly to asmcgocall
	// because its initial stack pointer can be outside the system stack bounds,
	// and Windows stops the stack unwinding without calling the exception handler
	// when it reaches that point.
	MOVQ	BX, CX		// CX = first argument in Win64
	CALL	AX
	// The exception handler is not called if the next instruction is part of
	// the epilogue, which includes the RET instruction, so we need to add a NOP here.
	BYTE	$0x90
	ADJSP	$-32
	RET
#endif
	// Tail call AX on non-Windows, as the extra stack frame is not needed.
	MOVQ	BX, DI		// DI = first argument in AMD64 ABI
	JMP	AX

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
	// come in on the m->g0 stack already. Or we might already
	// be on the m->gsignal stack.
	get_tls(CX)
	MOVQ	g(CX), DI
	CMPQ	DI, $0
	JEQ	nosave
	MOVQ	g_m(DI), R8
	MOVQ	m_gsignal(R8), SI
	CMPQ	DI, SI
	JEQ	nosave
	MOVQ	m_g0(R8), SI
	CMPQ	DI, SI
	JEQ	nosave

	// Switch to system stack.
	// The original frame pointer is stored in BP,
	// which is useful for stack unwinding.
	CALL	gosave_systemstack_switch<>(SB)
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), SP

	// Now on a scheduling stack (a pthread-created stack).
	SUBQ	$16, SP
	ANDQ	$~15, SP	// alignment for gcc ABI
	MOVQ	DI, 8(SP)	// save g
	MOVQ	(g_stack+stack_hi)(DI), DI
	SUBQ	DX, DI
	MOVQ	DI, 0(SP)	// save depth in stack (can't just save SP, as stack might be copied during a callback)
	CALL	runtime·asmcgocall_landingpad(SB)

	// Restore registers, g, stack pointer.
	get_tls(CX)
	MOVQ	8(SP), DI
	MOVQ	(g_stack+stack_hi)(DI), SI
	SUBQ	0(SP), SI
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
	SUBQ	$16, SP
	ANDQ	$~15, SP
	MOVQ	$0, 8(SP)		// where above code stores g, in case someone looks during debugging
	MOVQ	DX, 0(SP)	// save original stack pointer
	CALL	runtime·asmcgocall_landingpad(SB)
	MOVQ	0(SP), SI	// restore original stack pointer
	MOVQ	SI, SP
	MOVL	AX, ret+16(FP)
	RET

#ifdef GOOS_windows
// Dummy TLS that's used on Windows so that we don't crash trying
// to restore the G register in needm. needm and its callees are
// very careful never to actually use the G, the TLS just can't be
// unset since we're in Go code.
GLOBL zeroTLS<>(SB),RODATA,$const_tlsSize
#endif

// func cgocallback(fn, frame unsafe.Pointer, ctxt uintptr)
// See cgocall.go for more details.
TEXT ·cgocallback(SB),NOSPLIT,$24-24
	NO_LOCAL_POINTERS

	// Skip cgocallbackg, just dropm when fn is nil, and frame is the saved g.
	// It is used to dropm while thread is exiting.
	MOVQ	fn+0(FP), AX
	CMPQ	AX, $0
	JNE	loadg
	// Restore the g from frame.
	get_tls(CX)
	MOVQ	frame+8(FP), BX
	MOVQ	BX, g(CX)
	JMP	dropm

loadg:
	// If g is nil, Go did not create the current thread,
	// or if this thread never called into Go on pthread platforms.
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
	MOVQ	BX, savedm-8(SP)	// saved copy of oldm
	JMP	havem
needm:
#ifdef GOOS_windows
	// Set up a dummy TLS value. needm is careful not to use it,
	// but it needs to be there to prevent autogenerated code from
	// crashing when it loads from it.
	// We don't need to clear it or anything later because needm
	// will set up TLS properly.
	MOVQ	$zeroTLS<>(SB), DI
	CALL	runtime·settls(SB)
#endif
	// On some platforms (Windows) we cannot call needm through
	// an ABI wrapper because there's no TLS set up, and the ABI
	// wrapper will try to restore the G register (R14) from TLS.
	// Clear X15 because Go expects it and we're not calling
	// through a wrapper, but otherwise avoid setting the G
	// register in the wrapper and call needm directly. It
	// takes no arguments and doesn't return any values so
	// there's no need to handle that. Clear R14 so that there's
	// a bad value in there, in case needm tries to use it.
	XORPS	X15, X15
	XORQ    R14, R14
	MOVQ	$runtime·needAndBindM<ABIInternal>(SB), AX
	CALL	AX
	MOVQ	$0, savedm-8(SP)
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
	// To save m->curg->sched.pc, we push it onto the curg stack and
	// open a frame the same size as cgocallback's g0 frame.
	// Once we switch to the curg stack, the pushed PC will appear
	// to be the return PC of cgocallback, so that the traceback
	// will seamlessly trace back into the earlier calls.
	MOVQ	m_curg(BX), SI
	MOVQ	SI, g(CX)
	MOVQ	(g_sched+gobuf_sp)(SI), DI  // prepare stack as DI
	MOVQ	(g_sched+gobuf_pc)(SI), BX
	MOVQ	BX, -8(DI)  // "push" return PC on the g stack
	// Gather our arguments into registers.
	MOVQ	fn+0(FP), BX
	MOVQ	frame+8(FP), CX
	MOVQ	ctxt+16(FP), DX
	// Compute the size of the frame, including return PC and, if
	// GOEXPERIMENT=framepointer, the saved base pointer
	LEAQ	fn+0(FP), AX
	SUBQ	SP, AX   // AX is our actual frame size
	SUBQ	AX, DI   // Allocate the same frame size on the g stack
	MOVQ	DI, SP

	MOVQ	BX, 0(SP)
	MOVQ	CX, 8(SP)
	MOVQ	DX, 16(SP)
	MOVQ	$runtime·cgocallbackg(SB), AX
	CALL	AX	// indirect call to bypass nosplit check. We're on a different stack now.

	// Compute the size of the frame again. FP and SP have
	// completely different values here than they did above,
	// but only their difference matters.
	LEAQ	fn+0(FP), AX
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

	// If the m on entry was nil, we called needm above to borrow an m,
	// 1. for the duration of the call on non-pthread platforms,
	// 2. or the duration of the C thread alive on pthread platforms.
	// If the m on entry wasn't nil,
	// 1. the thread might be a Go thread,
	// 2. or it wasn't the first call from a C thread on pthread platforms,
	//    since then we skip dropm to reuse the m in the first call.
	MOVQ	savedm-8(SP), BX
	CMPQ	BX, $0
	JNE	done

	// Skip dropm to reuse it in the next call, when a pthread key has been created.
	MOVQ	_cgo_pthread_key_created(SB), AX
	// It means cgo is disabled when _cgo_pthread_key_created is a nil pointer, need dropm.
	CMPQ	AX, $0
	JEQ	dropm
	CMPQ	(AX), $0
	JNE	done

dropm:
	MOVQ	$runtime·dropm(SB), AX
	CALL	AX
#ifdef GOOS_windows
	// We need to clear the TLS pointer in case the next
	// thread that comes into Go tries to reuse that space
	// but uses the same M.
	XORQ	DI, DI
	CALL	runtime·settls(SB)
#endif
done:

	// Done!
	RET

// func setg(gg *g)
// set g. for use by needm.
TEXT runtime·setg(SB), NOSPLIT, $0-8
	MOVQ	gg+0(FP), BX
	get_tls(CX)
	MOVQ	BX, g(CX)
	RET

// void setg_gcc(G*); set g called from gcc.
TEXT setg_gcc<>(SB),NOSPLIT,$0
	get_tls(AX)
	MOVQ	DI, g(AX)
	MOVQ	DI, R14 // set the g register
	RET

TEXT runtime·abort(SB),NOSPLIT,$0-0
	INT	$3
loop:
	JMP	loop

// check that SP is in range [g->stack.lo, g->stack.hi)
TEXT runtime·stackcheck(SB), NOSPLIT|NOFRAME, $0-0
	get_tls(CX)
	MOVQ	g(CX), AX
	CMPQ	(g_stack+stack_hi)(AX), SP
	JHI	2(PC)
	CALL	runtime·abort(SB)
	CMPQ	SP, (g_stack+stack_lo)(AX)
	JHI	2(PC)
	CALL	runtime·abort(SB)
	RET

// func cputicks() int64
TEXT runtime·cputicks(SB),NOSPLIT,$0-0
	CMPB	internal∕cpu·X86+const_offsetX86HasRDTSCP(SB), $1
	JNE	fences
	// Instruction stream serializing RDTSCP is supported.
	// RDTSCP is supported by Intel Nehalem (2008) and
	// AMD K8 Rev. F (2006) and newer.
	RDTSCP
done:
	SHLQ	$32, DX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET
fences:
	// MFENCE is instruction stream serializing and flushes the
	// store buffers on AMD. The serialization semantics of LFENCE on AMD
	// are dependent on MSR C001_1029 and CPU generation.
	// LFENCE on Intel does wait for all previous instructions to have executed.
	// Intel recommends MFENCE;LFENCE in its manuals before RDTSC to have all
	// previous instructions executed and all previous loads and stores to globally visible.
	// Using MFENCE;LFENCE here aligns the serializing properties without
	// runtime detection of CPU manufacturer.
	MFENCE
	LFENCE
	RDTSC
	JMP done

// func memhash(p unsafe.Pointer, h, s uintptr) uintptr
// hash function using AES hardware instructions
TEXT runtime·memhash<ABIInternal>(SB),NOSPLIT,$0-32
	// AX = ptr to data
	// BX = seed
	// CX = size
	CMPB	runtime·useAeshash(SB), $0
	JEQ	noaes
	JMP	aeshashbody<>(SB)
noaes:
	JMP	runtime·memhashFallback<ABIInternal>(SB)

// func strhash(p unsafe.Pointer, h uintptr) uintptr
TEXT runtime·strhash<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to string struct
	// BX = seed
	CMPB	runtime·useAeshash(SB), $0
	JEQ	noaes
	MOVQ	8(AX), CX	// length of string
	MOVQ	(AX), AX	// string data
	JMP	aeshashbody<>(SB)
noaes:
	JMP	runtime·strhashFallback<ABIInternal>(SB)

// AX: data
// BX: hash seed
// CX: length
// At return: AX = return value
TEXT aeshashbody<>(SB),NOSPLIT,$0-0
	// Fill an SSE register with our seeds.
	MOVQ	BX, X0				// 64 bits of per-table hash seed
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
	MOVQ	X1, AX	// return X1
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
	MOVQ	X0, AX	// return X0
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
	MOVQ	X2, AX	// return X2
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
	MOVQ	X4, AX	// return X4
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
	// X15 must be zero on return
	PXOR	X15, X15
	MOVQ	X8, AX	// return X8
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

	PCALIGN $16
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
	// X15 must be zero on return
	PXOR	X15, X15
	MOVQ	X8, AX	// return X8
	RET

// func memhash32(p unsafe.Pointer, h uintptr) uintptr
// ABIInternal for performance.
TEXT runtime·memhash32<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to data
	// BX = seed
	CMPB	runtime·useAeshash(SB), $0
	JEQ	noaes
	MOVQ	BX, X0	// X0 = seed
	PINSRD	$2, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+32(SB), X0
	MOVQ	X0, AX	// return X0
	RET
noaes:
	JMP	runtime·memhash32Fallback<ABIInternal>(SB)

// func memhash64(p unsafe.Pointer, h uintptr) uintptr
// ABIInternal for performance.
TEXT runtime·memhash64<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to data
	// BX = seed
	CMPB	runtime·useAeshash(SB), $0
	JEQ	noaes
	MOVQ	BX, X0	// X0 = seed
	PINSRQ	$1, (AX), X0	// data
	AESENC	runtime·aeskeysched+0(SB), X0
	AESENC	runtime·aeskeysched+16(SB), X0
	AESENC	runtime·aeskeysched+32(SB), X0
	MOVQ	X0, AX	// return X0
	RET
noaes:
	JMP	runtime·memhash64Fallback<ABIInternal>(SB)

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

// func checkASM() bool
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
TEXT runtime·goexit(SB),NOSPLIT|TOPFRAME|NOFRAME,$0-0
	BYTE	$0x90	// NOP
	CALL	runtime·goexit1(SB)	// does not return
	// traceback from goexit1 must hit code range of goexit
	BYTE	$0x90	// NOP

// This is called from .init_array and follows the platform, not Go, ABI.
TEXT runtime·addmoduledata(SB),NOSPLIT,$0-0
	PUSHQ	R15 // The access to global variables below implicitly uses R15, which is callee-save
	MOVQ	runtime·lastmoduledatap(SB), AX
	MOVQ	DI, moduledata_next(AX)
	MOVQ	DI, runtime·lastmoduledatap(SB)
	POPQ	R15
	RET

// Initialize special registers then jump to sigpanic.
// This function is injected from the signal handler for panicking
// signals. It is quite painful to set X15 in the signal context,
// so we do it here.
TEXT ·sigpanic0(SB),NOSPLIT,$0-0
	get_tls(R14)
	MOVQ	g(R14), R14
	XORPS	X15, X15
	JMP	·sigpanic<ABIInternal>(SB)

// gcWriteBarrier informs the GC about heap pointer writes.
//
// gcWriteBarrier returns space in a write barrier buffer which
// should be filled in by the caller.
// gcWriteBarrier does NOT follow the Go ABI. It accepts the
// number of bytes of buffer needed in R11, and returns a pointer
// to the buffer space in R11.
// It clobbers FLAGS. It does not clobber any general-purpose registers,
// but may clobber others (e.g., SSE registers).
// Typical use would be, when doing *(CX+88) = AX
//     CMPL    $0, runtime.writeBarrier(SB)
//     JEQ     dowrite
//     CALL    runtime.gcBatchBarrier2(SB)
//     MOVQ    AX, (R11)
//     MOVQ    88(CX), DX
//     MOVQ    DX, 8(R11)
// dowrite:
//     MOVQ    AX, 88(CX)
TEXT gcWriteBarrier<>(SB),NOSPLIT,$112
	// Save the registers clobbered by the fast path. This is slightly
	// faster than having the caller spill these.
	MOVQ	R12, 96(SP)
	MOVQ	R13, 104(SP)
retry:
	// TODO: Consider passing g.m.p in as an argument so they can be shared
	// across a sequence of write barriers.
	MOVQ	g_m(R14), R13
	MOVQ	m_p(R13), R13
	// Get current buffer write position.
	MOVQ	(p_wbBuf+wbBuf_next)(R13), R12	// original next position
	ADDQ	R11, R12			// new next position
	// Is the buffer full?
	CMPQ	R12, (p_wbBuf+wbBuf_end)(R13)
	JA	flush
	// Commit to the larger buffer.
	MOVQ	R12, (p_wbBuf+wbBuf_next)(R13)
	// Make return value (the original next position)
	SUBQ	R11, R12
	MOVQ	R12, R11
	// Restore registers.
	MOVQ	96(SP), R12
	MOVQ	104(SP), R13
	RET

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
	MOVQ	DI, 0(SP)
	MOVQ	AX, 8(SP)
	MOVQ	BX, 16(SP)
	MOVQ	CX, 24(SP)
	MOVQ	DX, 32(SP)
	// DI already saved
	MOVQ	SI, 40(SP)
	MOVQ	BP, 48(SP)
	MOVQ	R8, 56(SP)
	MOVQ	R9, 64(SP)
	MOVQ	R10, 72(SP)
	MOVQ	R11, 80(SP)
	// R12 already saved
	// R13 already saved
	// R14 is g
	MOVQ	R15, 88(SP)

	CALL	runtime·wbBufFlush(SB)

	MOVQ	0(SP), DI
	MOVQ	8(SP), AX
	MOVQ	16(SP), BX
	MOVQ	24(SP), CX
	MOVQ	32(SP), DX
	MOVQ	40(SP), SI
	MOVQ	48(SP), BP
	MOVQ	56(SP), R8
	MOVQ	64(SP), R9
	MOVQ	72(SP), R10
	MOVQ	80(SP), R11
	MOVQ	88(SP), R15
	JMP	retry

TEXT runtime·gcWriteBarrier1<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $8, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier2<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $16, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier3<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $24, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier4<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $32, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier5<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $40, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier6<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $48, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier7<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $56, R11
	JMP     gcWriteBarrier<>(SB)
TEXT runtime·gcWriteBarrier8<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVL   $64, R11
	JMP     gcWriteBarrier<>(SB)

DATA	debugCallFrameTooLarge<>+0x00(SB)/20, $"call frame too large"
GLOBL	debugCallFrameTooLarge<>(SB), RODATA, $20	// Size duplicated below

// debugCallV2 is the entry point for debugger-injected function
// calls on running goroutines. It informs the runtime that a
// debug call has been injected and creates a call frame for the
// debugger to fill in.
//
// To inject a function call, a debugger should:
// 1. Check that the goroutine is in state _Grunning and that
//    there are at least 256 bytes free on the stack.
// 2. Push the current PC on the stack (updating SP).
// 3. Write the desired argument frame size at SP-16 (using the SP
//    after step 2).
// 4. Save all machine registers (including flags and XMM registers)
//    so they can be restored later by the debugger.
// 5. Set the PC to debugCallV2 and resume execution.
//
// If the goroutine is in state _Grunnable, then it's not generally
// safe to inject a call because it may return out via other runtime
// operations. Instead, the debugger should unwind the stack to find
// the return to non-runtime code, add a temporary breakpoint there,
// and inject the call once that breakpoint is hit.
//
// If the goroutine is in any other state, it's not safe to inject a call.
//
// This function communicates back to the debugger by setting R12 and
// invoking INT3 to raise a breakpoint signal. See the comments in the
// implementation for the protocol the debugger is expected to
// follow. InjectDebugCall in the runtime tests demonstrates this protocol.
//
// The debugger must ensure that any pointers passed to the function
// obey escape analysis requirements. Specifically, it must not pass
// a stack pointer to an escaping argument. debugCallV2 cannot check
// this invariant.
//
// This is ABIInternal because Go code injects its PC directly into new
// goroutine stacks.
TEXT runtime·debugCallV2<ABIInternal>(SB),NOSPLIT,$152-0
	// Save all registers that may contain pointers so they can be
	// conservatively scanned.
	//
	// We can't do anything that might clobber any of these
	// registers before this.
	MOVQ	R15, r15-(14*8+8)(SP)
	MOVQ	R14, r14-(13*8+8)(SP)
	MOVQ	R13, r13-(12*8+8)(SP)
	MOVQ	R12, r12-(11*8+8)(SP)
	MOVQ	R11, r11-(10*8+8)(SP)
	MOVQ	R10, r10-(9*8+8)(SP)
	MOVQ	R9, r9-(8*8+8)(SP)
	MOVQ	R8, r8-(7*8+8)(SP)
	MOVQ	DI, di-(6*8+8)(SP)
	MOVQ	SI, si-(5*8+8)(SP)
	MOVQ	BP, bp-(4*8+8)(SP)
	MOVQ	BX, bx-(3*8+8)(SP)
	MOVQ	DX, dx-(2*8+8)(SP)
	// Save the frame size before we clobber it. Either of the last
	// saves could clobber this depending on whether there's a saved BP.
	MOVQ	frameSize-24(FP), DX	// aka -16(RSP) before prologue
	MOVQ	CX, cx-(1*8+8)(SP)
	MOVQ	AX, ax-(0*8+8)(SP)

	// Save the argument frame size.
	MOVQ	DX, frameSize-128(SP)

	// Perform a safe-point check.
	MOVQ	retpc-8(FP), AX	// Caller's PC
	MOVQ	AX, 0(SP)
	CALL	runtime·debugCallCheck(SB)
	MOVQ	8(SP), AX
	TESTQ	AX, AX
	JZ	good
	// The safety check failed. Put the reason string at the top
	// of the stack.
	MOVQ	AX, 0(SP)
	MOVQ	16(SP), AX
	MOVQ	AX, 8(SP)
	// Set R12 to 8 and invoke INT3. The debugger should get the
	// reason a call can't be injected from the top of the stack
	// and resume execution.
	MOVQ	$8, R12
	BYTE	$0xcc
	JMP	restore

good:
	// Registers are saved and it's safe to make a call.
	// Open up a call frame, moving the stack if necessary.
	//
	// Once the frame is allocated, this will set R12 to 0 and
	// invoke INT3. The debugger should write the argument
	// frame for the call at SP, set up argument registers, push
	// the trapping PC on the stack, set the PC to the function to
	// call, set RDX to point to the closure (if a closure call),
	// and resume execution.
	//
	// If the function returns, this will set R12 to 1 and invoke
	// INT3. The debugger can then inspect any return value saved
	// on the stack at SP and in registers and resume execution again.
	//
	// If the function panics, this will set R12 to 2 and invoke INT3.
	// The interface{} value of the panic will be at SP. The debugger
	// can inspect the panic value and resume execution again.
#define DEBUG_CALL_DISPATCH(NAME,MAXSIZE)	\
	CMPQ	AX, $MAXSIZE;			\
	JA	5(PC);				\
	MOVQ	$NAME(SB), AX;			\
	MOVQ	AX, 0(SP);			\
	CALL	runtime·debugCallWrap(SB);	\
	JMP	restore

	MOVQ	frameSize-128(SP), AX
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
	MOVQ	$debugCallFrameTooLarge<>(SB), AX
	MOVQ	AX, 0(SP)
	MOVQ	$20, 8(SP) // length of debugCallFrameTooLarge string
	MOVQ	$8, R12
	BYTE	$0xcc
	JMP	restore

restore:
	// Calls and failures resume here.
	//
	// Set R12 to 16 and invoke INT3. The debugger should restore
	// all registers except RIP and RSP and resume execution.
	MOVQ	$16, R12
	BYTE	$0xcc
	// We must not modify flags after this point.

	// Restore pointer-containing registers, which may have been
	// modified from the debugger's copy by stack copying.
	MOVQ	ax-(0*8+8)(SP), AX
	MOVQ	cx-(1*8+8)(SP), CX
	MOVQ	dx-(2*8+8)(SP), DX
	MOVQ	bx-(3*8+8)(SP), BX
	MOVQ	bp-(4*8+8)(SP), BP
	MOVQ	si-(5*8+8)(SP), SI
	MOVQ	di-(6*8+8)(SP), DI
	MOVQ	r8-(7*8+8)(SP), R8
	MOVQ	r9-(8*8+8)(SP), R9
	MOVQ	r10-(9*8+8)(SP), R10
	MOVQ	r11-(10*8+8)(SP), R11
	MOVQ	r12-(11*8+8)(SP), R12
	MOVQ	r13-(12*8+8)(SP), R13
	MOVQ	r14-(13*8+8)(SP), R14
	MOVQ	r15-(14*8+8)(SP), R15

	RET

// runtime.debugCallCheck assumes that functions defined with the
// DEBUG_CALL_FN macro are safe points to inject calls.
#define DEBUG_CALL_FN(NAME,MAXSIZE)		\
TEXT NAME(SB),WRAPPER,$MAXSIZE-0;		\
	NO_LOCAL_POINTERS;			\
	MOVQ	$0, R12;				\
	BYTE	$0xcc;				\
	MOVQ	$1, R12;				\
	BYTE	$0xcc;				\
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
	// Copy the panic value to the top of stack.
	MOVQ	val_type+0(FP), AX
	MOVQ	AX, 0(SP)
	MOVQ	val_data+8(FP), AX
	MOVQ	AX, 8(SP)
	MOVQ	$2, R12
	BYTE	$0xcc
	RET

// Note: these functions use a special calling convention to save generated code space.
// Arguments are passed in registers, but the space for those arguments are allocated
// in the caller's stack frame. These stubs write the args into that stack space and
// then tail call to the corresponding runtime handler.
// The tail call makes these stubs disappear in backtraces.
// Defined as ABIInternal since they do not use the stack-based Go ABI.
TEXT runtime·panicIndex<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, BX
	JMP	runtime·goPanicIndex<ABIInternal>(SB)
TEXT runtime·panicIndexU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, BX
	JMP	runtime·goPanicIndexU<ABIInternal>(SB)
TEXT runtime·panicSliceAlen<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, AX
	MOVQ	DX, BX
	JMP	runtime·goPanicSliceAlen<ABIInternal>(SB)
TEXT runtime·panicSliceAlenU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, AX
	MOVQ	DX, BX
	JMP	runtime·goPanicSliceAlenU<ABIInternal>(SB)
TEXT runtime·panicSliceAcap<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, AX
	MOVQ	DX, BX
	JMP	runtime·goPanicSliceAcap<ABIInternal>(SB)
TEXT runtime·panicSliceAcapU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, AX
	MOVQ	DX, BX
	JMP	runtime·goPanicSliceAcapU<ABIInternal>(SB)
TEXT runtime·panicSliceB<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, BX
	JMP	runtime·goPanicSliceB<ABIInternal>(SB)
TEXT runtime·panicSliceBU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, BX
	JMP	runtime·goPanicSliceBU<ABIInternal>(SB)
TEXT runtime·panicSlice3Alen<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	DX, AX
	JMP	runtime·goPanicSlice3Alen<ABIInternal>(SB)
TEXT runtime·panicSlice3AlenU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	DX, AX
	JMP	runtime·goPanicSlice3AlenU<ABIInternal>(SB)
TEXT runtime·panicSlice3Acap<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	DX, AX
	JMP	runtime·goPanicSlice3Acap<ABIInternal>(SB)
TEXT runtime·panicSlice3AcapU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	DX, AX
	JMP	runtime·goPanicSlice3AcapU<ABIInternal>(SB)
TEXT runtime·panicSlice3B<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, AX
	MOVQ	DX, BX
	JMP	runtime·goPanicSlice3B<ABIInternal>(SB)
TEXT runtime·panicSlice3BU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, AX
	MOVQ	DX, BX
	JMP	runtime·goPanicSlice3BU<ABIInternal>(SB)
TEXT runtime·panicSlice3C<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, BX
	JMP	runtime·goPanicSlice3C<ABIInternal>(SB)
TEXT runtime·panicSlice3CU<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	CX, BX
	JMP	runtime·goPanicSlice3CU<ABIInternal>(SB)
TEXT runtime·panicSliceConvert<ABIInternal>(SB),NOSPLIT,$0-16
	MOVQ	DX, AX
	JMP	runtime·goPanicSliceConvert<ABIInternal>(SB)

#ifdef GOOS_android
// Use the free TLS_SLOT_APP slot #2 on Android Q.
// Earlier androids are set up in gcc_android.c.
DATA runtime·tls_g+0(SB)/8, $16
GLOBL runtime·tls_g+0(SB), NOPTR, $8
#endif
#ifdef GOOS_windows
GLOBL runtime·tls_g+0(SB), NOPTR, $8
#endif

// The compiler and assembler's -spectre=ret mode rewrites
// all indirect CALL AX / JMP AX instructions to be
// CALL retpolineAX / JMP retpolineAX.
// See https://support.google.com/faqs/answer/7625886.
#define RETPOLINE(reg) \
	/*   CALL setup */     BYTE $0xE8; BYTE $(2+2); BYTE $0; BYTE $0; BYTE $0;	\
	/* nospec: */									\
	/*   PAUSE */           BYTE $0xF3; BYTE $0x90;					\
	/*   JMP nospec */      BYTE $0xEB; BYTE $-(2+2);				\
	/* setup: */									\
	/*   MOVQ AX, 0(SP) */  BYTE $0x48|((reg&8)>>1); BYTE $0x89;			\
	                        BYTE $0x04|((reg&7)<<3); BYTE $0x24;			\
	/*   RET */             BYTE $0xC3

TEXT runtime·retpolineAX(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(0)
TEXT runtime·retpolineCX(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(1)
TEXT runtime·retpolineDX(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(2)
TEXT runtime·retpolineBX(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(3)
/* SP is 4, can't happen / magic encodings */
TEXT runtime·retpolineBP(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(5)
TEXT runtime·retpolineSI(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(6)
TEXT runtime·retpolineDI(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(7)
TEXT runtime·retpolineR8(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(8)
TEXT runtime·retpolineR9(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(9)
TEXT runtime·retpolineR10(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(10)
TEXT runtime·retpolineR11(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(11)
TEXT runtime·retpolineR12(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(12)
TEXT runtime·retpolineR13(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(13)
TEXT runtime·retpolineR14(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(14)
TEXT runtime·retpolineR15(SB),NOSPLIT|NOFRAME,$0; RETPOLINE(15)

TEXT ·getfp<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	MOVQ BP, AX
	RET
