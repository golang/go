// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, SunOS
// /usr/include/sys/syscall.h for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// This is needed by asm_amd64.s
TEXT runtime·settls(SB),NOSPLIT,$8
	RET

// void libc_miniterrno(void *(*___errno)(void));
//
// Set the TLS errno pointer in M.
//
// Called using runtime·asmcgocall from os_solaris.c:/minit.
// NOT USING GO CALLING CONVENTION.
TEXT runtime·miniterrno(SB),NOSPLIT,$0
	// asmcgocall will put first argument into DI.
	CALL	DI	// SysV ABI so returns in AX
	get_tls(CX)
	MOVQ	g(CX), BX
	MOVQ	g_m(BX), BX
	MOVQ	AX,	(m_mOS+mOS_perrno)(BX)
	RET

// Call a library function with SysV calling conventions.
// The called function can take a maximum of 6 INTEGER class arguments,
// see
//   Michael Matz, Jan Hubicka, Andreas Jaeger, and Mark Mitchell
//   System V Application Binary Interface
//   AMD64 Architecture Processor Supplement
// section 3.2.3.
//
// Called by runtime·asmcgocall or runtime·cgocall.
// NOT USING GO CALLING CONVENTION.
TEXT runtime·asmsysvicall6(SB),NOSPLIT,$0
	// asmcgocall will put first argument into DI.
	PUSHQ	DI			// save for later
	MOVQ	libcall_fn(DI), AX
	MOVQ	libcall_args(DI), R11
	MOVQ	libcall_n(DI), R10

	get_tls(CX)
	MOVQ	g(CX), BX
	CMPQ	BX, $0
	JEQ	skiperrno1
	MOVQ	g_m(BX), BX
	MOVQ	(m_mOS+mOS_perrno)(BX), DX
	CMPQ	DX, $0
	JEQ	skiperrno1
	MOVL	$0, 0(DX)

skiperrno1:
	CMPQ	R11, $0
	JEQ	skipargs
	// Load 6 args into correspondent registers.
	MOVQ	0(R11), DI
	MOVQ	8(R11), SI
	MOVQ	16(R11), DX
	MOVQ	24(R11), CX
	MOVQ	32(R11), R8
	MOVQ	40(R11), R9
skipargs:

	// Call SysV function
	CALL	AX

	// Return result
	POPQ	DI
	MOVQ	AX, libcall_r1(DI)
	MOVQ	DX, libcall_r2(DI)

	get_tls(CX)
	MOVQ	g(CX), BX
	CMPQ	BX, $0
	JEQ	skiperrno2
	MOVQ	g_m(BX), BX
	MOVQ	(m_mOS+mOS_perrno)(BX), AX
	CMPQ	AX, $0
	JEQ	skiperrno2
	MOVL	0(AX), AX
	MOVQ	AX, libcall_err(DI)

skiperrno2:
	RET

// uint32 tstart_sysvicall(M *newm);
TEXT runtime·tstart_sysvicall(SB),NOSPLIT,$0
	// DI contains first arg newm
	MOVQ	m_g0(DI), DX		// g

	// Make TLS entries point at g and m.
	get_tls(BX)
	MOVQ	DX, g(BX)
	MOVQ	DI, g_m(DX)

	// Layout new m scheduler stack on os stack.
	MOVQ	SP, AX
	MOVQ	AX, (g_stack+stack_hi)(DX)
	SUBQ	$(0x100000), AX		// stack size
	MOVQ	AX, (g_stack+stack_lo)(DX)
	ADDQ	$const_stackGuard, AX
	MOVQ	AX, g_stackguard0(DX)
	MOVQ	AX, g_stackguard1(DX)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·stackcheck(SB)	// clobbers AX,CX
	CALL	runtime·mstart(SB)

	XORL	AX, AX			// return 0 == success
	MOVL	AX, ret+8(FP)
	RET

// Careful, this is called by __sighndlr, a libc function. We must preserve
// registers as per AMD 64 ABI.
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME|NOFRAME,$0
	// Note that we are executing on altsigstack here, so we have
	// more stack available than NOSPLIT would have us believe.
	// To defeat the linker, we make our own stack frame with
	// more space:
	SUBQ    $168, SP
	// save registers
	MOVQ    BX, 24(SP)
	MOVQ    BP, 32(SP)
	MOVQ	R12, 40(SP)
	MOVQ	R13, 48(SP)
	MOVQ	R14, 56(SP)
	MOVQ	R15, 64(SP)

	get_tls(BX)
	// check that g exists
	MOVQ	g(BX), R10
	CMPQ	R10, $0
	JNE	allgood
	MOVQ	SI, 72(SP)
	MOVQ	DX, 80(SP)
	LEAQ	72(SP), AX
	MOVQ	DI, 0(SP)
	MOVQ	AX, 8(SP)
	MOVQ	$runtime·badsignal(SB), AX
	CALL	AX
	JMP	exit

allgood:
	// Save m->libcall and m->scratch. We need to do this because we
	// might get interrupted by a signal in runtime·asmcgocall.

	// save m->libcall
	MOVQ	g_m(R10), BP
	LEAQ	m_libcall(BP), R11
	MOVQ	libcall_fn(R11), R10
	MOVQ	R10, 72(SP)
	MOVQ	libcall_args(R11), R10
	MOVQ	R10, 80(SP)
	MOVQ	libcall_n(R11), R10
	MOVQ	R10, 88(SP)
	MOVQ    libcall_r1(R11), R10
	MOVQ    R10, 152(SP)
	MOVQ    libcall_r2(R11), R10
	MOVQ    R10, 160(SP)

	// save m->scratch
	LEAQ	(m_mOS+mOS_scratch)(BP), R11
	MOVQ	0(R11), R10
	MOVQ	R10, 96(SP)
	MOVQ	8(R11), R10
	MOVQ	R10, 104(SP)
	MOVQ	16(R11), R10
	MOVQ	R10, 112(SP)
	MOVQ	24(R11), R10
	MOVQ	R10, 120(SP)
	MOVQ	32(R11), R10
	MOVQ	R10, 128(SP)
	MOVQ	40(R11), R10
	MOVQ	R10, 136(SP)

	// save errno, it might be EINTR; stuff we do here might reset it.
	MOVQ	(m_mOS+mOS_perrno)(BP), R10
	MOVL	0(R10), R10
	MOVQ	R10, 144(SP)

	// prepare call
	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	MOVQ	DX, 16(SP)
	CALL	runtime·sigtrampgo(SB)

	get_tls(BX)
	MOVQ	g(BX), BP
	MOVQ	g_m(BP), BP
	// restore libcall
	LEAQ	m_libcall(BP), R11
	MOVQ	72(SP), R10
	MOVQ	R10, libcall_fn(R11)
	MOVQ	80(SP), R10
	MOVQ	R10, libcall_args(R11)
	MOVQ	88(SP), R10
	MOVQ	R10, libcall_n(R11)
	MOVQ    152(SP), R10
	MOVQ    R10, libcall_r1(R11)
	MOVQ    160(SP), R10
	MOVQ    R10, libcall_r2(R11)

	// restore scratch
	LEAQ	(m_mOS+mOS_scratch)(BP), R11
	MOVQ	96(SP), R10
	MOVQ	R10, 0(R11)
	MOVQ	104(SP), R10
	MOVQ	R10, 8(R11)
	MOVQ	112(SP), R10
	MOVQ	R10, 16(R11)
	MOVQ	120(SP), R10
	MOVQ	R10, 24(R11)
	MOVQ	128(SP), R10
	MOVQ	R10, 32(R11)
	MOVQ	136(SP), R10
	MOVQ	R10, 40(R11)

	// restore errno
	MOVQ	(m_mOS+mOS_perrno)(BP), R11
	MOVQ	144(SP), R10
	MOVL	R10, 0(R11)

exit:
	// restore registers
	MOVQ	24(SP), BX
	MOVQ	32(SP), BP
	MOVQ	40(SP), R12
	MOVQ	48(SP), R13
	MOVQ	56(SP), R14
	MOVQ	64(SP), R15
	ADDQ    $168, SP
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVQ	fn+0(FP),    AX
	MOVL	sig+8(FP),   DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP),  DX
	MOVQ	SP, BX		// callee-saved
	ANDQ	$~15, SP	// alignment for x86_64 ABI
	CALL	AX
	MOVQ	BX, SP
	RET

// Called from runtime·usleep (Go). Can be called on Go stack, on OS stack,
// can also be called in cgo callback path without a g->m.
TEXT runtime·usleep1(SB),NOSPLIT,$0
	MOVL	usec+0(FP), DI
	MOVQ	$usleep2<>(SB), AX // to hide from 6l

	// Execute call on m->g0.
	get_tls(R15)
	CMPQ	R15, $0
	JE	noswitch

	MOVQ	g(R15), R13
	CMPQ	R13, $0
	JE	noswitch
	MOVQ	g_m(R13), R13
	CMPQ	R13, $0
	JE	noswitch
	// TODO(aram): do something about the cpu profiler here.

	MOVQ	m_g0(R13), R14
	CMPQ	g(R15), R14
	JNE	switch
	// executing on m->g0 already
	CALL	AX
	RET

switch:
	// Switch to m->g0 stack and back.
	MOVQ	(g_sched+gobuf_sp)(R14), R14
	MOVQ	SP, -8(R14)
	LEAQ	-8(R14), SP
	CALL	AX
	MOVQ	0(SP), SP
	RET

noswitch:
	// Not a Go-managed thread. Do not switch stack.
	CALL	AX
	RET

// Runs on OS stack. duration (in µs units) is in DI.
TEXT usleep2<>(SB),NOSPLIT,$0
	LEAQ	libc_usleep(SB), AX
	CALL	AX
	RET

// Runs on OS stack, called from runtime·osyield.
TEXT runtime·osyield1(SB),NOSPLIT,$0
	LEAQ	libc_sched_yield(SB), AX
	CALL	AX
	RET
