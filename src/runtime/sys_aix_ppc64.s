// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for ppc64, Aix
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "asm_ppc64x.h"

// This function calls a C function with the function descriptor in R12
TEXT callCfunction<>(SB),	NOSPLIT|NOFRAME,$0
	MOVD	0(R12), R12
	MOVD	R2, 40(R1)
	MOVD	0(R12), R0
	MOVD	8(R12), R2
	MOVD	R0, CTR
	BR	(CTR)


// asmsyscall6 calls a library function with a function descriptor
// stored in libcall_fn and store the results in libcall structure
// Up to 6 arguments can be passed to this C function
// Called by runtime.asmcgocall
// It reserves a stack of 288 bytes for the C function. It must
// follow AIX convention, thus the first local variable must
// be stored at the offset 112, after the linker area (48 bytes)
// and the argument area (64).
// The AIX convention is described here:
// https://www.ibm.com/docs/en/aix/7.2?topic=overview-runtime-process-stack
// NOT USING GO CALLING CONVENTION
// runtime.asmsyscall6 is a function descriptor to the real asmsyscall6.
DATA	runtime·asmsyscall6+0(SB)/8, $asmsyscall6<>(SB)
DATA	runtime·asmsyscall6+8(SB)/8, $TOC(SB)
DATA	runtime·asmsyscall6+16(SB)/8, $0
GLOBL	runtime·asmsyscall6(SB), NOPTR, $24

TEXT asmsyscall6<>(SB),NOSPLIT,$256
	// Save libcall for later
	MOVD	R3, 112(R1)
	MOVD	libcall_fn(R3), R12
	MOVD	libcall_args(R3), R9
	MOVD	0(R9), R3
	MOVD	8(R9), R4
	MOVD	16(R9), R5
	MOVD	24(R9), R6
	MOVD	32(R9), R7
	MOVD	40(R9), R8
	BL	callCfunction<>(SB)

	// Restore R0 and TOC
	XOR	R0, R0
	MOVD	40(R1), R2

	// Store result in libcall
	MOVD	112(R1), R5
	MOVD	R3, (libcall_r1)(R5)
	MOVD	$-1, R6
	CMP	R6, R3
	BNE	skiperrno

	// Save errno in libcall
	BL	runtime·load_g(SB)
	MOVD	g_m(g), R4
	MOVD	(m_mOS + mOS_perrno)(R4), R9
	MOVW	0(R9), R9
	MOVD	R9, (libcall_err)(R5)
	RET
skiperrno:
	// Reset errno if no error has been returned
	MOVD	R0, (libcall_err)(R5)
	RET


TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R3
	MOVD	info+16(FP), R4
	MOVD	ctx+24(FP), R5
	MOVD	fn+0(FP), R12
	// fn is a function descriptor
	// R2 must be saved on restore
	MOVD	0(R12), R0
	MOVD	R2, 40(R1)
	MOVD	8(R12), R2
	MOVD	R0, CTR
	BL	(CTR)
	MOVD	40(R1), R2
	BL	runtime·reginit(SB)
	RET


// runtime.sigtramp is a function descriptor to the real sigtramp.
DATA	runtime·sigtramp+0(SB)/8, $sigtramp<>(SB)
DATA	runtime·sigtramp+8(SB)/8, $TOC(SB)
DATA	runtime·sigtramp+16(SB)/8, $0
GLOBL	runtime·sigtramp(SB), NOPTR, $24

// This function must not have any frame as we want to control how
// every registers are used.
// TODO(aix): Implement SetCgoTraceback handler.
TEXT sigtramp<>(SB),NOSPLIT|NOFRAME|TOPFRAME,$0
	MOVD	LR, R0
	MOVD	R0, 16(R1)
	// initialize essential registers (just in case)
	BL	runtime·reginit(SB)

	// Note that we are executing on altsigstack here, so we have
	// more stack available than NOSPLIT would have us believe.
	// To defeat the linker, we make our own stack frame with
	// more space.
	SUB	$144+FIXED_FRAME, R1

	// Save registers
	MOVD	R31, 56(R1)
	MOVD	g, 64(R1)
	MOVD	R29, 72(R1)
	MOVD	R14, 80(R1)
	MOVD	R15, 88(R1)

	BL	runtime·load_g(SB)

	CMP	$0, g
	BEQ	sigtramp // g == nil
	MOVD	g_m(g), R6
	CMP	$0, R6
	BEQ	sigtramp	// g.m == nil

	// Save m->libcall. We need to do this because we
	// might get interrupted by a signal in runtime·asmcgocall.
	MOVD	(m_libcall+libcall_fn)(R6), R7
	MOVD	R7, 96(R1)
	MOVD	(m_libcall+libcall_args)(R6), R7
	MOVD	R7, 104(R1)
	MOVD	(m_libcall+libcall_n)(R6), R7
	MOVD	R7, 112(R1)
	MOVD	(m_libcall+libcall_r1)(R6), R7
	MOVD	R7, 120(R1)
	MOVD	(m_libcall+libcall_r2)(R6), R7
	MOVD	R7, 128(R1)

	// save errno, it might be EINTR; stuff we do here might reset it.
	MOVD	(m_mOS+mOS_perrno)(R6), R8
	MOVD	0(R8), R8
	MOVD	R8, 136(R1)

sigtramp:
	MOVW	R3, FIXED_FRAME+0(R1)
	MOVD	R4, FIXED_FRAME+8(R1)
	MOVD	R5, FIXED_FRAME+16(R1)
	MOVD	$runtime·sigtrampgo(SB), R12
	MOVD	R12, CTR
	BL	(CTR)

	CMP	$0, g
	BEQ	exit // g == nil
	MOVD	g_m(g), R6
	CMP	$0, R6
	BEQ	exit	// g.m == nil

	// restore libcall
	MOVD	96(R1), R7
	MOVD	R7, (m_libcall+libcall_fn)(R6)
	MOVD	104(R1), R7
	MOVD	R7, (m_libcall+libcall_args)(R6)
	MOVD	112(R1), R7
	MOVD	R7, (m_libcall+libcall_n)(R6)
	MOVD	120(R1), R7
	MOVD	R7, (m_libcall+libcall_r1)(R6)
	MOVD	128(R1), R7
	MOVD	R7, (m_libcall+libcall_r2)(R6)

	// restore errno
	MOVD	(m_mOS+mOS_perrno)(R6), R7
	MOVD	136(R1), R8
	MOVD	R8, 0(R7)

exit:
	// restore registers
	MOVD	56(R1),R31
	MOVD	64(R1),g
	MOVD	72(R1),R29
	MOVD	80(R1), R14
	MOVD	88(R1), R15

	// Don't use RET because we need to restore R31 !
	ADD $144+FIXED_FRAME, R1
	MOVD	16(R1), R0
	MOVD	R0, LR
	BR (LR)

// runtime.tstart is a function descriptor to the real tstart.
DATA	runtime·tstart+0(SB)/8, $tstart<>(SB)
DATA	runtime·tstart+8(SB)/8, $TOC(SB)
DATA	runtime·tstart+16(SB)/8, $0
GLOBL	runtime·tstart(SB), NOPTR, $24

TEXT tstart<>(SB),NOSPLIT,$0
	XOR	 R0, R0 // reset R0

	// set g
	MOVD	m_g0(R3), g
	BL	runtime·save_g(SB)
	MOVD	R3, g_m(g)

	// Layout new m scheduler stack on os stack.
	MOVD	R1, R3
	MOVD	R3, (g_stack+stack_hi)(g)
	SUB	$(const_threadStackSize), R3		// stack size
	MOVD	R3, (g_stack+stack_lo)(g)
	ADD	$const_stackGuard, R3
	MOVD	R3, g_stackguard0(g)
	MOVD	R3, g_stackguard1(g)

	BL	runtime·mstart(SB)

	MOVD R0, R3
	RET


#define CSYSCALL()			\
	MOVD	0(R12), R12		\
	MOVD	R2, 40(R1)		\
	MOVD	0(R12), R0		\
	MOVD	8(R12), R2		\
	MOVD	R0, CTR			\
	BL	(CTR)			\
	MOVD	40(R1), R2		\
	BL runtime·reginit(SB)


// Runs on OS stack, called from runtime·osyield.
TEXT runtime·osyield1(SB),NOSPLIT,$0
	MOVD	$libc_sched_yield(SB), R12
	CSYSCALL()
	RET


// Runs on OS stack, called from runtime·sigprocmask.
TEXT runtime·sigprocmask1(SB),NOSPLIT,$0-24
	MOVD	how+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVD	$libpthread_sigthreadmask(SB), R12
	CSYSCALL()
	RET

// Runs on OS stack, called from runtime·usleep.
TEXT runtime·usleep1(SB),NOSPLIT,$0-4
	MOVW	us+0(FP), R3
	MOVD	$libc_usleep(SB), R12
	CSYSCALL()
	RET

// Runs on OS stack, called from runtime·exit.
TEXT runtime·exit1(SB),NOSPLIT,$0-4
	MOVW	code+0(FP), R3
	MOVD	$libc_exit(SB), R12
	CSYSCALL()
	RET

// Runs on OS stack, called from runtime·write1.
TEXT runtime·write2(SB),NOSPLIT,$0-28
	MOVD	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	MOVD	$libc_write(SB), R12
	CSYSCALL()
	MOVW	R3, ret+24(FP)
	RET

// Runs on OS stack, called from runtime·pthread_attr_init.
TEXT runtime·pthread_attr_init1(SB),NOSPLIT,$0-12
	MOVD	attr+0(FP), R3
	MOVD	$libpthread_attr_init(SB), R12
	CSYSCALL()
	MOVW	R3, ret+8(FP)
	RET

// Runs on OS stack, called from runtime·pthread_attr_setstacksize.
TEXT runtime·pthread_attr_setstacksize1(SB),NOSPLIT,$0-20
	MOVD	attr+0(FP), R3
	MOVD	size+8(FP), R4
	MOVD	$libpthread_attr_setstacksize(SB), R12
	CSYSCALL()
	MOVW	R3, ret+16(FP)
	RET

// Runs on OS stack, called from runtime·pthread_setdetachstate.
TEXT runtime·pthread_attr_setdetachstate1(SB),NOSPLIT,$0-20
	MOVD	attr+0(FP), R3
	MOVW	state+8(FP), R4
	MOVD	$libpthread_attr_setdetachstate(SB), R12
	CSYSCALL()
	MOVW	R3, ret+16(FP)
	RET

// Runs on OS stack, called from runtime·pthread_create.
TEXT runtime·pthread_create1(SB),NOSPLIT,$0-36
	MOVD	tid+0(FP), R3
	MOVD	attr+8(FP), R4
	MOVD	fn+16(FP), R5
	MOVD	arg+24(FP), R6
	MOVD	$libpthread_create(SB), R12
	CSYSCALL()
	MOVW	R3, ret+32(FP)
	RET

// Runs on OS stack, called from runtime·sigaction.
TEXT runtime·sigaction1(SB),NOSPLIT,$0-24
	MOVD	sig+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVD	$libc_sigaction(SB), R12
	CSYSCALL()
	RET
