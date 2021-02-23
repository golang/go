// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for arm64, OpenBSD
// System calls are implemented in libc/libpthread, this file
// contains trampolines that convert from Go to C calling convention.
// Some direct system call implementations currently remain.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME	$0
#define	CLOCK_MONOTONIC	$3

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$160
	// R0 points to the m.
	// We are already on m's g0 stack.

	// Save callee-save registers.
	MOVD	R19, 8(RSP)
	MOVD	R20, 16(RSP)
	MOVD	R21, 24(RSP)
	MOVD	R22, 32(RSP)
	MOVD	R23, 40(RSP)
	MOVD	R24, 48(RSP)
	MOVD	R25, 56(RSP)
	MOVD	R26, 64(RSP)
	MOVD	R27, 72(RSP)
	MOVD	g, 80(RSP)
	MOVD	R29, 88(RSP)
	FMOVD	F8, 96(RSP)
	FMOVD	F9, 104(RSP)
	FMOVD	F10, 112(RSP)
	FMOVD	F11, 120(RSP)
	FMOVD	F12, 128(RSP)
	FMOVD	F13, 136(RSP)
	FMOVD	F14, 144(RSP)
	FMOVD	F15, 152(RSP)

	MOVD    m_g0(R0), g
	BL	runtime·save_g(SB)

	BL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOVD	8(RSP), R19
	MOVD	16(RSP), R20
	MOVD	24(RSP), R21
	MOVD	32(RSP), R22
	MOVD	40(RSP), R23
	MOVD	48(RSP), R24
	MOVD	56(RSP), R25
	MOVD	64(RSP), R26
	MOVD	72(RSP), R27
	MOVD	80(RSP), g
	MOVD	88(RSP), R29
	FMOVD	96(RSP), F8
	FMOVD	104(RSP), F9
	FMOVD	112(RSP), F10
	FMOVD	120(RSP), F11
	FMOVD	128(RSP), F12
	FMOVD	136(RSP), F13
	FMOVD	144(RSP), F14
	FMOVD	152(RSP), F15

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	MOVD	$0, R0

	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)			// Alignment for ELF ABI?
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$192
	// Save callee-save registers in the case of signal forwarding.
	// Please refer to https://golang.org/issue/31827 .
	MOVD	R19, 8*4(RSP)
	MOVD	R20, 8*5(RSP)
	MOVD	R21, 8*6(RSP)
	MOVD	R22, 8*7(RSP)
	MOVD	R23, 8*8(RSP)
	MOVD	R24, 8*9(RSP)
	MOVD	R25, 8*10(RSP)
	MOVD	R26, 8*11(RSP)
	MOVD	R27, 8*12(RSP)
	MOVD	g, 8*13(RSP)
	MOVD	R29, 8*14(RSP)
	FMOVD	F8, 8*15(RSP)
	FMOVD	F9, 8*16(RSP)
	FMOVD	F10, 8*17(RSP)
	FMOVD	F11, 8*18(RSP)
	FMOVD	F12, 8*19(RSP)
	FMOVD	F13, 8*20(RSP)
	FMOVD	F14, 8*21(RSP)
	FMOVD	F15, 8*22(RSP)

	// If called from an external code context, g will not be set.
	// Save R0, since runtime·load_g will clobber it.
	MOVW	R0, 8(RSP)		// signum
	BL	runtime·load_g(SB)

	MOVD	R1, 16(RSP)
	MOVD	R2, 24(RSP)
	BL	runtime·sigtrampgo(SB)

	// Restore callee-save registers.
	MOVD	8*4(RSP), R19
	MOVD	8*5(RSP), R20
	MOVD	8*6(RSP), R21
	MOVD	8*7(RSP), R22
	MOVD	8*8(RSP), R23
	MOVD	8*9(RSP), R24
	MOVD	8*10(RSP), R25
	MOVD	8*11(RSP), R26
	MOVD	8*12(RSP), R27
	MOVD	8*13(RSP), g
	MOVD	8*14(RSP), R29
	FMOVD	8*15(RSP), F8
	FMOVD	8*16(RSP), F9
	FMOVD	8*17(RSP), F10
	FMOVD	8*18(RSP), F11
	FMOVD	8*19(RSP), F12
	FMOVD	8*20(RSP), F13
	FMOVD	8*21(RSP), F14
	FMOVD	8*22(RSP), F15

	RET

//
// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in R0.
// A single int32 result is returned in R0.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0		// arg 1 - attr
	CALL	libc_pthread_attr_init(SB)
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0		// arg 1 - attr
	CALL	libc_pthread_attr_destroy(SB)
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - size
	MOVD	0(R0), R0		// arg 1 - attr
	CALL	libc_pthread_attr_getstacksize(SB)
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - state
	MOVD	0(R0), R0		// arg 1 - attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R1		// arg 2 - attr
	MOVD	8(R0), R2		// arg 3 - start
	MOVD	16(R0), R3		// arg 4 - arg
	SUB	$16, RSP
	MOVD	RSP, R0			// arg 1 - &threadid (discard)
	CALL	libc_pthread_create(SB)
	ADD	$16, RSP
	RET

TEXT runtime·thrkill_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1		// arg 2 - signal
	MOVD	$0, R2			// arg 3 - tcb
	MOVW	0(R0), R0		// arg 1 - tid
	CALL	libc_thrkill(SB)
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1		// arg 2 - clock_id
	MOVD	16(R0), R2		// arg 3 - abstime
	MOVD	24(R0), R3		// arg 4 - lock
	MOVD	32(R0), R4		// arg 5 - abort
	MOVD	0(R0), R0		// arg 1 - id
	CALL	libc_thrsleep(SB)
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1		// arg 2 - count
	MOVD	0(R0), R0		// arg 1 - id
	CALL	libc_thrwakeup(SB)
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0		// arg 1 - status
	CALL	libc_exit(SB)
	MOVD	$0, R0			// crash on failure
	MOVD	R0, (R0)
	RET

TEXT runtime·getthrid_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19			// pointer to args
	CALL	libc_getthrid(SB)
	MOVW	R0, 0(R19)		// return value
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19			// pointer to args
	CALL	libc_getpid(SB)		// arg 1 - pid
	MOVW	0(R19), R1		// arg 2 - signal
	CALL	libc_kill(SB)
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$0
	CALL	libc_sched_yield(SB)
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args
	MOVD	0(R19), R0		// arg 1 - addr
	MOVD	8(R19), R1		// arg 2 - len
	MOVW	16(R19), R2		// arg 3 - prot
	MOVW	20(R19), R3		// arg 4 - flags
	MOVW	24(R19), R4		// arg 5 - fid
	MOVW	28(R19), R5		// arg 6 - offset
	CALL	libc_mmap(SB)
	MOVD	$0, R1
	CMP	$-1, R0
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R0), R1		// errno
	MOVD	$0, R0
noerr:
	MOVD	R0, 32(R19)
	MOVD	R1, 40(R19)
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - len
	MOVD	0(R0), R0		// arg 1 - addr
	CALL	libc_munmap(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVD	$0, R0			// crash on failure
	MOVD	R0, (R0)
	RET

TEXT runtime·madvise_trampoline(SB), NOSPLIT, $0
	MOVD	8(R0), R1		// arg 2 - len
	MOVW	16(R0), R2		// arg 3 - advice
	MOVD	0(R0), R0		// arg 1 - addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1		// arg 2 - flags
	MOVW	12(R0), R2		// arg 3 - mode
	MOVD	0(R0), R0		// arg 1 - path
	MOVD	$0, R3			// varargs
	CALL	libc_open(SB)
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0		// arg 1 - fd
	CALL	libc_close(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - buf
	MOVW	16(R0), R2		// arg 3 - count
	MOVW	0(R0), R0		// arg 1 - fd
	CALL	libc_read(SB)
	CMP	$-1, R0
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	NEG	R0, R0			// caller expects negative errno value
noerr:
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - buf
	MOVW	16(R0), R2		// arg 3 - count
	MOVW	0(R0), R0		// arg 1 - fd
	CALL	libc_write(SB)
	CMP	$-1, R0
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	NEG	R0, R0			// caller expects negative errno value
noerr:
	RET

TEXT runtime·pipe2_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1		// arg 2 - flags
	MOVD	0(R0), R0		// arg 1 - filedes
	CALL	libc_pipe2(SB)
	CMP	$-1, R0
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	NEG	R0, R0			// caller expects negative errno value
noerr:
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - new
	MOVD	16(R0), R2		// arg 3 - old
	MOVW	0(R0), R0		// arg 1 - which
	CALL	libc_setitimer(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0		// arg 1 - usec
	CALL	libc_usleep(SB)
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1		// arg 2 - miblen
	MOVD	16(R0), R2		// arg 3 - out
	MOVD	24(R0), R3		// arg 4 - size
	MOVD	32(R0), R4		// arg 5 - dst
	MOVD	40(R0), R5		// arg 6 - ndst
	MOVD	0(R0), R0		// arg 1 - mib
	CALL	libc_sysctl(SB)
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	CALL	libc_kqueue(SB)
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - keventt
	MOVW	16(R0), R2		// arg 3 - nch
	MOVD	24(R0), R3		// arg 4 - ev
	MOVW	32(R0), R4		// arg 5 - nev
	MOVD	40(R0), R5		// arg 6 - ts
	MOVW	0(R0), R0		// arg 1 - kq
	CALL	libc_kevent(SB)
	CMP	$-1, R0
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	NEG	R0, R0			// caller expects negative errno value
noerr:
	RET

TEXT runtime·clock_gettime_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - tp
	MOVD	0(R0), R0		// arg 1 - clock_id
	CALL	libc_clock_gettime(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVD	$0, R0			// crash on failure
	MOVD	R0, (R0)
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1		// arg 2 - cmd
	MOVW	8(R0), R2		// arg 3 - arg
	MOVW	0(R0), R0		// arg 1 - fd
	MOVD	$0, R3			// vararg
	CALL	libc_fcntl(SB)
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - new
	MOVD	16(R0), R2		// arg 3 - old
	MOVW	0(R0), R0		// arg 1 - sig
	CALL	libc_sigaction(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVD	$0, R0			// crash on syscall failure
	MOVD	R0, (R0)
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - new
	MOVD	16(R0), R2		// arg 3 - old
	MOVW	0(R0), R0		// arg 1 - how
	CALL	libc_pthread_sigmask(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVD	$0, R0			// crash on syscall failure
	MOVD	R0, (R0)
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - old
	MOVD	0(R0), R0		// arg 1 - new
	CALL	libc_sigaltstack(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVD	$0, R0			// crash on syscall failure
	MOVD	R0, (R0)
	RET

// syscall calls a function in libc on behalf of the syscall package.
// syscall takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args

	MOVD	(0*8)(R19), R11		// fn
	MOVD	(1*8)(R19), R0		// a1
	MOVD	(2*8)(R19), R1		// a2
	MOVD	(3*8)(R19), R2		// a3
	MOVD	$0, R3			// vararg

	CALL	R11

	MOVD	R0, (4*8)(R19)		// r1
	MOVD	R1, (5*8)(R19)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPW	$-1, R0
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R0), R0
	MOVD	R0, (6*8)(R19)		// err

ok:
	RET

// syscallX calls a function in libc on behalf of the syscall package.
// syscallX takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscallX must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscallX is like syscall but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscallX(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args

	MOVD	(0*8)(R19), R11		// fn
	MOVD	(1*8)(R19), R0		// a1
	MOVD	(2*8)(R19), R1		// a2
	MOVD	(3*8)(R19), R2		// a3
	MOVD	$0, R3			// vararg

	CALL	R11

	MOVD	R0, (4*8)(R19)		// r1
	MOVD	R1, (5*8)(R19)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMP	$-1, R0
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R0), R0
	MOVD	R0, (6*8)(R19)		// err

ok:
	RET

// syscall6 calls a function in libc on behalf of the syscall package.
// syscall6 takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall6 must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall6 expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall6(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args

	MOVD	(0*8)(R19), R11		// fn
	MOVD	(1*8)(R19), R0		// a1
	MOVD	(2*8)(R19), R1		// a2
	MOVD	(3*8)(R19), R2		// a3
	MOVD	(4*8)(R19), R3		// a4
	MOVD	(5*8)(R19), R4		// a5
	MOVD	(6*8)(R19), R5		// a6
	MOVD	$0, R6			// vararg

	CALL	R11

	MOVD	R0, (7*8)(R19)		// r1
	MOVD	R1, (8*8)(R19)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPW	$-1, R0
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R0), R0
	MOVD	R0, (9*8)(R19)		// err

ok:
	RET

// syscall6X calls a function in libc on behalf of the syscall package.
// syscall6X takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall6X must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall6X is like syscall6 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall6X(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args

	MOVD	(0*8)(R19), R11		// fn
	MOVD	(1*8)(R19), R0		// a1
	MOVD	(2*8)(R19), R1		// a2
	MOVD	(3*8)(R19), R2		// a3
	MOVD	(4*8)(R19), R3		// a4
	MOVD	(5*8)(R19), R4		// a5
	MOVD	(6*8)(R19), R5		// a6
	MOVD	$0, R6			// vararg

	CALL	R11

	MOVD	R0, (7*8)(R19)		// r1
	MOVD	R1, (8*8)(R19)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMP	$-1, R0
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R0), R0
	MOVD	R0, (9*8)(R19)		// err

ok:
	RET

// syscall10 calls a function in libc on behalf of the syscall package.
// syscall10 takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	a7    uintptr
//	a8    uintptr
//	a9    uintptr
//	a10   uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall10 must be called on the g0 stack with the
// C calling convention (use libcCall).
TEXT runtime·syscall10(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args

	MOVD	(0*8)(R19), R11		// fn
	MOVD	(1*8)(R19), R0		// a1
	MOVD	(2*8)(R19), R1		// a2
	MOVD	(3*8)(R19), R2		// a3
	MOVD	(4*8)(R19), R3		// a4
	MOVD	(5*8)(R19), R4		// a5
	MOVD	(6*8)(R19), R5		// a6
	MOVD	(7*8)(R19), R6		// a7
	MOVD	(8*8)(R19), R7		// a8
	MOVD	(9*8)(R19), R8		// a9
	MOVD	(10*8)(R19), R9		// a10
	MOVD	$0, R10			// vararg

	CALL	R11

	MOVD	R0, (11*8)(R19)		// r1
	MOVD	R1, (12*8)(R19)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPW	$-1, R0
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R0), R0
	MOVD	R0, (13*8)(R19)		// err

ok:
	RET

// syscall10X calls a function in libc on behalf of the syscall package.
// syscall10X takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	a7    uintptr
//	a8    uintptr
//	a9    uintptr
//	a10   uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall10X must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall10X is like syscall10 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall10X(SB),NOSPLIT,$0
	MOVD    R0, R19			// pointer to args

	MOVD	(0*8)(R19), R11		// fn
	MOVD	(1*8)(R19), R0		// a1
	MOVD	(2*8)(R19), R1		// a2
	MOVD	(3*8)(R19), R2		// a3
	MOVD	(4*8)(R19), R3		// a4
	MOVD	(5*8)(R19), R4		// a5
	MOVD	(6*8)(R19), R5		// a6
	MOVD	(7*8)(R19), R6		// a7
	MOVD	(8*8)(R19), R7		// a8
	MOVD	(9*8)(R19), R8		// a9
	MOVD	(10*8)(R19), R9		// a10
	MOVD	$0, R10			// vararg

	CALL	R11

	MOVD	R0, (11*8)(R19)		// r1
	MOVD	R1, (12*8)(R19)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMP	$-1, R0
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R0), R0
	MOVD	R0, (13*8)(R19)		// err

ok:
	RET
