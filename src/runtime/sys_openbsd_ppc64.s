// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for ppc64, OpenBSD
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
TEXT runtime·mstart_stub(SB),NOSPLIT,$32
	// R3 points to the m.
	// We are already on m's g0 stack.

	// Go relies on R0 being $0.
	XOR	R0, R0

	// TODO(jsing): Save callee-save registers (R14-R31, F14-F31, V20-V31).

	MOVD    m_g0(R3), g
	BL	runtime·save_g(SB)

	BL	runtime·mstart(SB)

	// TODO(jsing): Restore callee-save registers (R14-R31, F14-F31, V20-V31).

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	MOVD	$0, R3

	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R3
	MOVD	info+16(FP), R4
	MOVD	ctx+24(FP), R5
	MOVD	fn+0(FP), R12
	MOVD	R12, CTR
	CALL	(CTR)			// Alignment for ELF ABI?
	RET

TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$16
	// Go relies on R0 being $0 and we may have been executing non-Go code.
	XOR	R0, R0

	// TODO(jsing): Save callee-save registers (R2, R14-R31, F14-F31).
	// in the case of signal forwarding.
	// Please refer to https://golang.org/issue/31827 .

	// If called from an external code context, g will not be set.
	BL	runtime·load_g(SB)

	BL	runtime·sigtrampgo<ABIInternal>(SB)

	// TODO(jsing): Restore callee-save registers.

	RET

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in R3.
// A single int32 result is returned in R3.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$32
	MOVD	0(R3), R3		// arg 1 - attr
	CALL	libc_pthread_attr_init(SB)
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$32
	MOVD	0(R3), R3		// arg 1 - attr
	CALL	libc_pthread_attr_destroy(SB)
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - size
	MOVD	0(R3), R3		// arg 1 - attr
	CALL	libc_pthread_attr_getstacksize(SB)
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - state
	MOVD	0(R3), R3		// arg 1 - attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$32
	MOVD	0(R3), R4		// arg 2 - attr
	MOVD	8(R3), R5		// arg 3 - start
	MOVD	16(R3), R6		// arg 4 - arg

	MOVD	R1, R15
	SUB	$64, R1
	RLDCR	$0, R1, $~15, R1
	MOVD	R1, R3			// arg 1 - &threadid (discard)
	CALL	libc_pthread_create(SB)
	MOVD	R15, R1

	RET

TEXT runtime·thrkill_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - signal (int64)
	MOVD	$0, R5			// arg 3 - tcb
	MOVW	0(R3), R3		// arg 1 - tid
	CALL	libc_thrkill(SB)
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$32
	MOVW	8(R3), R4		// arg 2 - clock_id
	MOVD	16(R3), R5		// arg 3 - abstime
	MOVD	24(R3), R6		// arg 4 - lock
	MOVD	32(R3), R7		// arg 5 - abort
	MOVD	0(R3), R3		// arg 1 - id
	CALL	libc_thrsleep(SB)
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$32
	MOVW	8(R3), R4		// arg 2 - count
	MOVD	0(R3), R3		// arg 1 - id
	CALL	libc_thrwakeup(SB)
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT,$32
	MOVW	0(R3), R3		// arg 1 - status
	CALL	libc_exit(SB)
	MOVD	$0, R3			// crash on failure
	MOVD	R3, (R3)
	RET

TEXT runtime·getthrid_trampoline(SB),NOSPLIT,$32
	MOVD	R3, R14			// pointer to args
	CALL	libc_getthrid(SB)
	MOVW	R3, 0(R14)		// return value
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$32
	MOVD	R3, R14			// pointer to args
	CALL	libc_getpid(SB)		// arg 1 - pid
	MOVW	0(R14), R4		// arg 2 - signal
	CALL	libc_kill(SB)
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$32
	CALL	libc_sched_yield(SB)
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args
	MOVD	0(R14), R3		// arg 1 - addr
	MOVD	8(R14), R4		// arg 2 - len
	MOVW	16(R14), R5		// arg 3 - prot
	MOVW	20(R14), R6		// arg 4 - flags
	MOVW	24(R14), R7		// arg 5 - fid
	MOVW	28(R14), R8		// arg 6 - offset
	CALL	libc_mmap(SB)
	MOVD	$0, R4
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R4		// errno
	MOVD	$0, R3
noerr:
	MOVD	R3, 32(R14)
	MOVD	R4, 40(R14)
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - len
	MOVD	0(R3), R3		// arg 1 - addr
	CALL	libc_munmap(SB)
	CMP	R3, $-1
	BNE	3(PC)
	MOVD	$0, R3			// crash on failure
	MOVD	R3, (R3)
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - len
	MOVW	16(R3), R5		// arg 3 - advice
	MOVD	0(R3), R3		// arg 1 - addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$32
	MOVW	8(R3), R4		// arg 2 - flags
	MOVW	12(R3), R5		// arg 3 - mode
	MOVD	0(R3), R3		// arg 1 - path
	MOVD	$0, R6			// varargs
	CALL	libc_open(SB)
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$32
	MOVW	0(R3), R3		// arg 1 - fd
	CALL	libc_close(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - buf
	MOVW	16(R3), R5		// arg 3 - count
	MOVW	0(R3), R3		// arg 1 - fd (int32)
	CALL	libc_read(SB)
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R3		// errno
	NEG	R3, R3			// caller expects negative errno value
noerr:
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - buf
	MOVW	16(R3), R5		// arg 3 - count
	MOVD	0(R3), R3		// arg 1 - fd (uintptr)
	CALL	libc_write(SB)
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R3		// errno
	NEG	R3, R3			// caller expects negative errno value
noerr:
	RET

TEXT runtime·pipe2_trampoline(SB),NOSPLIT,$32
	MOVW	8(R3), R4		// arg 2 - flags
	MOVD	0(R3), R3		// arg 1 - filedes
	CALL	libc_pipe2(SB)
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R3		// errno
	NEG	R3, R3			// caller expects negative errno value
noerr:
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - new
	MOVD	16(R3), R5		// arg 3 - old
	MOVW	0(R3), R3		// arg 1 - which
	CALL	libc_setitimer(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$32
	MOVW	0(R3), R3		// arg 1 - usec
	CALL	libc_usleep(SB)
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$32
	MOVW	8(R3), R4		// arg 2 - miblen
	MOVD	16(R3), R5		// arg 3 - out
	MOVD	24(R3), R6		// arg 4 - size
	MOVD	32(R3), R7		// arg 5 - dst
	MOVD	40(R3), R8		// arg 6 - ndst
	MOVD	0(R3), R3		// arg 1 - mib
	CALL	libc_sysctl(SB)
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$32
	CALL	libc_kqueue(SB)
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - keventt
	MOVW	16(R3), R5		// arg 3 - nch
	MOVD	24(R3), R6		// arg 4 - ev
	MOVW	32(R3), R7		// arg 5 - nev
	MOVD	40(R3), R8		// arg 6 - ts
	MOVW	0(R3), R3		// arg 1 - kq
	CALL	libc_kevent(SB)
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R3		// errno
	NEG	R3, R3			// caller expects negative errno value
noerr:
	RET

TEXT runtime·clock_gettime_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - tp
	MOVW	0(R3), R3		// arg 1 - clock_id
	CALL	libc_clock_gettime(SB)
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R3		// errno
	NEG	R3, R3			// caller expects negative errno value
noerr:
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args
	MOVW	0(R14), R3		// arg 1 - fd
	MOVW	4(R14), R4		// arg 2 - cmd
	MOVW	8(R14), R5		// arg 3 - arg
	MOVD	$0, R6			// vararg
	CALL	libc_fcntl(SB)
	MOVD	$0, R4
	CMP	R3, $-1
	BNE	noerr
	CALL	libc_errno(SB)
	MOVW	(R3), R4		// errno
	MOVW	$-1, R3
noerr:
	MOVW	R3, 12(R14)
	MOVW	R4, 16(R14)
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - new
	MOVD	16(R3), R5		// arg 3 - old
	MOVW	0(R3), R3		// arg 1 - sig
	CALL	libc_sigaction(SB)
	CMP	R3, $-1
	BNE	3(PC)
	MOVD	$0, R3			// crash on syscall failure
	MOVD	R3, (R3)
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - new
	MOVD	16(R3), R5		// arg 3 - old
	MOVW	0(R3), R3		// arg 1 - how
	CALL	libc_pthread_sigmask(SB)
	CMP	R3, $-1
	BNE	3(PC)
	MOVD	$0, R3			// crash on syscall failure
	MOVD	R3, (R3)
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$32
	MOVD	8(R3), R4		// arg 2 - old
	MOVD	0(R3), R3		// arg 1 - new
	CALL	libc_sigaltstack(SB)
	CMP	R3, $-1
	BNE	3(PC)
	MOVD	$0, R3			// crash on syscall failure
	MOVD	R3, (R3)
	RET

TEXT runtime·issetugid_trampoline(SB),NOSPLIT,$32
	MOVD	R3, R14			// pointer to args
	CALL	libc_getthrid(SB)
	MOVW	R3, 0(R14)		// return value
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
TEXT runtime·syscall(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args

	MOVD	(0*8)(R14), R12		// fn
	MOVD	(1*8)(R14), R3		// a1
	MOVD	(2*8)(R14), R4		// a2
	MOVD	(3*8)(R14), R5		// a3
	MOVD	$0, R6			// vararg

	MOVD	R12, CTR
	CALL	(CTR)

	MOVD	R3, (4*8)(R14)		// r1
	MOVD	R4, (5*8)(R14)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPW	R3, $-1
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R3), R3
	MOVD	R3, (6*8)(R14)		// err

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
TEXT runtime·syscallX(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args

	MOVD	(0*8)(R14), R12		// fn
	MOVD	(1*8)(R14), R3		// a1
	MOVD	(2*8)(R14), R4		// a2
	MOVD	(3*8)(R14), R5		// a3
	MOVD	$0, R6			// vararg

	MOVD	R12, CTR
	CALL	(CTR)

	MOVD	R3, (4*8)(R14)		// r1
	MOVD	R4, (5*8)(R14)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMP	R3, $-1
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R3), R3
	MOVD	R3, (6*8)(R14)		// err

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
TEXT runtime·syscall6(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args

	MOVD	(0*8)(R14), R12		// fn
	MOVD	(1*8)(R14), R3		// a1
	MOVD	(2*8)(R14), R4		// a2
	MOVD	(3*8)(R14), R5		// a3
	MOVD	(4*8)(R14), R6		// a4
	MOVD	(5*8)(R14), R7		// a5
	MOVD	(6*8)(R14), R8		// a6
	MOVD	$0, R9			// vararg

	MOVD	R12, CTR
	CALL	(CTR)

	MOVD	R3, (7*8)(R14)		// r1
	MOVD	R4, (8*8)(R14)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPW	R3, $-1
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R3), R3
	MOVD	R3, (9*8)(R14)		// err

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
TEXT runtime·syscall6X(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args

	MOVD	(0*8)(R14), R12		// fn
	MOVD	(1*8)(R14), R3		// a1
	MOVD	(2*8)(R14), R4		// a2
	MOVD	(3*8)(R14), R5		// a3
	MOVD	(4*8)(R14), R6		// a4
	MOVD	(5*8)(R14), R7		// a5
	MOVD	(6*8)(R14), R8		// a6
	MOVD	$0, R9			// vararg

	MOVD	R12, CTR
	CALL	(CTR)

	MOVD	R3, (7*8)(R14)		// r1
	MOVD	R4, (8*8)(R14)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMP	R3, $-1
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R3), R3
	MOVD	R3, (9*8)(R14)		// err

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
// C calling convention (use libcCall). Note that this is
// really syscall8 as a maximum of eight parameters can be
// passed via registers (and current usage does not exceed
// this).
TEXT runtime·syscall10(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args

	MOVD	(0*8)(R14), R12		// fn
	MOVD	(1*8)(R14), R3		// a1
	MOVD	(2*8)(R14), R4		// a2
	MOVD	(3*8)(R14), R5		// a3
	MOVD	(4*8)(R14), R6		// a4
	MOVD	(5*8)(R14), R7		// a5
	MOVD	(6*8)(R14), R8		// a6
	MOVD	(7*8)(R14), R9		// a7
	MOVD	(8*8)(R14), R10		// a8

	MOVD	R12, CTR
	CALL	(CTR)

	MOVD	R3, (11*8)(R14)		// r1
	MOVD	R4, (12*8)(R14)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPW	R3, $-1
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R3), R3
	MOVD	R3, (13*8)(R14)		// err

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
// C calling convention (use libcCall). Note that this is
// really syscall8X as a maximum of eight parameters can be
// passed via registers (and current usage does not exceed
// this).
//
// syscall10X is like syscall10 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall10X(SB),NOSPLIT,$32
	MOVD    R3, R14			// pointer to args

	MOVD	(0*8)(R14), R12		// fn
	MOVD	(1*8)(R14), R3		// a1
	MOVD	(2*8)(R14), R4		// a2
	MOVD	(3*8)(R14), R5		// a3
	MOVD	(4*8)(R14), R6		// a4
	MOVD	(5*8)(R14), R7		// a5
	MOVD	(6*8)(R14), R8		// a6
	MOVD	(7*8)(R14), R9		// a7
	MOVD	(8*8)(R14), R10		// a8

	MOVD	R12, CTR
	CALL	(CTR)

	MOVD	R3, (11*8)(R14)		// r1
	MOVD	R4, (12*8)(R14)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMP	R3, $-1
	BNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(R3), R3
	MOVD	R3, (13*8)(R14)		// err

ok:
	RET
