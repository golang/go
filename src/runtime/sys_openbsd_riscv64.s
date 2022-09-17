// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for riscv64, OpenBSD
// System calls are implemented in libc/libpthread, this file
// contains trampolines that convert from Go to C calling convention.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME	$0
#define	CLOCK_MONOTONIC	$3

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$200
	// X10 points to the m.
	// We are already on m's g0 stack.

	// Save callee-save registers (X8, X9, X18..X27, F8, F9, F18..F27)
	MOV	X8, (1*8)(X2)
	MOV	X9, (2*8)(X2)
	MOV	X18, (3*8)(X2)
	MOV	X19, (4*8)(X2)
	MOV	X20, (5*8)(X2)
	MOV	X21, (6*8)(X2)
	MOV	X22, (7*8)(X2)
	MOV	X23, (8*8)(X2)
	MOV	X24, (9*8)(X2)
	MOV	X25, (10*8)(X2)
	MOV	X26, (11*8)(X2)
	MOV	g, (12*8)(X2)
	MOVF	F8, (13*8)(X2)
	MOVF	F9, (14*8)(X2)
	MOVF	F18, (15*8)(X2)
	MOVF	F19, (16*8)(X2)
	MOVF	F20, (17*8)(X2)
	MOVF	F21, (18*8)(X2)
	MOVF	F22, (19*8)(X2)
	MOVF	F23, (20*8)(X2)
	MOVF	F24, (21*8)(X2)
	MOVF	F25, (22*8)(X2)
	MOVF	F26, (23*8)(X2)
	MOVF	F27, (24*8)(X2)

	MOV	m_g0(X10), g
	CALL	runtime·save_g(SB)

	CALL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOV	(1*8)(X2), X8
	MOV	(2*8)(X2), X9
	MOV	(3*8)(X2), X18
	MOV	(4*8)(X2), X19
	MOV	(5*8)(X2), X20
	MOV	(6*8)(X2), X21
	MOV	(7*8)(X2), X22
	MOV	(8*8)(X2), X23
	MOV	(9*8)(X2), X24
	MOV	(10*8)(X2), X25
	MOV	(11*8)(X2), X26
	MOV	(12*8)(X2), g
	MOVF	(13*8)(X2), F8
	MOVF	(14*8)(X2), F9
	MOVF	(15*8)(X2), F18
	MOVF	(16*8)(X2), F19
	MOVF	(17*8)(X2), F20
	MOVF	(18*8)(X2), F21
	MOVF	(19*8)(X2), F22
	MOVF	(20*8)(X2), F23
	MOVF	(21*8)(X2), F24
	MOVF	(22*8)(X2), F25
	MOVF	(23*8)(X2), F26
	MOVF	(24*8)(X2), F27

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	MOV	$0, X10

	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), X10
	MOV	info+16(FP), X11
	MOV	ctx+24(FP), X12
	MOV	fn+0(FP), X5
	JALR	X1, X5
	RET

TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$224
	// Save callee-save registers (X8, X9, X18..X27, F8, F9, F18..F27)
	MOV	X8, (4*8)(X2)
	MOV	X9, (5*8)(X2)
	MOV	X18, (6*8)(X2)
	MOV	X19, (7*8)(X2)
	MOV	X20, (8*8)(X2)
	MOV	X21, (9*8)(X2)
	MOV	X22, (10*8)(X2)
	MOV	X23, (11*8)(X2)
	MOV	X24, (12*8)(X2)
	MOV	X25, (13*8)(X2)
	MOV	X26, (14*8)(X2)
	MOV	g, (15*8)(X2)
	MOVF	F8, (16*8)(X2)
	MOVF	F9, (17*8)(X2)
	MOVF	F18, (18*8)(X2)
	MOVF	F19, (19*8)(X2)
	MOVF	F20, (20*8)(X2)
	MOVF	F21, (21*8)(X2)
	MOVF	F22, (22*8)(X2)
	MOVF	F23, (23*8)(X2)
	MOVF	F24, (24*8)(X2)
	MOVF	F25, (25*8)(X2)
	MOVF	F26, (26*8)(X2)
	MOVF	F27, (27*8)(X2)

	// this might be called in external code context,
	// where g is not set.
	CALL	runtime·load_g(SB)

	MOVW	X10, 8(X2)
	MOV	X11, 16(X2)
	MOV	X12, 24(X2)
	MOV	$runtime·sigtrampgo(SB), X5
	JALR	X1, X5

	// Restore callee-save registers.
	MOV	(4*8)(X2), X8
	MOV	(5*8)(X2), X9
	MOV	(6*8)(X2), X18
	MOV	(7*8)(X2), X19
	MOV	(8*8)(X2), X20
	MOV	(9*8)(X2), X21
	MOV	(10*8)(X2), X22
	MOV	(11*8)(X2), X23
	MOV	(12*8)(X2), X24
	MOV	(13*8)(X2), X25
	MOV	(14*8)(X2), X26
	MOV	(15*8)(X2), g
	MOVF	(16*8)(X2), F8
	MOVF	(17*8)(X2), F9
	MOVF	(18*8)(X2), F18
	MOVF	(19*8)(X2), F19
	MOVF	(20*8)(X2), F20
	MOVF	(21*8)(X2), F21
	MOVF	(22*8)(X2), F22
	MOVF	(23*8)(X2), F23
	MOVF	(24*8)(X2), F24
	MOVF	(25*8)(X2), F25
	MOVF	(26*8)(X2), F26
	MOVF	(27*8)(X2), F27

	RET

//
// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in R0.
// A single int32 result is returned in R0.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$8
	MOV	0(X10), X10		// arg 1 - attr
	CALL	libc_pthread_attr_init(SB)
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$8
	MOV	0(X10), X10		// arg 1 - attr
	CALL	libc_pthread_attr_destroy(SB)
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - size
	MOV	0(X10), X10		// arg 1 - attr
	CALL	libc_pthread_attr_getstacksize(SB)
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - state
	MOV	0(X10), X10		// arg 1 - attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$8
	MOV	0(X10), X11		// arg 2 - attr
	MOV	8(X10), X12		// arg 3 - start
	MOV	16(X10), X13		// arg 4 - arg
	ADD	$-16, X2
	MOV	X2, X10			// arg 1 - &threadid (discard)
	CALL	libc_pthread_create(SB)
	ADD	$16, X2
	RET

TEXT runtime·thrkill_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - signal
	MOV	$0, X12			// arg 3 - tcb
	MOVW	0(X10), X10		// arg 1 - tid
	CALL	libc_thrkill(SB)
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$8
	MOVW	8(X10), X11		// arg 2 - clock_id
	MOV	16(X10), X12		// arg 3 - abstime
	MOV	24(X10), X13		// arg 4 - lock
	MOV	32(X10), X14		// arg 5 - abort
	MOV	0(X10), X10		// arg 1 - id
	CALL	libc_thrsleep(SB)
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$8
	MOVW	8(X10), X11		// arg 2 - count
	MOV	0(X10), X10		// arg 1 - id
	CALL	libc_thrwakeup(SB)
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT,$8
	MOVW	0(X10), X10		// arg 1 - status
	CALL	libc_exit(SB)
	MOV	$0, X5			// crash on failure
	MOV	X5, (X5)
	RET

TEXT runtime·getthrid_trampoline(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args
	CALL	libc_getthrid(SB)
	MOVW	X10, 0(X9)		// return value
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args
	CALL	libc_getpid(SB)		// arg 1 - pid (result in X10)
	MOVW	0(X9), X11		// arg 2 - signal
	CALL	libc_kill(SB)
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$8
	CALL	libc_sched_yield(SB)
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args
	MOV	0(X9), X10		// arg 1 - addr
	MOV	8(X9), X11		// arg 2 - len
	MOVW	16(X9), X12		// arg 3 - prot
	MOVW	20(X9), X13		// arg 4 - flags
	MOVW	24(X9), X14		// arg 5 - fid
	MOVW	28(X9), X15		// arg 6 - offset
	CALL	libc_mmap(SB)
	MOV	$0, X5
	MOV	$-1, X6
	BNE	X6, X10, noerr
	CALL	libc_errno(SB)
	MOVW	(X10), X5		// errno
	MOV	$0, X10
noerr:
	MOV	X10, 32(X9)
	MOV	X5, 40(X9)
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - len
	MOV	0(X10), X10		// arg 1 - addr
	CALL	libc_munmap(SB)
	MOV	$-1, X5
	BNE	X5, X10, 3(PC)
	MOV	$0, X5			// crash on failure
	MOV	X5, (X5)
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - len
	MOVW	16(X10), X12		// arg 3 - advice
	MOV	0(X10), X10		// arg 1 - addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$8
	MOVW	8(X10), X11		// arg 2 - flags
	MOVW	12(X10), X12		// arg 3 - mode
	MOV	0(X10), X10		// arg 1 - path
	MOV	$0, X13			// varargs
	CALL	libc_open(SB)
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$8
	MOVW	0(X10), X10		// arg 1 - fd
	CALL	libc_close(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - buf
	MOVW	16(X10), X12		// arg 3 - count
	MOVW	0(X10), X10		// arg 1 - fd (int32 from read)
	CALL	libc_read(SB)
	MOV	$-1, X5
	BNE	X5, X10, noerr
	CALL	libc_errno(SB)
	MOVW	(X10), X10		// errno
	NEG	X10			// caller expects negative errno
noerr:
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - buf
	MOVW	16(X10), X12		// arg 3 - count
	MOV	0(X10), X10		// arg 1 - fd (uintptr from write1)
	CALL	libc_write(SB)
	MOV	$-1, X5
	BNE	X5, X10, noerr
	CALL	libc_errno(SB)
	MOVW	(X10), X10		// errno
	NEG	X10			// caller expects negative errno
noerr:
	RET

TEXT runtime·pipe2_trampoline(SB),NOSPLIT,$8
	MOVW	8(X10), X11		// arg 2 - flags
	MOV	0(X10), X10		// arg 1 - filedes
	CALL	libc_pipe2(SB)
	MOV	$-1, X5
	BNE	X5, X10, noerr
	CALL	libc_errno(SB)
	MOVW	(X10), X10		// errno
	NEG	X10			// caller expects negative errno
noerr:
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - new
	MOV	16(X10), X12		// arg 3 - old
	MOVW	0(X10), X10		// arg 1 - which
	CALL	libc_setitimer(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$8
	MOVW	0(X10), X10		// arg 1 - usec
	CALL	libc_usleep(SB)
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$8
	MOVW	8(X10), X11		// arg 2 - miblen
	MOV	16(X10), X12		// arg 3 - out
	MOV	24(X10), X13		// arg 4 - size
	MOV	32(X10), X14		// arg 5 - dst
	MOV	40(X10), X15		// arg 6 - ndst
	MOV	0(X10), X10		// arg 1 - mib
	CALL	libc_sysctl(SB)
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$8
	CALL	libc_kqueue(SB)
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - keventt
	MOVW	16(X10), X12		// arg 3 - nch
	MOV	24(X10), X13		// arg 4 - ev
	MOVW	32(X10), X14		// arg 5 - nev
	MOV	40(X10), X15		// arg 6 - ts
	MOVW	0(X10), X10		// arg 1 - kq
	CALL	libc_kevent(SB)
	MOV	$-1, X5
	BNE	X5, X10, noerr
	CALL	libc_errno(SB)
	MOVW	(X10), X10		// errno
	NEG	X10			// caller expects negative errno
noerr:
	RET

TEXT runtime·clock_gettime_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - tp
	MOVW	0(X10), X10		// arg 1 - clock_id
	CALL	libc_clock_gettime(SB)
	MOV	$-1, X5
	BNE	X5, X10, 3(PC)
	MOV	$0, X5			// crash on failure
	MOV	X5, (X5)
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args
	MOVW	0(X9), X10		// arg 1 - fd
	MOVW	4(X9), X11		// arg 2 - cmd
	MOVW	8(X9), X12		// arg 3 - arg
	MOV	$0, X13			// vararg
	CALL	libc_fcntl(SB)
	MOV	$-1, X5
	MOV	$0, X11
	BNE	X5, X10, noerr
	CALL	libc_errno(SB)
	MOVW	(X10), X11		// errno
	MOV	$-1, X10
noerr:
	MOVW	X10, 12(X9)
	MOVW	X11, 16(X9)
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - new
	MOV	16(X10), X12		// arg 3 - old
	MOVW	0(X10), X10		// arg 1 - sig
	CALL	libc_sigaction(SB)
	MOV	$-1, X5
	BNE	X5, X10, 3(PC)
	MOV	$0, X5			// crash on failure
	MOV	X5, (X5)
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - new
	MOV	16(X10), X12		// arg 3 - old
	MOVW	0(X10), X10		// arg 1 - how
	CALL	libc_pthread_sigmask(SB)
	MOV	$-1, X5
	BNE	X5, X10, 3(PC)
	MOV	$0, X5			// crash on failure
	MOV	X5, (X5)
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$8
	MOV	8(X10), X11		// arg 2 - old
	MOV	0(X10), X10		// arg 1 - new
	CALL	libc_sigaltstack(SB)
	MOV	$-1, X5
	BNE	X5, X10, 3(PC)
	MOV	$0, X5			// crash on failure
	MOV	X5, (X5)
	RET

TEXT runtime·issetugid_trampoline(SB),NOSPLIT,$0
	MOV	X10, X9			// pointer to args
	CALL	libc_issetugid(SB)
	MOVW	X10, 0(X9)		// return value
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
TEXT runtime·syscall(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args

	MOV	(0*8)(X9), X5		// fn
	MOV	(1*8)(X9), X10		// a1
	MOV	(2*8)(X9), X11		// a2
	MOV	(3*8)(X9), X12		// a3
	MOV	$0, X13			// vararg

	JALR	X1, X5

	MOV	X10, (4*8)(X9)		// r1
	MOV	X11, (5*8)(X9)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	MOV	$-1, X5
	MOVW	X10, X11
	BNE	X5, X11, ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(X10), X10
	MOV	X10, (6*8)(X9)		// err

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
TEXT runtime·syscallX(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args

	MOV	(0*8)(X9), X5		// fn
	MOV	(1*8)(X9), X10		// a1
	MOV	(2*8)(X9), X11		// a2
	MOV	(3*8)(X9), X12		// a3
	MOV	$0, X13			// vararg

	JALR	X1, X5

	MOV	X10, (4*8)(X9)		// r1
	MOV	X11, (5*8)(X9)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	MOV	$-1, X5
	BNE	X5, X10, ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(X10), X10
	MOV	X10, (6*8)(X9)		// err

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
TEXT runtime·syscall6(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args

	MOV	(0*8)(X9), X5		// fn
	MOV	(1*8)(X9), X10		// a1
	MOV	(2*8)(X9), X11		// a2
	MOV	(3*8)(X9), X12		// a3
	MOV	(4*8)(X9), X13		// a4
	MOV	(5*8)(X9), X14		// a5
	MOV	(6*8)(X9), X15		// a6
	MOV	$0, X16			// vararg

	JALR	X1, X5

	MOV	X10, (7*8)(X9)		// r1
	MOV	X11, (8*8)(X9)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	MOV	$-1, X5
	MOVW	X10, X11
	BNE	X5, X11, ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(X10), X10
	MOV	X10, (9*8)(X9)		// err

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
TEXT runtime·syscall6X(SB),NOSPLIT,$8
	MOV	X10, X9			// pointer to args

	MOV	(0*8)(X9), X5		// fn
	MOV	(1*8)(X9), X10		// a1
	MOV	(2*8)(X9), X11		// a2
	MOV	(3*8)(X9), X12		// a3
	MOV	(4*8)(X9), X13		// a4
	MOV	(5*8)(X9), X14		// a5
	MOV	(6*8)(X9), X15		// a6
	MOV	$0, X16			// vararg

	JALR	X1, X5

	MOV	X10, (7*8)(X9)		// r1
	MOV	X11, (8*8)(X9)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	MOV	$-1, X5
	BNE	X5, X10, ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(X10), X10
	MOV	X10, (9*8)(X9)		// err

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
//
// The openbsd/riscv64 kernel only accepts eight syscall arguments.
TEXT runtime·syscall10(SB),NOSPLIT,$0
	MOV	X10, X9			// pointer to args

	ADD	$-16, X2

	MOV	(0*8)(X9), X5		// fn
	MOV	(1*8)(X9), X10		// a1
	MOV	(2*8)(X9), X11		// a2
	MOV	(3*8)(X9), X12		// a3
	MOV	(4*8)(X9), X13		// a4
	MOV	(5*8)(X9), X14		// a5
	MOV	(6*8)(X9), X15		// a6
	MOV	(7*8)(X9), X16		// a7
	MOV	(8*8)(X9), X17		// a8

	JALR	X1, X5

	MOV	X10, (11*8)(X9)		// r1
	MOV	X11, (12*8)(X9)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	MOV	$-1, X5
	MOVW	X10, X11
	BNE	X5, X11, ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(X10), X10
	MOV	X10, (13*8)(X9)		// err

ok:
	ADD	$16, X2
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
//
// The openbsd/riscv64 kernel only accepts eight syscall arguments.
TEXT runtime·syscall10X(SB),NOSPLIT,$0
	MOV	X10, X9			// pointer to args

	ADD	$-16, X2

	MOV	(0*8)(X9), X5		// fn
	MOV	(1*8)(X9), X10		// a1
	MOV	(2*8)(X9), X11		// a2
	MOV	(3*8)(X9), X12		// a3
	MOV	(4*8)(X9), X13		// a4
	MOV	(5*8)(X9), X14		// a5
	MOV	(6*8)(X9), X15		// a6
	MOV	(7*8)(X9), X16		// a7
	MOV	(8*8)(X9), X17		// a8

	JALR	X1, X5

	MOV	X10, (11*8)(X9)		// r1
	MOV	X11, (12*8)(X9)		// r2

	// Standard libc functions return -1 on error
	// and set errno.
	MOV	$-1, X5
	BNE	X5, X10, ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVW	(X10), X10
	MOV	X10, (13*8)(X9)		// err

ok:
	ADD	$16, X2
	RET
