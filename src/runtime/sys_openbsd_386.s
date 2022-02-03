// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, OpenBSD
// System calls are implemented in libc/libpthread, this file
// contains trampolines that convert from Go to C calling convention.
// Some direct system call implementations currently remain.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define	CLOCK_MONOTONIC	$3

TEXT runtime·setldt(SB),NOSPLIT,$0
	// Nothing to do, pthread already set thread-local storage up.
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$28
	NOP	SP	// tell vet SP changed - stop checking offsets

	// We are already on m's g0 stack.

	// Save callee-save registers.
	MOVL	BX, bx-4(SP)
	MOVL	BP, bp-8(SP)
	MOVL	SI, si-12(SP)
	MOVL	DI, di-16(SP)

	MOVL	32(SP), AX	// m
	MOVL	m_g0(AX), DX
	get_tls(CX)
	MOVL	DX, g(CX)

	CALL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOVL	di-16(SP), DI
	MOVL	si-12(SP), SI
	MOVL	bp-8(SP),  BP
	MOVL	bx-4(SP),  BX

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	MOVL	$0, AX
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-16
	MOVL	fn+0(FP), AX
	MOVL	sig+4(FP), BX
	MOVL	info+8(FP), CX
	MOVL	ctx+12(FP), DX
	MOVL	SP, SI
	SUBL	$32, SP
	ANDL	$~15, SP	// align stack: handler might be a C function
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)	// save SI: handler might be a Go function
	CALL	AX
	MOVL	12(SP), AX
	MOVL	AX, SP
	RET

// Called by OS using C ABI.
TEXT runtime·sigtramp(SB),NOSPLIT,$28
	NOP	SP	// tell vet SP changed - stop checking offsets
	// Save callee-saved C registers, since the caller may be a C signal handler.
	MOVL	BX, bx-4(SP)
	MOVL	BP, bp-8(SP)
	MOVL	SI, si-12(SP)
	MOVL	DI, di-16(SP)
	// We don't save mxcsr or the x87 control word because sigtrampgo doesn't
	// modify them.

	MOVL	32(SP), BX // signo
	MOVL	BX, 0(SP)
	MOVL	36(SP), BX // info
	MOVL	BX, 4(SP)
	MOVL	40(SP), BX // context
	MOVL	BX, 8(SP)
	CALL	runtime·sigtrampgo(SB)

	MOVL	di-16(SP), DI
	MOVL	si-12(SP), SI
	MOVL	bp-8(SP),  BP
	MOVL	bx-4(SP),  BX
	RET

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall - note that while asmcgocall does
// stack alignment, creation of a frame undoes it again.
// A pointer to the arguments is passed on the stack.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$4, SP
	MOVL	12(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	AX, 0(SP)		// arg 1 - attr
	CALL	libc_pthread_attr_init(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$4, SP
	MOVL	12(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	AX, 0(SP)		// arg 1 - attr
	CALL	libc_pthread_attr_destroy(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - attr
	MOVL	BX, 4(SP)		// arg 2 - size
	CALL	libc_pthread_attr_getstacksize(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - attr
	MOVL	BX, 4(SP)		// arg 2 - state
	CALL	libc_pthread_attr_setdetachstate(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$20, SP
	MOVL	28(SP), DX		// pointer to args
	LEAL	16(SP), AX
	MOVL	AX, 0(SP)		// arg 1 - &threadid (discarded)
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 4(SP)		// arg 2 - attr
	MOVL	BX, 8(SP)		// arg 3 - start
	MOVL	CX, 12(SP)		// arg 4 - arg
	CALL	libc_pthread_create(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·thrkill_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - tid
	MOVL	BX, 4(SP)		// arg 2 - signal
	MOVL	$0, 8(SP)		// arg 3 - tcb
	CALL	libc_thrkill(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$20, SP
	MOVL	28(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - id
	MOVL	BX, 4(SP)		// arg 2 - clock_id
	MOVL	CX, 8(SP)		// arg 3 - abstime
	MOVL	12(DX), AX
	MOVL	16(DX), BX
	MOVL	AX, 12(SP)		// arg 4 - lock
	MOVL	BX, 16(SP)		// arg 5 - abort
	CALL	libc_thrsleep(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - id
	MOVL	BX, 4(SP)		// arg 2 - count
	CALL	libc_thrwakeup(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$4, SP
	MOVL	12(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	AX, 0(SP)		// arg 1 - status
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1		// crash on failure
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·getthrid_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	CALL	libc_getthrid(SB)
	NOP	SP			// tell vet SP changed - stop checking offsets
	MOVL	8(SP), DX		// pointer to return value
	MOVL	AX, 0(DX)
	POPL	BP
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX
	MOVL	0(DX), BX
	CALL	libc_getpid(SB)
	MOVL	AX, 0(SP)		// arg 1 - pid
	MOVL	BX, 4(SP)		// arg 2 - signal
	CALL	libc_kill(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	CALL	libc_sched_yield(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$32, SP
	MOVL	40(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - addr
	MOVL	BX, 4(SP)		// arg 2 - len
	MOVL	CX, 8(SP)		// arg 3 - prot
	MOVL	12(DX), AX
	MOVL	16(DX), BX
	MOVL	20(DX), CX
	MOVL	AX, 12(SP)		// arg 4 - flags
	MOVL	BX, 16(SP)		// arg 5 - fid
	MOVL	$0, 20(SP)		// pad
	MOVL	CX, 24(SP)		// arg 6 - offset (low 32 bits)
	MOVL	$0, 28(SP)		// offset (high 32 bits)
	CALL	libc_mmap(SB)
	MOVL	$0, BX
	CMPL	AX, $-1
	JNE	ok
	CALL	libc_errno(SB)
	MOVL	(AX), BX
	MOVL	$0, AX
ok:
	MOVL	40(SP), DX
	MOVL	AX, 24(DX)
	MOVL	BX, 28(DX)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - addr
	MOVL	BX, 4(SP)		// arg 2 - len
	CALL	libc_munmap(SB)
	CMPL	AX, $-1
	JNE	2(PC)
	MOVL	$0xf1, 0xf1		// crash on failure
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - addr
	MOVL	BX, 4(SP)		// arg 2 - len
	MOVL	CX, 8(SP)		// arg 3 - advice
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$16, SP
	MOVL	24(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - path
	MOVL	BX, 4(SP)		// arg 2 - flags
	MOVL	CX, 8(SP)		// arg 3 - mode
	MOVL	$0, 12(SP)		// vararg
	CALL	libc_open(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$4, SP
	MOVL	12(SP), DX
	MOVL	0(DX), AX
	MOVL	AX, 0(SP)		// arg 1 - fd
	CALL	libc_close(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - fd
	MOVL	BX, 4(SP)		// arg 2 - buf
	MOVL	CX, 8(SP)		// arg 3 - count
	CALL	libc_read(SB)
	CMPL	AX, $-1
	JNE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - fd
	MOVL	BX, 4(SP)		// arg 2 - buf
	MOVL	CX, 8(SP)		// arg 3 - count
	CALL	libc_write(SB)
	CMPL	AX, $-1
	JNE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pipe2_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - fds
	MOVL	BX, 4(SP)		// arg 2 - flags
	CALL	libc_pipe2(SB)
	CMPL	AX, $-1
	JNE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - which
	MOVL	BX, 4(SP)		// arg 2 - new
	MOVL	CX, 8(SP)		// arg 3 - old
	CALL	libc_setitimer(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$4, SP
	MOVL	12(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	AX, 0(SP)
	CALL	libc_usleep(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - name
	MOVL	BX, 4(SP)		// arg 2 - namelen
	MOVL	CX, 8(SP)		// arg 3 - old
	MOVL	12(DX), AX
	MOVL	16(DX), BX
	MOVL	20(DX), CX
	MOVL	AX, 12(SP)		// arg 4 - oldlenp
	MOVL	BX, 16(SP)		// arg 5 - newp
	MOVL	CX, 20(SP)		// arg 6 - newlen
	CALL	libc_sysctl(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	CALL	libc_kqueue(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - kq
	MOVL	BX, 4(SP)		// arg 2 - keventt
	MOVL	CX, 8(SP)		// arg 3 - nch
	MOVL	12(DX), AX
	MOVL	16(DX), BX
	MOVL	20(DX), CX
	MOVL	AX, 12(SP)		// arg 4 - ev
	MOVL	BX, 16(SP)		// arg 5 - nev
	MOVL	CX, 20(SP)		// arg 6 - ts
	CALL	libc_kevent(SB)
	CMPL	AX, $-1
	JNE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·clock_gettime_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - tp
	MOVL	BX, 4(SP)		// arg 2 - clock_id
	CALL	libc_clock_gettime(SB)
	CMPL	AX, $-1
	JNE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$16, SP
	MOVL	24(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - fd
	MOVL	BX, 4(SP)		// arg 2 - cmd
	MOVL	CX, 8(SP)		// arg 3 - arg
	MOVL	$0, 12(SP)		// vararg
	CALL	libc_fcntl(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - sig
	MOVL	BX, 4(SP)		// arg 2 - new
	MOVL	CX, 8(SP)		// arg 3 - old
	CALL	libc_sigaction(SB)
	CMPL	AX, $-1
	JNE	2(PC)
	MOVL	$0xf1, 0xf1		// crash on failure
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$12, SP
	MOVL	20(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	8(DX), CX
	MOVL	AX, 0(SP)		// arg 1 - how
	MOVL	BX, 4(SP)		// arg 2 - new
	MOVL	CX, 8(SP)		// arg 3 - old
	CALL	libc_pthread_sigmask(SB)
	CMPL	AX, $-1
	JNE	2(PC)
	MOVL	$0xf1, 0xf1		// crash on failure
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), DX		// pointer to args
	MOVL	0(DX), AX
	MOVL	4(DX), BX
	MOVL	AX, 0(SP)		// arg 1 - new
	MOVL	BX, 4(SP)		// arg 2 - old
	CALL	libc_sigaltstack(SB)
	CMPL	AX, $-1
	JNE	2(PC)
	MOVL	$0xf1, 0xf1		// crash on failure
	MOVL	BP, SP
	POPL	BP
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
	PUSHL	BP
	MOVL	SP, BP

	SUBL	$12, SP
	MOVL	20(SP), BX		// pointer to args

	MOVL	(1*4)(BX), AX
	MOVL	(2*4)(BX), CX
	MOVL	(3*4)(BX), DX
	MOVL	AX, (0*4)(SP)		// a1
	MOVL	CX, (1*4)(SP)		// a2
	MOVL	DX, (2*4)(SP)		// a3

	MOVL	(0*4)(BX), AX		// fn
	CALL	AX

	MOVL	AX, (4*4)(BX)		// r1
	MOVL	DX, (5*4)(BX)		// r2

	// Standard libc functions return -1 on error and set errno.
	CMPL	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	MOVW	AX, (6*4)(BX)		// err

ok:
	MOVL	$0, AX			// no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
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
	PUSHL	BP
	MOVL	SP, BP

	SUBL	$12, SP
	MOVL	20(SP), BX		// pointer to args

	MOVL	(1*4)(BX), AX
	MOVL	(2*4)(BX), CX
	MOVL	(3*4)(BX), DX
	MOVL	AX, (0*4)(SP)		// a1
	MOVL	CX, (1*4)(SP)		// a2
	MOVL	DX, (2*4)(SP)		// a3

	MOVL	(0*4)(BX), AX		// fn
	CALL	AX

	MOVL	AX, (4*4)(BX)		// r1
	MOVL	DX, (5*4)(BX)		// r2

	// Standard libc functions return -1 on error and set errno.
	CMPL	AX, $-1
	JNE	ok
	CMPL	DX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	MOVW	AX, (6*4)(BX)		// err

ok:
	MOVL	$0, AX			// no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
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
	PUSHL	BP
	MOVL	SP, BP

	SUBL	$24, SP
	MOVL	32(SP), BX		// pointer to args

	MOVL	(1*4)(BX), AX
	MOVL	(2*4)(BX), CX
	MOVL	(3*4)(BX), DX
	MOVL	AX, (0*4)(SP)		// a1
	MOVL	CX, (1*4)(SP)		// a2
	MOVL	DX, (2*4)(SP)		// a3
	MOVL	(4*4)(BX), AX
	MOVL	(5*4)(BX), CX
	MOVL	(6*4)(BX), DX
	MOVL	AX, (3*4)(SP)		// a4
	MOVL	CX, (4*4)(SP)		// a5
	MOVL	DX, (5*4)(SP)		// a6

	MOVL	(0*4)(BX), AX		// fn
	CALL	AX

	MOVL	AX, (7*4)(BX)		// r1
	MOVL	DX, (8*4)(BX)		// r2

	// Standard libc functions return -1 on error and set errno.
	CMPL	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	MOVW	AX, (9*4)(BX)		// err

ok:
	MOVL	$0, AX			// no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
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
	PUSHL	BP
	MOVL	SP, BP

	SUBL	$24, SP
	MOVL	32(SP), BX		// pointer to args

	MOVL	(1*4)(BX), AX
	MOVL	(2*4)(BX), CX
	MOVL	(3*4)(BX), DX
	MOVL	AX, (0*4)(SP)		// a1
	MOVL	CX, (1*4)(SP)		// a2
	MOVL	DX, (2*4)(SP)		// a3
	MOVL	(4*4)(BX), AX
	MOVL	(5*4)(BX), CX
	MOVL	(6*4)(BX), DX
	MOVL	AX, (3*4)(SP)		// a4
	MOVL	CX, (4*4)(SP)		// a5
	MOVL	DX, (5*4)(SP)		// a6

	MOVL	(0*4)(BX), AX		// fn
	CALL	AX

	MOVL	AX, (7*4)(BX)		// r1
	MOVL	DX, (8*4)(BX)		// r2

	// Standard libc functions return -1 on error and set errno.
	CMPL	AX, $-1
	JNE	ok
	CMPL	DX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	MOVW	AX, (9*4)(BX)		// err

ok:
	MOVL	$0, AX			// no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
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
	PUSHL	BP
	MOVL	SP, BP

	SUBL	$40, SP
	MOVL	48(SP), BX		// pointer to args

	MOVL	(1*4)(BX), AX
	MOVL	(2*4)(BX), CX
	MOVL	(3*4)(BX), DX
	MOVL	AX, (0*4)(SP)		// a1
	MOVL	CX, (1*4)(SP)		// a2
	MOVL	DX, (2*4)(SP)		// a3
	MOVL	(4*4)(BX), AX
	MOVL	(5*4)(BX), CX
	MOVL	(6*4)(BX), DX
	MOVL	AX, (3*4)(SP)		// a4
	MOVL	CX, (4*4)(SP)		// a5
	MOVL	DX, (5*4)(SP)		// a6
	MOVL	(7*4)(BX), AX
	MOVL	(8*4)(BX), CX
	MOVL	(9*4)(BX), DX
	MOVL	AX, (6*4)(SP)		// a7
	MOVL	CX, (7*4)(SP)		// a8
	MOVL	DX, (8*4)(SP)		// a9
	MOVL	(10*4)(BX), AX
	MOVL	AX, (9*4)(SP)		// a10

	MOVL	(0*4)(BX), AX		// fn
	CALL	AX

	MOVL	AX, (11*4)(BX)		// r1
	MOVL	DX, (12*4)(BX)		// r2

	// Standard libc functions return -1 on error and set errno.
	CMPL	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	MOVW	AX, (13*4)(BX)		// err

ok:
	MOVL	$0, AX			// no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
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
// syscall10X is like syscall9 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall10X(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP

	SUBL	$40, SP
	MOVL	48(SP), BX		// pointer to args

	MOVL	(1*4)(BX), AX
	MOVL	(2*4)(BX), CX
	MOVL	(3*4)(BX), DX
	MOVL	AX, (0*4)(SP)		// a1
	MOVL	CX, (1*4)(SP)		// a2
	MOVL	DX, (2*4)(SP)		// a3
	MOVL	(4*4)(BX), AX
	MOVL	(5*4)(BX), CX
	MOVL	(6*4)(BX), DX
	MOVL	AX, (3*4)(SP)		// a4
	MOVL	CX, (4*4)(SP)		// a5
	MOVL	DX, (5*4)(SP)		// a6
	MOVL	(7*4)(BX), AX
	MOVL	(8*4)(BX), CX
	MOVL	(9*4)(BX), DX
	MOVL	AX, (6*4)(SP)		// a7
	MOVL	CX, (7*4)(SP)		// a8
	MOVL	DX, (8*4)(SP)		// a9
	MOVL	(10*4)(BX), AX
	MOVL	AX, (9*4)(SP)		// a10

	MOVL	(0*4)(BX), AX		// fn
	CALL	AX

	MOVL	AX, (11*4)(BX)		// r1
	MOVL	DX, (12*4)(BX)		// r2

	// Standard libc functions return -1 on error and set errno.
	CMPL	AX, $-1
	JNE	ok
	CMPL	DX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVL	(AX), AX
	MOVW	AX, (13*4)(BX)		// err

ok:
	MOVL	$0, AX			// no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
	RET
