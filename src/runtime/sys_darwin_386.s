// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for 386, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Exit the entire program (like C exit)
TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP   	// allocate space for callee args (must be 8 mod 16)
	MOVL	16(SP), CX	// arg ptr
	MOVL	0(CX), AX	// arg 1 exit status
	MOVL	AX, 0(SP)
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 name
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 mode
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 perm
	MOVL	AX, 8(SP)
	CALL	libc_open(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	CALL	libc_close(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 buf
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 count
	MOVL	AX, 8(SP)
	CALL	libc_read(SB)
	TESTL	AX, AX
	JGE	noerr
	CALL	libc_error(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno value
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 buf
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 count
	MOVL	AX, 8(SP)
	CALL	libc_write(SB)
	TESTL	AX, AX
	JGE	noerr
	CALL	libc_error(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno value
noerr:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pipe_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), AX		// arg 1 pipefd
	MOVL	AX, 0(SP)
	CALL	libc_pipe(SB)
	TESTL	AX, AX
	JEQ	3(PC)
	CALL	libc_error(SB)		// return negative errno value
	NEGL	AX
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 addr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 len
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 prot
	MOVL	AX, 8(SP)
	MOVL	12(CX), AX		// arg 4 flags
	MOVL	AX, 12(SP)
	MOVL	16(CX), AX		// arg 5 fid
	MOVL	AX, 16(SP)
	MOVL	20(CX), AX		// arg 6 offset
	MOVL	AX, 20(SP)
	CALL	libc_mmap(SB)
	XORL	DX, DX
	CMPL	AX, $-1
	JNE	ok
	CALL	libc_error(SB)
	MOVL	(AX), DX		// errno
	XORL	AX, AX
ok:
	MOVL	32(SP), CX
	MOVL	AX, 24(CX)		// result pointer
	MOVL	DX, 28(CX)		// errno
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 addr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 len
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 advice
	MOVL	AX, 8(SP)
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX		// arg 1 addr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 len
	MOVL	AX, 4(SP)
	CALL	libc_munmap(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 mode
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 new
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 old
	MOVL	AX, 8(SP)
	CALL	libc_setitimer(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·walltime_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), AX
	MOVL	AX, 0(SP)	// *timeval
	MOVL	$0, 4(SP)	// no timezone needed
	CALL	libc_gettimeofday(SB)
	MOVL	BP, SP
	POPL	BP
	RET

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8+(machTimebaseInfo__size+15)/16*16, SP
	CALL	libc_mach_absolute_time(SB)
	MOVL	16+(machTimebaseInfo__size+15)/16*16(SP), CX
	MOVL	AX, 0(CX)
	MOVL	DX, 4(CX)
	MOVL	timebase<>+machTimebaseInfo_denom(SB), DI // atomic read
	MOVL	timebase<>+machTimebaseInfo_numer(SB), SI
	TESTL	DI, DI
	JNE	initialized

	LEAL	4(SP), AX
	MOVL	AX, 0(SP)
	CALL	libc_mach_timebase_info(SB)
	MOVL	4+machTimebaseInfo_numer(SP), SI
	MOVL	4+machTimebaseInfo_denom(SP), DI

	MOVL	SI, timebase<>+machTimebaseInfo_numer(SB)
	MOVL	DI, AX
	XCHGL	AX, timebase<>+machTimebaseInfo_denom(SB) // atomic write
	MOVL	16+(machTimebaseInfo__size+15)/16*16(SP), CX

initialized:
	MOVL	SI, 8(CX)
	MOVL	DI, 12(CX)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 sig
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 new
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 old
	MOVL	AX, 8(SP)
	CALL	libc_sigaction(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 how
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 new
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 old
	MOVL	AX, 8(SP)
	CALL	libc_pthread_sigmask(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX		// arg 1 new
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 old
	MOVL	AX, 4(SP)
	CALL	libc_sigaltstack(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	CALL	libc_getpid(SB)
	MOVL	AX, 0(SP)	// arg 1 pid
	MOVL	16(SP), CX
	MOVL	0(CX), AX
	MOVL	AX, 4(SP)	// arg 2 signal
	CALL	libc_kill(SB)
	MOVL	BP, SP
	POPL	BP
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

// Sigtramp's job is to call the actual signal handler.
// It is called with the C calling convention, and calls out
// to sigtrampgo with the Go calling convention.
TEXT runtime·sigtramp(SB),NOSPLIT,$0
	SUBL	$28, SP

	// Save callee-save registers.
	MOVL	BP, 12(SP)
	MOVL	BX, 16(SP)
	MOVL	SI, 20(SP)
	MOVL	DI, 24(SP)

	MOVL	32(SP), AX
	MOVL	AX, 0(SP)	// arg 1 signal number
	MOVL	36(SP), AX
	MOVL	AX, 4(SP)	// arg 2 siginfo
	MOVL	40(SP), AX
	MOVL	AX, 8(SP)	// arg 3 ctxt
	CALL	runtime·sigtrampgo(SB)

	// Restore callee-save registers.
	MOVL	12(SP), BP
	MOVL	16(SP), BX
	MOVL	20(SP), SI
	MOVL	24(SP), DI

	ADDL	$28, SP
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 usec
	MOVL	AX, 0(SP)
	CALL	libc_usleep(SB)
	MOVL	BP, SP
	POPL	BP
	RET

// func setldt(entry int, address int, limit int)
TEXT runtime·setldt(SB),NOSPLIT,$32
	// Nothing to do on Darwin, pthread already set thread-local storage up.
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 mib
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 miblen
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 out
	MOVL	AX, 8(SP)
	MOVL	12(CX), AX		// arg 4 size
	MOVL	AX, 12(SP)
	MOVL	16(CX), AX		// arg 5 dst
	MOVL	AX, 16(SP)
	MOVL	20(CX), AX		// arg 6 ndst
	MOVL	AX, 20(SP)
	CALL	libc_sysctl(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	CALL	libc_kqueue(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 kq
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 ch
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 nch
	MOVL	AX, 8(SP)
	MOVL	12(CX), AX		// arg 4 ev
	MOVL	AX, 12(SP)
	MOVL	16(CX), AX		// arg 5 nev
	MOVL	AX, 16(SP)
	MOVL	20(CX), AX		// arg 6 ts
	MOVL	AX, 20(SP)
	CALL	libc_kevent(SB)
	CMPL	AX, $-1
	JNE	ok
	CALL	libc_error(SB)
	MOVL	(AX), AX		// errno
	NEGL	AX			// caller wants it as a negative error code
ok:
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 cmd
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 arg
	MOVL	AX, 8(SP)
	CALL	libc_fcntl(SB)
	MOVL	BP, SP
	POPL	BP
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// The value at SP+4 points to the m.
	// We are already on m's g0 stack.

	// Save callee-save registers.
	SUBL	$16, SP
	MOVL	BP, 0(SP)
	MOVL	BX, 4(SP)
	MOVL	SI, 8(SP)
	MOVL	DI, 12(SP)

	MOVL	SP, AX       // hide argument read from vet (vet thinks this function is using the Go calling convention)
	MOVL	20(AX), DI   // m
	MOVL	m_g0(DI), DX // g

	// Initialize TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVL	DX, 0x18(GS)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOVL	0(SP), BP
	MOVL	4(SP), BX
	MOVL	8(SP), SI
	MOVL	12(SP), DI

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX

	ADDL	$16, SP
	RET

TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 0(SP)
	CALL	libc_pthread_attr_init(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 size
	MOVL	AX, 4(SP)
	CALL	libc_pthread_attr_getstacksize(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 state
	MOVL	AX, 4(SP)
	CALL	libc_pthread_attr_setdetachstate(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	LEAL	16(SP), AX	// arg "0" &threadid (which we throw away)
	MOVL	AX, 0(SP)
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 4(SP)
	MOVL	4(CX), AX	// arg 2 start
	MOVL	AX, 8(SP)
	MOVL	8(CX), AX	// arg 3 arg
	MOVL	AX, 12(SP)
	CALL	libc_pthread_create(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·raise_trampoline(SB),NOSPLIT,$0
	PUSHL   BP
	MOVL    SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL    0(CX), AX	// arg 1 sig
	MOVL	AX, 0(SP)
	CALL    libc_raise(SB)
	MOVL    BP, SP
	POPL    BP
	RET

TEXT runtime·pthread_mutex_init_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 mutex
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 attr
	MOVL	AX, 4(SP)
	CALL	libc_pthread_mutex_init(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_mutex_lock_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 mutex
	MOVL	AX, 0(SP)
	CALL	libc_pthread_mutex_lock(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_mutex_unlock_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 mutex
	MOVL	AX, 0(SP)
	CALL	libc_pthread_mutex_unlock(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_cond_init_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 cond
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 attr
	MOVL	AX, 4(SP)
	CALL	libc_pthread_cond_init(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_cond_wait_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 cond
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 mutex
	MOVL	AX, 4(SP)
	CALL	libc_pthread_cond_wait(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_cond_timedwait_relative_np_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX	// arg 1 cond
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 mutex
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX	// arg 3 timeout
	MOVL	AX, 8(SP)
	CALL	libc_pthread_cond_timedwait_relative_np(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_cond_signal_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 cond
	MOVL	AX, 0(SP)
	CALL	libc_pthread_cond_signal(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_self_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	NOP	SP	// hide SP from vet
	CALL	libc_pthread_self(SB)
	MOVL	8(SP), CX
	MOVL	AX, 0(CX)		// return value
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_kill_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 thread
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 sig
	MOVL	AX, 4(SP)
	CALL	libc_pthread_kill(SB)
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
TEXT runtime·syscall(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	(0*4)(CX), AX // fn
	MOVL	(1*4)(CX), DX // a1
	MOVL	DX, 0(SP)
	MOVL	(2*4)(CX), DX // a2
	MOVL	DX, 4(SP)
	MOVL	(3*4)(CX), DX // a3
	MOVL	DX, 8(SP)

	CALL	AX

	MOVL	32(SP), CX
	MOVL	AX, (4*4)(CX) // r1
	MOVL	DX, (5*4)(CX) // r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPL	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_error(SB)
	MOVL	(AX), AX
	MOVL	32(SP), CX
	MOVL	AX, (6*4)(CX) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
	RET

// syscallPtr is like syscall except the libc function reports an
// error by returning NULL and setting errno.
TEXT runtime·syscallPtr(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	(0*4)(CX), AX // fn
	MOVL	(1*4)(CX), DX // a1
	MOVL	DX, 0(SP)
	MOVL	(2*4)(CX), DX // a2
	MOVL	DX, 4(SP)
	MOVL	(3*4)(CX), DX // a3
	MOVL	DX, 8(SP)

	CALL	AX

	MOVL	32(SP), CX
	MOVL	AX, (4*4)(CX) // r1
	MOVL	DX, (5*4)(CX) // r2

	// syscallPtr libc functions return NULL on error
	// and set errno.
	TESTL	AX, AX
	JNE	ok

	// Get error code from libc.
	CALL	libc_error(SB)
	MOVL	(AX), AX
	MOVL	32(SP), CX
	MOVL	AX, (6*4)(CX) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
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
TEXT runtime·syscall6(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	(0*4)(CX), AX // fn
	MOVL	(1*4)(CX), DX // a1
	MOVL	DX, 0(SP)
	MOVL	(2*4)(CX), DX // a2
	MOVL	DX, 4(SP)
	MOVL	(3*4)(CX), DX // a3
	MOVL	DX, 8(SP)
	MOVL	(4*4)(CX), DX // a4
	MOVL	DX, 12(SP)
	MOVL	(5*4)(CX), DX // a5
	MOVL	DX, 16(SP)
	MOVL	(6*4)(CX), DX // a6
	MOVL	DX, 20(SP)

	CALL	AX

	MOVL	32(SP), CX
	MOVL	AX, (7*4)(CX) // r1
	MOVL	DX, (8*4)(CX) // r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPL	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_error(SB)
	MOVL	(AX), AX
	MOVL	32(SP), CX
	MOVL	AX, (9*4)(CX) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
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
TEXT runtime·syscall6X(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	(0*4)(CX), AX // fn
	MOVL	(1*4)(CX), DX // a1
	MOVL	DX, 0(SP)
	MOVL	(2*4)(CX), DX // a2
	MOVL	DX, 4(SP)
	MOVL	(3*4)(CX), DX // a3
	MOVL	DX, 8(SP)
	MOVL	(4*4)(CX), DX // a4
	MOVL	DX, 12(SP)
	MOVL	(5*4)(CX), DX // a5
	MOVL	DX, 16(SP)
	MOVL	(6*4)(CX), DX // a6
	MOVL	DX, 20(SP)

	CALL	AX

	MOVL	32(SP), CX
	MOVL	AX, (7*4)(CX) // r1
	MOVL	DX, (8*4)(CX) // r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPL	AX, $-1
	JNE	ok
	CMPL	DX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_error(SB)
	MOVL	(AX), AX
	MOVL	32(SP), CX
	MOVL	AX, (9*4)(CX) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
	RET

// syscall9 calls a function in libc on behalf of the syscall package.
// syscall9 takes a pointer to a struct like:
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
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall9 must be called on the g0 stack with the
// C calling convention (use libcCall).
TEXT runtime·syscall9(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$40, SP
	MOVL	48(SP), CX
	MOVL	(0*4)(CX), AX // fn
	MOVL	(1*4)(CX), DX // a1
	MOVL	DX, 0(SP)
	MOVL	(2*4)(CX), DX // a2
	MOVL	DX, 4(SP)
	MOVL	(3*4)(CX), DX // a3
	MOVL	DX, 8(SP)
	MOVL	(4*4)(CX), DX // a4
	MOVL	DX, 12(SP)
	MOVL	(5*4)(CX), DX // a5
	MOVL	DX, 16(SP)
	MOVL	(6*4)(CX), DX // a6
	MOVL	DX, 20(SP)
	MOVL	(7*4)(CX), DX // a7
	MOVL	DX, 24(SP)
	MOVL	(8*4)(CX), DX // a8
	MOVL	DX, 28(SP)
	MOVL	(9*4)(CX), DX // a9
	MOVL	DX, 32(SP)

	CALL	AX

	MOVL	48(SP), CX
	MOVL	AX, (10*4)(CX) // r1
	MOVL	DX, (11*4)(CX) // r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPL	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_error(SB)
	MOVL	(AX), AX
	MOVL	48(SP), CX
	MOVL	AX, (12*4)(CX) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	MOVL	BP, SP
	POPL	BP
	RET
