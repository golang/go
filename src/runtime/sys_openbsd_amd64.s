// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, OpenBSD.
// System calls are implemented in libc/libpthread, this file
// contains trampolines that convert from Go to C calling convention.
// Some direct system call implementations currently remain.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_MONOTONIC	$3

TEXT runtime·settls(SB),NOSPLIT,$0
	// Nothing to do, pthread already set thread-local storage up.
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// DI points to the m.
	// We are already on m's g0 stack.

	// Save callee-save registers.
	SUBQ	$48, SP
	MOVQ	BX, 0(SP)
	MOVQ	BP, 8(SP)
	MOVQ	R12, 16(SP)
	MOVQ	R13, 24(SP)
	MOVQ	R14, 32(SP)
	MOVQ	R15, 40(SP)

	// Load g and save to TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVQ	m_g0(DI), DX // g
	MOVQ	DX, -8(FS)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOVQ	0(SP), BX
	MOVQ	8(SP), BP
	MOVQ	16(SP), R12
	MOVQ	24(SP), R13
	MOVQ	32(SP), R14
	MOVQ	40(SP), R15

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX

	ADDQ	$48, SP
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVQ	fn+0(FP),    AX
	MOVL	sig+8(FP),   DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP),  DX
	PUSHQ	BP
	MOVQ	SP, BP
	ANDQ	$~15, SP     // alignment for x86_64 ABI
	CALL	AX
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$72
	// Save callee-saved C registers, since the caller may be a C signal handler.
	MOVQ	BX,  bx-8(SP)
	MOVQ	BP,  bp-16(SP)  // save in case GOEXPERIMENT=noframepointer is set
	MOVQ	R12, r12-24(SP)
	MOVQ	R13, r13-32(SP)
	MOVQ	R14, r14-40(SP)
	MOVQ	R15, r15-48(SP)
	// We don't save mxcsr or the x87 control word because sigtrampgo doesn't
	// modify them.

	MOVQ	DX, ctx-56(SP)
	MOVQ	SI, info-64(SP)
	MOVQ	DI, signum-72(SP)
	CALL	runtime·sigtrampgo(SB)

	MOVQ	r15-48(SP), R15
	MOVQ	r14-40(SP), R14
	MOVQ	r13-32(SP), R13
	MOVQ	r12-24(SP), R12
	MOVQ	bp-16(SP),  BP
	MOVQ	bx-8(SP),   BX
	RET

//
// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in DI.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_init(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_destroy(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 - stacksize
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_getstacksize(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 - detachstate
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	SUBQ	$16, SP
	MOVQ	0(DI), SI		// arg 2 - attr
	MOVQ	8(DI), DX		// arg 3 - start
	MOVQ	16(DI), CX		// arg 4 - arg
	MOVQ	SP, DI			// arg 1 - &thread (discarded)
	CALL	libc_pthread_create(SB)
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	8(DI), SI		// arg 2 - clock_id
	MOVQ	16(DI), DX		// arg 3 - abstime
	MOVQ	24(DI), CX		// arg 3 - lock
	MOVQ	32(DI), R8		// arg 4 - abort
	MOVQ	0(DI), DI		// arg 1 - id
	CALL	libc_thrsleep(SB)
	POPQ	BP
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	8(DI), SI		// arg 2 - count
	MOVQ	0(DI), DI		// arg 1 - id
	CALL	libc_thrwakeup(SB)
	POPQ	BP
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	CALL	libc_sched_yield(SB)
	POPQ	BP
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVL	code+0(FP), DI		// arg 1 - exit status
	MOVL	$1, AX			// sys_exit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-8
	MOVQ	wait+0(FP), DI		// arg 1 - notdead
	MOVL	$302, AX		// sys___threxit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT,$-8
	MOVQ	name+0(FP), DI		// arg 1 pathname
	MOVL	mode+8(FP), SI		// arg 2 flags
	MOVL	perm+12(FP), DX		// arg 3 mode
	MOVL	$5, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-8
	MOVL	fd+0(FP), DI		// arg 1 fd
	MOVL	$6, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-8
	MOVL	fd+0(FP), DI		// arg 1 fd
	MOVQ	p+8(FP), SI		// arg 2 buf
	MOVL	n+16(FP), DX		// arg 3 count
	MOVL	$3, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX			// caller expects negative errno
	MOVL	AX, ret+24(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT,$0-12
	LEAQ	r+0(FP), DI
	MOVL	$263, AX
	SYSCALL
	MOVL	AX, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$0-20
	LEAQ	r+8(FP), DI
	MOVL	flags+0(FP), SI
	MOVL	$101, AX
	SYSCALL
	MOVL	AX, errno+16(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$-8
	MOVQ	fd+0(FP), DI		// arg 1 - fd
	MOVQ	p+8(FP), SI		// arg 2 - buf
	MOVL	n+16(FP), DX		// arg 3 - nbyte
	MOVL	$4, AX			// sys_write
	SYSCALL
	JCC	2(PC)
	NEGQ	AX			// caller expects negative errno
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVQ	AX, 0(SP)		// tv_sec
	MOVL	$1000, AX
	MULL	DX
	MOVQ	AX, 8(SP)		// tv_nsec

	MOVQ	SP, DI			// arg 1 - rqtp
	MOVQ	$0, SI			// arg 2 - rmtp
	MOVL	$91, AX			// sys_nanosleep
	SYSCALL
	RET

TEXT runtime·getthrid(SB),NOSPLIT,$0-4
	MOVL	$299, AX		// sys_getthrid
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·thrkill(SB),NOSPLIT,$0-16
	MOVL	tid+0(FP), DI		// arg 1 - tid
	MOVQ	sig+8(FP), SI		// arg 2 - signum
	MOVQ	$0, DX			// arg 3 - tcb
	MOVL	$119, AX		// sys_thrkill
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	MOVL	$20, AX			// sys_getpid
	SYSCALL
	MOVQ	AX, DI			// arg 1 - pid
	MOVL	sig+0(FP), SI		// arg 2 - signum
	MOVL	$122, AX		// sys_kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-8
	MOVL	mode+0(FP), DI		// arg 1 - which
	MOVQ	new+8(FP), SI		// arg 2 - itv
	MOVQ	old+16(FP), DX		// arg 3 - oitv
	MOVL	$69, AX			// sys_setitimer
	SYSCALL
	RET

// func walltime1() (sec int64, nsec int32)
TEXT runtime·walltime1(SB), NOSPLIT, $32
	MOVQ	$0, DI			// arg 1 - clock_id
	LEAQ	8(SP), SI		// arg 2 - tp
	MOVL	$87, AX			// sys_clock_gettime
	SYSCALL
	MOVQ	8(SP), AX		// sec
	MOVQ	16(SP), DX		// nsec

	// sec is in AX, nsec in DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$24
	MOVQ	CLOCK_MONOTONIC, DI	// arg 1 - clock_id
	LEAQ	8(SP), SI		// arg 2 - tp
	MOVL	$87, AX			// sys_clock_gettime
	SYSCALL
	MOVQ	8(SP), AX		// sec
	MOVQ	16(SP), DX		// nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVL	sig+0(FP), DI		// arg 1 - signum
	MOVQ	new+8(FP), SI		// arg 2 - nsa
	MOVQ	old+16(FP), DX		// arg 3 - osa
	MOVL	$46, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·obsdsigprocmask(SB),NOSPLIT,$0
	MOVL	how+0(FP), DI		// arg 1 - how
	MOVL	new+4(FP), SI		// arg 2 - set
	MOVL	$48, AX			// sys_sigprocmask
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 - addr
	MOVQ	n+8(FP), SI		// arg 2 - len
	MOVL	prot+16(FP), DX		// arg 3 - prot
	MOVL	flags+20(FP), R10		// arg 4 - flags
	MOVL	fd+24(FP), R8		// arg 5 - fd
	MOVL	off+28(FP), R9
	SUBQ	$16, SP
	MOVQ	R9, 8(SP)		// arg 7 - offset (passed on stack)
	MOVQ	$0, R9			// arg 6 - pad
	MOVL	$197, AX
	SYSCALL
	JCC	ok
	ADDQ	$16, SP
	MOVQ	$0, p+32(FP)
	MOVQ	AX, err+40(FP)
	RET
ok:
	ADDQ	$16, SP
	MOVQ	AX, p+32(FP)
	MOVQ	$0, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 - addr
	MOVQ	n+8(FP), SI		// arg 2 - len
	MOVL	$73, AX			// sys_munmap
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 - addr
	MOVQ	n+8(FP), SI		// arg 2 - len
	MOVL	flags+16(FP), DX	// arg 3 - behav
	MOVQ	$75, AX			// sys_madvise
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVQ	new+0(FP), DI		// arg 1 - nss
	MOVQ	old+8(FP), SI		// arg 2 - oss
	MOVQ	$288, AX		// sys_sigaltstack
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVQ	mib+0(FP), DI		// arg 1 - name
	MOVL	miblen+8(FP), SI		// arg 2 - namelen
	MOVQ	out+16(FP), DX		// arg 3 - oldp
	MOVQ	size+24(FP), R10		// arg 4 - oldlenp
	MOVQ	dst+32(FP), R8		// arg 5 - newp
	MOVQ	ndst+40(FP), R9		// arg 6 - newlen
	MOVQ	$202, AX		// sys___sysctl
	SYSCALL
	JCC	4(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+48(FP)
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$269, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	kq+0(FP), DI
	MOVQ	ch+8(FP), SI
	MOVL	nch+16(FP), DX
	MOVQ	ev+24(FP), R10
	MOVL	nev+32(FP), R8
	MOVQ	ts+40(FP), R9
	MOVL	$72, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL	fd+0(FP), DI	// fd
	MOVQ	$2, SI		// F_SETFD
	MOVQ	$1, DX		// FD_CLOEXEC
	MOVL	$92, AX		// fcntl
	SYSCALL
	RET

// func runtime·setNonblock(int32 fd)
TEXT runtime·setNonblock(SB),NOSPLIT,$0-4
	MOVL    fd+0(FP), DI  // fd
	MOVQ    $3, SI  // F_GETFL
	MOVQ    $0, DX
	MOVL	$92, AX // fcntl
	SYSCALL
	MOVL	fd+0(FP), DI // fd
	MOVQ	$4, SI // F_SETFL
	MOVQ	$4, DX // O_NONBLOCK
	ORL	AX, DX
	MOVL	$92, AX // fcntl
	SYSCALL
	RET
