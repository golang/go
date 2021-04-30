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

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	CALL	libc_sched_yield(SB)
	MOVL	BP, SP
	POPL	BP
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVL	$1, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVL	$302, AX		// sys___threxit
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT,$-4
	MOVL	$5, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-4
	MOVL	$6, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-4
	MOVL	$3, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+12(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT,$8-12
	MOVL	$263, AX
	LEAL	r+0(FP), BX
	MOVL	BX, 4(SP)
	INT	$0x80
	MOVL	AX, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$12-16
	MOVL	$101, AX
	LEAL	r+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	flags+0(FP), BX
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	AX, errno+12(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$-4
	MOVL	$4, AX			// sys_write
	INT	$0x80
	JAE	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$24
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 12(SP)		// tv_sec - l32
	MOVL	$0, 16(SP)		// tv_sec - h32
	MOVL	$1000, AX
	MULL	DX
	MOVL	AX, 20(SP)		// tv_nsec

	MOVL	$0, 0(SP)
	LEAL	12(SP), AX
	MOVL	AX, 4(SP)		// arg 1 - rqtp
	MOVL	$0, 8(SP)		// arg 2 - rmtp
	MOVL	$91, AX			// sys_nanosleep
	INT	$0x80
	RET

TEXT runtime·getthrid(SB),NOSPLIT,$0-4
	MOVL	$299, AX		// sys_getthrid
	INT	$0x80
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·thrkill(SB),NOSPLIT,$16-8
	MOVL	$0, 0(SP)
	MOVL	tid+0(FP), AX
	MOVL	AX, 4(SP)		// arg 1 - tid
	MOVL	sig+4(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signum
	MOVL	$0, 12(SP)		// arg 3 - tcb
	MOVL	$119, AX		// sys_thrkill
	INT	$0x80
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVL	$20, AX			// sys_getpid
	INT	$0x80
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)		// arg 1 - pid
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signum
	MOVL	$122, AX		// sys_kill
	INT	$0x80
	RET

TEXT runtime·mmap(SB),NOSPLIT,$36
	LEAL	addr+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - addr
	MOVSL				// arg 2 - len
	MOVSL				// arg 3 - prot
	MOVSL				// arg 4 - flags
	MOVSL				// arg 5 - fd
	MOVL	$0, AX
	STOSL				// arg 6 - pad
	MOVSL				// arg 7 - offset
	MOVL	$0, AX			// top 32 bits of file offset
	STOSL
	MOVL	$197, AX		// sys_mmap
	INT	$0x80
	JAE	ok
	MOVL	$0, p+24(FP)
	MOVL	AX, err+28(FP)
	RET
ok:
	MOVL	AX, p+24(FP)
	MOVL	$0, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$-4
	MOVL	$73, AX			// sys_munmap
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-4
	MOVL	$75, AX			// sys_madvise
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-4
	MOVL	$69, AX
	INT	$0x80
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)		// arg 1 - clock_id
	MOVL	BX, 8(SP)		// arg 2 - tp
	MOVL	$87, AX			// sys_clock_gettime
	INT	$0x80

	MOVL	12(SP), AX		// sec - l32
	MOVL	AX, sec_lo+0(FP)
	MOVL	16(SP), AX		// sec - h32
	MOVL	AX, sec_hi+4(FP)

	MOVL	20(SP), BX		// nsec
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime1(void) so really
// void nanotime1(int64 *nsec)
TEXT runtime·nanotime1(SB),NOSPLIT,$32
	LEAL	12(SP), BX
	MOVL	CLOCK_MONOTONIC, 4(SP)	// arg 1 - clock_id
	MOVL	BX, 8(SP)		// arg 2 - tp
	MOVL	$87, AX			// sys_clock_gettime
	INT	$0x80

	MOVL    16(SP), CX		// sec - h32
	IMULL   $1000000000, CX

	MOVL    12(SP), AX		// sec - l32
	MOVL    $1000000000, BX
	MULL    BX			// result in dx:ax

	MOVL	20(SP), BX		// nsec
	ADDL	BX, AX
	ADCL	CX, DX			// add high bits with carry

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$-4
	MOVL	$46, AX			// sys_sigaction
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·obsdsigprocmask(SB),NOSPLIT,$-4
	MOVL	$48, AX			// sys_sigprocmask
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVL	$288, AX		// sys_sigaltstack
	MOVL	new+0(FP), BX
	MOVL	old+4(FP), CX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

TEXT set_tcb<>(SB),NOSPLIT,$8
	// adjust for ELF: wants to use -4(GS) for g
	MOVL	tlsbase+0(FP), CX
	ADDL	$4, CX
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	CX, 4(SP)		// arg 1 - tcb
	MOVL	$329, AX		// sys___set_tcb
	INT	$0x80
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$28
	LEAL	mib+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - name
	MOVSL				// arg 2 - namelen
	MOVSL				// arg 3 - oldp
	MOVSL				// arg 4 - oldlenp
	MOVSL				// arg 5 - newp
	MOVSL				// arg 6 - newlen
	MOVL	$202, AX		// sys___sysctl
	INT	$0x80
	JCC	4(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+24(FP)
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$269, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	$72, AX			// sys_kevent
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

// int32 runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$32
	MOVL	$92, AX			// sys_fcntl
	// 0(SP) is where the caller PC would be; kernel skips it
	MOVL	fd+0(FP), BX
	MOVL	BX, 4(SP)	// fd
	MOVL	$2, 8(SP)	// F_SETFD
	MOVL	$1, 12(SP)	// FD_CLOEXEC
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	RET

// func runtime·setNonblock(fd int32)
TEXT runtime·setNonblock(SB),NOSPLIT,$16-4
	MOVL	$92, AX // fcntl
	MOVL	fd+0(FP), BX // fd
	MOVL	BX, 4(SP)
	MOVL	$3, 8(SP) // F_GETFL
	MOVL	$0, 12(SP)
	INT	$0x80
	MOVL	fd+0(FP), BX // fd
	MOVL	BX, 4(SP)
	MOVL	$4, 8(SP) // F_SETFL
	ORL	$4, AX // O_NONBLOCK
	MOVL	AX, 12(SP)
	MOVL	$92, AX // fcntl
	INT	$0x80
	RET

GLOBL runtime·tlsoffset(SB),NOPTR,$4
