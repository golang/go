// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, NetBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME		0
#define CLOCK_MONOTONIC		3

#define SYS_exit			1
#define SYS_read			3
#define SYS_write			4
#define SYS_open			5
#define SYS_close			6
#define SYS_getpid			20
#define SYS_kill			37
#define SYS_munmap			73
#define SYS_madvise			75
#define SYS_fcntl			92
#define SYS_mmap			197
#define SYS___sysctl			202
#define SYS___sigaltstack14		281
#define SYS___sigprocmask14		293
#define SYS_issetugid			305
#define SYS_getcontext			307
#define SYS_setcontext			308
#define SYS__lwp_create			309
#define SYS__lwp_exit			310
#define SYS__lwp_self			311
#define SYS__lwp_setprivate		317
#define SYS__lwp_kill			318
#define SYS__lwp_unpark			321
#define SYS___sigaction_sigtramp	340
#define SYS_kqueue			344
#define SYS_sched_yield			350
#define SYS___setitimer50		425
#define SYS___clock_gettime50		427
#define SYS___nanosleep50		430
#define SYS___kevent50			435
#define SYS____lwp_park60		478

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVL	$SYS_exit, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	RET

// func exitThread(wait *atomic.Uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVL	wait+0(FP), AX
	// We're done using the stack.
	MOVL	$0, (AX)
	MOVL	$SYS__lwp_exit, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT,$-4
	MOVL	$SYS_open, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-4
	MOVL	$SYS_close, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-4
	MOVL	$SYS_read, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+12(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$12-16
	MOVL	$453, AX
	LEAL	r+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	flags+0(FP), BX
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	AX, errno+12(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$-4
	MOVL	$SYS_write, AX
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
	MOVL	$SYS___nanosleep50, AX
	INT	$0x80
	RET

TEXT runtime·lwp_kill(SB),NOSPLIT,$12-8
	MOVL	$0, 0(SP)
	MOVL	tid+0(FP), AX
	MOVL	AX, 4(SP)		// arg 1 - target
	MOVL	sig+4(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signo
	MOVL	$SYS__lwp_kill, AX
	INT	$0x80
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVL	$SYS_getpid, AX
	INT	$0x80
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)		// arg 1 - pid
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signo
	MOVL	$SYS_kill, AX
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
	MOVL	$SYS_mmap, AX
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
	MOVL	$SYS_munmap, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-4
	MOVL	$SYS_madvise, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-4
	MOVL	$SYS___setitimer50, AX
	INT	$0x80
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	LEAL	12(SP), BX
	MOVL	$CLOCK_REALTIME, 4(SP)	// arg 1 - clock_id
	MOVL	BX, 8(SP)		// arg 2 - tp
	MOVL	$SYS___clock_gettime50, AX
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
	MOVL	$CLOCK_MONOTONIC, 4(SP)	// arg 1 - clock_id
	MOVL	BX, 8(SP)		// arg 2 - tp
	MOVL	$SYS___clock_gettime50, AX
	INT	$0x80

	MOVL	16(SP), CX		// sec - h32
	IMULL	$1000000000, CX

	MOVL	12(SP), AX		// sec - l32
	MOVL	$1000000000, BX
	MULL	BX			// result in dx:ax

	MOVL	20(SP), BX		// nsec
	ADDL	BX, AX
	ADCL	CX, DX			// add high bits with carry

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET

TEXT runtime·getcontext(SB),NOSPLIT,$-4
	MOVL	$SYS_getcontext, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$-4
	MOVL	$SYS___sigprocmask14, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT sigreturn_tramp<>(SB),NOSPLIT,$0
	LEAL	140(SP), AX		// Load address of ucontext
	MOVL	AX, 4(SP)
	MOVL	$SYS_setcontext, AX
	INT	$0x80
	MOVL	$-1, 4(SP)		// Something failed...
	MOVL	$SYS_exit, AX
	INT	$0x80

TEXT runtime·sigaction(SB),NOSPLIT,$24
	LEAL	sig+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - sig
	MOVSL				// arg 2 - act
	MOVSL				// arg 3 - oact
	LEAL	sigreturn_tramp<>(SB), AX
	STOSL				// arg 4 - tramp
	MOVL	$2, AX
	STOSL				// arg 5 - vers
	MOVL	$SYS___sigaction_sigtramp, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$12-16
	MOVL	fn+0(FP), AX
	MOVL	sig+4(FP), BX
	MOVL	info+8(FP), CX
	MOVL	ctx+12(FP), DX
	MOVL	SP, SI
	SUBL	$32, SP
	ANDL	$-15, SP	// align stack: handler might be a C function
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)	// save SI: handler might be a Go function
	CALL	AX
	MOVL	12(SP), AX
	MOVL	AX, SP
	RET

// Called by OS using C ABI.
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$28
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

// int32 lwp_create(void *context, uintptr flags, void *lwpid);
TEXT runtime·lwp_create(SB),NOSPLIT,$16
	MOVL	$0, 0(SP)
	MOVL	ctxt+0(FP), AX
	MOVL	AX, 4(SP)		// arg 1 - context
	MOVL	flags+4(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - flags
	MOVL	lwpid+8(FP), AX
	MOVL	AX, 12(SP)		// arg 3 - lwpid
	MOVL	$SYS__lwp_create, AX
	INT	$0x80
	JCC	2(PC)
	NEGL	AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·lwp_tramp(SB),NOSPLIT,$0

	// Set FS to point at m->tls
	LEAL	m_tls(BX), BP
	PUSHAL				// save registers
	PUSHL	BP
	CALL	lwp_setprivate<>(SB)
	POPL	AX
	POPAL

	// Now segment is established. Initialize m, g.
	get_tls(AX)
	MOVL	DX, g(AX)
	MOVL	BX, g_m(DX)

	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	MOVL	0(DX), DX		// paranoia; check they are not nil
	MOVL	0(BX), BX

	// more paranoia; check that stack splitting code works
	PUSHAL
	CALL	runtime·emptyfunc(SB)
	POPAL

	// Call fn
	CALL	SI

	// fn should never return
	MOVL	$0x1234, 0x1005
	RET

TEXT ·netbsdMstart(SB),NOSPLIT|TOPFRAME,$0
	CALL	·netbsdMstart0(SB)
	RET // not reached

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVL	$SYS___sigaltstack14, AX
	MOVL	new+0(FP), BX
	MOVL	old+4(FP), CX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

TEXT runtime·setldt(SB),NOSPLIT,$8
	// Under NetBSD we set the GS base instead of messing with the LDT.
	MOVL	base+4(FP), AX
	MOVL	AX, 0(SP)
	CALL	lwp_setprivate<>(SB)
	RET

TEXT lwp_setprivate<>(SB),NOSPLIT,$16
	// adjust for ELF: wants to use -4(GS) for g
	MOVL	base+0(FP), CX
	ADDL	$4, CX
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	CX, 4(SP)		// arg 1 - ptr
	MOVL	$SYS__lwp_setprivate, AX
	INT	$0x80
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$SYS_sched_yield, AX
	INT	$0x80
	RET

TEXT runtime·lwp_park(SB),NOSPLIT,$-4
	MOVL	$SYS____lwp_park60, AX
	INT	$0x80
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·lwp_unpark(SB),NOSPLIT,$-4
	MOVL	$SYS__lwp_unpark, AX
	INT	$0x80
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·lwp_self(SB),NOSPLIT,$-4
	MOVL	$SYS__lwp_self, AX
	INT	$0x80
	MOVL	AX, ret+0(FP)
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
	MOVL	$SYS___sysctl, AX
	INT	$0x80
	JAE	4(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+24(FP)
	RET

GLOBL runtime·tlsoffset(SB),NOPTR,$4

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$SYS_kqueue, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	$SYS___kevent50, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

// func fcntl(fd, cmd, arg int32) (int32, int32)
TEXT runtime·fcntl(SB),NOSPLIT,$-4
	MOVL	$SYS_fcntl, AX
	INT	$0x80
	JAE	noerr
	MOVL	$-1, ret+12(FP)
	MOVL	AX, errno+16(FP)
	RET
noerr:
	MOVL	AX, ret+12(FP)
	MOVL	$0, errno+16(FP)
	RET

// func issetugid() int32
TEXT runtime·issetugid(SB),NOSPLIT,$0
	MOVL	$SYS_issetugid, AX
	INT	$0x80
	MOVL	AX, ret+0(FP)
	RET
