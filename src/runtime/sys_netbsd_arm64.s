// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm64, NetBSD
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "cgo/abi_arm64.h"

#define CLOCK_REALTIME		0
#define CLOCK_MONOTONIC		3
#define FD_CLOEXEC		1
#define F_SETFD			2
#define F_GETFL			3
#define F_SETFL			4
#define O_NONBLOCK		4

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
#define SYS_getcontext			307
#define SYS_setcontext			308
#define SYS__lwp_create			309
#define SYS__lwp_exit			310
#define SYS__lwp_self			311
#define SYS__lwp_kill			318
#define SYS__lwp_unpark			321
#define SYS___sigaction_sigtramp	340
#define SYS_kqueue			344
#define SYS_sched_yield			350
#define SYS___setitimer50		425
#define SYS___clock_gettime50		427
#define SYS___nanosleep50		430
#define SYS___kevent50			435
#define SYS_pipe2			453
#define SYS_openat			468
#define SYS____lwp_park60		478

// int32 lwp_create(void *context, uintptr flags, void *lwpid)
TEXT runtime·lwp_create(SB),NOSPLIT,$0
	MOVD	ctxt+0(FP), R0
	MOVD	flags+8(FP), R1
	MOVD	lwpid+16(FP), R2
	SVC	$SYS__lwp_create
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·lwp_tramp(SB),NOSPLIT,$0
	CMP	$0, R1
	BEQ	nog
	CMP	$0, R2
	BEQ	nog

	MOVD	R0, g_m(R1)
	MOVD	R1, g
nog:
	CALL	(R2)

	MOVD	$0, R0  // crash (not reached)
	MOVD	R0, (R8)

TEXT ·netbsdMstart(SB),NOSPLIT|TOPFRAME,$0
	CALL	·netbsdMstart0(SB)
	RET // not reached

TEXT runtime·osyield(SB),NOSPLIT,$0
	SVC	$SYS_sched_yield
	RET

TEXT runtime·lwp_park(SB),NOSPLIT,$0
	MOVW	clockid+0(FP), R0	// arg 1 - clockid
	MOVW	flags+4(FP), R1		// arg 2 - flags
	MOVD	ts+8(FP), R2		// arg 3 - ts
	MOVW	unpark+16(FP), R3	// arg 4 - unpark
	MOVD	hint+24(FP), R4		// arg 5 - hint
	MOVD	unparkhint+32(FP), R5	// arg 6 - unparkhint
	SVC	$SYS____lwp_park60
	MOVW	R0, ret+40(FP)
	RET

TEXT runtime·lwp_unpark(SB),NOSPLIT,$0
	MOVW	lwp+0(FP), R0		// arg 1 - lwp
	MOVD	hint+8(FP), R1		// arg 2 - hint
	SVC	$SYS__lwp_unpark
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·lwp_self(SB),NOSPLIT,$0
	SVC	$SYS__lwp_self
	MOVW	R0, ret+0(FP)
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVW	code+0(FP), R0		// arg 1 - exit status
	SVC	$SYS_exit
	MOVD	$0, R0			// If we're still running,
	MOVD	R0, (R0)		// crash

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-8
	MOVD	wait+0(FP), R0
	// We're done using the stack.
	MOVW	$0, R1
	STLRW	R1, (R0)
	SVC	$SYS__lwp_exit
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$-8
	MOVD	name+0(FP), R0		// arg 1 - pathname
	MOVW	mode+8(FP), R1		// arg 2 - flags
	MOVW	perm+12(FP), R2		// arg 3 - mode
	SVC	$SYS_open
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-8
	MOVW	fd+0(FP), R0		// arg 1 - fd
	SVC	$SYS_close
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVD	p+8(FP), R1		// arg 2 - buf
	MOVW	n+16(FP), R2		// arg 3 - count
	SVC	$SYS_read
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+24(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	ADD	$16, RSP, R0
	MOVW	flags+0(FP), R1
	SVC	$SYS_pipe2
	BCC	pipe2ok
	NEG	R0, R0
pipe2ok:
	MOVW	R0, errno+16(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$-8
	MOVD	fd+0(FP), R0		// arg 1 - fd
	MOVD	p+8(FP), R1		// arg 2 - buf
	MOVW	n+16(FP), R2		// arg 3 - nbyte
	SVC	$SYS_write
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$24-4
	MOVWU	usec+0(FP), R3
	MOVD	R3, R5
	MOVW	$1000000, R4
	UDIV	R4, R3
	MOVD	R3, 8(RSP)		// sec
	MUL	R3, R4
	SUB	R4, R5
	MOVW	$1000, R4
	MUL	R4, R5
	MOVD	R5, 16(RSP)		// nsec

	MOVD	$8(RSP), R0		// arg 1 - rqtp
	MOVD	$0, R1			// arg 2 - rmtp
	SVC	$SYS___nanosleep50
	RET

TEXT runtime·lwp_kill(SB),NOSPLIT,$0-16
	MOVW	tid+0(FP), R0		// arg 1 - target
	MOVD	sig+8(FP), R1		// arg 2 - signo
	SVC	$SYS__lwp_kill
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	SVC	$SYS_getpid
					// arg 1 - pid (from getpid)
	MOVD	sig+0(FP), R1		// arg 2 - signo
	SVC	$SYS_kill
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-8
	MOVW	mode+0(FP), R0		// arg 1 - which
	MOVD	new+8(FP), R1		// arg 2 - itv
	MOVD	old+16(FP), R2		// arg 3 - oitv
	SVC	$SYS___setitimer50
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	MOVW	$CLOCK_REALTIME, R0	// arg 1 - clock_id
	MOVD	$8(RSP), R1		// arg 2 - tp
	SVC	$SYS___clock_gettime50

	MOVD	8(RSP), R0		// sec
	MOVD	16(RSP), R1		// nsec

	// sec is in R0, nsec in R1
	MOVD	R0, sec+0(FP)
	MOVW	R1, nsec+8(FP)
	RET

// int64 nanotime1(void) so really
// void nanotime1(int64 *nsec)
TEXT runtime·nanotime1(SB), NOSPLIT, $32
	MOVD	$CLOCK_MONOTONIC, R0	// arg 1 - clock_id
	MOVD	$8(RSP), R1		// arg 2 - tp
	SVC	$SYS___clock_gettime50
	MOVD	8(RSP), R0		// sec
	MOVD	16(RSP), R2		// nsec

	// sec is in R0, nsec in R2
	// return nsec in R2
	MOVD	$1000000000, R3
	MUL	R3, R0
	ADD	R2, R0

	MOVD	R0, ret+0(FP)
	RET

TEXT runtime·getcontext(SB),NOSPLIT,$-8
	MOVD	ctxt+0(FP), R0		// arg 1 - context
	SVC	$SYS_getcontext
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)		// crash

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW	how+0(FP), R0		// arg 1 - how
	MOVD	new+8(FP), R1		// arg 2 - set
	MOVD	old+16(FP), R2		// arg 3 - oset
	SVC	$SYS___sigprocmask14
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)		// crash

TEXT sigreturn_tramp<>(SB),NOSPLIT,$-8
	MOVD	g, R0
	SVC	$SYS_setcontext
	MOVD	$0, R0
	MOVD	R0, (R0)		// crash

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVW	sig+0(FP), R0		// arg 1 - signum
	MOVD	new+8(FP), R1		// arg 2 - nsa
	MOVD	old+16(FP), R2		// arg 3 - osa
					// arg 4 - tramp
	MOVD	$sigreturn_tramp<>(SB), R3
	MOVW	$2, R4			// arg 5 - vers
	SVC	$SYS___sigaction_sigtramp
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)		// crash

// XXX ???
TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$176
	// Save callee-save registers in the case of signal forwarding.
	// Please refer to https://golang.org/issue/31827 .
	SAVE_R19_TO_R28(8*4)
	SAVE_F8_TO_F15(8*14)
	// Unclobber g for now (kernel uses it as ucontext ptr)
	// See https://github.com/golang/go/issues/30824#issuecomment-492772426
	// This is only correct in the non-cgo case.
	// XXX should use lwp_getprivate as suggested.
	// 8*36 is ucontext.uc_mcontext.__gregs[_REG_X28]
	MOVD	8*36(g), g

	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVD	R0, 8(RSP)		// signum
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	// XXX branch destination
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	MOVD	R1, 16(RSP)
	MOVD	R2, 24(RSP)
	BL	runtime·sigtrampgo(SB)

	// Restore callee-save registers.
	RESTORE_R19_TO_R28(8*4)
	RESTORE_F8_TO_F15(8*14)

	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0		// arg 1 - addr
	MOVD	n+8(FP), R1		// arg 2 - len
	MOVW	prot+16(FP), R2		// arg 3 - prot
	MOVW	flags+20(FP), R3	// arg 4 - flags
	MOVW	fd+24(FP), R4		// arg 5 - fd
	MOVW	$0, R5			// arg 6 - pad
	MOVD	off+28(FP), R6		// arg 7 - offset
	SVC	$SYS_mmap
	BCS	fail
	MOVD	R0, p+32(FP)
	MOVD	$0, err+40(FP)
	RET
fail:
	MOVD	$0, p+32(FP)
	MOVD	R0, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0	// arg 1 - addr
	MOVD	n+8(FP), R1	// arg 2 - len
	SVC	$SYS_munmap
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0		// arg 1 - addr
	MOVD	n+8(FP), R1		// arg 2 - len
	MOVW	flags+16(FP), R2	// arg 3 - behav
	SVC	$SYS_madvise
	BCC	ok
	MOVD	$-1, R0
ok:
	MOVD	R0, ret+24(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVD	new+0(FP), R0		// arg 1 - nss
	MOVD	old+8(FP), R1		// arg 2 - oss
	SVC	$SYS___sigaltstack14
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)		// crash

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVD	mib+0(FP), R0		// arg 1 - name
	MOVW	miblen+8(FP), R1	// arg 2 - namelen
	MOVD	out+16(FP), R2		// arg 3 - oldp
	MOVD	size+24(FP), R3		// arg 4 - oldlenp
	MOVD	dst+32(FP), R4		// arg 5 - newp
	MOVD	ndst+40(FP), R5		// arg 6 - newlen
	SVC	$SYS___sysctl
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+48(FP)
	RET

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVD	$0, R0
	SVC	$SYS_kqueue
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW	kq+0(FP), R0	// arg 1 - kq
	MOVD	ch+8(FP), R1	// arg 2 - changelist
	MOVW	nch+16(FP), R2	// arg 3 - nchanges
	MOVD	ev+24(FP), R3	// arg 4 - eventlist
	MOVW	nev+32(FP), R4	// arg 5 - nevents
	MOVD	ts+40(FP), R5	// arg 6 - timeout
	SVC	$SYS___kevent50
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+48(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVW	$F_SETFD, R1
	MOVW	$FD_CLOEXEC, R2
	SVC	$SYS_fcntl
	RET
