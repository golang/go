// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build ppc64 ppc64le

//
// System calls and other sys.stuff for ppc64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define SYS_exit		  1
#define SYS_read		  3
#define SYS_write		  4
#define SYS_open		  5
#define SYS_close		  6
#define SYS_getpid		 20
#define SYS_kill		 37
#define SYS_fcntl		 55
#define SYS_gettimeofday	 78
#define SYS_select		 82	// always return -ENOSYS
#define SYS_mmap		 90
#define SYS_munmap		 91
#define SYS_setitimer		104
#define SYS_clone		120
#define SYS_newselect		142
#define SYS_sched_yield		158
#define SYS_rt_sigreturn	172
#define SYS_rt_sigaction	173
#define SYS_rt_sigprocmask	174
#define SYS_sigaltstack		185
#define SYS_ugetrlimit		190
#define SYS_madvise		205
#define SYS_mincore		206
#define SYS_gettid		207
#define SYS_tkill		208
#define SYS_futex		221
#define SYS_sched_getaffinity	223
#define SYS_exit_group		234
#define SYS_epoll_create	236
#define SYS_epoll_ctl		237
#define SYS_epoll_wait		238
#define SYS_clock_gettime	246
#define SYS_epoll_create1	315

TEXT runtime·exit(SB),NOSPLIT,$-8-4
	MOVW	code+0(FP), R3
	SYSCALL	$SYS_exit_group
	RETURN

TEXT runtime·exit1(SB),NOSPLIT,$-8-4
	MOVW	code+0(FP), R3
	SYSCALL	$SYS_exit
	RETURN

TEXT runtime·open(SB),NOSPLIT,$-8-20
	MOVD	name+0(FP), R3
	MOVW	mode+8(FP), R4
	MOVW	perm+12(FP), R5
	SYSCALL	$SYS_open
	MOVW	R3, ret+16(FP)
	RETURN

TEXT runtime·close(SB),NOSPLIT,$-8-12
	MOVW	fd+0(FP), R3
	SYSCALL	$SYS_close
	MOVW	R3, ret+8(FP)
	RETURN

TEXT runtime·write(SB),NOSPLIT,$-8-28
	MOVD	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	SYSCALL	$SYS_write
	MOVW	R3, ret+24(FP)
	RETURN

TEXT runtime·read(SB),NOSPLIT,$-8-28
	MOVW	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	SYSCALL	$SYS_read
	MOVW	R3, ret+24(FP)
	RETURN

TEXT runtime·getrlimit(SB),NOSPLIT,$-8-20
	MOVW	kind+0(FP), R3
	MOVD	limit+8(FP), R4
	SYSCALL	$SYS_ugetrlimit
	MOVW	R3, ret+16(FP)
	RETURN

TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVW	usec+0(FP), R3
	MOVD	R3, R5
	MOVW	$1000000, R4
	DIVD	R4, R3
	MOVD	R3, 8(R1)
	MULLD	R3, R4
	SUB	R4, R5
	MOVD	R5, 16(R1)

	// select(0, 0, 0, 0, &tv)
	MOVW	$0, R3
	MOVW	$0, R4
	MOVW	$0, R5
	MOVW	$0, R6
	ADD	$8, R1, R7
	SYSCALL	$SYS_newselect
	RETURN

TEXT runtime·raise(SB),NOSPLIT,$-8
	SYSCALL	$SYS_gettid
	MOVW	R3, R3	// arg 1 tid
	MOVW	sig+0(FP), R4	// arg 2
	SYSCALL	$SYS_tkill
	RETURN

TEXT runtime·raiseproc(SB),NOSPLIT,$-8
	SYSCALL	$SYS_getpid
	MOVW	R3, R3	// arg 1 pid
	MOVW	sig+0(FP), R4	// arg 2
	SYSCALL	$SYS_kill
	RETURN

TEXT runtime·setitimer(SB),NOSPLIT,$-8-24
	MOVW	mode+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	SYSCALL	$SYS_setitimer
	RETURN

TEXT runtime·mincore(SB),NOSPLIT,$-8-28
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	MOVD	dst+16(FP), R5
	SYSCALL	$SYS_mincore
	MOVW	R3, ret+24(FP)
	RETURN

// func now() (sec int64, nsec int32)
TEXT time·now(SB),NOSPLIT,$16
	MOVD	$0(R1), R3
	MOVD	$0, R4
	SYSCALL	$SYS_gettimeofday
	MOVD	0(R1), R3	// sec
	MOVD	8(R1), R5	// usec
	MOVD	$1000, R4
	MULLD	R4, R5
	MOVD	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RETURN

TEXT runtime·nanotime(SB),NOSPLIT,$16
	MOVW	$1, R3 // CLOCK_MONOTONIC
	MOVD	$0(R1), R4
	SYSCALL	$SYS_clock_gettime
	MOVD	0(R1), R3	// sec
	MOVD	8(R1), R5	// nsec
	// sec is in R3, nsec in R5
	// return nsec in R3
	MOVD	$1000000000, R4
	MULLD	R4, R3
	ADD	R5, R3
	MOVD	R3, ret+0(FP)
	RETURN

TEXT runtime·rtsigprocmask(SB),NOSPLIT,$-8-28
	MOVW	sig+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVW	size+24(FP), R6
	SYSCALL	$SYS_rt_sigprocmask
	BVC	2(PC)
	MOVD	R0, 0xf1(R0)	// crash
	RETURN

TEXT runtime·rt_sigaction(SB),NOSPLIT,$-8-36
	MOVD	sig+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVD	size+24(FP), R6
	SYSCALL	$SYS_rt_sigaction
	MOVW	R3, ret+32(FP)
	RETURN

#ifdef GOARCH_ppc64le
// ppc64le doesn't need function descriptors
TEXT runtime·sigtramp(SB),NOSPLIT,$64
#else
// function descriptor for the real sigtramp
TEXT runtime·sigtramp(SB),NOSPLIT,$-8
	DWORD	$runtime·_sigtramp(SB)
	DWORD	$0
	DWORD	$0
TEXT runtime·_sigtramp(SB),NOSPLIT,$64
#endif
	// initialize essential registers (just in case)
	BL	runtime·reginit(SB)

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R6
	CMP 	R6, $0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	// check that g exists
	CMP	g, $0
	BNE	6(PC)
	MOVD	R3, 8(R1)
	MOVD	$runtime·badsignal(SB), R31
	MOVD	R31, CTR
	BL	(CTR)
	RETURN

	// save g
	MOVD	g, 40(R1)
	MOVD	g, R6

	// g = m->gsignal
	MOVD	g_m(g), R7
	MOVD	m_gsignal(R7), g

	MOVW	R3, 8(R1)
	MOVD	R4, 16(R1)
	MOVD	R5, 24(R1)
	MOVD	R6, 32(R1)

	BL	runtime·sighandler(SB)

	// restore g
	MOVD	40(R1), g

	RETURN

TEXT runtime·mmap(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	MOVW	prot+16(FP), R5
	MOVW	flags+20(FP), R6
	MOVW	fd+24(FP), R7
	MOVW	off+28(FP), R8

	SYSCALL	$SYS_mmap
	MOVD	R3, ret+32(FP)
	RETURN

TEXT runtime·munmap(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	SYSCALL	$SYS_munmap
	BVC	2(PC)
	MOVD	R0, 0xf3(R0)
	RETURN

TEXT runtime·madvise(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	MOVW	flags+16(FP), R5
	SYSCALL	$SYS_madvise
	// ignore failure - maybe pages are locked
	RETURN

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R3
	MOVW	op+8(FP), R4
	MOVW	val+12(FP), R5
	MOVD	ts+16(FP), R6
	MOVD	addr2+24(FP), R7
	MOVW	val3+32(FP), R8
	SYSCALL	$SYS_futex
	MOVW	R3, ret+40(FP)
	RETURN

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT,$-8
	MOVW	flags+0(FP), R3
	MOVD	stk+8(FP), R4

	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVD	mm+16(FP), R7
	MOVD	gg+24(FP), R8
	MOVD	fn+32(FP), R12

	MOVD	R7, -8(R4)
	MOVD	R8, -16(R4)
	MOVD	R12, -24(R4)
	MOVD	$1234, R7
	MOVD	R7, -32(R4)

	SYSCALL $SYS_clone

	// In parent, return.
	CMP	R3, $0
	BEQ	3(PC)
	MOVW	R3, ret+40(FP)
	RETURN

	// In child, on new stack.
	// initialize essential registers
	BL	runtime·reginit(SB)
	MOVD	-32(R1), R7
	CMP	R7, $1234
	BEQ	2(PC)
	MOVD	R0, 0(R0)

	// Initialize m->procid to Linux tid
	SYSCALL $SYS_gettid

	MOVD	-24(R1), R12
	MOVD	-16(R1), R8
	MOVD	-8(R1), R7

	MOVD	R3, m_procid(R7)

	// TODO: setup TLS.

	// In child, set up new stack
	MOVD	R7, g_m(R8)
	MOVD	R8, g
	//CALL	runtime·stackcheck(SB)

	// Call fn
	MOVD	R12, CTR
	BL	(CTR)

	// It shouldn't return.  If it does, exit
	MOVW	$111, R3
	SYSCALL $SYS_exit_group
	BR	-2(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVD	new+0(FP), R3
	MOVD	old+8(FP), R4
	SYSCALL	$SYS_sigaltstack
	BVC	2(PC)
	MOVD	R0, 0xf1(R0)  // crash
	RETURN

TEXT runtime·osyield(SB),NOSPLIT,$-8
	SYSCALL	$SYS_sched_yield
	RETURN

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$-8
	MOVD	pid+0(FP), R3
	MOVD	len+8(FP), R4
	MOVD	buf+16(FP), R5
	SYSCALL	$SYS_sched_getaffinity
	MOVW	R3, ret+24(FP)
	RETURN

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT,$-8
	MOVW    size+0(FP), R3
	SYSCALL	$SYS_epoll_create
	MOVW	R3, ret+8(FP)
	RETURN

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT,$-8
	MOVW	flags+0(FP), R3
	SYSCALL	$SYS_epoll_create1
	MOVW	R3, ret+8(FP)
	RETURN

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$-8
	MOVW	epfd+0(FP), R3
	MOVW	op+4(FP), R4
	MOVW	fd+8(FP), R5
	MOVD	ev+16(FP), R6
	SYSCALL	$SYS_epoll_ctl
	MOVW	R3, ret+24(FP)
	RETURN

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT,$-8
	MOVW	epfd+0(FP), R3
	MOVD	ev+8(FP), R4
	MOVW	nev+16(FP), R5
	MOVW	timeout+20(FP), R6
	SYSCALL	$SYS_epoll_wait
	MOVW	R3, ret+24(FP)
	RETURN

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$-8
	MOVW    fd+0(FP), R3  // fd
	MOVD    $2, R4  // F_SETFD
	MOVD    $1, R5  // FD_CLOEXEC
	SYSCALL	$SYS_fcntl
	RETURN
