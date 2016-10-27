// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define AT_FDCWD -100

#define SYS_exit		93
#define SYS_read		63
#define SYS_write		64
#define SYS_openat		56
#define SYS_close		57
#define SYS_fcntl		25
#define SYS_gettimeofday	169
#define SYS_pselect6		72
#define SYS_mmap		222
#define SYS_munmap		215
#define SYS_setitimer		103
#define SYS_clone		220
#define SYS_sched_yield		124
#define SYS_rt_sigreturn	139
#define SYS_rt_sigaction	134
#define SYS_rt_sigprocmask	135
#define SYS_sigaltstack		132
#define SYS_getrlimit		163
#define SYS_madvise		233
#define SYS_mincore		232
#define SYS_getpid		172
#define SYS_gettid		178
#define SYS_kill		129
#define SYS_tkill		130
#define SYS_futex		98
#define SYS_sched_getaffinity	123
#define SYS_exit_group		94
#define SYS_epoll_create1	20
#define SYS_epoll_ctl		21
#define SYS_epoll_pwait		22
#define SYS_clock_gettime	113
#define SYS_faccessat		48
#define SYS_socket		198
#define SYS_connect		203

TEXT runtime·exit(SB),NOSPLIT,$-8-4
	MOVW	code+0(FP), R0
	MOVD	$SYS_exit_group, R8
	SVC
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8-4
	MOVW	code+0(FP), R0
	MOVD	$SYS_exit, R8
	SVC
	RET

TEXT runtime·open(SB),NOSPLIT,$-8-20
	MOVD	$AT_FDCWD, R0
	MOVD	name+0(FP), R1
	MOVW	mode+8(FP), R2
	MOVW	perm+12(FP), R3
	MOVD	$SYS_openat, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVW	$-1, R0
done:
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-8-12
	MOVW	fd+0(FP), R0
	MOVD	$SYS_close, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVW	$-1, R0
done:
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-8-28
	MOVD	fd+0(FP), R0
	MOVD	p+8(FP), R1
	MOVW	n+16(FP), R2
	MOVD	$SYS_write, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVW	$-1, R0
done:
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-8-28
	MOVW	fd+0(FP), R0
	MOVD	p+8(FP), R1
	MOVW	n+16(FP), R2
	MOVD	$SYS_read, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVW	$-1, R0
done:
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$-8-20
	MOVW	kind+0(FP), R0
	MOVD	limit+8(FP), R1
	MOVD	$SYS_getrlimit, R8
	SVC
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$24-4
	MOVWU	usec+0(FP), R3
	MOVD	R3, R5
	MOVW	$1000000, R4
	UDIV	R4, R3
	MOVD	R3, 8(RSP)
	MUL	R3, R4
	SUB	R4, R5
	MOVW	$1000, R4
	MUL	R4, R5
	MOVD	R5, 16(RSP)

	// pselect6(0, 0, 0, 0, &ts, 0)
	MOVD	$0, R0
	MOVD	R0, R1
	MOVD	R0, R2
	MOVD	R0, R3
	ADD	$8, RSP, R4
	MOVD	R0, R5
	MOVD	$SYS_pselect6, R8
	SVC
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVD	$SYS_gettid, R8
	SVC
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$-8
	MOVD	$SYS_gettid, R8
	SVC
	MOVW	R0, R0	// arg 1 tid
	MOVW	sig+0(FP), R1	// arg 2
	MOVD	$SYS_tkill, R8
	SVC
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$-8
	MOVD	$SYS_getpid, R8
	SVC
	MOVW	R0, R0		// arg 1 pid
	MOVW	sig+0(FP), R1	// arg 2
	MOVD	$SYS_kill, R8
	SVC
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-8-24
	MOVW	mode+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	$SYS_setitimer, R8
	SVC
	RET

TEXT runtime·mincore(SB),NOSPLIT,$-8-28
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	dst+16(FP), R2
	MOVD	$SYS_mincore, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB),NOSPLIT,$24-12
	MOVW	$0, R0 // CLOCK_REALTIME
	MOVD	RSP, R1
	MOVD	$SYS_clock_gettime, R8
	SVC
	MOVD	0(RSP), R3	// sec
	MOVD	8(RSP), R5	// nsec
	MOVD	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB),NOSPLIT,$24-8
	MOVW	$1, R0 // CLOCK_MONOTONIC
	MOVD	RSP, R1
	MOVD	$SYS_clock_gettime, R8
	SVC
	MOVD	0(RSP), R3	// sec
	MOVD	8(RSP), R5	// nsec
	// sec is in R3, nsec in R5
	// return nsec in R3
	MOVD	$1000000000, R4
	MUL	R4, R3
	ADD	R5, R3
	MOVD	R3, ret+0(FP)
	RET

TEXT runtime·rtsigprocmask(SB),NOSPLIT,$-8-28
	MOVW	how+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVW	size+24(FP), R3
	MOVD	$SYS_rt_sigprocmask, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash
done:
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT,$-8-36
	MOVD	sig+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	size+24(FP), R3
	MOVD	$SYS_rt_sigaction, R8
	SVC
	MOVW	R0, ret+32(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$24
	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVW	R0, 8(RSP)
	MOVBU	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	MOVD	R1, 16(RSP)
	MOVD	R2, 24(RSP)
	MOVD	$runtime·sigtrampgo(SB), R0
	BL	(R0)
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	MOVD	$runtime·sigtramp(SB), R3
	B	(R3)

TEXT runtime·mmap(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	prot+16(FP), R2
	MOVW	flags+20(FP), R3
	MOVW	fd+24(FP), R4
	MOVW	off+28(FP), R5

	MOVD	$SYS_mmap, R8
	SVC
	CMN	$4095, R0
	BCC	2(PC)
	NEG	R0,R0
	MOVD	R0, ret+32(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	$SYS_munmap, R8
	SVC
	CMN	$4095, R0
	BCC	cool
	MOVD	R0, 0xf0(R0)
cool:
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	flags+16(FP), R2
	MOVD	$SYS_madvise, R8
	SVC
	// ignore failure - maybe pages are locked
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$-8
	MOVD	addr+0(FP), R0
	MOVW	op+8(FP), R1
	MOVW	val+12(FP), R2
	MOVD	ts+16(FP), R3
	MOVD	addr2+24(FP), R4
	MOVW	val3+32(FP), R5
	MOVD	$SYS_futex, R8
	SVC
	MOVW	R0, ret+40(FP)
	RET

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT,$-8
	MOVW	flags+0(FP), R0
	MOVD	stk+8(FP), R1

	// Copy mp, gp, fn off parent stack for use by child.
	MOVD	mp+16(FP), R10
	MOVD	gp+24(FP), R11
	MOVD	fn+32(FP), R12

	MOVD	R10, -8(R1)
	MOVD	R11, -16(R1)
	MOVD	R12, -24(R1)
	MOVD	$1234, R10
	MOVD	R10, -32(R1)

	MOVD	$SYS_clone, R8
	SVC

	// In parent, return.
	CMP	ZR, R0
	BEQ	child
	MOVW	R0, ret+40(FP)
	RET
child:

	// In child, on new stack.
	MOVD	-32(RSP), R10
	MOVD	$1234, R0
	CMP	R0, R10
	BEQ	good
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash

good:
	// Initialize m->procid to Linux tid
	MOVD	$SYS_gettid, R8
	SVC

	MOVD	-24(RSP), R12     // fn
	MOVD	-16(RSP), R11     // g
	MOVD	-8(RSP), R10      // m

	CMP	$0, R10
	BEQ	nog
	CMP	$0, R11
	BEQ	nog

	MOVD	R0, m_procid(R10)

	// TODO: setup TLS.

	// In child, set up new stack
	MOVD	R10, g_m(R11)
	MOVD	R11, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	MOVD	R12, R0
	BL	(R0)

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R0
again:
	MOVD	$SYS_exit, R8
	SVC
	B	again	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVD	new+0(FP), R0
	MOVD	old+8(FP), R1
	MOVD	$SYS_sigaltstack, R8
	SVC
	CMN	$4095, R0
	BCC	ok
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash
ok:
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-8
	MOVD	$SYS_sched_yield, R8
	SVC
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$-8
	MOVD	pid+0(FP), R0
	MOVD	len+8(FP), R1
	MOVD	buf+16(FP), R2
	MOVD	$SYS_sched_getaffinity, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT,$-8
	MOVW	$0, R0
	MOVD	$SYS_epoll_create1, R8
	SVC
	MOVW	R0, ret+8(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT,$-8
	MOVW	flags+0(FP), R0
	MOVD	$SYS_epoll_create1, R8
	SVC
	MOVW	R0, ret+8(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$-8
	MOVW	epfd+0(FP), R0
	MOVW	op+4(FP), R1
	MOVW	fd+8(FP), R2
	MOVD	ev+16(FP), R3
	MOVD	$SYS_epoll_ctl, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT,$-8
	MOVW	epfd+0(FP), R0
	MOVD	ev+8(FP), R1
	MOVW	nev+16(FP), R2
	MOVW	timeout+20(FP), R3
	MOVD	$0, R4
	MOVD	$SYS_epoll_pwait, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$-8
	MOVW	fd+0(FP), R0  // fd
	MOVD	$2, R1	// F_SETFD
	MOVD	$1, R2	// FD_CLOEXEC
	MOVD	$SYS_fcntl, R8
	SVC
	RET

// int access(const char *name, int mode)
TEXT runtime·access(SB),NOSPLIT,$0-20
	MOVD	$AT_FDCWD, R0
	MOVD	name+0(FP), R1
	MOVW	mode+8(FP), R2
	MOVD	$SYS_faccessat, R8
	SVC
	MOVW	R0, ret+16(FP)
	RET

// int connect(int fd, const struct sockaddr *addr, socklen_t len)
TEXT runtime·connect(SB),NOSPLIT,$0-28
	MOVW	fd+0(FP), R0
	MOVD	addr+8(FP), R1
	MOVW	len+16(FP), R2
	MOVD	$SYS_connect, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int socket(int domain, int typ, int prot)
TEXT runtime·socket(SB),NOSPLIT,$0-20
	MOVW	domain+0(FP), R0
	MOVW	typ+4(FP), R1
	MOVW	prot+8(FP), R2
	MOVD	$SYS_socket, R8
	SVC
	MOVW	R0, ret+16(FP)
	RET
