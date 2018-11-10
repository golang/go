// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips64 mips64le

//
// System calls and other sys.stuff for mips64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define SYS_exit		5058
#define SYS_read		5000
#define SYS_write		5001
#define SYS_open		5002
#define SYS_close		5003
#define SYS_getpid		5038
#define SYS_kill		5060
#define SYS_fcntl		5080
#define SYS_gettimeofday	5094
#define SYS_mmap		5009
#define SYS_munmap		5011
#define SYS_setitimer		5036
#define SYS_clone		5055
#define SYS_newselect		5022
#define SYS_sched_yield		5023
#define SYS_rt_sigreturn	5211
#define SYS_rt_sigaction	5013
#define SYS_rt_sigprocmask	5014
#define SYS_sigaltstack		5129
#define SYS_getrlimit		5095
#define SYS_madvise		5027
#define SYS_mincore		5026
#define SYS_gettid		5178
#define SYS_tkill		5192
#define SYS_futex		5194
#define SYS_sched_getaffinity	5196
#define SYS_exit_group		5205
#define SYS_epoll_create	5207
#define SYS_epoll_ctl		5208
#define SYS_epoll_wait		5209
#define SYS_clock_gettime	5222
#define SYS_epoll_create1	5285
#define SYS_brk			5012

TEXT runtime·exit(SB),NOSPLIT,$-8-4
	MOVW	code+0(FP), R4
	MOVV	$SYS_exit_group, R2
	SYSCALL
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8-4
	MOVW	code+0(FP), R4
	MOVV	$SYS_exit, R2
	SYSCALL
	RET

TEXT runtime·open(SB),NOSPLIT,$-8-20
	MOVV	name+0(FP), R4
	MOVW	mode+8(FP), R5
	MOVW	perm+12(FP), R6
	MOVV	$SYS_open, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-8-12
	MOVW	fd+0(FP), R4
	MOVV	$SYS_close, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+8(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-8-28
	MOVV	fd+0(FP), R4
	MOVV	p+8(FP), R5
	MOVW	n+16(FP), R6
	MOVV	$SYS_write, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-8-28
	MOVW	fd+0(FP), R4
	MOVV	p+8(FP), R5
	MOVW	n+16(FP), R6
	MOVV	$SYS_read, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$-8-20
	MOVW	kind+0(FP), R4	// _RLIMIT_AS = 6 on linux/mips
	MOVV	limit+8(FP), R5
	MOVV	$SYS_getrlimit, R2
	SYSCALL
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVWU	usec+0(FP), R3
	MOVV	R3, R5
	MOVW	$1000000, R4
	DIVVU	R4, R3
	MOVV	LO, R3
	MOVV	R3, 8(R29)
	MULVU	R3, R4
	MOVV	LO, R4
	SUBVU	R4, R5
	MOVV	R5, 16(R29)

	// select(0, 0, 0, 0, &tv)
	MOVW	$0, R4
	MOVW	$0, R5
	MOVW	$0, R6
	MOVW	$0, R7
	ADDV	$8, R29, R8
	MOVV	$SYS_newselect, R2
	SYSCALL
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVV	$SYS_gettid, R2
	SYSCALL
	MOVW	R2, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$-8
	MOVV	$SYS_gettid, R2
	SYSCALL
	MOVW	R2, R4	// arg 1 tid
	MOVW	sig+0(FP), R5	// arg 2
	MOVV	$SYS_tkill, R2
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$-8
	MOVV	$SYS_getpid, R2
	SYSCALL
	MOVW	R2, R4	// arg 1 pid
	MOVW	sig+0(FP), R5	// arg 2
	MOVV	$SYS_kill, R2
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-8-24
	MOVW	mode+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVV	$SYS_setitimer, R2
	SYSCALL
	RET

TEXT runtime·mincore(SB),NOSPLIT,$-8-28
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	dst+16(FP), R6
	MOVV	$SYS_mincore, R2
	SYSCALL
	SUBVU	R2, R0, R2	// caller expects negative errno
	MOVW	R2, ret+24(FP)
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB),NOSPLIT,$16
	MOVW	$0, R4 // CLOCK_REALTIME
	MOVV	$0(R29), R5
	MOVV	$SYS_clock_gettime, R2
	SYSCALL
	MOVV	0(R29), R3	// sec
	MOVV	8(R29), R5	// nsec
	MOVV	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB),NOSPLIT,$16
	MOVW	$1, R4 // CLOCK_MONOTONIC
	MOVV	$0(R29), R5
	MOVV	$SYS_clock_gettime, R2
	SYSCALL
	MOVV	0(R29), R3	// sec
	MOVV	8(R29), R5	// nsec
	// sec is in R3, nsec in R5
	// return nsec in R3
	MOVV	$1000000000, R4
	MULVU	R4, R3
	MOVV	LO, R3
	ADDVU	R5, R3
	MOVV	R3, ret+0(FP)
	RET

TEXT runtime·rtsigprocmask(SB),NOSPLIT,$-8-28
	MOVW	how+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVW	size+24(FP), R7
	MOVV	$SYS_rt_sigprocmask, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT,$-8-36
	MOVV	sig+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVV	size+24(FP), R7
	MOVV	$SYS_rt_sigaction, R2
	SYSCALL
	MOVW	R2, ret+32(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R4
	MOVV	info+16(FP), R5
	MOVV	ctx+24(FP), R6
	MOVV	fn+0(FP), R25
	JAL	(R25)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$64
	// initialize REGSB = PC&0xffffffff00000000
	BGEZAL	R0, 1(PC)
	SRLV	$32, R31, RSB
	SLLV	$32, RSB

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R1
	BEQ	R1, 2(PC)
	JAL	runtime·load_g(SB)

	MOVW	R4, 8(R29)
	MOVV	R5, 16(R29)
	MOVV	R6, 24(R29)
	MOVV	$runtime·sigtrampgo(SB), R1
	JAL	(R1)
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

TEXT runtime·mmap(SB),NOSPLIT,$-8
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVW	prot+16(FP), R6
	MOVW	flags+20(FP), R7
	MOVW	fd+24(FP), R8
	MOVW	off+28(FP), R9

	MOVV	$SYS_mmap, R2
	SYSCALL
	MOVV	R2, ret+32(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$-8
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	$SYS_munmap, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVV	R0, 0xf3(R0)	// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-8
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVW	flags+16(FP), R6
	MOVV	$SYS_madvise, R2
	SYSCALL
	// ignore failure - maybe pages are locked
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$-8
	MOVV	addr+0(FP), R4
	MOVW	op+8(FP), R5
	MOVW	val+12(FP), R6
	MOVV	ts+16(FP), R7
	MOVV	addr2+24(FP), R8
	MOVW	val3+32(FP), R9
	MOVV	$SYS_futex, R2
	SYSCALL
	MOVW	R2, ret+40(FP)
	RET

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT,$-8
	MOVW	flags+0(FP), R4
	MOVV	stk+8(FP), R5

	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVV	mp+16(FP), R16
	MOVV	gp+24(FP), R17
	MOVV	fn+32(FP), R18

	MOVV	R16, -8(R5)
	MOVV	R17, -16(R5)
	MOVV	R18, -24(R5)
	MOVV	$1234, R16
	MOVV	R16, -32(R5)

	MOVV	$SYS_clone, R2
	SYSCALL

	// In parent, return.
	BEQ	R2, 3(PC)
	MOVW	R2, ret+40(FP)
	RET

	// In child, on new stack.
	MOVV	-32(R29), R16
	MOVV	$1234, R1
	BEQ	R16, R1, 2(PC)
	MOVV	R0, 0(R0)

	// Initialize m->procid to Linux tid
	MOVV	$SYS_gettid, R2
	SYSCALL

	MOVV	-24(R29), R18		// fn
	MOVV	-16(R29), R17		// g
	MOVV	-8(R29), R16		// m

	BEQ	R16, nog
	BEQ	R17, nog

	MOVV	R2, m_procid(R16)

	// TODO: setup TLS.

	// In child, set up new stack
	MOVV	R16, g_m(R17)
	MOVV	R17, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	JAL	(R18)

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R4
	MOVV	$SYS_exit, R2
	SYSCALL
	JMP	-3(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVV	new+0(FP), R4
	MOVV	old+8(FP), R5
	MOVV	$SYS_sigaltstack, R2
	SYSCALL
	BEQ	R7, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-8
	MOVV	$SYS_sched_yield, R2
	SYSCALL
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$-8
	MOVV	pid+0(FP), R4
	MOVV	len+8(FP), R5
	MOVV	buf+16(FP), R6
	MOVV	$SYS_sched_getaffinity, R2
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT,$-8
	MOVW    size+0(FP), R4
	MOVV	$SYS_epoll_create, R2
	SYSCALL
	MOVW	R2, ret+8(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT,$-8
	MOVW	flags+0(FP), R4
	MOVV	$SYS_epoll_create1, R2
	SYSCALL
	MOVW	R2, ret+8(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$-8
	MOVW	epfd+0(FP), R4
	MOVW	op+4(FP), R5
	MOVW	fd+8(FP), R6
	MOVV	ev+16(FP), R7
	MOVV	$SYS_epoll_ctl, R2
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT,$-8
	MOVW	epfd+0(FP), R4
	MOVV	ev+8(FP), R5
	MOVW	nev+16(FP), R6
	MOVW	timeout+20(FP), R7
	MOVV	$SYS_epoll_wait, R2
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$-8
	MOVW    fd+0(FP), R4  // fd
	MOVV    $2, R5  // F_SETFD
	MOVV    $1, R6  // FD_CLOEXEC
	MOVV	$SYS_fcntl, R2
	SYSCALL
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT,$-8-8
	// Implemented as brk(NULL).
	MOVV	$0, R4
	MOVV	$SYS_brk, R2
	SYSCALL
	MOVV	R2, ret+0(FP)
	RET
