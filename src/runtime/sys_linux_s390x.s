// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other system stuff for Linux s390x; see
// /usr/include/asm/unistd.h for the syscall number definitions.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define SYS_exit                  1
#define SYS_read                  3
#define SYS_write                 4
#define SYS_open                  5
#define SYS_close                 6
#define SYS_getpid               20
#define SYS_kill                 37
#define SYS_fcntl                55
#define SYS_gettimeofday         78
#define SYS_mmap                 90
#define SYS_munmap               91
#define SYS_setitimer           104
#define SYS_clone               120
#define SYS_select              142
#define SYS_sched_yield         158
#define SYS_rt_sigreturn        173
#define SYS_rt_sigaction        174
#define SYS_rt_sigprocmask      175
#define SYS_sigaltstack         186
#define SYS_ugetrlimit          191
#define SYS_madvise             219
#define SYS_mincore             218
#define SYS_gettid              236
#define SYS_tkill               237
#define SYS_futex               238
#define SYS_sched_getaffinity   240
#define SYS_exit_group          248
#define SYS_epoll_create        249
#define SYS_epoll_ctl           250
#define SYS_epoll_wait          251
#define SYS_clock_gettime       260
#define SYS_epoll_create1       327

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R2
	MOVW	$SYS_exit_group, R1
	SYSCALL
	RET

TEXT runtime·exit1(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R2
	MOVW	$SYS_exit, R1
	SYSCALL
	RET

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	name+0(FP), R2
	MOVW	mode+8(FP), R3
	MOVW	perm+12(FP), R4
	MOVW	$SYS_open, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R2
	MOVW	$SYS_close, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+8(FP)
	RET

TEXT runtime·write(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	fd+0(FP), R2
	MOVD	p+8(FP), R3
	MOVW	n+16(FP), R4
	MOVW	$SYS_write, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R2
	MOVD	p+8(FP), R3
	MOVW	n+16(FP), R4
	MOVW	$SYS_read, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT|NOFRAME,$0-20
	MOVW	kind+0(FP), R2
	MOVD	limit+8(FP), R3
	MOVW	$SYS_ugetrlimit, R1
	SYSCALL
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVW	usec+0(FP), R2
	MOVD	R2, R4
	MOVW	$1000000, R3
	DIVD	R3, R2
	MOVD	R2, 8(R15)
	MULLD	R2, R3
	SUB	R3, R4
	MOVD	R4, 16(R15)

	// select(0, 0, 0, 0, &tv)
	MOVW	$0, R2
	MOVW	$0, R3
	MOVW	$0, R4
	MOVW	$0, R5
	ADD	$8, R15, R6
	MOVW	$SYS_select, R1
	SYSCALL
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVW	$SYS_gettid, R1
	SYSCALL
	MOVW	R2, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_gettid, R1
	SYSCALL
	MOVW	R2, R2	// arg 1 tid
	MOVW	sig+0(FP), R3	// arg 2
	MOVW	$SYS_tkill, R1
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_getpid, R1
	SYSCALL
	MOVW	R2, R2	// arg 1 pid
	MOVW	sig+0(FP), R3	// arg 2
	MOVW	$SYS_kill, R1
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R2
	MOVD	new+8(FP), R3
	MOVD	old+16(FP), R4
	MOVW	$SYS_setitimer, R1
	SYSCALL
	RET

TEXT runtime·mincore(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVD	dst+16(FP), R4
	MOVW	$SYS_mincore, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB),NOSPLIT,$16
	MOVD	$0(R15), R2
	MOVD	$0, R3
	MOVW	$SYS_gettimeofday, R1
	SYSCALL
	MOVD	0(R15), R2	// sec
	MOVD	8(R15), R4	// usec
	MOVD	$1000, R3
	MULLD	R3, R4
	MOVD	R2, sec+0(FP)
	MOVW	R4, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB),NOSPLIT,$16
	MOVW	$1, R2 // CLOCK_MONOTONIC
	MOVD	$0(R15), R3
	MOVW	$SYS_clock_gettime, R1
	SYSCALL
	MOVD	0(R15), R2	// sec
	MOVD	8(R15), R4	// nsec
	// sec is in R2, nsec in R4
	// return nsec in R2
	MOVD	$1000000000, R3
	MULLD	R3, R2
	ADD	R4, R2
	MOVD	R2, ret+0(FP)
	RET

TEXT runtime·rtsigprocmask(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	sig+0(FP), R2
	MOVD	new+8(FP), R3
	MOVD	old+16(FP), R4
	MOVW	size+24(FP), R5
	MOVW	$SYS_rt_sigprocmask, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVD	R0, 0(R0) // crash
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT|NOFRAME,$0-36
	MOVD	sig+0(FP), R2
	MOVD	new+8(FP), R3
	MOVD	old+16(FP), R4
	MOVD	size+24(FP), R5
	MOVW	$SYS_rt_sigaction, R1
	SYSCALL
	MOVW	R2, ret+32(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R2
	MOVD	info+16(FP), R3
	MOVD	ctx+24(FP), R4
	MOVD	fn+0(FP), R5
	BL	R5
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$64
	// initialize essential registers (just in case)
	XOR	R0, R0

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R6
	CMPBEQ	R6, $0, 2(PC)
	BL	runtime·load_g(SB)

	MOVW	R2, 8(R15)
	MOVD	R3, 16(R15)
	MOVD	R4, 24(R15)
	MOVD	$runtime·sigtrampgo(SB), R5
	BL	R5
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	BR	runtime·sigtramp(SB)

// func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer
TEXT runtime·mmap(SB),NOSPLIT,$48-40
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVW	prot+16(FP), R4
	MOVW	flags+20(FP), R5
	MOVW	fd+24(FP), R6
	MOVWZ	off+28(FP), R7

	// s390x uses old_mmap, so the arguments need to be placed into
	// a struct and a pointer to the struct passed to mmap.
	MOVD	R2, addr-48(SP)
	MOVD	R3, n-40(SP)
	MOVD	R4, prot-32(SP)
	MOVD	R5, flags-24(SP)
	MOVD	R6, fd-16(SP)
	MOVD	R7, off-8(SP)

	MOVD	$addr-48(SP), R2
	MOVW	$SYS_mmap, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	NEG	R2
	MOVD	R2, ret+32(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVW	$SYS_munmap, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVD	R0, 0(R0) // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVW	flags+16(FP), R4
	MOVW	$SYS_madvise, R1
	SYSCALL
	// ignore failure - maybe pages are locked
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R2
	MOVW	op+8(FP), R3
	MOVW	val+12(FP), R4
	MOVD	ts+16(FP), R5
	MOVD	addr2+24(FP), R6
	MOVW	val3+32(FP),  R7
	MOVW	$SYS_futex, R1
	SYSCALL
	MOVW	R2, ret+40(FP)
	RET

// int32 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R3
	MOVD	stk+8(FP), R2

	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVD	mm+16(FP), R7
	MOVD	gg+24(FP), R8
	MOVD	fn+32(FP), R9

	MOVD	R7, -8(R2)
	MOVD	R8, -16(R2)
	MOVD	R9, -24(R2)
	MOVD	$1234, R7
	MOVD	R7, -32(R2)

	SYSCALL $SYS_clone

	// In parent, return.
	CMPBEQ	R2, $0, 3(PC)
	MOVW	R2, ret+40(FP)
	RET

	// In child, on new stack.
	// initialize essential registers
	XOR	R0, R0
	MOVD	-32(R15), R7
	CMP	R7, $1234
	BEQ	2(PC)
	MOVD	R0, 0(R0)

	// Initialize m->procid to Linux tid
	SYSCALL $SYS_gettid

	MOVD	-24(R15), R9        // fn
	MOVD	-16(R15), R8        // g
	MOVD	-8(R15), R7         // m

	CMPBEQ	R7, $0, nog
	CMP	R8, $0
	BEQ	nog

	MOVD	R2, m_procid(R7)

	// In child, set up new stack
	MOVD	R7, g_m(R8)
	MOVD	R8, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	BL	R9

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R2
	MOVW	$SYS_exit, R1
	SYSCALL
	BR	-2(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVD	new+0(FP), R2
	MOVD	old+8(FP), R3
	MOVW	$SYS_sigaltstack, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVD	R0, 0(R0) // crash
	RET

TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_sched_yield, R1
	SYSCALL
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT|NOFRAME,$0
	MOVD	pid+0(FP), R2
	MOVD	len+8(FP), R3
	MOVD	buf+16(FP), R4
	MOVW	$SYS_sched_getaffinity, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT|NOFRAME,$0
	MOVW    size+0(FP), R2
	MOVW	$SYS_epoll_create, R1
	SYSCALL
	MOVW	R2, ret+8(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R2
	MOVW	$SYS_epoll_create1, R1
	SYSCALL
	MOVW	R2, ret+8(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R2
	MOVW	op+4(FP), R3
	MOVW	fd+8(FP), R4
	MOVD	ev+16(FP), R5
	MOVW	$SYS_epoll_ctl, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R2
	MOVD	ev+8(FP), R3
	MOVW	nev+16(FP), R4
	MOVW	timeout+20(FP), R5
	MOVW	$SYS_epoll_wait, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT|NOFRAME,$0
	MOVW    fd+0(FP), R2  // fd
	MOVD    $2, R3  // F_SETFD
	MOVD    $1, R4  // FD_CLOEXEC
	MOVW	$SYS_fcntl, R1
	SYSCALL
	RET
