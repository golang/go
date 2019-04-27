// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for arm64, OpenBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME	$0
#define	CLOCK_MONOTONIC	$3

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0
	MOVW	code+0(FP), R0		// arg 1 - status
	MOVD	$1, R8			// sys_exit
	SVC
	BCC	3(PC)
	MOVD	$0, R0			// crash on syscall failure
	MOVD	R0, (R0)
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0
	MOVD	wait+0(FP), R0		// arg 1 - notdead
	MOVD	$302, R8		// sys___threxit
	SVC
	MOVD	$0, R0			// crash on syscall failure
	MOVD	R0, (R0)
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0
	MOVD	name+0(FP), R0		// arg 1 - path
	MOVW	mode+8(FP), R1		// arg 2 - mode
	MOVW	perm+12(FP), R2		// arg 3 - perm
	MOVD	$5, R8			// sys_open
	SVC
	BCC	2(PC)
	MOVW	$-1, R0
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVD	$6, R8			// sys_close
	SVC
	BCC	2(PC)
	MOVW	$-1, R0
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVD	p+8(FP), R1		// arg 2 - buf
	MOVW	n+16(FP), R2		// arg 3 - nbyte
	MOVD	$3, R8			// sys_read
	SVC
	BCC	2(PC)
	MOVW	$-1, R0
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·write(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVD	p+8(FP), R1		// arg 2 - buf
	MOVW	n+16(FP), R2		// arg 3 - nbyte
	MOVD	$4, R8			// sys_write
	SVC
	BCC	2(PC)
	MOVW	$-1, R0
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$24-4
	MOVWU	usec+0(FP), R3
	MOVD	R3, R5
	MOVW	$1000000, R4
	UDIV	R4, R3
	MOVD	R3, 8(RSP)		// tv_sec
	MUL	R3, R4
	SUB	R4, R5
	MOVW	$1000, R4
	MUL	R4, R5
	MOVD	R5, 16(RSP)		// tv_nsec

	ADD	$8, RSP, R0		// arg 1 - rqtp
	MOVD	$0, R1			// arg 2 - rmtp
	MOVD	$91, R8			// sys_nanosleep
	SVC
	RET

TEXT runtime·raise(SB),NOSPLIT,$0
	MOVD	$299, R8		// sys_getthrid
	SVC
					// arg 1 - tid, already in R0
	MOVW	sig+0(FP), R1		// arg 2 - signum
	MOVW	$0, R2			// arg 3 - tcb
	MOVD	$119, R8		// sys_thrkill
	SVC
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$0
	MOVD	$20, R8			// sys_getpid
	SVC
					// arg 1 - pid, already in R0
	MOVW	sig+0(FP), R1		// arg 2 - signum
	MOVD	$122, R8		// sys_kill
	SVC
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0		// arg 1 - addr
	MOVD	n+8(FP), R1		// arg 2 - len
	MOVW	prot+16(FP), R2		// arg 3 - prot
	MOVW	flags+20(FP), R3	// arg 4 - flags
	MOVW	fd+24(FP), R4		// arg 5 - fd
	MOVW	$0, R5			// arg 6 - pad
	MOVW	off+28(FP), R6		// arg 7 - offset
	MOVD	$197, R8		// sys_mmap
	SVC
	MOVD	$0, R1
	BCC	3(PC)
	MOVD	R0, R1			// if error, move to R1
	MOVD	$0, R0
	MOVD	R0, p+32(FP)
	MOVD	R1, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0		// arg 1 - addr
	MOVD	n+8(FP), R1		// arg 2 - len
	MOVD	$73, R8			// sys_munmap
	SVC
	BCC	3(PC)
	MOVD	$0, R0			// crash on syscall failure
	MOVD	R0, (R0)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0		// arg 1 - addr
	MOVD	n+8(FP), R1		// arg 2 - len
	MOVW	flags+16(FP), R2	// arg 2 - flags
	MOVD	$75, R8			// sys_madvise
	SVC
	BCC	2(PC)
	MOVW	$-1, R0
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0		// arg 1 - mode
	MOVD	new+8(FP), R1		// arg 2 - new value
	MOVD	old+16(FP), R2		// arg 3 - old value
	MOVD	$69, R8			// sys_setitimer
	SVC
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	MOVW	CLOCK_REALTIME, R0	// arg 1 - clock_id
	MOVD	$8(RSP), R1		// arg 2 - tp
	MOVD	$87, R8			// sys_clock_gettime
	SVC

	MOVD	8(RSP), R0		// sec
	MOVD	16(RSP), R1		// nsec
	MOVD	R0, sec+0(FP)
	MOVW	R1, nsec+8(FP)

	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB),NOSPLIT,$32
	MOVW	CLOCK_MONOTONIC, R0	// arg 1 - clock_id
	MOVD	$8(RSP), R1		// arg 2 - tp
	MOVD	$87, R8			// sys_clock_gettime
	SVC

	MOVW	8(RSP), R3		// sec
	MOVW	16(RSP), R5		// nsec

	MOVD	$1000000000, R4
	MUL	R4, R3
	ADD	R5, R3
	MOVD	R3, ret+0(FP)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0
	MOVW	sig+0(FP), R0		// arg 1 - signum
	MOVD	new+8(FP), R1		// arg 2 - new sigaction
	MOVD	old+16(FP), R2		// arg 3 - old sigaction
	MOVD	$46, R8			// sys_sigaction
	SVC
	BCC	3(PC)
	MOVD	$3, R0			// crash on syscall failure
	MOVD	R0, (R0)
	RET

TEXT runtime·obsdsigprocmask(SB),NOSPLIT,$0
	MOVW	how+0(FP), R0		// arg 1 - mode
	MOVW	new+4(FP), R1		// arg 2 - new
	MOVD	$48, R8			// sys_sigprocmask
	SVC
	BCC	3(PC)
	MOVD	$3, R8			// crash on syscall failure
	MOVD	R8, (R8)
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)			// Alignment for ELF ABI?
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$32
	// If called from an external code context, g will not be set.
	// Save R0, since runtime·load_g will clobber it.
	MOVW	R0, 8(RSP)		// signum
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	MOVD	R1, 16(RSP)
	MOVD	R2, 24(RSP)
	BL	runtime·sigtrampgo(SB)
	RET

// int32 tfork(void *param, uintptr psize, M *mp, G *gp, void (*fn)(void));
TEXT runtime·tfork(SB),NOSPLIT,$0

	// Copy mp, gp and fn off parent stack for use by child.
	MOVD	mm+16(FP), R4
	MOVD	gg+24(FP), R5
	MOVD	fn+32(FP), R6

	MOVD	param+0(FP), R0		// arg 1 - param
	MOVD	psize+8(FP), R1		// arg 2 - psize
	MOVD	$8, R8			// sys___tfork
	SVC

	// Return if syscall failed.
	BCC	4(PC)
	NEG	R0,  R0
	MOVW	R0, ret+40(FP)
	RET

	// In parent, return.
	CMP	$0, R0
	BEQ	3(PC)
	MOVW	R0, ret+40(FP)
	RET

	// Initialise m, g.
	MOVD	R5, g
	MOVD	R4, g_m(g)

	// Call fn.
	BL	(R6)

	// fn should never return.
	MOVD	$2, R8			// crash if reached
	MOVD	R8, (R8)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVD	new+0(FP), R0		// arg 1 - new sigaltstack
	MOVD	old+8(FP), R1		// arg 2 - old sigaltstack
	MOVD	$288, R8		// sys_sigaltstack
	SVC
	BCC	3(PC)
	MOVD	$0, R8			// crash on syscall failure
	MOVD	R8, (R8)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVD	$298, R8		// sys_sched_yield
	SVC
	RET

TEXT runtime·thrsleep(SB),NOSPLIT,$0
	MOVD	ident+0(FP), R0		// arg 1 - ident
	MOVW	clock_id+8(FP), R1	// arg 2 - clock_id
	MOVD	tsp+16(FP), R2		// arg 3 - tsp
	MOVD	lock+24(FP), R3		// arg 4 - lock
	MOVD	abort+32(FP), R4	// arg 5 - abort
	MOVD	$94, R8			// sys___thrsleep
	SVC
	MOVW	R0, ret+40(FP)
	RET

TEXT runtime·thrwakeup(SB),NOSPLIT,$0
	MOVD	ident+0(FP), R0		// arg 1 - ident
	MOVW	n+8(FP), R1		// arg 2 - n
	MOVD	$301, R8		// sys___thrwakeup
	SVC
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVD	mib+0(FP), R0		// arg 1 - mib
	MOVW	miblen+8(FP), R1	// arg 2 - miblen
	MOVD	out+16(FP), R2		// arg 3 - out
	MOVD	size+24(FP), R3		// arg 4 - size
	MOVD	dst+32(FP), R4		// arg 5 - dest
	MOVD	ndst+40(FP), R5		// arg 6 - newlen
	MOVD	$202, R8		// sys___sysctl
	SVC
	BCC	2(PC)
	NEG	R0, R0
	MOVW	R0, ret+48(FP)
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVD	$269, R8		// sys_kqueue
	SVC
	BCC	2(PC)
	NEG	R0, R0
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW	kq+0(FP), R0		// arg 1 - kq
	MOVD	ch+8(FP), R1		// arg 2 - changelist
	MOVW	nch+16(FP), R2		// arg 3 - nchanges
	MOVD	ev+24(FP), R3		// arg 4 - eventlist
	MOVW	nev+32(FP), R4		// arg 5 - nevents
	MOVD	ts+40(FP), R5		// arg 6 - timeout
	MOVD	$72, R8			// sys_kevent
	SVC
	BCC	2(PC)
	NEG	R0, R0
	MOVW	R0, ret+48(FP)
	RET

// func closeonexec(fd int32)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVD	$2, R1			// arg 2 - cmd (F_SETFD)
	MOVD	$1, R2			// arg 3 - arg (FD_CLOEXEC)
	MOVD	$92, R8			// sys_fcntl
	SVC
	RET
