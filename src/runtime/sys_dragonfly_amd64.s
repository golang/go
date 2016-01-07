// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
	
TEXT runtime·sys_umtx_sleep(SB),NOSPLIT,$0
	MOVQ addr+0(FP), DI		// arg 1 - ptr
	MOVL val+8(FP), SI		// arg 2 - value
	MOVL timeout+12(FP), DX		// arg 3 - timeout
	MOVL $469, AX		// umtx_sleep
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·sys_umtx_wakeup(SB),NOSPLIT,$0
	MOVQ addr+0(FP), DI		// arg 1 - ptr
	MOVL val+8(FP), SI		// arg 2 - count
	MOVL $470, AX		// umtx_wakeup
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·lwp_create(SB),NOSPLIT,$0
	MOVQ param+0(FP), DI		// arg 1 - params
	MOVL $495, AX		// lwp_create
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·lwp_start(SB),NOSPLIT,$0
	MOVQ	DI, R13 // m

	// set up FS to point at m->tls
	LEAQ	m_tls(R13), DI
	CALL	runtime·settls(SB)	// smashes DI

	// set up m, g
	get_tls(CX)
	MOVQ	m_g0(R13), DI
	MOVQ	R13, g_m(DI)
	MOVQ	DI, g(CX)

	CALL	runtime·stackcheck(SB)
	CALL	runtime·mstart(SB)

	MOVQ 0, AX			// crash (not reached)

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVL	code+0(FP), DI		// arg 1 exit status
	MOVL	$1, AX
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8
	MOVL	code+0(FP), DI		// arg 1 exit status
	MOVL	$431, AX
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

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
	MOVL	$-1, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-8
	MOVQ	fd+0(FP), DI		// arg 1 fd
	MOVQ	p+8(FP), SI		// arg 2 buf
	MOVL	n+16(FP), DX		// arg 3 count
	MOVL	$4, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$-8
	MOVL	kind+0(FP), DI
	MOVQ	limit+8(FP), SI
	MOVL	$194, AX
	SYSCALL
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$16
	MOVL	$496, AX	// lwp_gettid
	SYSCALL
	MOVQ	$-1, DI		// arg 1 - pid
	MOVQ	AX, SI		// arg 2 - tid
	MOVL	sig+0(FP), DX	// arg 3 - signum
	MOVL	$497, AX	// lwp_kill
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$0
	MOVL	$20, AX		// getpid
	SYSCALL
	MOVQ	AX, DI		// arg 1 - pid
	MOVL	sig+0(FP), SI	// arg 2 - signum
	MOVL	$37, AX		// kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $-8
	MOVL	mode+0(FP), DI
	MOVQ	new+8(FP), SI
	MOVQ	old+16(FP), DX
	MOVL	$83, AX
	SYSCALL
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVL	$232, AX
	MOVQ	$0, DI  	// CLOCK_REALTIME
	LEAQ	8(SP), SI
	SYSCALL
	MOVQ	8(SP), AX	// sec
	MOVQ	16(SP), DX	// nsec

	// sec is in AX, nsec in DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB), NOSPLIT, $32
	MOVL	$232, AX
	MOVQ	$4, DI  	// CLOCK_MONOTONIC
	LEAQ	8(SP), SI
	SYSCALL
	MOVQ	8(SP), AX	// sec
	MOVQ	16(SP), DX	// nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVL	sig+0(FP), DI		// arg 1 sig
	MOVQ	new+8(FP), SI		// arg 2 act
	MOVQ	old+16(FP), DX		// arg 3 oact
	MOVL	$342, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVL	sig+8(FP), DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP), DX
	MOVQ	fn+0(FP), AX
	CALL	AX
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$24
	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	MOVQ	DX, 16(SP)
	CALL	runtime·sigtrampgo(SB)
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
	ADDQ	$16, SP
	MOVQ	AX, ret+32(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 addr
	MOVQ	n+8(FP), SI		// arg 2 len
	MOVL	$73, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI
	MOVQ	n+8(FP), SI
	MOVL	flags+16(FP), DX
	MOVQ	$75, AX	// madvise
	SYSCALL
	// ignore failure - maybe pages are locked
	RET
	
TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVQ	new+8(SP), DI
	MOVQ	old+16(SP), SI
	MOVQ	$53, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
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
	MOVL	$240, AX		// sys_nanosleep
	SYSCALL
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$16
	ADDQ	$8, DI	// adjust for ELF: wants to use -8(FS) for g
	MOVQ	DI, 0(SP)
	MOVQ	$16, 8(SP)
	MOVQ	$0, DI			// arg 1 - which
	MOVQ	SP, SI			// arg 2 - tls_info
	MOVQ	$16, DX			// arg 3 - infosize
	MOVQ	$472, AX		// set_tls_area
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
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
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+48(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$331, AX		// sys_sched_yield
	SYSCALL
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	how+0(FP), DI		// arg 1 - how
	MOVQ	new+8(FP), SI		// arg 2 - set
	MOVQ	old+16(FP), DX		// arg 3 - oset
	MOVL	$340, AX		// sys_sigprocmask
	SYSCALL
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVQ	$0, DI
	MOVQ	$0, SI
	MOVQ	$0, DX
	MOVL	$362, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	fd+0(FP), DI
	MOVQ	ev1+8(FP), SI
	MOVL	nev1+16(FP), DX
	MOVQ	ev2+24(FP), R10
	MOVL	nev2+32(FP), R8
	MOVQ	ts+40(FP), R9
	MOVL	$363, AX
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
