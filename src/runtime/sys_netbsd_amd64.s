// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, NetBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// int32 lwp_create(void *context, uintptr flags, void *lwpid)
TEXT runtime·lwp_create(SB),NOSPLIT,$0
	MOVQ	ctxt+0(FP), DI
	MOVQ	flags+8(FP), SI
	MOVQ	lwpid+16(FP), DX
	MOVL	$309, AX		// sys__lwp_create
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·lwp_tramp(SB),NOSPLIT,$0
	
	// Set FS to point at m->tls.
	LEAQ	m_tls(R8), DI
	CALL	runtime·settls(SB)

	// Set up new stack.
	get_tls(CX)
	MOVQ	R8, g_m(R9)
	MOVQ	R9, g(CX)
	CALL	runtime·stackcheck(SB)

	// Call fn
	CALL	R12

	// It shouldn't return. If it does, exit.
	MOVL	$310, AX		// sys__lwp_exit
	SYSCALL
	JMP	-3(PC)			// keep exiting

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVL	$350, AX		// sys_sched_yield
	SYSCALL
	RET

TEXT runtime·lwp_park(SB),NOSPLIT,$0
	MOVQ	abstime+0(FP), DI		// arg 1 - abstime
	MOVL	unpark+8(FP), SI		// arg 2 - unpark
	MOVQ	hint+16(FP), DX		// arg 3 - hint
	MOVQ	unparkhint+24(FP), R10		// arg 4 - unparkhint
	MOVL	$434, AX		// sys__lwp_park
	SYSCALL
	MOVL	AX, ret+32(FP)
	RET

TEXT runtime·lwp_unpark(SB),NOSPLIT,$0
	MOVL	lwp+0(FP), DI		// arg 1 - lwp
	MOVQ	hint+8(FP), SI		// arg 2 - hint
	MOVL	$321, AX		// sys__lwp_unpark
	SYSCALL
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·lwp_self(SB),NOSPLIT,$0
	MOVL	$311, AX		// sys__lwp_self
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVL	code+0(FP), DI		// arg 1 - exit status
	MOVL	$1, AX			// sys_exit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8
	MOVL	$310, AX		// sys__lwp_exit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
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
	MOVQ	fd+0(FP), DI		// arg 1 - fd
	MOVQ	p+8(FP), SI		// arg 2 - buf
	MOVL	n+16(FP), DX		// arg 3 - nbyte
	MOVL	$4, AX			// sys_write
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
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
	MOVL	$430, AX		// sys_nanosleep
	SYSCALL
	RET

TEXT runtime·raise(SB),NOSPLIT,$16
	MOVL	$311, AX		// sys__lwp_self
	SYSCALL
	MOVQ	AX, DI			// arg 1 - target
	MOVL	sig+0(FP), SI		// arg 2 - signo
	MOVL	$318, AX		// sys__lwp_kill
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	MOVL	$20, AX			// sys_getpid
	SYSCALL
	MOVQ	AX, DI			// arg 1 - pid
	MOVL	sig+0(FP), SI		// arg 2 - signo
	MOVL	$37, AX			// sys_kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-8
	MOVL	mode+0(FP), DI		// arg 1 - which
	MOVQ	new+8(FP), SI		// arg 2 - itv
	MOVQ	old+16(FP), DX		// arg 3 - oitv
	MOVL	$425, AX		// sys_setitimer
	SYSCALL
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	MOVQ	$0, DI			// arg 1 - clock_id
	LEAQ	8(SP), SI		// arg 2 - tp
	MOVL	$427, AX		// sys_clock_gettime
	SYSCALL
	MOVQ	8(SP), AX		// sec
	MOVL	16(SP), DX		// nsec

	// sec is in AX, nsec in DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB),NOSPLIT,$32
	MOVQ	$0, DI			// arg 1 - clock_id
	LEAQ	8(SP), SI		// arg 2 - tp
	MOVL	$427, AX		// sys_clock_gettime
	SYSCALL
	MOVQ	8(SP), AX		// sec
	MOVL	16(SP), DX		// nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET

TEXT runtime·getcontext(SB),NOSPLIT,$-8
	MOVQ	ctxt+0(FP), DI		// arg 1 - context
	MOVL	$307, AX		// sys_getcontext
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	how+0(FP), DI		// arg 1 - how
	MOVQ	new+8(FP), SI		// arg 2 - set
	MOVQ	old+16(FP), DX		// arg 3 - oset
	MOVL	$293, AX		// sys_sigprocmask
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigreturn_tramp(SB),NOSPLIT,$-8
	MOVQ	R15, DI			// Load address of ucontext
	MOVQ	$308, AX		// sys_setcontext
	SYSCALL
	MOVQ	$-1, DI			// Something failed...
	MOVL	$1, AX			// sys_exit
	SYSCALL

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVL	sig+0(FP), DI		// arg 1 - signum
	MOVQ	new+8(FP), SI		// arg 2 - nsa
	MOVQ	old+16(FP), DX		// arg 3 - osa
					// arg 4 - tramp
	LEAQ	runtime·sigreturn_tramp(SB), R10
	MOVQ	$2, R8			// arg 5 - vers
	MOVL	$340, AX		// sys___sigaction_sigtramp
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
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
	MOVL	$197, AX		// sys_mmap
	SYSCALL
	ADDQ	$16, SP
	MOVQ	AX, ret+32(FP)
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
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVQ	new+0(FP), DI		// arg 1 - nss
	MOVQ	old+8(FP), SI		// arg 2 - oss
	MOVQ	$281, AX		// sys___sigaltstack14
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$8
	// adjust for ELF: wants to use -8(FS) for g
	ADDQ	$8, DI			// arg 1 - ptr
	MOVQ	$317, AX		// sys__lwp_setprivate
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
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+48(FP)
	RET

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVQ	$0, DI
	MOVL	$344, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	kq+0(FP), DI
	MOVQ	ch+8(FP), SI
	MOVL	nch+16(FP), DX
	MOVQ	ev+24(FP), R10
	MOVL	nev+32(FP), R8
	MOVQ	ts+40(FP), R9
	MOVL	$435, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL	fd+0(FP), DI	// fd
	MOVQ	$2, SI		// F_SETFD
	MOVQ	$1, DX		// FD_CLOEXEC
	MOVL	$92, AX		// fcntl
	SYSCALL
	RET
