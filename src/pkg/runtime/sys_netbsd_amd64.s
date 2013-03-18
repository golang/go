// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, NetBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"

// int32 lwp_create(void *context, uintptr flags, void *lwpid)
TEXT runtime·lwp_create(SB),7,$0
	MOVQ	context+0(FP), DI
	MOVQ	flags+8(FP), SI
	MOVQ	lwpid+16(FP), DX
	MOVL	$309, AX		// sys__lwp_create
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	RET

TEXT runtime·lwp_tramp(SB),7,$0
	
	// Set FS to point at m->tls.
	LEAQ	m_tls(R8), DI
	CALL	runtime·settls(SB)

	// Set up new stack.
	get_tls(CX)
	MOVQ	R8, m(CX)
	MOVQ	R9, g(CX)
	CALL	runtime·stackcheck(SB)

	// Call fn
	CALL	R12

	// It shouldn't return.  If it does, exit.
	MOVL	$310, AX		// sys__lwp_exit
	SYSCALL
	JMP	-3(PC)			// keep exiting

TEXT runtime·osyield(SB),7,$0
	MOVL	$350, AX		// sys_sched_yield
	SYSCALL
	RET

TEXT runtime·lwp_park(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 - abstime
	MOVL	16(SP), SI		// arg 2 - unpark
	MOVQ	24(SP), DX		// arg 3 - hint
	MOVQ	32(SP), R10		// arg 4 - unparkhint
	MOVL	$434, AX		// sys__lwp_park
	SYSCALL
	RET

TEXT runtime·lwp_unpark(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 - lwp
	MOVL	16(SP), SI		// arg 2 - hint
	MOVL	$321, AX		// sys__lwp_unpark
	SYSCALL
	RET

TEXT runtime·lwp_self(SB),7,$0
	MOVL	$311, AX		// sys__lwp_self
	SYSCALL
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 - exit status
	MOVL	$1, AX			// sys_exit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·exit1(SB),7,$-8
	MOVL	$310, AX		// sys__lwp_exit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·open(SB),7,$-8
	MOVQ	8(SP), DI		// arg 1 pathname
	MOVL	16(SP), SI		// arg 2 flags
	MOVL	20(SP), DX		// arg 3 mode
	MOVL	$5, AX
	SYSCALL
	RET

TEXT runtime·close(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 fd
	MOVL	$6, AX
	SYSCALL
	RET

TEXT runtime·read(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 fd
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$3, AX
	SYSCALL
	RET

TEXT runtime·write(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 - fd
	MOVQ	16(SP), SI		// arg 2 - buf
	MOVL	24(SP), DX		// arg 3 - nbyte
	MOVL	$4, AX			// sys_write
	SYSCALL
	RET

TEXT runtime·usleep(SB),7,$16
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

TEXT runtime·raise(SB),7,$16
	MOVL	$311, AX		// sys__lwp_self
	SYSCALL
	MOVQ	AX, DI			// arg 1 - target
	MOVL	sig+0(FP), SI		// arg 2 - signo
	MOVL	$318, AX		// sys__lwp_kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 - which
	MOVQ	16(SP), SI		// arg 2 - itv
	MOVQ	24(SP), DX		// arg 3 - oitv
	MOVL	$425, AX		// sys_setitimer
	SYSCALL
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), 7, $32
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

TEXT runtime·nanotime(SB),7,$32
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
	RET

TEXT runtime·getcontext(SB),7,$-8
	MOVQ	8(SP), DI		// arg 1 - context
	MOVL	$307, AX		// sys_getcontext
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigprocmask(SB),7,$0
	MOVL	8(SP), DI		// arg 1 - how
	MOVQ	16(SP), SI		// arg 2 - set
	MOVQ	24(SP), DX		// arg 3 - oset
	MOVL	$293, AX		// sys_sigprocmask
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigreturn_tramp(SB),7,$-8
	MOVQ	R15, DI			// Load address of ucontext
	MOVQ	$308, AX		// sys_setcontext
	SYSCALL
	MOVQ	$-1, DI			// Something failed...
	MOVL	$1, AX			// sys_exit
	SYSCALL

TEXT runtime·sigaction(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 - signum
	MOVQ	16(SP), SI		// arg 2 - nsa
	MOVQ	24(SP), DX		// arg 3 - osa
					// arg 4 - tramp
	LEAQ	runtime·sigreturn_tramp(SB), R10
	MOVQ	$2, R8			// arg 5 - vers
	MOVL	$340, AX		// sys___sigaction_sigtramp
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigtramp(SB),7,$64
	get_tls(BX)

	// check that m exists
	MOVQ	m(BX), BP
	CMPQ	BP, $0
	JNE	4(PC)
	MOVQ	DI, 0(SP)
	CALL	runtime·badsignal(SB)
	RET

	// save g
	MOVQ	g(BX), R10
	MOVQ	R10, 40(SP)

	// g = m->signal
	MOVQ	m_gsignal(BP), BP
	MOVQ	BP, g(BX)

	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	MOVQ	DX, 16(SP)
	MOVQ	R10, 24(SP)

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(BX)
	MOVQ	40(SP), R10
	MOVQ	R10, g(BX)
	RET

TEXT runtime·mmap(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 - addr
	MOVQ	16(SP), SI		// arg 2 - len
	MOVL	24(SP), DX		// arg 3 - prot
	MOVL	28(SP), R10		// arg 4 - flags
	MOVL	32(SP), R8		// arg 5 - fd
	MOVQ	36(SP), R9
	SUBQ	$16, SP
	MOVQ	R9, 8(SP)		// arg 7 - offset (passed on stack)
	MOVQ	$0, R9			// arg 6 - pad
	MOVL	$197, AX		// sys_mmap
	SYSCALL
	ADDQ	$16, SP
	RET

TEXT runtime·munmap(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 - addr
	MOVQ	16(SP), SI		// arg 2 - len
	MOVL	$73, AX			// sys_munmap
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET


TEXT runtime·madvise(SB),7,$0
	MOVQ	addr+0(FP), DI		// arg 1 - addr
	MOVQ	len+8(FP), SI		// arg 2 - len
	MOVQ	behav+16(FP), DX	// arg 3 - behav
	MOVQ	$75, AX			// sys_madvise
	SYSCALL
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·sigaltstack(SB),7,$-8
	MOVQ	new+8(SP), DI		// arg 1 - nss
	MOVQ	old+16(SP), SI		// arg 2 - oss
	MOVQ	$281, AX		// sys___sigaltstack14
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

// set tls base to DI
TEXT runtime·settls(SB),7,$8
	// adjust for ELF: wants to use -16(FS) and -8(FS) for g and m
	ADDQ	$16, DI			// arg 1 - ptr
	MOVQ	$317, AX		// sys__lwp_setprivate
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sysctl(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 - name
	MOVL	16(SP), SI		// arg 2 - namelen
	MOVQ	24(SP), DX		// arg 3 - oldp
	MOVQ	32(SP), R10		// arg 4 - oldlenp
	MOVQ	40(SP), R8		// arg 5 - newp
	MOVQ	48(SP), R9		// arg 6 - newlen
	MOVQ	$202, AX		// sys___sysctl
	SYSCALL
	JCC 3(PC)
	NEGQ	AX
	RET
	MOVL	$0, AX
	RET

