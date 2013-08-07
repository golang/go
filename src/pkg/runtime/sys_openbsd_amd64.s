// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, OpenBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"
#include "../../cmd/ld/textflag.h"

// int64 tfork(void *param, uintptr psize, M *mp, G *gp, void (*fn)(void));
TEXT runtime·tfork(SB),NOSPLIT,$32

	// Copy mp, gp and fn off parent stack for use by child.
	MOVQ	mm+16(FP), R8
	MOVQ	gg+24(FP), R9
	MOVQ	fn+32(FP), R12

	MOVQ	param+0(FP), DI
	MOVQ	psize+8(FP), SI
	MOVL	$8, AX			// sys___tfork
	SYSCALL

	// Return if tfork syscall failed.
	JCC	3(PC)
	NEGQ	AX
	RET

	// In parent, return.
	CMPL	AX, $0
	JEQ	2(PC)
	RET

	// Set FS to point at m->tls.
	LEAQ	m_tls(R8), DI
	CALL	runtime·settls(SB)

	// In child, set up new stack.
	get_tls(CX)
	MOVQ	R8, m(CX)
	MOVQ	R9, g(CX)
	CALL	runtime·stackcheck(SB)

	// Call fn
	CALL	R12

	// It shouldn't return.  If it does, exit
	MOVQ	$0, DI			// arg 1 - notdead
	MOVL	$302, AX		// sys___threxit
	SYSCALL
	JMP	-3(PC)			// keep exiting

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVL	$298, AX		// sys_sched_yield
	SYSCALL
	RET

TEXT runtime·thrsleep(SB),NOSPLIT,$0
	MOVQ	8(SP), DI		// arg 1 - ident
	MOVL	16(SP), SI		// arg 2 - clock_id
	MOVQ	24(SP), DX		// arg 3 - tp
	MOVQ	32(SP), R10		// arg 4 - lock
	MOVQ	40(SP), R8		// arg 5 - abort
	MOVL	$300, AX		// sys___thrsleep
	SYSCALL
	RET

TEXT runtime·thrwakeup(SB),NOSPLIT,$0
	MOVQ	8(SP), DI		// arg 1 - ident
	MOVL	16(SP), SI		// arg 2 - n
	MOVL	$301, AX		// sys___thrwakeup
	SYSCALL
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 - exit status
	MOVL	$1, AX			// sys_exit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8
	MOVQ	$0, DI			// arg 1 - notdead
	MOVL	$302, AX		// sys___threxit
	SYSCALL
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·open(SB),NOSPLIT,$-8
	MOVQ	8(SP), DI		// arg 1 pathname
	MOVL	16(SP), SI		// arg 2 flags
	MOVL	20(SP), DX		// arg 3 mode
	MOVL	$5, AX
	SYSCALL
	RET

TEXT runtime·close(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 fd
	MOVL	$6, AX
	SYSCALL
	RET

TEXT runtime·read(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 fd
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$3, AX
	SYSCALL
	RET

TEXT runtime·write(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 - fd
	MOVQ	16(SP), SI		// arg 2 - buf
	MOVL	24(SP), DX		// arg 3 - nbyte
	MOVL	$4, AX			// sys_write
	SYSCALL
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

TEXT runtime·raise(SB),NOSPLIT,$16
	MOVL	$299, AX		// sys_getthrid
	SYSCALL
	MOVQ	AX, DI			// arg 1 - pid
	MOVL	sig+0(FP), SI			// arg 2 - signum
	MOVL	$37, AX			// sys_kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 - which
	MOVQ	16(SP), SI		// arg 2 - itv
	MOVQ	24(SP), DX		// arg 3 - oitv
	MOVL	$83, AX			// sys_setitimer
	SYSCALL
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVQ	$0, DI			// arg 1 - clock_id
	LEAQ	8(SP), SI		// arg 2 - tp
	MOVL	$232, AX		// sys_clock_gettime
	SYSCALL
	MOVL	8(SP), AX		// sec
	MOVQ	16(SP), DX		// nsec

	// sec is in AX, nsec in DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB),NOSPLIT,$24
	MOVQ	$0, DI			// arg 1 - clock_id
	LEAQ	8(SP), SI		// arg 2 - tp
	MOVL	$232, AX		// sys_clock_gettime
	SYSCALL
	MOVL	8(SP), AX		// sec
	MOVQ	16(SP), DX		// nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 - signum
	MOVQ	16(SP), SI		// arg 2 - nsa
	MOVQ	24(SP), DX		// arg 3 - osa
	MOVL	$46, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	8(SP), DI		// arg 1 - how
	MOVL	12(SP), SI		// arg 2 - set
	MOVL	$48, AX			// sys_sigprocmask
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	MOVL	AX, oset+0(FP)		// Return oset
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$64
	get_tls(BX)
	
	// check that m exists
	MOVQ	m(BX), BP
	CMPQ	BP, $0
	JNE	5(PC)
	MOVQ	DI, 0(SP)
	MOVQ	$runtime·badsignal(SB), AX
	CALL	AX
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

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVQ	8(SP), DI		// arg 1 - addr
	MOVQ	16(SP), SI		// arg 2 - len
	MOVL	24(SP), DX		// arg 3 - prot
	MOVL	28(SP), R10		// arg 4 - flags
	MOVL	32(SP), R8		// arg 5 - fd
	MOVQ	36(SP), R9
	SUBQ	$16, SP
	MOVQ	R9, 8(SP)		// arg 7 - offset (passed on stack)
	MOVQ	$0, R9			// arg 6 - pad
	MOVL	$197, AX
	SYSCALL
	ADDQ	$16, SP
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVQ	8(SP), DI		// arg 1 - addr
	MOVQ	16(SP), SI		// arg 2 - len
	MOVL	$73, AX			// sys_munmap
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 - addr
	MOVQ	len+8(FP), SI		// arg 2 - len
	MOVQ	behav+16(FP), DX	// arg 3 - behav
	MOVQ	$75, AX			// sys_madvise
	SYSCALL
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVQ	new+8(SP), DI		// arg 1 - nss
	MOVQ	old+16(SP), SI		// arg 2 - oss
	MOVQ	$288, AX		// sys_sigaltstack
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$0
	// adjust for ELF: wants to use -16(FS) and -8(FS) for g and m
	ADDQ	$16, DI
	MOVQ	$329, AX		// sys___settcb
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVQ	8(SP), DI		// arg 1 - name
	MOVL	16(SP), SI		// arg 2 - namelen
	MOVQ	24(SP), DX		// arg 3 - oldp
	MOVQ	32(SP), R10		// arg 4 - oldlenp
	MOVQ	40(SP), R8		// arg 5 - newp
	MOVQ	48(SP), R9		// arg 6 - newlen
	MOVQ	$202, AX		// sys___sysctl
	SYSCALL
	JCC	3(PC)
	NEGQ	AX
	RET
	MOVL	$0, AX
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVQ	$0, DI
	MOVQ	$0, SI
	MOVQ	$0, DX
	MOVL	$269, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVQ	32(SP), R10
	MOVL	40(SP), R8
	MOVQ	48(SP), R9
	MOVL	$270, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL	8(SP), DI	// fd
	MOVQ	$2, SI		// F_SETFD
	MOVQ	$1, DX		// FD_CLOEXEC
	MOVL	$92, AX		// fcntl
	SYSCALL
	RET
