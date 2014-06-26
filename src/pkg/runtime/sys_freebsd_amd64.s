// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"
#include "../../cmd/ld/textflag.h"

// FreeBSD 8, FreeBSD 9, and older versions that I have checked
// do not restore R10 on exit from a "restarted" system call
// if you use the SYSCALL instruction. This means that, for example,
// if a signal arrives while the wait4 system call is executing,
// the wait4 internally returns ERESTART, which makes the kernel
// back up the PC to execute the SYSCALL instruction a second time.
// However, since the kernel does not restore R10, the fourth
// argument to the system call has been lost. (FreeBSD 9 also fails
// to restore the fifth and sixth arguments, R8 and R9, although
// some earlier versions did restore those correctly.)
// The broken code is in fast_syscall in FreeBSD's amd64/amd64/exception.S.
// It restores only DI, SI, DX, AX, and RFLAGS on system call return.
// http://fxr.watson.org/fxr/source/amd64/amd64/exception.S?v=FREEBSD91#L399
//
// The INT $0x80 system call path (int0x80_syscall in FreeBSD's 
// amd64/ia32/ia32_exception.S) does not have this problem,
// but it expects the third argument in R10. Instead of rewriting
// all the assembly in this file, #define SYSCALL to a safe simulation
// using INT $0x80.
//
// INT $0x80 is a little slower than SYSCALL, but correctness wins.
//
// See golang.org/issue/6372.
#define SYSCALL MOVQ R10, CX; INT $0x80
	
TEXT runtime·sys_umtx_op(SB),NOSPLIT,$0
	MOVQ 8(SP), DI
	MOVL 16(SP), SI
	MOVL 20(SP), DX
	MOVQ 24(SP), R10
	MOVQ 32(SP), R8
	MOVL $454, AX
	SYSCALL
	RET

TEXT runtime·thr_new(SB),NOSPLIT,$0
	MOVQ 8(SP), DI
	MOVQ 16(SP), SI
	MOVL $455, AX
	SYSCALL
	RET

TEXT runtime·thr_start(SB),NOSPLIT,$0
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
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$1, AX
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8
	MOVQ	8(SP), DI		// arg 1 exit status
	MOVL	$431, AX
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
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
	MOVL	8(SP), DI		// arg 1 fd
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$4, AX
	SYSCALL
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$194, AX
	SYSCALL
	RET

TEXT runtime·raise(SB),NOSPLIT,$16
	// thr_self(&8(SP))
	LEAQ	8(SP), DI	// arg 1 &8(SP)
	MOVL	$432, AX
	SYSCALL
	// thr_kill(self, SIGPIPE)
	MOVQ	8(SP), DI	// arg 1 id
	MOVL	sig+0(FP), SI	// arg 2
	MOVL	$433, AX
	SYSCALL
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$83, AX
	SYSCALL
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVL	$232, AX
	MOVQ	$0, DI		// CLOCK_REALTIME
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
	// We can use CLOCK_MONOTONIC_FAST here when we drop
	// support for FreeBSD 8-STABLE.
	MOVQ	$4, DI		// CLOCK_MONOTONIC
	LEAQ	8(SP), SI
	SYSCALL
	MOVQ	8(SP), AX	// sec
	MOVQ	16(SP), DX	// nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVL	8(SP), DI		// arg 1 sig
	MOVQ	16(SP), SI		// arg 2 act
	MOVQ	24(SP), DX		// arg 3 oact
	MOVL	$416, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$64
	get_tls(BX)

	// check that g exists
	MOVQ	g(BX), R10
	CMPQ	R10, $0
	JNE	5(PC)
	MOVQ	DI, 0(SP)
	MOVQ	$runtime·badsignal(SB), AX
	CALL	AX
	RET

	// save g
	MOVQ	R10, 40(SP)
	
	// g = m->signal
	MOVQ	g_m(R10), BP
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
	MOVQ	8(SP), DI		// arg 1 addr
	MOVQ	16(SP), SI		// arg 2 len
	MOVL	24(SP), DX		// arg 3 prot
	MOVL	28(SP), R10		// arg 4 flags
	MOVL	32(SP), R8		// arg 5 fid
	MOVL	36(SP), R9		// arg 6 offset
	MOVL	$477, AX
	SYSCALL
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVQ	8(SP), DI		// arg 1 addr
	MOVQ	16(SP), SI		// arg 2 len
	MOVL	$73, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
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
TEXT runtime·settls(SB),NOSPLIT,$8
	ADDQ	$16, DI	// adjust for ELF: wants to use -16(FS) and -8(FS) for g and m
	MOVQ	DI, 0(SP)
	MOVQ	SP, SI
	MOVQ	$129, DI	// AMD64_SET_FSBASE
	MOVQ	$165, AX	// sysarch
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
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
	JCC 3(PC)
	NEGQ	AX
	RET
	MOVL	$0, AX
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$331, AX		// sys_sched_yield
	SYSCALL
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	$3, DI			// arg 1 - how (SIG_SETMASK)
	MOVQ	8(SP), SI		// arg 2 - set
	MOVQ	16(SP), DX		// arg 3 - oset
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
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVQ	32(SP), R10
	MOVL	40(SP), R8
	MOVQ	48(SP), R9
	MOVL	$363, AX
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
