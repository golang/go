// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, NetBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVL	$1, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-4
	MOVL	$310, AX		// sys__lwp_exit
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·open(SB),NOSPLIT,$-4
	MOVL	$5, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-4
	MOVL	$6, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-4
	MOVL	$3, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-4
	MOVL	$4, AX			// sys_write
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$24
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 12(SP)		// tv_sec - l32
	MOVL	$0, 16(SP)		// tv_sec - h32
	MOVL	$1000, AX
	MULL	DX
	MOVL	AX, 20(SP)		// tv_nsec

	MOVL	$0, 0(SP)
	LEAL	12(SP), AX
	MOVL	AX, 4(SP)		// arg 1 - rqtp
	MOVL	$0, 8(SP)		// arg 2 - rmtp
	MOVL	$430, AX		// sys_nanosleep
	INT	$0x80
	RET

TEXT runtime·raise(SB),NOSPLIT,$12
	MOVL	$311, AX		// sys__lwp_self
	INT	$0x80
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)		// arg 1 - target
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signo
	MOVL	$318, AX		// sys__lwp_kill
	INT	$0x80
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVL	$20, AX			// sys_getpid
	INT	$0x80
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)		// arg 1 - pid
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signo
	MOVL	$37, AX			// sys_kill
	INT	$0x80
	RET

TEXT runtime·mmap(SB),NOSPLIT,$36
	LEAL	addr+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - addr
	MOVSL				// arg 2 - len
	MOVSL				// arg 3 - prot
	MOVSL				// arg 4 - flags
	MOVSL				// arg 5 - fd
	MOVL	$0, AX
	STOSL				// arg 6 - pad
	MOVSL				// arg 7 - offset
	MOVL	$0, AX			// top 32 bits of file offset
	STOSL
	MOVL	$197, AX		// sys_mmap
	INT	$0x80
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$-4
	MOVL	$73, AX			// sys_munmap
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-4
	MOVL	$75, AX			// sys_madvise
	INT	$0x80
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-4
	MOVL	$425, AX		// sys_setitimer
	INT	$0x80
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)		// arg 1 - clock_id
	MOVL	BX, 8(SP)		// arg 2 - tp
	MOVL	$427, AX		// sys_clock_gettime
	INT	$0x80

	MOVL	12(SP), AX		// sec - l32
	MOVL	AX, sec_lo+0(FP)
	MOVL	16(SP), AX		// sec - h32
	MOVL	AX, sec_hi+4(FP)

	MOVL	20(SP), BX		// nsec
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB),NOSPLIT,$32
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)		// arg 1 - clock_id
	MOVL	BX, 8(SP)		// arg 2 - tp
	MOVL	$427, AX		// sys_clock_gettime
	INT	$0x80

	MOVL	16(SP), CX		// sec - h32
	IMULL	$1000000000, CX

	MOVL	12(SP), AX		// sec - l32
	MOVL	$1000000000, BX
	MULL	BX			// result in dx:ax

	MOVL	20(SP), BX		// nsec
	ADDL	BX, AX
	ADCL	CX, DX			// add high bits with carry

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET

TEXT runtime·getcontext(SB),NOSPLIT,$-4
	MOVL	$307, AX		// sys_getcontext
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$-4
	MOVL	$293, AX		// sys_sigprocmask
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigreturn_tramp(SB),NOSPLIT,$0
	LEAL	140(SP), AX		// Load address of ucontext
	MOVL	AX, 4(SP)
	MOVL	$308, AX		// sys_setcontext
	INT	$0x80
	MOVL	$-1, 4(SP)		// Something failed...
	MOVL	$1, AX			// sys_exit
	INT	$0x80

TEXT runtime·sigaction(SB),NOSPLIT,$24
	LEAL	sig+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - sig
	MOVSL				// arg 2 - act
	MOVSL				// arg 3 - oact
	LEAL	runtime·sigreturn_tramp(SB), AX
	STOSL				// arg 4 - tramp
	MOVL	$2, AX
	STOSL				// arg 5 - vers
	MOVL	$340, AX		// sys___sigaction_sigtramp
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$12-16
	MOVL	fn+0(FP), AX
	MOVL	sig+4(FP), BX
	MOVL	info+8(FP), CX
	MOVL	ctx+12(FP), DX
	MOVL	SP, SI
	SUBL	$32, SP
	ANDL	$-15, SP	// align stack: handler might be a C function
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)	// save SI: handler might be a Go function
	CALL	AX
	MOVL	12(SP), AX
	MOVL	AX, SP
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$28
	// Save callee-saved C registers, since the caller may be a C signal handler.
	MOVL	BX, bx-4(SP)
	MOVL	BP, bp-8(SP)
	MOVL	SI, si-12(SP)
	MOVL	DI, di-16(SP)
	// We don't save mxcsr or the x87 control word because sigtrampgo doesn't
	// modify them.

	MOVL	signo+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	info+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	context+8(FP), BX
	MOVL	BX, 8(SP)
	CALL	runtime·sigtrampgo(SB)

	MOVL	di-16(SP), DI
	MOVL	si-12(SP), SI
	MOVL	bp-8(SP),  BP
	MOVL	bx-4(SP),  BX
	RET

// int32 lwp_create(void *context, uintptr flags, void *lwpid);
TEXT runtime·lwp_create(SB),NOSPLIT,$16
	MOVL	$0, 0(SP)
	MOVL	ctxt+0(FP), AX
	MOVL	AX, 4(SP)		// arg 1 - context
	MOVL	flags+4(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - flags
	MOVL	lwpid+8(FP), AX
	MOVL	AX, 12(SP)		// arg 3 - lwpid
	MOVL	$309, AX		// sys__lwp_create
	INT	$0x80
	JCC	2(PC)
	NEGL	AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·lwp_tramp(SB),NOSPLIT,$0

	// Set FS to point at m->tls
	LEAL	m_tls(BX), BP
	PUSHAL				// save registers
	PUSHL	BP
	CALL	runtime·settls(SB)
	POPL	AX
	POPAL

	// Now segment is established. Initialize m, g.
	get_tls(AX)
	MOVL	DX, g(AX)
	MOVL	BX, g_m(DX)

	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	MOVL	0(DX), DX		// paranoia; check they are not nil
	MOVL	0(BX), BX

	// more paranoia; check that stack splitting code works
	PUSHAL
	CALL	runtime·emptyfunc(SB)
	POPAL

	// Call fn
	CALL	SI

	CALL	runtime·exit1(SB)
	MOVL	$0x1234, 0x1005
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVL	$281, AX		// sys___sigaltstack14
	MOVL	new+0(FP), BX
	MOVL	old+4(FP), CX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

TEXT runtime·setldt(SB),NOSPLIT,$8
	// Under NetBSD we set the GS base instead of messing with the LDT.
	MOVL	16(SP), AX		// tls0
	MOVL	AX, 0(SP)
	CALL	runtime·settls(SB)
	RET

TEXT runtime·settls(SB),NOSPLIT,$16
	// adjust for ELF: wants to use -4(GS) for g
	MOVL	base+0(FP), CX
	ADDL	$4, CX
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	CX, 4(SP)		// arg 1 - ptr
	MOVL	$317, AX		// sys__lwp_setprivate
	INT	$0x80
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$350, AX		// sys_sched_yield
	INT	$0x80
	RET

TEXT runtime·lwp_park(SB),NOSPLIT,$-4
	MOVL	$434, AX		// sys__lwp_park
	INT	$0x80
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·lwp_unpark(SB),NOSPLIT,$-4
	MOVL	$321, AX		// sys__lwp_unpark
	INT	$0x80
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·lwp_self(SB),NOSPLIT,$-4
	MOVL	$311, AX		// sys__lwp_self
	INT	$0x80
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$28
	LEAL	mib+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - name
	MOVSL				// arg 2 - namelen
	MOVSL				// arg 3 - oldp
	MOVSL				// arg 4 - oldlenp
	MOVSL				// arg 5 - newp
	MOVSL				// arg 6 - newlen
	MOVL	$202, AX		// sys___sysctl
	INT	$0x80
	JCC	3(PC)
	NEGL	AX
	RET
	MOVL	$0, AX
	RET

GLOBL runtime·tlsoffset(SB),NOPTR,$4

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$344, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	$435, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

// int32 runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$32
	MOVL	$92, AX		// fcntl
	// 0(SP) is where the caller PC would be; kernel skips it
	MOVL	fd+0(FP), BX
	MOVL	BX, 4(SP)	// fd
	MOVL	$2, 8(SP)	// F_SETFD
	MOVL	$1, 12(SP)	// FD_CLOEXEC
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	RET
