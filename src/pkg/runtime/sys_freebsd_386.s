// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"
	
TEXT runtime·sys_umtx_op(SB),7,$-4
	MOVL	$454, AX
	INT	$0x80
	RET

TEXT runtime·thr_new(SB),7,$-4
	MOVL	$455, AX
	INT	$0x80
	RET

TEXT runtime·thr_start(SB),7,$0
	MOVL	mm+0(FP), AX
	MOVL	m_g0(AX), BX
	LEAL	m_tls(AX), BP
	MOVL	0(BP), DI
	ADDL	$7, DI
	PUSHAL
	PUSHL	$32
	PUSHL	BP
	PUSHL	DI
	CALL	runtime·setldt(SB)
	POPL	AX
	POPL	AX
	POPL	AX
	POPAL
	get_tls(CX)
	MOVL	BX, g(CX)
	
	MOVL	AX, m(CX)
	CALL	runtime·stackcheck(SB)		// smashes AX
	CALL	runtime·mstart(SB)

	MOVL	0, AX			// crash (not reached)

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),7,$-4
	MOVL	$1, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·exit1(SB),7,$-4
	MOVL	$431, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·open(SB),7,$-4
	MOVL	$5, AX
	INT	$0x80
	RET

TEXT runtime·close(SB),7,$-4
	MOVL	$6, AX
	INT	$0x80
	RET

TEXT runtime·read(SB),7,$-4
	MOVL	$3, AX
	INT	$0x80
	RET

TEXT runtime·write(SB),7,$-4
	MOVL	$4, AX
	INT	$0x80
	RET

TEXT runtime·getrlimit(SB),7,$-4
	MOVL	$194, AX
	INT	$0x80
	RET

TEXT runtime·raise(SB),7,$16
	// thr_self(&8(SP))
	LEAL	8(SP), AX
	MOVL	AX, 4(SP)
	MOVL	$432, AX
	INT	$0x80
	// thr_kill(self, SIGPIPE)
	MOVL	8(SP), AX
	MOVL	AX, 4(SP)
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)
	MOVL	$433, AX
	INT	$0x80
	RET

TEXT runtime·mmap(SB),7,$32
	LEAL arg0+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVL	$0, AX	// top 32 bits of file offset
	STOSL
	MOVL	$477, AX
	INT	$0x80
	RET

TEXT runtime·munmap(SB),7,$-4
	MOVL	$73, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise(SB),7,$-4
	MOVL	$75, AX	// madvise
	INT	$0x80
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·setitimer(SB), 7, $-4
	MOVL	$83, AX
	INT	$0x80
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), 7, $32
	MOVL	$232, AX
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	12(SP), AX	// sec
	MOVL	16(SP), BX	// nsec

	// sec is in AX, nsec in BX
	MOVL	AX, sec+0(FP)
	MOVL	$0, sec+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB), 7, $32
	MOVL	$232, AX
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	12(SP), AX	// sec
	MOVL	16(SP), BX	// nsec

	// sec is in AX, nsec in BX
	// convert to DX:AX nsec
	MOVL	$1000000000, CX
	MULL	CX
	ADDL	BX, AX
	ADCL	$0, DX

	MOVL	ret+0(FP), DI
	MOVL	AX, 0(DI)
	MOVL	DX, 4(DI)
	RET


TEXT runtime·sigaction(SB),7,$-4
	MOVL	$416, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigtramp(SB),7,$44
	get_tls(CX)

	// check that m exists
	MOVL	m(CX), BX
	CMPL	BX, $0
	JNE	5(PC)
	MOVL	signo+0(FP), BX
	MOVL	BX, 0(SP)
	CALL	runtime·badsignal(SB)
	RET

	// save g
	MOVL	g(CX), DI
	MOVL	DI, 20(SP)
	
	// g = m->gsignal
	MOVL	m_gsignal(BX), BX
	MOVL	BX, g(CX)

	// copy arguments for call to sighandler
	MOVL	signo+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	info+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	context+8(FP), BX
	MOVL	BX, 8(SP)
	MOVL	DI, 12(SP)

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(CX)
	MOVL	20(SP), BX
	MOVL	BX, g(CX)
	
	// call sigreturn
	MOVL	context+8(FP), AX
	MOVL	$0, 0(SP)	// syscall gap
	MOVL	AX, 4(SP)
	MOVL	$417, AX	// sigreturn(ucontext)
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack(SB),7,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep(SB),7,$20
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 12(SP)		// tv_sec
	MOVL	$1000, AX
	MULL	DX
	MOVL	AX, 16(SP)		// tv_nsec

	MOVL	$0, 0(SP)
	LEAL	12(SP), AX
	MOVL	AX, 4(SP)		// arg 1 - rqtp
	MOVL	$0, 8(SP)		// arg 2 - rmtp
	MOVL	$240, AX		// sys_nanosleep
	INT	$0x80
	RET

/*
descriptor entry format for system call
is the native machine format, ugly as it is:

	2-byte limit
	3-byte base
	1-byte: 0x80=present, 0x60=dpl<<5, 0x1F=type
	1-byte: 0x80=limit is *4k, 0x40=32-bit operand size,
		0x0F=4 more bits of limit
	1 byte: 8 more bits of base

int i386_get_ldt(int, union ldt_entry *, int);
int i386_set_ldt(int, const union ldt_entry *, int);

*/

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),7,$32
	MOVL	address+4(FP), BX	// aka base
	// see comment in sys_linux_386.s; freebsd is similar
	ADDL	$0x8, BX

	// set up data_desc
	LEAL	16(SP), AX	// struct data_desc
	MOVL	$0, 0(AX)
	MOVL	$0, 4(AX)

	MOVW	BX, 2(AX)
	SHRL	$16, BX
	MOVB	BX, 4(AX)
	SHRL	$8, BX
	MOVB	BX, 7(AX)

	MOVW	$0xffff, 0(AX)
	MOVB	$0xCF, 6(AX)	// 32-bit operand, 4k limit unit, 4 more bits of limit

	MOVB	$0xF2, 5(AX)	// r/w data descriptor, dpl=3, present

	// call i386_set_ldt(entry, desc, 1)
	MOVL	$0xffffffff, 0(SP)	// auto-allocate entry and return in AX
	MOVL	AX, 4(SP)
	MOVL	$1, 8(SP)
	CALL	runtime·i386_set_ldt(SB)

	// compute segment selector - (entry*8+7)
	SHLL	$3, AX
	ADDL	$7, AX
	MOVW	AX, GS
	RET

TEXT runtime·i386_set_ldt(SB),7,$16
	LEAL	args+0(FP), AX	// 0(FP) == 4(SP) before SP got moved
	MOVL	$0, 0(SP)	// syscall gap
	MOVL	$1, 4(SP)
	MOVL	AX, 8(SP)
	MOVL	$165, AX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

TEXT runtime·sysctl(SB),7,$28
	LEAL	arg0+0(FP), SI
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

TEXT runtime·osyield(SB),7,$-4
	MOVL	$331, AX		// sys_sched_yield
	INT	$0x80
	RET

TEXT runtime·sigprocmask(SB),7,$16
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	$3, 4(SP)		// arg 1 - how (SIG_SETMASK)
	MOVL	args+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - set
	MOVL	args+4(FP), AX
	MOVL	AX, 12(SP)		// arg 3 - oset
	MOVL	$340, AX		// sys_sigprocmask
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

GLOBL runtime·tlsoffset(SB),$4
