// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, OpenBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),7,$-4
	MOVL	$1, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·exit1(SB),7,$8
	MOVL	$0, 0(SP)
	MOVL	$0, 4(SP)		// arg 1 - notdead
	MOVL	$302, AX		// sys___threxit
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
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
	MOVL	$4, AX			// sys_write
	INT	$0x80
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

TEXT runtime·raise(SB),7,$12
	MOVL	$299, AX		// sys_getthrid
	INT	$0x80
	MOVL	$0, 0(SP)
	MOVL	AX, 4(SP)		// arg 1 - pid
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - signum
	MOVL	$37, AX			// sys_kill
	INT	$0x80
	RET

TEXT runtime·mmap(SB),7,$36
	LEAL	arg0+0(FP), SI
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
	RET

TEXT runtime·munmap(SB),7,$-4
	MOVL	$73, AX			// sys_munmap
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·madvise(SB),7,$-4
	MOVL	$75, AX			// sys_madvise
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·setitimer(SB),7,$-4
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
	MOVL	12(SP), AX		// sec
	MOVL	16(SP), BX		// nsec

	// sec is in AX, nsec in BX
	MOVL	AX, sec+0(FP)
	MOVL	$0, sec+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB),7,$32
	MOVL	$232, AX
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	12(SP), AX		// sec
	MOVL	16(SP), BX		// nsec

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
	MOVL	$46, AX			// sys_sigaction
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·sigprocmask(SB),7,$-4
	MOVL	$48, AX			// sys_sigprocmask
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	MOVL	AX, oset+0(FP)
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
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	AX, 4(SP)		// arg 1 - sigcontext
	MOVL	$103, AX		// sys_sigreturn
	INT	$0x80
	MOVL	$0xf1, 0xf1		// crash
	RET

// int32 tfork(void *param, uintptr psize, M *mp, G *gp, void (*fn)(void));
TEXT runtime·tfork(SB),7,$12

	// Copy mp, gp and fn from the parent stack onto the child stack.
	MOVL	params+4(FP), AX
	MOVL	8(AX), CX		// tf_stack
	SUBL	$16, CX
	MOVL	CX, 8(AX)
	MOVL	mm+12(FP), SI
	MOVL	SI, 0(CX)
	MOVL	gg+16(FP), SI
	MOVL	SI, 4(CX)
	MOVL	fn+20(FP), SI
	MOVL	SI, 8(CX)
	MOVL	$1234, 12(CX)

	MOVL	$0, 0(SP)		// syscall gap
	MOVL	params+4(FP), AX
	MOVL	AX, 4(SP)		// arg 1 - param
	MOVL	psize+8(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - psize
	MOVL	$8, AX			// sys___tfork
	INT	$0x80

	// Return if tfork syscall failed.
	JCC	5(PC)
	NEGL	AX
	MOVL	ret+0(FP), DX
	MOVL	AX, 0(DX)
	RET

	// In parent, return.
	CMPL	AX, $0
	JEQ	4(PC)
	MOVL	ret+0(FP), DX
	MOVL	AX, 0(DX)
	RET

	// Paranoia: check that SP is as we expect.
	MOVL	12(SP), BP
	CMPL	BP, $1234
	JEQ	2(PC)
	INT	$3

	// Reload registers.
	MOVL	0(SP), BX		// m
	MOVL	4(SP), DX		// g
	MOVL	8(SP), SI		// fn

	// Set FS to point at m->tls.
	LEAL	m_tls(BX), BP
	PUSHAL				// save registers
	PUSHL	BP
	CALL	runtime·settls(SB)
	POPL	AX
	POPAL
	
	// Now segment is established.  Initialize m, g.
	get_tls(AX)
	MOVL	DX, g(AX)
	MOVL	BX, m(AX)

	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	MOVL	0(DX), DX		// paranoia; check they are not nil
	MOVL	0(BX), BX

	// More paranoia; check that stack splitting code works.
	PUSHAL
	CALL	runtime·emptyfunc(SB)
	POPAL

	// Call fn.
	CALL	SI

	CALL	runtime·exit1(SB)
	MOVL	$0x1234, 0x1005
	RET

TEXT runtime·sigaltstack(SB),7,$-8
	MOVL	$288, AX		// sys_sigaltstack
	MOVL	new+4(SP), BX
	MOVL	old+8(SP), CX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

TEXT runtime·setldt(SB),7,$4
	// Under OpenBSD we set the GS base instead of messing with the LDT.
	MOVL	tls0+4(FP), AX
	MOVL	AX, 0(SP)
	CALL	runtime·settls(SB)
	RET

TEXT runtime·settls(SB),7,$8
	// adjust for ELF: wants to use -8(GS) and -4(GS) for g and m
	MOVL	tlsbase+0(FP), CX
	ADDL	$8, CX
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	CX, 4(SP)		// arg 1 - tcb
	MOVL	$329, AX		// sys___set_tcb
	INT	$0x80
	JCC	2(PC)
	MOVL	$0xf1, 0xf1		// crash
	RET

TEXT runtime·osyield(SB),7,$-4
	MOVL	$298, AX		// sys_sched_yield
	INT	$0x80
	RET

TEXT runtime·thrsleep(SB),7,$-4
	MOVL	$300, AX		// sys___thrsleep
	INT	$0x80
	RET

TEXT runtime·thrwakeup(SB),7,$-4
	MOVL	$301, AX		// sys___thrwakeup
	INT	$0x80
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

GLOBL runtime·tlsoffset(SB),$4
