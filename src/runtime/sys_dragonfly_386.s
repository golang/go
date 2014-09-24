// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"
#include "textflag.h"
	
TEXT runtime·sys_umtx_sleep(SB),NOSPLIT,$-4
	MOVL	$469, AX		// umtx_sleep
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·sys_umtx_wakeup(SB),NOSPLIT,$-4
	MOVL	$470, AX		// umtx_wakeup
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·lwp_create(SB),NOSPLIT,$-4
	MOVL	$495, AX		// lwp_create
	INT	$0x80
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·lwp_start(SB),NOSPLIT,$0

	// Set GS to point at m->tls.
	MOVL	mm+0(FP), BX
	MOVL	m_g0(BX), DX
	LEAL	m_tls(BX), BP
	PUSHAL
	PUSHL	BP
	CALL	runtime·settls(SB)
	POPL	AX
	POPAL
	
	// Now segment is established.  Initialize m, g.
	get_tls(CX)
	MOVL	BX, g_m(DX)
	MOVL	DX, g(CX)

	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	MOVL	0(DX), DX		// paranoia; check they are not nil
	MOVL	0(BX), BX

	// More paranoia; check that stack splitting code works.
	PUSHAL
	CALL	runtime·emptyfunc(SB)
	POPAL

	CALL	runtime·mstart(SB)

	CALL	runtime·exit1(SB)
	MOVL	$0x1234, 0x1005
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVL	$1, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·exit1(SB),NOSPLIT,$16
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	$0x10000, 4(SP)		// arg 1 - how (EXTEXIT_LWP)
	MOVL	$0, 8(SP)		// arg 2 - status
	MOVL	$0, 12(SP)		// arg 3 - addr
	MOVL	$494, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·open(SB),NOSPLIT,$-4
	MOVL	$5, AX
	INT	$0x80
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·close(SB),NOSPLIT,$-4
	MOVL	$6, AX
	INT	$0x80
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-4
	MOVL	$3, AX
	INT	$0x80
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-4
	MOVL	$4, AX
	INT	$0x80
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$-4
	MOVL	$194, AX
	INT	$0x80
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$16
	MOVL	$496, AX		// lwp_gettid
	INT	$0x80
	MOVL	$0, 0(SP)
	MOVL	$-1, 4(SP)		// arg 1 - pid
	MOVL	AX, 8(SP)		// arg 2 - tid
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)		// arg 3 - signum
	MOVL	$497, AX		// lwp_kill
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
	MOVL	$73, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-4
	MOVL	$75, AX	// madvise
	INT	$0x80
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $-4
	MOVL	$83, AX
	INT	$0x80
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVL	$232, AX
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)	// CLOCK_REALTIME
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
TEXT runtime·nanotime(SB), NOSPLIT, $32
	MOVL	$232, AX
	LEAL	12(SP), BX
	MOVL	$4, 4(SP)	// CLOCK_MONOTONIC
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

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET


TEXT runtime·sigaction(SB),NOSPLIT,$-4
	MOVL	$342, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$44
	get_tls(CX)

	// check that g exists
	MOVL	g(CX), DI
	CMPL	DI, $0
	JNE	6(PC)
	MOVL	signo+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	$runtime·badsignal(SB), AX
	CALL	AX
	JMP 	sigtramp_ret

	// save g
	MOVL	DI, 20(SP)
	
	// g = m->gsignal
	MOVL	g_m(DI), BX
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

sigtramp_ret:
	// call sigreturn
	MOVL	context+8(FP), AX
	MOVL	$0, 0(SP)	// syscall gap
	MOVL	AX, 4(SP)
	MOVL	$344, AX	// sigreturn(ucontext)
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep(SB),NOSPLIT,$20
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

TEXT runtime·setldt(SB),NOSPLIT,$4
	// Under DragonFly we set the GS base instead of messing with the LDT.
	MOVL	tls0+4(FP), AX
	MOVL	AX, 0(SP)
	CALL	runtime·settls(SB)
	RET

TEXT runtime·settls(SB),NOSPLIT,$24
	// adjust for ELF: wants to use -8(GS) and -4(GS) for g and m
	MOVL	tlsbase+0(FP), CX
	ADDL	$8, CX

	// Set up a struct tls_info - a size of -1 maps the whole address
	// space and is required for direct-tls access of variable data
	// via negative offsets.
	LEAL	16(SP), BX
	MOVL	CX, 16(SP)		// base
	MOVL	$-1, 20(SP)		// size

	// set_tls_area returns the descriptor that needs to be loaded into GS.
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	$0, 4(SP)		// arg 1 - which
	MOVL	BX, 8(SP)		// arg 2 - tls_info
	MOVL	$8, 12(SP)		// arg 3 - infosize
	MOVL    $472, AX                // set_tls_area
	INT     $0x80
	JCC     2(PC)
	MOVL    $0xf1, 0xf1             // crash
	MOVW	AX, GS
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
	JCC	4(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$331, AX		// sys_sched_yield
	INT	$0x80
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$16
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	$3, 4(SP)		// arg 1 - how (SIG_SETMASK)
	MOVL	new+0(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - set
	MOVL	old+4(FP), AX
	MOVL	AX, 12(SP)		// arg 3 - oset
	MOVL	$340, AX		// sys_sigprocmask
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$362, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	$363, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

// int32 runtime·closeonexec(int32 fd);
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

GLOBL runtime·tlsoffset(SB),NOPTR,$4
