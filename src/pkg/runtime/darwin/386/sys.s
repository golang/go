// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for 386, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.

#include "386/asm.h"

TEXT runtime·notok(SB),7,$0
	MOVL	$0xf1, 0xf1
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),7,$0
	MOVL	$1, AX
	INT	$0x80
	CALL	runtime·notok(SB)
	RET

// Exit this OS thread (like pthread_exit, which eventually
// calls __bsdthread_terminate).
TEXT runtime·exit1(SB),7,$0
	MOVL	$361, AX
	INT	$0x80
	JAE 2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·write(SB),7,$0
	MOVL	$4, AX
	INT	$0x80
	RET

TEXT runtime·raisesigpipe(SB),7,$8
	get_tls(CX)
	MOVL	m(CX), DX
	MOVL	m_procid(DX), DX
	MOVL	DX, 0(SP)	// thread_port
	MOVL	$13, 4(SP)	// signal: SIGPIPE
	MOVL	$328, AX	// __pthread_kill
	INT	$0x80
	RET

TEXT runtime·mmap(SB),7,$0
	MOVL	$197, AX
	INT	$0x80
	RET

TEXT runtime·munmap(SB),7,$0
	MOVL	$73, AX
	INT	$0x80
	JAE	2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·setitimer(SB),7,$0
	MOVL	$83, AX
	INT	$0x80
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), 7, $32
	LEAL	12(SP), AX	// must be non-nil, unused
	MOVL	AX, 4(SP)
	MOVL	$0, 8(SP)	// time zone pointer
	MOVL	$116, AX
	INT	$0x80
	MOVL	DX, BX

	// sec is in AX, usec in BX
	MOVL	AX, sec+0(FP)
	MOVL	$0, sec+4(FP)
	IMULL	$1000, BX
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB), 7, $32
	LEAL	12(SP), AX	// must be non-nil, unused
	MOVL	AX, 4(SP)
	MOVL	$0, 8(SP)	// time zone pointer
	MOVL	$116, AX
	INT	$0x80
	MOVL	DX, BX

	// sec is in AX, usec in BX
	// convert to DX:AX nsec
	MOVL	$1000000000, CX
	MULL	CX
	IMULL	$1000, BX
	ADDL	BX, AX
	ADCL	$0, DX
	
	MOVL	ret+0(FP), DI
	MOVL	AX, 0(DI)
	MOVL	DX, 4(DI)
	RET

TEXT runtime·sigaction(SB),7,$0
	MOVL	$46, AX
	INT	$0x80
	JAE	2(PC)
	CALL	runtime·notok(SB)
	RET

// Sigtramp's job is to call the actual signal handler.
// It is called with the following arguments on the stack:
//	0(FP)	"return address" - ignored
//	4(FP)	actual handler
//	8(FP)	siginfo style - ignored
//	12(FP)	signal number
//	16(FP)	siginfo
//	20(FP)	context
TEXT runtime·sigtramp(SB),7,$40
	get_tls(CX)

	// save g
	MOVL	g(CX), DI
	MOVL	DI, 20(SP)

	// g = m->gsignal
	MOVL	m(CX), BP
	MOVL	m_gsignal(BP), BP
	MOVL	BP, g(CX)

	// copy arguments to sighandler
	MOVL	sig+8(FP), BX
	MOVL	BX, 0(SP)
	MOVL	info+12(FP), BX
	MOVL	BX, 4(SP)
	MOVL	context+16(FP), BX
	MOVL	BX, 8(SP)
	MOVL	DI, 12(SP)

	MOVL	handler+0(FP), BX
	CALL	BX

	// restore g
	get_tls(CX)
	MOVL	20(SP), DI
	MOVL	DI, g(CX)

	// call sigreturn
	MOVL	context+16(FP), CX
	MOVL	style+4(FP), BX
	MOVL	$0, 0(SP)	// "caller PC" - ignored
	MOVL	CX, 4(SP)
	MOVL	BX, 8(SP)
	MOVL	$184, AX	// sigreturn(ucontext, infostyle)
	INT	$0x80
	CALL	runtime·notok(SB)
	RET

TEXT runtime·sigaltstack(SB),7,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·usleep(SB),7,$32
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 24(SP)  // sec
	MOVL	DX, 28(SP)  // usec

	// select(0, 0, 0, 0, &tv)
	MOVL	$0, 0(SP)  // "return PC" - ignored
	MOVL	$0, 4(SP)
	MOVL	$0, 8(SP)
	MOVL	$0, 12(SP)
	MOVL	$0, 16(SP)
	LEAL	24(SP), AX
	MOVL	AX, 20(SP)
	MOVL	$93, AX
	INT	$0x80
	RET

// void bsdthread_create(void *stk, M *m, G *g, void (*fn)(void))
// System call args are: func arg stack pthread flags.
TEXT runtime·bsdthread_create(SB),7,$32
	MOVL	$360, AX
	// 0(SP) is where the caller PC would be; kernel skips it
	MOVL	func+12(FP), BX
	MOVL	BX, 4(SP)	// func
	MOVL	mm+4(FP), BX
	MOVL	BX, 8(SP)	// arg
	MOVL	stk+0(FP), BX
	MOVL	BX, 12(SP)	// stack
	MOVL	gg+8(FP), BX
	MOVL	BX, 16(SP)	// pthread
	MOVL	$0x1000000, 20(SP)	// flags = PTHREAD_START_CUSTOM
	INT	$0x80
	JAE	3(PC)
	NEGL	AX
	RET
	MOVL	$0, AX
	RET

// The thread that bsdthread_create creates starts executing here,
// because we registered this function using bsdthread_register
// at startup.
//	AX = "pthread" (= g)
//	BX = mach thread port
//	CX = "func" (= fn)
//	DX = "arg" (= m)
//	DI = stack top
//	SI = flags (= 0x1000000)
//	SP = stack - C_32_STK_ALIGN
TEXT runtime·bsdthread_start(SB),7,$0
	// set up ldt 7+id to point at m->tls.
	// m->tls is at m+40.  newosproc left
	// the m->id in tls[0].
	LEAL	m_tls(DX), BP
	MOVL	0(BP), DI
	ADDL	$7, DI	// m0 is LDT#7. count up.
	// setldt(tls#, &tls, sizeof tls)
	PUSHAL	// save registers
	PUSHL	$32	// sizeof tls
	PUSHL	BP	// &tls
	PUSHL	DI	// tls #
	CALL	runtime·setldt(SB)
	POPL	AX
	POPL	AX
	POPL	AX
	POPAL

	// Now segment is established.  Initialize m, g.
	get_tls(BP)
	MOVL	AX, g(BP)
	MOVL	DX, m(BP)
	MOVL	BX, m_procid(DX)	// m->procid = thread port (for debuggers)
	CALL	runtime·stackcheck(SB)		// smashes AX
	CALL	CX	// fn()
	CALL	runtime·exit1(SB)
	RET

// void bsdthread_register(void)
// registers callbacks for threadstart (see bsdthread_create above
// and wqthread and pthsize (not used).  returns 0 on success.
TEXT runtime·bsdthread_register(SB),7,$40
	MOVL	$366, AX
	// 0(SP) is where kernel expects caller PC; ignored
	MOVL	$runtime·bsdthread_start(SB), 4(SP)	// threadstart
	MOVL	$0, 8(SP)	// wqthread, not used by us
	MOVL	$0, 12(SP)	// pthsize, not used by us
	MOVL	$0, 16(SP)	// dummy_value [sic]
	MOVL	$0, 20(SP)	// targetconc_ptr
	MOVL	$0, 24(SP)	// dispatchqueue_offset
	INT	$0x80
	JAE	2(PC)
	CALL	runtime·notok(SB)
	RET

// Invoke Mach system call.
// Assumes system call number in AX,
// caller PC on stack, caller's caller PC next,
// and then the system call arguments.
//
// Can be used for BSD too, but we don't,
// because if you use this interface the BSD
// system call numbers need an extra field
// in the high 16 bits that seems to be the
// argument count in bytes but is not always.
// INT $0x80 works fine for those.
TEXT runtime·sysenter(SB),7,$0
	POPL	DX
	MOVL	SP, CX
	BYTE $0x0F; BYTE $0x34;  // SYSENTER
	// returns to DX with SP set to CX

TEXT runtime·mach_msg_trap(SB),7,$0
	MOVL	$-31, AX
	CALL	runtime·sysenter(SB)
	RET

TEXT runtime·mach_reply_port(SB),7,$0
	MOVL	$-26, AX
	CALL	runtime·sysenter(SB)
	RET

TEXT runtime·mach_task_self(SB),7,$0
	MOVL	$-28, AX
	CALL	runtime·sysenter(SB)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// uint32 mach_semaphore_wait(uint32)
TEXT runtime·mach_semaphore_wait(SB),7,$0
	MOVL	$-36, AX
	CALL	runtime·sysenter(SB)
	RET

// uint32 mach_semaphore_timedwait(uint32, uint32, uint32)
TEXT runtime·mach_semaphore_timedwait(SB),7,$0
	MOVL	$-38, AX
	CALL	runtime·sysenter(SB)
	RET

// uint32 mach_semaphore_signal(uint32)
TEXT runtime·mach_semaphore_signal(SB),7,$0
	MOVL	$-33, AX
	CALL	runtime·sysenter(SB)
	RET

// uint32 mach_semaphore_signal_all(uint32)
TEXT runtime·mach_semaphore_signal_all(SB),7,$0
	MOVL	$-34, AX
	CALL	runtime·sysenter(SB)
	RET

// setldt(int entry, int address, int limit)
// entry and limit are ignored.
TEXT runtime·setldt(SB),7,$32
	MOVL	address+4(FP), BX	// aka base

	/*
	 * When linking against the system libraries,
	 * we use its pthread_create and let it set up %gs
	 * for us.  When we do that, the private storage
	 * we get is not at 0(GS) but at 0x468(GS).
	 * To insulate the rest of the tool chain from this ugliness,
	 * 8l rewrites 0(GS) into 0x468(GS) for us.
	 * To accommodate that rewrite, we translate the
	 * address and limit here so that 0x468(GS) maps to 0(address).
	 *
	 * See ../../../../libcgo/darwin_386.c for the derivation
	 * of the constant.
	 */
	SUBL	$0x468, BX

	/*
	 * Must set up as USER_CTHREAD segment because
	 * Darwin forces that value into %gs for signal handlers,
	 * and if we don't set one up, we'll get a recursive
	 * fault trying to get into the signal handler.
	 * Since we have to set one up anyway, it might as
	 * well be the value we want.  So don't bother with
	 * i386_set_ldt.
	 */
	MOVL	BX, 4(SP)
	MOVL	$3, AX	// thread_fast_set_cthread_self - machdep call #3
	INT	$0x82	// sic: 0x82, not 0x80, for machdep call

	XORL	AX, AX
	MOVW	GS, AX
	RET

TEXT runtime·sysctl(SB),7,$0
	MOVL	$202, AX
	INT	$0x80
	JAE	3(PC)
	NEGL	AX
	RET
	MOVL	$0, AX
	RET
