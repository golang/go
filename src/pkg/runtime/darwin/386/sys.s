// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for 386, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.

#include "386/asm.h"

TEXT notok(SB),7,$0
	MOVL	$0xf1, 0xf1
	RET

// Exit the entire program (like C exit)
TEXT exit(SB),7,$0
	MOVL	$1, AX
	INT	$0x80
	CALL	notok(SB)
	RET

// Exit this OS thread (like pthread_exit, which eventually
// calls __bsdthread_terminate).
TEXT exit1(SB),7,$0
	MOVL	$361, AX
	INT	$0x80
	JAE 2(PC)
	CALL	notok(SB)
	RET

TEXT write(SB),7,$0
	MOVL	$4, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

TEXT sys·mmap(SB),7,$0
	MOVL	$197, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

TEXT sigaction(SB),7,$0
	MOVL	$46, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

// Sigtramp's job is to call the actual signal handler.
// It is called with the following arguments on the stack:
//	0(FP)	"return address" - ignored
//	4(FP)	actual handler
//	8(FP)	siginfo style - ignored
//	12(FP)	signal number
//	16(FP)	siginfo
//	20(FP)	context
TEXT sigtramp(SB),7,$40
	// g = m->gsignal
	MOVL	m, BP
	MOVL	m_gsignal(BP), BP
	MOVL	BP, g

	MOVL	handler+4(FP), DI
	MOVL	signo+12(FP), AX
	MOVL	siginfo+16(FP), BX
	MOVL	context+20(FP), CX

	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	MOVL	CX, 8(SP)
	CALL	DI

	MOVL	context+20(FP), CX
	MOVL	style+8(FP), BX

	MOVL	$0, 0(SP)	// "caller PC" - ignored
	MOVL	CX, 4(SP)
	MOVL	BX, 8(SP)
	MOVL	$184, AX	// sigreturn(ucontext, infostyle)
	INT	$0x80
	CALL	notok(SB)
	RET

TEXT sigaltstack(SB),7,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

// void bsdthread_create(void *stk, M *m, G *g, void (*fn)(void))
// System call args are: func arg stack pthread flags.
TEXT bsdthread_create(SB),7,$32
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
	JAE	2(PC)
	CALL	notok(SB)
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
TEXT bsdthread_start(SB),7,$0
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
	CALL	setldt(SB)
	POPL	AX
	POPL	AX
	POPL	AX
	POPAL
	SHLL	$3, DI	// segment# is ldt*8 + 7.
	ADDL	$7, DI
	MOVW	DI, GS

	// Now segment is established.  Initialize m, g.
	MOVL	AX, g
	MOVL	DX, m
	MOVL	BX, m_procid(DX)	// m->procid = thread port (for debuggers)
	CALL	CX	// fn()
	CALL	exit1(SB)
	RET

// void bsdthread_register(void)
// registers callbacks for threadstart (see bsdthread_create above
// and wqthread and pthsize (not used).  returns 0 on success.
TEXT bsdthread_register(SB),7,$40
	MOVL	$366, AX
	// 0(SP) is where kernel expects caller PC; ignored
	MOVL	$bsdthread_start(SB), 4(SP)	// threadstart
	MOVL	$0, 8(SP)	// wqthread, not used by us
	MOVL	$0, 12(SP)	// pthsize, not used by us
	MOVL	$0, 16(SP)	// paranoia
	MOVL	$0, 20(SP)
	MOVL	$0, 24(SP)
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
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
TEXT sysenter(SB),7,$0
	POPL	DX
	MOVL	SP, CX
	BYTE $0x0F; BYTE $0x34;  // SYSENTER
	// returns to DX with SP set to CX

TEXT mach_msg_trap(SB),7,$0
	MOVL	$-31, AX
	CALL	sysenter(SB)
	RET

TEXT mach_reply_port(SB),7,$0
	MOVL	$-26, AX
	CALL	sysenter(SB)
	RET

TEXT mach_task_self(SB),7,$0
	MOVL	$-28, AX
	CALL	sysenter(SB)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// uint32 mach_semaphore_wait(uint32)
TEXT mach_semaphore_wait(SB),7,$0
	MOVL	$-36, AX
	CALL	sysenter(SB)
	RET

// uint32 mach_semaphore_timedwait(uint32, uint32, uint32)
TEXT mach_semaphore_timedwait(SB),7,$0
	MOVL	$-38, AX
	CALL	sysenter(SB)
	RET

// uint32 mach_semaphore_signal(uint32)
TEXT mach_semaphore_signal(SB),7,$0
	MOVL	$-33, AX
	CALL	sysenter(SB)
	RET

// uint32 mach_semaphore_signal_all(uint32)
TEXT mach_semaphore_signal_all(SB),7,$0
	MOVL	$-34, AX
	CALL	sysenter(SB)
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
TEXT setldt(SB),7,$32
	// set up data_desc
	LEAL	16(SP), AX	// struct data_desc
	MOVL	$0, 0(AX)
	MOVL	$0, 4(AX)

	MOVL	address+4(FP), BX	// aka base
	MOVW	BX, 2(AX)
	SHRL	$16, BX
	MOVB	BX, 4(AX)
	SHRL	$8, BX
	MOVB	BX, 7(AX)

	MOVL	limit+8(FP), BX
	MOVW	BX, 0(AX)
	SHRL	$16, BX
	ANDL	$0x0F, BX
	ORL	$0x40, BX		// 32-bit operand size
	MOVB	BX, 6(AX)

	MOVL	$0xF2, 5(AX)	// r/w data descriptor, dpl=3, present

	// call i386_set_ldt(entry, desc, 1)
	MOVL	entry+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	AX, 4(SP)
	MOVL	$1, 8(SP)
	CALL	i386_set_ldt(SB)
	RET

TEXT i386_set_ldt(SB),7,$0
	MOVL	$5, AX
	INT	$0x82	// sic
	JAE	2(PC)
	CALL	notok(SB)
	RET

