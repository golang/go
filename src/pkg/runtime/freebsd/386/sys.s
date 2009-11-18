// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "386/asm.h"
	
TEXT sys_umtx_op(SB),7,$-4
	MOVL $454, AX
	INT $0x80
	RET

TEXT thr_new(SB),7,$-4
	MOVL $455, AX
	INT $0x80
	RET

TEXT thr_start(SB),7,$0
	MOVL mm+0(FP), AX
	MOVL m_g0(AX), BX
	LEAL	m_tls(AX), BP
	MOVL	0(BP), DI
	ADDL	$7, DI
	PUSHAL
	PUSHL	$32
	PUSHL	BP
	PUSHL	DI
	CALL	setldt(SB)
	POPL	AX
	POPL	AX
	POPL	AX
	POPAL
	MOVL BX, g
	MOVL AX, m
	CALL mstart(SB)
	MOVL 0, AX			// crash (not reached)

// Exit the entire program (like C exit)
TEXT exit(SB),7,$-4
	MOVL	$1, AX
	INT	$0x80
	CALL	notok(SB)
	RET

TEXT exit1(SB),7,$-4
	MOVL	$431, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

TEXT write(SB),7,$-4
	MOVL	$4, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

TEXT	notok(SB),7,$0
	MOVL	$0xf1, 0xf1
	RET

TEXT runtime·mmap(SB),7,$-4
	MOVL	$477, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

TEXT sigaction(SB),7,$-4
	MOVL	$416, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
	RET

TEXT sigtramp(SB),7,$40
	// g = m->gsignal
	MOVL	m, BP
	MOVL	m_gsignal(BP), BP
	MOVL	BP, g

	MOVL	signo+0(FP), AX
	MOVL	siginfo+4(FP), BX
	MOVL	context+8(FP), CX

	MOVL	AX, 0(SP)
	MOVL	BX, 4(SP)
	MOVL	CX, 8(SP)
	CALL	sighandler(SB)

	// g = m->curg
	MOVL	m, BP
	MOVL	m_curg(BP), BP
	MOVL	BP, g

	MOVL	context+8(FP), AX

	MOVL	$0, 0(SP)	// syscall gap
	MOVL	AX, 4(SP)
	MOVL	$417, AX	// sigreturn(ucontext)
	INT	$0x80
	CALL	notok(SB)
	RET

TEXT sigaltstack(SB),7,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	CALL	notok(SB)
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
	MOVL	address+4(FP), BX	// aka base
	MOVL	limit+8(FP), CX

	// set up data_desc
	LEAL	16(SP), AX	// struct data_desc
	MOVL	$0, 0(AX)
	MOVL	$0, 4(AX)

	MOVW	BX, 2(AX)
	SHRL	$16, BX
	MOVB	BX, 4(AX)
	SHRL	$8, BX
	MOVB	BX, 7(AX)

	MOVW	CX, 0(AX)
	SHRL	$16, CX
	ANDL	$0x0F, CX
	ORL	$0x40, CX		// 32-bit operand size
	MOVB	CX, 6(AX)

	MOVB	$0xF2, 5(AX)	// r/w data descriptor, dpl=3, present

	// call i386_set_ldt(entry, desc, 1)
	MOVL	$0xffffffff, 0(SP)	// auto-allocate entry and return in AX
	MOVL	AX, 4(SP)
	MOVL	$1, 8(SP)
	CALL	i386_set_ldt(SB)

	// compute segment selector - (entry*8+7)
	SHLL	$3, AX
	ADDL	$7, AX
	MOVW	AX, GS
	RET

TEXT i386_set_ldt(SB),7,$16
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

GLOBL tlsoffset(SB),$4
