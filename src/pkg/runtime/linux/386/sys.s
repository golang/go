// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for 386, Linux
//

#include "386/asm.h"

TEXT exit(SB),7,$0
	MOVL	$252, AX	// syscall number
	MOVL	4(SP), BX
	INT	$0x80
	INT $3	// not reached
	RET

TEXT exit1(SB),7,$0
	MOVL	$1, AX	// exit - exit the current os thread
	MOVL	4(SP), BX
	INT	$0x80
	INT $3	// not reached
	RET

TEXT write(SB),7,$0
	MOVL	$4, AX		// syscall - write
	MOVL	4(SP),  BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	INT	$0x80
	RET

TEXT rt_sigaction(SB),7,$0
	MOVL	$174, AX		// syscall - rt_sigaction
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	INT	$0x80
	RET

TEXT sigtramp(SB),7,$0
	MOVL	m, BP
	MOVL	m_gsignal(BP), AX
	MOVL	AX, g
	JMP	sighandler(SB)

TEXT sigignore(SB),7,$0
	RET

TEXT sigreturn(SB),7,$0
	// g = m->curg
	MOVL	m, BP
	MOVL	m_curg(BP), BP
	MOVL	BP, g
	MOVL	$173, AX	// rt_sigreturn
	INT $0x80
	INT $3	// not reached
	RET

TEXT runtime·mmap(SB),7,$0
	MOVL	$192, AX	// mmap2
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	MOVL	20(SP), DI
	MOVL	24(SP), BP
	SHRL	$12, BP
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	3(PC)
	NOTL	AX
	INCL	AX
	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT futex(SB),7,$0
	MOVL	$240, AX	// futex
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	MOVL	20(SP), DI
	MOVL	24(SP), BP
	INT	$0x80
	RET

// int32 clone(int32 flags, void *stack, M *m, G *g, void (*fn)(void));
TEXT clone(SB),7,$0
	MOVL	$120, AX	// clone
	MOVL	flags+4(SP), BX
	MOVL	stack+8(SP), CX
	MOVL	$0, DX	// parent tid ptr
	MOVL	$0, DI	// child tid ptr

	// Copy m, g, fn off parent stack for use by child.
	SUBL	$16, CX
	MOVL	mm+12(SP), SI
	MOVL	SI, 0(CX)
	MOVL	gg+16(SP), SI
	MOVL	SI, 4(CX)
	MOVL	fn+20(SP), SI
	MOVL	SI, 8(CX)
	MOVL	$1234, 12(CX)

	INT	$0x80

	// In parent, return.
	CMPL	AX, $0
	JEQ	2(PC)
	RET

	// Paranoia: check that SP is as we expect.
	MOVL	12(SP), BP
	CMPL	BP, $1234
	JEQ	2(PC)
	INT	$3

	// Initialize AX to Linux tid
	MOVL	$224, AX
	INT	$0x80

	// In child on new stack.  Reload registers (paranoia).
	MOVL	0(SP), BX	// m
	MOVL	4(SP), DX	// g
	MOVL	8(SP), CX	// fn

	MOVL	AX, m_procid(BX)	// save tid as m->procid

	// set up ldt 7+id to point at m->tls.
	// m->tls is at m+40.  newosproc left the id in tls[0].
	LEAL	m_tls(BX), BP
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
	SHLL	$3, DI	// segment# is ldt*8 + 7 (different 7 than above)
	ADDL	$7, DI
	MOVW	DI, GS

	// Now segment is established.  Initialize m, g.
	MOVL	DX, g
	MOVL	BX, m

	CALL	stackcheck(SB)	// smashes AX
	MOVL	0(DX), DX	// paranoia; check they are not nil
	MOVL	0(BX), BX

	// more paranoia; check that stack splitting code works
	PUSHAL
	CALL	emptyfunc(SB)
	POPAL

	CALL	CX	// fn()
	CALL	exit1(SB)
	MOVL	$0x1234, 0x1005
	RET

TEXT sigaltstack(SB),7,$-8
	MOVL	$186, AX	// sigaltstack
	MOVL	new+4(SP), BX
	MOVL	old+8(SP), CX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

// <asm-i386/ldt.h>
// struct user_desc {
//	unsigned int  entry_number;
//	unsigned long base_addr;
//	unsigned int  limit;
//	unsigned int  seg_32bit:1;
//	unsigned int  contents:2;
//	unsigned int  read_exec_only:1;
//	unsigned int  limit_in_pages:1;
//	unsigned int  seg_not_present:1;
//	unsigned int  useable:1;
// };
#define SEG_32BIT 0x01
// contents are the 2 bits 0x02 and 0x04.
#define CONTENTS_DATA 0x00
#define CONTENTS_STACK 0x02
#define CONTENTS_CODE 0x04
#define READ_EXEC_ONLY 0x08
#define LIMIT_IN_PAGES 0x10
#define SEG_NOT_PRESENT 0x20
#define USEABLE 0x40

// setldt(int entry, int address, int limit)
TEXT setldt(SB),7,$32
	MOVL	entry+0(FP), BX	// entry
	MOVL	address+4(FP), CX	// base address

	/*
	 * When linking against the system libraries,
	 * we use its pthread_create and let it set up %gs
	 * for us.  When we do that, the private storage
	 * we get is not at 0(GS), 4(GS), but -8(GS), -4(GS).
	 * To insulate the rest of the tool chain from this
	 * ugliness, 8l rewrites 0(GS) into -8(GS) for us.
	 * To accommodate that rewrite, we translate
	 * the address here and bump the limit to 0xffffffff (no limit)
	 * so that -8(GS) maps to 0(address).
	 */
	ADDL	$0x8, CX	// address

	// set up user_desc
	LEAL	16(SP), AX	// struct user_desc
	MOVL	BX, 0(AX)
	MOVL	CX, 4(AX)
	MOVL	$0xfffff, 8(AX)
	MOVL	$(SEG_32BIT|LIMIT_IN_PAGES|USEABLE|CONTENTS_DATA), 12(AX)	// flag bits

	// call modify_ldt
	MOVL	$1, BX	// func = 1 (write)
	MOVL	AX, CX	// user_desc
	MOVL	$16, DX	// sizeof(user_desc)
	MOVL	$123, AX	// syscall - modify_ldt
	INT	$0x80

	// breakpoint on error
	CMPL AX, $0xfffff001
	JLS 2(PC)
	INT $3

	// compute segment selector - (entry*8+7)
	MOVL	entry+0(FP), AX
	SHLL	$3, AX
	ADDL	$7, AX
	MOVW	AX, GS

	RET

