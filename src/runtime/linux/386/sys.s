// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for 386, Linux
//

TEXT syscall(SB),7,$0
	MOVL 4(SP), AX	// syscall number
	MOVL 8(SP), BX	// arg1
	MOVL 12(SP), CX	// arg2
	MOVL 16(SP), DX	// arg3
	MOVL 20(SP), SI	// arg4
	MOVL 24(SP), DI	// arg5
	MOVL 28(SP), BP	// arg6
	INT $0x80
	CMPL AX, $0xfffff001
	JLS 2(PC)
	INT $3	// not reached
	RET

TEXT sys·Exit(SB),7,$0
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

TEXT getpid(SB),7,$0
	MOVL	$20, AX
	INT	$0x80
	RET

TEXT kill(SB),7,$0
	MOVL	$37, AX
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	INT	$0x80
	RET

TEXT sys·write(SB),7,$0
	MOVL	$4, AX		// syscall - write
	MOVL	4(SP), BX
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
	MOVL	4(FS), BP	// m
	MOVL	20(BP), AX	// m->gsignal
	MOVL	AX, 0(FS)	// g = m->gsignal
	JMP	sighandler(SB)

TEXT sigignore(SB),7,$0
	RET

TEXT sigreturn(SB),7,$0
	MOVL	4(FS), BP	// m
	MOVL	32(BP), BP	// m->curg
	MOVL	BP, 0(FS)	// g = m->curg
	MOVL	$173, AX	// rt_sigreturn
	INT $0x80
	INT $3	// not reached
	RET

TEXT sys·mmap(SB),7,$0
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
	JLS	2(PC)
	INT	$3
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
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

// int64 clone(int32 flags, void *stack, M *m, G *g, void (*fn)(void));
TEXT clone(SB),7,$0
	MOVL	$120, AX	// clone
	MOVL	flags+4(SP), BX
	MOVL	stack+8(SP), CX

	// Copy m, g, fn off parent stack for use by child.
	SUBL	$12, CX
	MOVL	m+12(SP), DX
	MOVL	DX, 0(CX)
	MOVL	g+16(SP), DX
	MOVL	DX, 4(CX)
	MOVL	fn+20(SP), DX
	MOVL	DX, 8(CX)

	MOVL	$120, AX
	INT	$0x80

	// In parent, return.
	CMPL	AX, $0
	JEQ	2(PC)
	RET

	// In child, set up new stack, etc.
	MOVL	0(CX), BX	// m
	MOVL	12(AX), AX	// fs (= m->cret)
	MOVW	AX, FS
	MOVL	8(CX), DX	// fn
	ADDL	$12, CX
	MOVL	CX, SP

	// fn is now on top of stack.

	// initialize m->procid to Linux tid
	MOVL	$224, AX
	INT	$0x80
	MOVL	AX, 20(BX)

	// call fn
	CALL	DX

	// It shouldn't return; if it does, exit.
	MOVL	$111, DI
	MOVL	$1, AX
	INT	$0x80
	JMP	-3(PC)	// keep exiting

TEXT sigaltstack(SB),7,$-8
	MOVL	$186, AX	// sigaltstack
	MOVL	new+4(SP), BX
	MOVL	old+8(SP), CX
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

//	// fake the per-goroutine and per-mach registers
//	LEAL	m0(SB),

// TODO(rsc): move to linux.s
// <asm-i386/ldt.h>
// struct user_desc {
// 	unsigned int  entry_number;
// 	unsigned long base_addr;
// 	unsigned int  limit;
// 	unsigned int  seg_32bit:1;
// 	unsigned int  contents:2;
// 	unsigned int  read_exec_only:1;
// 	unsigned int  limit_in_pages:1;
// 	unsigned int  seg_not_present:1;
// 	unsigned int  useable:1;
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
	// set up user_desc
	LEAL	16(SP), AX	// struct user_desc
	MOVL	entry+0(FP), BX	// entry
	MOVL	BX, 0(AX)
	MOVL	address+4(FP), BX	// base address
	MOVL	BX, 4(AX)
	MOVL	limit+8(FP), BX	// limit
	MOVL	BX, 8(AX)
	MOVL	$(SEG_32BIT|USEABLE|CONTENTS_DATA), 12(AX)	// flag bits

	// call modify_ldt
	MOVL	$123, 0(SP)	// syscall - modify_ldt
	MOVL	$1, 4(SP)	// func = 1 (write)
	MOVL	AX, 8(SP)	// user_desc
	MOVL	$16, 12(SP)	// sizeof(user_desc)
	CALL	syscall(SB)
	RET

