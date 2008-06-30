// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


TEXT	_rt0_amd64_linux(SB),7,$-8

	// copy arguments forward on an even stack

	MOVQ	0(SP), AX		// argc
	LEAQ	8(SP), BX		// argv
	SUBQ	$(4*8+7), SP		// 2args 2auto
	ANDQ	$~7, SP
	MOVQ	AX, 16(SP)
	MOVQ	BX, 24(SP)

	// allocate the per-user block

	LEAQ	peruser<>(SB), R15	// dedicated u. register

	LEAQ	(-4096+104+4*8)(SP), AX
	MOVQ	AX, 0(R15)		// 0(R15) is stack limit (w 104b guard)

	MOVL	$1024, AX
	MOVL	AX, 0(SP)
	CALL	mal(SB)

	LEAQ	104(AX), BX
	MOVQ	BX, 16(R15)		// 16(R15) is limit of istack (w 104b guard)

	ADDQ	0(SP), AX
	LEAQ	(-4*8)(AX), BX
	MOVQ	BX, 24(R15)		// 24(R15) is base of istack (w auto*4)

	CALL	check(SB)

	// process the arguments

	MOVL	16(SP), AX		// copy argc
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX		// copy argv
	MOVQ	AX, 8(SP)
	CALL	args(SB)

	CALL	main·main(SB)

	MOVQ	$0, AX
	MOVQ	AX, 0(SP)		// exit status
	CALL	sys·exit(SB)

	CALL	notok(SB)		// fault
	RET

//
// the calling sequence for a routine that
// needs N bytes stack, A args.
//
//	N1 = (N+160 > 4096)? N+160: 0
//	A1 = A
//
// if N <= 75
//	CMPQ	SP, 0(R15)
//	JHI	3(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	CALL	_morestack
//
// if N > 75
//	LEAQ	(-N-75)(SP), AX
//	CMPQ	AX, 0(R15)
//	JHI	3(PC)
//	MOVQ	$(N1<<0) | (A1<<32)), AX
//	CALL	_morestack
//

TEXT	_morestack(SB), 7, $0
	// save stuff on interrupt stack

	MOVQ	24(R15), BX		// istack
	MOVQ	SP, 8(BX)		// old SP
	MOVQ	AX, 16(BX)		// magic number
	MOVQ	0(R15), AX		// old limit
	MOVQ	AX, 24(BX)

	// switch and set up new limit

	MOVQ	BX, SP
	MOVQ	16(R15), AX		// istack limit
	MOVQ	AX, 0(R15)

	// allocate a new stack max of request and 4k

	MOVL	16(SP), AX		// magic number
	CMPL	AX, $4096
	JHI	2(PC)
	MOVL	$4096, AX
	MOVL	AX, 0(SP)
	CALL	mal(SB)

	// switch to new stack

	MOVQ	SP, BX			// istack
	ADDQ	$104, AX		// new stack limit
	MOVQ	AX, 0(R15)
	ADDQ	0(SP), AX
	LEAQ	(-104-4*8)(AX), SP	// new SP
	MOVQ	8(R15), AX
	MOVQ	AX, 0(SP)		// old base
	MOVQ	SP, 8(R15)		// new base

	// copy needed stuff from istack to new stack

	MOVQ	16(BX), AX		// magic number
	MOVQ	AX, 16(SP)
	MOVQ	24(BX), AX		// old limit
	MOVQ	AX, 24(SP)
	MOVQ	8(BX), AX		// old SP
	MOVQ	AX, 8(SP)

// are there parameters

	MOVL	20(SP), CX		// copy count
	CMPL	CX, $0
	JEQ	easy

// copy in

	LEAQ	16(AX), SI
	SUBQ	CX, SP
	MOVQ	SP, DI
	SHRL	$3, CX
	CLD
	REP
	MOVSQ

	// call the intended
	CALL	0(AX)

// copy out

	MOVQ	SP, SI
	MOVQ	8(R15), BX		// new base
	MOVQ	8(BX), AX		// old SP
	LEAQ	16(AX), DI
	MOVL	20(BX), CX		// copy count
	SHRL	$3, CX
	CLD
	REP
	MOVSQ

	// restore old SP and limit
	MOVQ	8(R15), SP		// new base
	MOVQ	24(SP), AX		// old limit
	MOVQ	AX, 0(R15)
	MOVQ	0(SP), AX
	MOVQ	AX, 8(R15)		// old base
	MOVQ	8(SP), AX		// old SP
	MOVQ	AX, SP

	// and return to the call behind mine
	ADDQ	$8, SP
	RET

easy:
	CALL	0(AX)

	// restore old SP and limit
	MOVQ	24(SP), AX		// old limit
	MOVQ	AX, 0(R15)
	MOVQ	0(SP), AX
	MOVQ	AX, 8(R15)		// old base
	MOVQ	8(SP), AX		// old SP
	MOVQ	AX, SP

	// and return to the call behind mine
	ADDQ	$8, SP
	RET

TEXT	FLUSH(SB),7,$-8
	RET

TEXT	sys·exit(SB),1,$-8
	MOVL	8(SP), DI
	MOVL	$60, AX
	SYSCALL
	RET

TEXT	sys·write(SB),1,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$1, AX			// syscall entry
	SYSCALL
	RET

TEXT	open(SB),1,$-8
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	$2, AX			// syscall entry
	SYSCALL
	RET

TEXT	close(SB),1,$-8
	MOVL	8(SP), DI
	MOVL	$3, AX			// syscall entry
	SYSCALL
	RET

TEXT	fstat(SB),1,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$5, AX			// syscall entry
	SYSCALL
	RET

TEXT	read(SB),1,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$0, AX			// syscall entry
	SYSCALL
	RET

TEXT	sys·rt_sigaction(SB),1,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVQ	32(SP), CX
	MOVL	CX, R10
	MOVL	$13, AX			// syscall entry
	SYSCALL
	RET

TEXT	sigtramp(SB),1,$24
	MOVQ	DI,0(SP)
	MOVQ	SI,8(SP)
	MOVQ	DX,16(SP)
	CALL	sighandler(SB)
	RET

TEXT	sys·breakpoint(SB),1,$-8
	BYTE	$0xcc
	RET

TEXT	sys·mmap(SB),1,$-8
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	20(SP), DX
	MOVL	24(SP), CX
	MOVL	28(SP), R8
	MOVL	32(SP), R9

/* flags arg for ANON is 1000 but sb 20 */
	MOVL	CX, AX
	ANDL	$~0x1000, CX
	ANDL	$0x1000, AX
	SHRL	$7, AX
	ORL	AX, CX

	MOVL	CX, R10
	MOVL	$9, AX			// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	2(PC)
	CALL	notok(SB)
	RET

TEXT	notok(SB),1,$-8
	MOVL	$0xf1, BP
	MOVQ	BP, (BP)
	RET

TEXT	sys·memclr(SB),1,$-8
	MOVQ	8(SP), DI		// arg 1 addr
	MOVL	16(SP), CX		// arg 2 count (cannot be zero)
	ADDL	$7, CX
	SHRL	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	RET

TEXT	sys·getcallerpc+0(SB),1,$0
	MOVQ	x+0(FP),AX
	MOVQ	-8(AX),AX
	RET

GLOBL	peruser<>(SB),$64
