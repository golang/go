// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


TEXT	_rt0_amd64_linux(SB),7,$-8

// copy arguments forward on an even stack


	MOVQ	0(SP), AX		// argc
	LEAQ	8(SP), BX		// argv
	ANDQ	$~7, SP
	SUBQ	$32, SP
	MOVQ	AX, 16(SP)
	MOVQ	BX, 24(SP)

// allocate the per-user block

	LEAQ	peruser<>(SB), R15	// dedicated u. register
	MOVQ	SP, AX
	SUBQ	$4096, AX
	MOVQ	AX, 0(R15)

	CALL	check(SB)

// process the arguments

	MOVL	16(SP), AX
	MOVL	AX, 0(SP)
	MOVQ	24(SP), AX
	MOVQ	AX, 8(SP)
	CALL	args(SB)

	CALL	main·main(SB)

	MOVQ	$0, AX
	MOVQ	AX, 0(SP)		// exit status
	CALL	sys·exit(SB)

	CALL	notok(SB)

	ADDQ	$32, SP
	RET

TEXT	_morestack(SB), 7, $0
	MOVQ	SP, AX
	SUBQ	$1024, AX
	MOVQ	AX, 0(R15)
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
	MOVL	16(SP), CX		// arg 2 count
	ADDL	$7, CX
	SHRL	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	RET

TEXT	sys·getcallerpc+0(SB),0,$0
	MOVQ	x+0(FP),AX
	MOVQ	-8(AX),AX
	RET

GLOBL	peruser<>(SB),$64
