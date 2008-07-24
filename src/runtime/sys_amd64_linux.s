// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for AMD64, Linux
//

TEXT	sys·exit(SB),1,$0-8
	MOVL	8(SP), DI
	MOVL	$60, AX
	SYSCALL
	RET

TEXT	open(SB),1,$0-16
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	20(SP), DX
	MOVL	$2, AX			// syscall entry
	SYSCALL
	RET

TEXT	close(SB),1,$0-8
	MOVL	8(SP), DI
	MOVL	$3, AX			// syscall entry
	SYSCALL
	RET

TEXT	fstat(SB),1,$0-16
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$5, AX			// syscall entry
	SYSCALL
	RET

TEXT	read(SB),1,$0-24
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$0, AX			// syscall entry
	SYSCALL
	RET

TEXT	write(SB),1,$0-24
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$1, AX			// syscall entry
	SYSCALL
	RET

TEXT	sys·write(SB),1,$0-24
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$1, AX			// syscall entry
	SYSCALL
	RET

TEXT	sys·rt_sigaction(SB),1,$0-32
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVQ	32(SP), CX
	MOVL	CX, R10
	MOVL	$13, AX			// syscall entry
	SYSCALL
	RET

TEXT	sigtramp(SB),1,$24-16
	MOVQ	DI,0(SP)
	MOVQ	SI,8(SP)
	MOVQ	DX,16(SP)
	CALL	sighandler(SB)
	RET

TEXT	sys·mmap(SB),1,$0-32
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

TEXT	notok(SB),7,$0
	MOVL	$0xf1, BP
	MOVQ	BP, (BP)
	RET

TEXT	sys·memclr(SB),1,$0-16
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
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	-8(AX),AX		// get calling pc
	RET

TEXT	sys·setcallerpc+0(SB),1,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	x+8(FP), BX
	MOVQ	BX, -8(AX)		// set calling pc
	RET
