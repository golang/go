// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for AMD64, Linux
//

//TEXT	sys·exit(SB),1,$0-8
//	MOVL	8(SP), DI
//	MOVL	$60, AX
//	SYSCALL
//	RET

TEXT	syscall·open(SB),1,$0-16
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	$0, DX
	MOVQ	$2, AX			// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	5(PC)
	MOVQ	$-1, 24(SP)
	NEGQ	AX
	MOVQ	AX, 32(SP)
	RET
	MOVQ	AX, 24(SP)
	MOVQ	$0, 32(SP)
	RET

TEXT	syscall·close(SB),1,$0-16
	MOVQ	8(SP), DI
	MOVL	$3, AX			// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	5(PC)
	MOVQ	$-1, 16(SP)
	NEGQ	AX
	MOVQ	AX, 24(SP)
	RET
	MOVQ	AX, 16(SP)
	MOVQ	$0, 24(SP)
	RET

//TEXT	fstat(SB),1,$0-16
//	MOVL	8(SP), DI
//	MOVQ	16(SP), SI
//	MOVL	$5, AX			// syscall entry
//	SYSCALL
//	RET

TEXT	syscall·read(SB),1,$0-16
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$0, AX			// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	5(PC)
	MOVQ	$-1, 32(SP)
	NEGQ	AX
	MOVQ	AX, 40(SP)
	RET
	MOVQ	AX, 32(SP)
	MOVQ	$0, 40(SP)
	RET

TEXT	syscall·write(SB),1,$0-16
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$1, AX			// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	5(PC)
	MOVQ	$-1, 32(SP)
	NEGQ	AX
	MOVQ	AX, 40(SP)
	RET
	MOVQ	AX, 32(SP)
	MOVQ	$0, 40(SP)
	RET

//TEXT	sys·rt_sigaction(SB),1,$0-32
//	MOVL	8(SP), DI
//	MOVQ	16(SP), SI
//	MOVQ	24(SP), DX
//	MOVQ	32(SP), CX
//	MOVL	CX, R10
//	MOVL	$13, AX			// syscall entry
//	SYSCALL
//	RET
//
//TEXT	sigtramp(SB),1,$24-16
//	MOVQ	DI,0(SP)
//	MOVQ	SI,8(SP)
//	MOVQ	DX,16(SP)
//	CALL	sighandler(SB)
//	RET
//
//TEXT	sys·mmap(SB),1,$0-32
//	MOVQ	8(SP), DI
//	MOVL	16(SP), SI
//	MOVL	20(SP), DX
//	MOVL	24(SP), CX
//	MOVL	28(SP), R8
//	MOVL	32(SP), R9
//
///* flags arg for ANON is 1000 but sb 20 */
//	MOVL	CX, AX
//	ANDL	$~0x1000, CX
//	ANDL	$0x1000, AX
//	SHRL	$7, AX
//	ORL	AX, CX
//
//	MOVL	CX, R10
//	MOVL	$9, AX			// syscall entry
//	SYSCALL
//	CMPQ	AX, $0xfffffffffffff001
//	JLS	2(PC)
//	CALL	notok(SB)
//	RET
//
