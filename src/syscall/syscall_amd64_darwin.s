// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for AMD64, Darwin
//

//TEXT	syscall·exit(SB),1,$-8
//	MOVL	8(SP), DI		// arg 1 exit status
//	MOVL	$(0x2000000+1), AX	// syscall entry
//	SYSCALL
//	CALL	notok(SB)
//	RET

TEXT	syscall·open(SB),1,$-8
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	$0, R10
	MOVL	$(0x2000000+5), AX	// syscall entry
	SYSCALL
	JCC	4(PC)
	MOVQ	$-1, 24(SP)
	MOVQ	AX, 32(SP)
	RET
	MOVQ	AX, 24(SP)
	MOVQ	$0, 32(SP)
	RET

TEXT	syscall·close(SB),1,$-8
	MOVL	8(SP), DI
	MOVL	$(0x2000000+6), AX	// syscall entry
	SYSCALL
	JCC	4(PC)
	MOVQ	$-1, 16(SP)
	MOVQ	AX, 24(SP)
	RET
	MOVQ	AX, 16(SP)
	MOVQ	$0, 24(SP)
	RET

TEXT	syscall·read(SB),1,$-8
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$(0x2000000+3), AX	// syscall entry
	SYSCALL
	JCC	4(PC)
	MOVQ	$-1, 32(SP)
	MOVQ	AX, 40(SP)
	RET
	MOVQ	AX, 32(SP)
	MOVQ	$0, 40(SP)
	RET

TEXT	syscall·write(SB),1,$-8
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$(0x2000000+4), AX	// syscall entry
	SYSCALL
	JCC	4(PC)
	MOVQ	$-1, 32(SP)
	MOVQ	AX, 40(SP)
	RET
	MOVQ	AX, 32(SP)
	MOVQ	$0, 40(SP)
	RET

//TEXT	fstat(SB),1,$-8
//	MOVL	8(SP), DI
//	MOVQ	16(SP), SI
//	MOVL	$(0x2000000+339), AX	// syscall entry; really fstat64
//	SYSCALL
//	RET
//
//TEXT	syscall·sigaction(SB),1,$-8
//	MOVL	8(SP), DI		// arg 1 sig
//	MOVQ	16(SP), SI		// arg 2 act
//	MOVQ	24(SP), DX		// arg 3 oact
//	MOVQ	24(SP), CX		// arg 3 oact
//	MOVQ	24(SP), R10		// arg 3 oact
//	MOVL	$(0x2000000+46), AX	// syscall entry
//	SYSCALL
//	JCC	2(PC)
//	CALL	notok(SB)
//	RET
//
//TEXT sigtramp(SB),1,$24
//	MOVL	DX,0(SP)
//	MOVQ	CX,8(SP)
//	MOVQ	R8,16(SP)
//	CALL	sighandler(SB)
//	RET
//
//TEXT	syscall·mmap(SB),1,$-8
//	MOVQ	8(SP), DI		// arg 1 addr
//	MOVL	16(SP), SI		// arg 2 len
//	MOVL	20(SP), DX		// arg 3 prot
//	MOVL	24(SP), R10		// arg 4 flags
//	MOVL	28(SP), R8		// arg 5 fid
//	MOVL	32(SP), R9		// arg 6 offset
//	MOVL	$(0x2000000+197), AX	// syscall entry
//	SYSCALL
//	JCC	2(PC)
//	CALL	notok(SB)
//	RET
