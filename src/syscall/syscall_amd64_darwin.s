// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for AMD64, Darwin
//

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

TEXT	syscall·stat(SB),1,$-8
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$(0x2000000+338), AX	// syscall entry
	SYSCALL
	JCC	4(PC)
	MOVQ	$-1, 24(SP)
	MOVQ	AX, 32(SP)
	RET
	MOVQ	AX, 24(SP)
	MOVQ	$0, 32(SP)
	RET

TEXT	syscall·fstat(SB),1,$-8
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$(0x2000000+339), AX	// syscall entry
	SYSCALL
	JCC	4(PC)
	MOVQ	$-1, 24(SP)
	MOVQ	AX, 32(SP)
	RET
	MOVQ	AX, 24(SP)
	MOVQ	$0, 32(SP)
	RET
