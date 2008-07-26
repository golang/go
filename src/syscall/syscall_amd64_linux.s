// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for AMD64, Linux
//

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

TEXT	syscall·stat(SB),1,$0-16
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	$0, DX
	MOVQ	$5, AX			// syscall entry
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

TEXT	syscall·fstat(SB),1,$0-16
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	$0, DX
	MOVQ	$5, AX			// syscall entry
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
