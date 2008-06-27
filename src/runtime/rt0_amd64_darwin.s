// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


TEXT	_rt0_amd64_darwin(SB),1,$-8
	PUSHQ	$0
	MOVQ	SP, BP
	ANDQ	$~15, SP
	MOVQ	8(BP), DI	// argc
	LEAQ	16(BP), SI	// argv
	MOVL	DI, DX
	ADDL	$1, DX
	SHLL	$3, DX
	ADDQ	SI, DX
	MOVQ	DX, CX
	CMPQ	(CX), $0
	JEQ	done

loop:
	ADDQ	$8, CX
	CMPQ	(CX), $0
	JNE	loop

done:
	ADDQ	$8, CX
	SUBQ	$16, SP
	MOVL	DI, 0(SP)
	MOVQ	SI, 8(SP)
	CALL	args(SB)
	ADDQ	$16, SP
	CALL	check(SB)
	CALL	main·main(SB)
	CALL	sys·exit(SB)
	CALL	notok(SB)
	POPQ	AX
	RET

TEXT	FLUSH(SB),7,$-8
	RET

TEXT	sys·exit(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$(0x2000000+1), AX
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT	sys·write(SB),1,$-8
	MOVL	8(SP), DI		// arg 1 fid
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$(0x2000000+4), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT	open(SB),1,$-8
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVQ	$0, R10
	MOVL	$(0x2000000+5), AX	// syscall entry
	SYSCALL
	RET

TEXT	close(SB),1,$-8
	MOVL	8(SP), DI
	MOVL	$(0x2000000+6), AX	// syscall entry
	SYSCALL
	RET

TEXT	fstat(SB),1,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$(0x2000000+339), AX	// syscall entry; really fstat64
	SYSCALL
	RET

TEXT	read(SB),1,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$(0x2000000+3), AX	// syscall entry
	SYSCALL
	RET

TEXT	sys·sigaction(SB),1,$-8
	MOVL	8(SP), DI		// arg 1 sig
	MOVQ	16(SP), SI		// arg 2 act
	MOVQ	24(SP), DX		// arg 3 oact
	MOVQ	24(SP), CX		// arg 3 oact
	MOVQ	24(SP), R10		// arg 3 oact
	MOVL	$(0x2000000+46), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT sigtramp(SB),1,$24
	MOVL	DX,0(SP)
	MOVQ	CX,8(SP)
	MOVQ	R8,16(SP)
	CALL	sighandler(SB)
	RET

TEXT	sys·breakpoint(SB),1,$-8
	BYTE	$0xcc
	RET

TEXT	sys·mmap(SB),1,$-8
	MOVQ	8(SP), DI		// arg 1 addr
	MOVL	16(SP), SI		// arg 2 len
	MOVL	20(SP), DX		// arg 3 prot
	MOVL	24(SP), R10		// arg 4 flags
	MOVL	28(SP), R8		// arg 5 fid
	MOVL	32(SP), R9		// arg 6 offset
	MOVL	$(0x2000000+197), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
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
