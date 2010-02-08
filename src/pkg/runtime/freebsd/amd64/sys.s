// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "amd64/asm.h"
	
TEXT sys_umtx_op(SB),7,$0
	MOVQ 8(SP), DI
	MOVL 16(SP), SI
	MOVL 20(SP), DX
	MOVQ 24(SP), R10
	MOVQ 32(SP), R8
	MOVL $454, AX
	SYSCALL
	RET

TEXT thr_new(SB),7,$0
	MOVQ 8(SP), DI
	MOVQ 16(SP), SI
	MOVL $455, AX
	SYSCALL
	RET

TEXT thr_start(SB),7,$0
	MOVQ DI, m
	MOVQ m_g0(m), g
	CALL stackcheck(SB)
	CALL mstart(SB)
	MOVQ 0, AX			// crash (not reached)


// Exit the entire program (like C exit)
TEXT	exit(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$1, AX
	SYSCALL
	CALL	notok(SB)
	RET

TEXT	exit1(SB),7,$-8
	MOVQ	8(SP), DI		// arg 1 exit status
	MOVL	$431, AX
	SYSCALL
	CALL	notok(SB)
	RET

TEXT	write(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 fd
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$4, AX
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT gettime(SB), 7, $32
	MOVL	$116, AX
	LEAQ	8(SP), DI
	SYSCALL

	MOVQ	8(SP), BX	// sec
	MOVQ	sec+0(FP), DI
	MOVQ	BX, (DI)

	MOVL	16(SP), BX	// usec
	MOVQ	usec+8(FP), DI
	MOVL	BX, (DI)
	RET


TEXT	sigaction(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 sig
	MOVQ	16(SP), SI		// arg 2 act
	MOVQ	24(SP), DX		// arg 3 oact
	MOVL	$416, AX
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT	sigtramp(SB),7,$24-16
	MOVQ	m_gsignal(m), g
	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	MOVQ	DX, 16(SP)
	CALL	sighandler(SB)
	RET

TEXT	·mmap(SB),7,$-8
	MOVQ	8(SP), DI		// arg 1 addr
	MOVL	16(SP), SI		// arg 2 len
	MOVL	20(SP), DX		// arg 3 prot
	MOVL	24(SP), R10		// arg 4 flags
	MOVL	28(SP), R8		// arg 5 fid
	MOVL	32(SP), R9		// arg 6 offset
	MOVL	$477, AX
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT	notok(SB),7,$-8
	MOVL	$0xf1, BP
	MOVQ	BP, (BP)
	RET

TEXT	·memclr(SB),7,$-8
	MOVQ	8(SP), DI		// arg 1 addr
	MOVL	16(SP), CX		// arg 2 count
	ADDL	$7, CX
	SHRL	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	RET

TEXT	·getcallerpc+0(SB),7,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	-8(AX),AX		// get calling pc
	RET

TEXT	·setcallerpc+0(SB),7,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	x+8(FP), BX
	MOVQ	BX, -8(AX)		// set calling pc
	RET

TEXT sigaltstack(SB),7,$-8
	MOVQ	new+8(SP), DI
	MOVQ	old+16(SP), SI
	MOVQ	$53, AX
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET
