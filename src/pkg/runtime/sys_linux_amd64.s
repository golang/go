// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for AMD64, Linux
//

#include "zasm_GOOS_GOARCH.h"

TEXT runtime·exit(SB),7,$0-8
	MOVL	8(SP), DI
	MOVL	$231, AX	// exitgroup - force all os threads to exit
	SYSCALL
	RET

TEXT runtime·exit1(SB),7,$0-8
	MOVL	8(SP), DI
	MOVL	$60, AX	// exit - exit the current os thread
	SYSCALL
	RET

TEXT runtime·open(SB),7,$0-16
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	20(SP), DX
	MOVL	$2, AX			// syscall entry
	SYSCALL
	RET

TEXT runtime·close(SB),7,$0-16
	MOVL	8(SP), DI
	MOVL	$3, AX			// syscall entry
	SYSCALL
	RET

TEXT runtime·write(SB),7,$0-24
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$1, AX			// syscall entry
	SYSCALL
	RET

TEXT runtime·read(SB),7,$0-24
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$0, AX			// syscall entry
	SYSCALL
	RET

TEXT runtime·usleep(SB),7,$16
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVQ	AX, 0(SP)
	MOVQ	DX, 8(SP)

	// select(0, 0, 0, 0, &tv)
	MOVL	$0, DI
	MOVL	$0, SI
	MOVL	$0, DX
	MOVL	$0, R10
	MOVQ	SP, R8
	MOVL	$23, AX
	SYSCALL
	RET

TEXT runtime·raisesigpipe(SB),7,$12
	MOVL	$186, AX	// syscall - gettid
	SYSCALL
	MOVL	AX, DI	// arg 1 tid
	MOVL	$13, SI	// arg 2 SIGPIPE
	MOVL	$200, AX	// syscall - tkill
	SYSCALL
	RET

TEXT runtime·setitimer(SB),7,$0-24
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$38, AX			// syscall entry
	SYSCALL
	RET

TEXT runtime·mincore(SB),7,$0-24
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$27, AX			// syscall entry
	SYSCALL
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), 7, $32
	LEAQ	8(SP), DI
	MOVQ	$0, SI
	MOVQ	$0xffffffffff600000, AX
	CALL	AX
	MOVQ	8(SP), AX	// sec
	MOVL	16(SP), DX	// usec

	// sec is in AX, usec in DX
	MOVQ	AX, sec+0(FP)
	IMULQ	$1000, DX
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB), 7, $32
	LEAQ	8(SP), DI
	MOVQ	$0, SI
	MOVQ	$0xffffffffff600000, AX
	CALL	AX
	MOVQ	8(SP), AX	// sec
	MOVL	16(SP), DX	// usec

	// sec is in AX, usec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	IMULQ	$1000, DX
	ADDQ	DX, AX
	RET

TEXT runtime·rtsigprocmask(SB),7,$0-32
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	32(SP), R10
	MOVL	$14, AX			// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·rt_sigaction(SB),7,$0-32
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVQ	32(SP), R10
	MOVL	$13, AX			// syscall entry
	SYSCALL
	RET

TEXT runtime·sigtramp(SB),7,$64
	get_tls(BX)

	// save g
	MOVQ	g(BX), R10
	MOVQ	R10, 40(SP)

	// g = m->gsignal
	MOVQ	m(BX), BP
	MOVQ	m_gsignal(BP), BP
	MOVQ	BP, g(BX)

	MOVQ	DI, 0(SP)
	MOVQ	SI, 8(SP)
	MOVQ	DX, 16(SP)
	MOVQ	R10, 24(SP)

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(BX)
	MOVQ	40(SP), R10
	MOVQ	R10, g(BX)
	RET

TEXT runtime·sigreturn(SB),7,$0
	MOVL	$15, AX	// rt_sigreturn
	SYSCALL
	INT $3	// not reached

TEXT runtime·mmap(SB),7,$0
	MOVQ	8(SP), DI
	MOVQ	$0, SI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	28(SP), R10
	MOVL	32(SP), R8
	MOVL	36(SP), R9

	MOVL	$9, AX			// mmap
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	3(PC)
	NOTQ	AX
	INCQ	AX
	RET

TEXT runtime·munmap(SB),7,$0
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	$11, AX	// munmap
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·madvise(SB),7,$0
	MOVQ	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVQ	$28, AX	// madvise
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·notok(SB),7,$0
	MOVQ	$0xf1, BP
	MOVQ	BP, (BP)
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),7,$0
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	20(SP), DX
	MOVQ	24(SP), R10
	MOVQ	32(SP), R8
	MOVL	40(SP), R9
	MOVL	$202, AX
	SYSCALL
	RET

// int64 clone(int32 flags, void *stack, M *m, G *g, void (*fn)(void));
TEXT runtime·clone(SB),7,$0
	MOVL	flags+8(SP), DI
	MOVQ	stack+16(SP), SI

	// Copy m, g, fn off parent stack for use by child.
	// Careful: Linux system call clobbers CX and R11.
	MOVQ	mm+24(SP), R8
	MOVQ	gg+32(SP), R9
	MOVQ	fn+40(SP), R12

	MOVL	$56, AX
	SYSCALL

	// In parent, return.
	CMPQ	AX, $0
	JEQ	2(PC)
	RET

	// In child, on new stack.
	MOVQ	SI, SP

	// Initialize m->procid to Linux tid
	MOVL	$186, AX	// gettid
	SYSCALL
	MOVQ	AX, m_procid(R8)

	// Set FS to point at m->tls.
	LEAQ	m_tls(R8), DI
	CALL	runtime·settls(SB)

	// In child, set up new stack
	get_tls(CX)
	MOVQ	R8, m(CX)
	MOVQ	R9, g(CX)
	CALL	runtime·stackcheck(SB)

	// Call fn
	CALL	R12

	// It shouldn't return.  If it does, exit
	MOVL	$111, DI
	MOVL	$60, AX
	SYSCALL
	JMP	-3(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),7,$-8
	MOVQ	new+8(SP), DI
	MOVQ	old+16(SP), SI
	MOVQ	$131, AX
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	2(PC)
	CALL	runtime·notok(SB)
	RET

// set tls base to DI
TEXT runtime·settls(SB),7,$32
	ADDQ	$16, DI	// ELF wants to use -16(FS), -8(FS)

	MOVQ	DI, SI
	MOVQ	$0x1002, DI	// ARCH_SET_FS
	MOVQ	$158, AX	// arch_prctl
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	2(PC)
	CALL	runtime·notok(SB)
	RET

TEXT runtime·osyield(SB),7,$0
	MOVL	$24, AX
	SYSCALL
	RET
