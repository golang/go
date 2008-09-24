// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for AMD64, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.
//

// Exit the entire program (like C exit)
TEXT	sys·exit(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$(0x2000000+1), AX	// syscall entry
	SYSCALL
	CALL	notok(SB)
	RET

// Exit this OS thread (like pthread_exit, which eventually
// calls __bsdthread_terminate).
TEXT	exit1(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$(0x2000000+361), AX	// syscall entry
	SYSCALL
	CALL	notok(SB)
	RET

TEXT	sys·write(SB),7,$-8
	MOVL	8(SP), DI		// arg 1 fid
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$(0x2000000+4), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

TEXT	open(SB),7,$-8
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	20(SP), DX
	MOVQ	$0, R10
	MOVL	$(0x2000000+5), AX	// syscall entry
	SYSCALL
	RET

TEXT	close(SB),7,$-8
	MOVL	8(SP), DI
	MOVL	$(0x2000000+6), AX	// syscall entry
	SYSCALL
	RET

TEXT	fstat(SB),7,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	$(0x2000000+339), AX	// syscall entry; really fstat64
	SYSCALL
	RET

TEXT	read(SB),7,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$(0x2000000+3), AX	// syscall entry
	SYSCALL
	RET

TEXT	write(SB),7,$-8
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVL	24(SP), DX
	MOVL	$(0x2000000+4), AX	// syscall entry
	SYSCALL
	RET

TEXT	sys·sigaction(SB),7,$-8
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

TEXT sigtramp(SB),7,$24
	MOVQ	32(R14), R15	// g = m->gsignal
	MOVL	DX,0(SP)
	MOVQ	CX,8(SP)
	MOVQ	R8,16(SP)
	CALL	sighandler(SB)
	RET

TEXT	sys·mmap(SB),7,$-8
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

TEXT	notok(SB),7,$-8
	MOVL	$0xf1, BP
	MOVQ	BP, (BP)
	RET

TEXT	sys·memclr(SB),7,$-8
	MOVQ	8(SP), DI		// arg 1 addr
	MOVL	16(SP), CX		// arg 2 count
	ADDL	$7, CX
	SHRL	$3, CX
	MOVQ	$0, AX
	CLD
	REP
	STOSQ
	RET

TEXT	sys·getcallerpc+0(SB),7,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	-8(AX),AX		// get calling pc
	RET

TEXT	sys·setcallerpc+0(SB),7,$0
	MOVQ	x+0(FP),AX		// addr of first arg
	MOVQ	x+8(FP), BX
	MOVQ	BX, -8(AX)		// set calling pc
	RET

TEXT sigaltstack(SB),7,$-8
	MOVQ	new+8(SP), DI
	MOVQ	old+16(SP), SI
	MOVQ	$(0x2000000+53), AX
	SYSCALL
	JCC	2(PC)
	CALL	notok(SB)
	RET

// void bsdthread_create(void *stk, M *m, G *g, void (*fn)(void))
TEXT bsdthread_create(SB),7,$-8
	// Set up arguments to bsdthread_create system call.
	// The ones in quotes pass through to the thread callback
	// uninterpreted, so we can put whatever we want there.
	MOVQ	fn+32(SP), DI	// "func"
	MOVQ	m+16(SP), SI	// "arg"
	MOVQ	stk+8(SP), DX	// stack
	MOVQ	g+24(SP), R10	// "pthread"
	MOVQ	$0, R10	// flags
	MOVQ	$(0x2000000+360), AX	// bsdthread_create
	SYSCALL
	JCC 2(PC)
	CALL	notok(SB)
	RET

// The thread that bsdthread_create creates starts executing here,
// because we registered this function using bsdthread_register
// at startup.
//	DI = "pthread" (= g)
//	SI = mach thread port
//	DX = "func" (= fn)
//	CX = "arg" (= m)
//	R8 = stack
//	R9 = flags (= 0)
//	SP = stack - C_64_REDZONE_LEN (= stack - 128)
TEXT bsdthread_start(SB),7,$-8
	MOVQ	CX, R14	// m
	MOVQ	DI, R15	// g
	MOVQ	SI, 24(R14)	// thread port is m->procid
	CALL	DX	// fn
	CALL	exit1(SB)
	RET

// void bsdthread_register(void)
// registers callbacks for threadstart (see bsdthread_create above
// and wqthread and pthsize (not used).  returns 0 on success.
TEXT bsdthread_register(SB),7,$-8
	MOVQ	$bsdthread_start(SB), DI	// threadstart
	MOVQ	$0, SI	// wqthread, not used by us
	MOVQ	$0, DX	// pthsize, not used by us
	MOVQ	$(0x2000000+366), AX	// bsdthread_register
	SYSCALL
	JCC 2(PC)
	CALL	notok(SB)
	RET

// Mach system calls use 0x1000000 instead of the BSD's 0x2000000.

// uint32 mach_msg_trap(void*, uint32, uint32, uint32, uint32, uint32, uint32)
TEXT mach_msg_trap(SB),7,$0
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVL	20(SP), DX
	MOVL	24(SP), R10
	MOVL	28(SP), R8
	MOVL	32(SP), R9
	MOVL	36(SP), R11
	PUSHQ	R11	// seventh arg, on stack
	MOVL	$(0x1000000+31), AX	// mach_msg_trap
	SYSCALL
	POPQ	R11
	RET

TEXT mach_task_self(SB),7,$0
	MOVL	$(0x1000000+28), AX	// task_self_trap
	SYSCALL
	RET

TEXT mach_thread_self(SB),7,$0
	MOVL	$(0x1000000+27), AX	// thread_self_trap
	SYSCALL
	RET

TEXT mach_reply_port(SB),7,$0
	MOVL	$(0x1000000+26), AX	// mach_reply_port
	SYSCALL
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// uint32 mach_semaphore_wait(uint32)
TEXT mach_semaphore_wait(SB),7,$0
	MOVL	8(SP), DI
	MOVL	$(0x1000000+36), AX	// semaphore_wait_trap
	SYSCALL
	RET

// uint32 mach_semaphore_timedwait(uint32, uint32, uint32)
TEXT mach_semaphore_timedwait(SB),7,$0
	MOVL	8(SP), DI
	MOVL	12(SP), SI
	MOVL	16(SP), DX
	MOVL	$(0x1000000+38), AX	// semaphore_timedwait_trap
	SYSCALL
	RET

// uint32 mach_semaphore_signal(uint32)
TEXT mach_semaphore_signal(SB),7,$0
	MOVL	8(SP), DI
	MOVL	$(0x1000000+33), AX	// semaphore_signal_trap
	SYSCALL
	RET

// uint32 mach_semaphore_signal_all(uint32)
TEXT mach_semaphore_signal_all(SB),7,$0
	MOVL	8(SP), DI
	MOVL	$(0x1000000+34), AX	// semaphore_signal_all_trap
	SYSCALL
	RET

