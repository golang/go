// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for AMD64, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.
//
// The low 24 bits are the system call number.
// The high 8 bits specify the kind of system call: 1=Mach, 2=BSD, 3=Machine-Dependent.
//

#include "zasm_GOOS_GOARCH.h"

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),7,$0
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$(0x2000000+1), AX	// syscall entry
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

// Exit this OS thread (like pthread_exit, which eventually
// calls __bsdthread_terminate).
TEXT runtime·exit1(SB),7,$0
	MOVL	8(SP), DI		// arg 1 exit status
	MOVL	$(0x2000000+361), AX	// syscall entry
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·open(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 pathname
	MOVL	16(SP), SI		// arg 2 flags
	MOVL	20(SP), DX		// arg 3 mode
	MOVL	$(0x2000000+5), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·close(SB),7,$0
	MOVL	8(SP), DI		// arg 1 fd
	MOVL	$(0x2000000+6), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·read(SB),7,$0
	MOVL	8(SP), DI		// arg 1 fd
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$(0x2000000+3), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·write(SB),7,$0
	MOVL	8(SP), DI		// arg 1 fd
	MOVQ	16(SP), SI		// arg 2 buf
	MOVL	24(SP), DX		// arg 3 count
	MOVL	$(0x2000000+4), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·raise(SB),7,$24
	MOVL	$(0x2000000+20), AX // getpid
	SYSCALL
	MOVQ	AX, DI	// arg 1 - pid
	MOVL	sig+0(FP), SI	// arg 2 - signal
	MOVL	$1, DX	// arg 3 - posix
	MOVL	$(0x2000000+37), AX // kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB), 7, $0
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$(0x2000000+83), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·madvise(SB), 7, $0
	MOVQ	8(SP), DI		// arg 1 addr
	MOVQ	16(SP), SI		// arg 2 len
	MOVL	24(SP), DX		// arg 3 advice
	MOVL	$(0x2000000+75), AX	// syscall entry madvise
	SYSCALL
	// ignore failure - maybe pages are locked
	RET

// OS X comm page time offsets
// http://www.opensource.apple.com/source/xnu/xnu-1699.26.8/osfmk/i386/cpu_capabilities.h
#define	nt_tsc_base	0x50
#define	nt_scale	0x58
#define	nt_shift	0x5c
#define	nt_ns_base	0x60
#define	nt_generation	0x68
#define	gtod_generation	0x6c
#define	gtod_ns_base	0x70
#define	gtod_sec_base	0x78

// int64 nanotime(void)
TEXT runtime·nanotime(SB), 7, $32
	MOVQ	$0x7fffffe00000, BP	/* comm page base */
	// Loop trying to take a consistent snapshot
	// of the time parameters.
timeloop:
	MOVL	gtod_generation(BP), R8
	TESTL	R8, R8
	JZ	systime
	MOVL	nt_generation(BP), R9
	TESTL	R9, R9
	JZ	timeloop
	RDTSC
	MOVQ	nt_tsc_base(BP), R10
	MOVL	nt_scale(BP), R11
	MOVQ	nt_ns_base(BP), R12
	CMPL	nt_generation(BP), R9
	JNE	timeloop
	MOVQ	gtod_ns_base(BP), R13
	MOVQ	gtod_sec_base(BP), R14
	CMPL	gtod_generation(BP), R8
	JNE	timeloop

	// Gathered all the data we need. Compute time.
	//	((tsc - nt_tsc_base) * nt_scale) >> 32 + nt_ns_base - gtod_ns_base + gtod_sec_base*1e9
	// The multiply and shift extracts the top 64 bits of the 96-bit product.
	SHLQ	$32, DX
	ADDQ	DX, AX
	SUBQ	R10, AX
	MULQ	R11
	SHRQ	$32, AX:DX
	ADDQ	R12, AX
	SUBQ	R13, AX
	IMULQ	$1000000000, R14
	ADDQ	R14, AX
	RET

systime:
	// Fall back to system call (usually first call in this thread).
	MOVQ	SP, DI	// must be non-nil, unused
	MOVQ	$0, SI
	MOVL	$(0x2000000+116), AX
	SYSCALL
	// sec is in AX, usec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	IMULQ	$1000, DX
	ADDQ	DX, AX
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB),7,$0
	CALL	runtime·nanotime(SB)

	// generated code for
	//	func f(x uint64) (uint64, uint64) { return x/1000000000, x%100000000 }
	// adapted to reduce duplication
	MOVQ	AX, CX
	MOVQ	$1360296554856532783, AX
	MULQ	CX
	ADDQ	CX, DX
	RCRQ	$1, DX
	SHRQ	$29, DX
	MOVQ	DX, sec+0(FP)
	IMULQ	$1000000000, DX
	SUBQ	DX, CX
	MOVL	CX, nsec+8(FP)
	RET

TEXT runtime·sigprocmask(SB),7,$0
	MOVL	8(SP), DI
	MOVQ	16(SP), SI
	MOVQ	24(SP), DX
	MOVL	$(0x2000000+329), AX  // pthread_sigmask (on OS X, sigprocmask==entire process)
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaction(SB),7,$0
	MOVL	8(SP), DI		// arg 1 sig
	MOVQ	16(SP), SI		// arg 2 act
	MOVQ	24(SP), DX		// arg 3 oact
	MOVQ	24(SP), CX		// arg 3 oact
	MOVQ	24(SP), R10		// arg 3 oact
	MOVL	$(0x2000000+46), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigtramp(SB),7,$64
	get_tls(BX)

	// check that m exists
	MOVQ	m(BX), BP
	CMPQ	BP, $0
	JNE	4(PC)
	MOVL	DX, 0(SP)
	CALL	runtime·badsignal(SB)
	RET

	// save g
	MOVQ	g(BX), R10
	MOVQ	R10, 48(SP)

	// g = m->gsignal
	MOVQ	m_gsignal(BP), BP
	MOVQ	BP, g(BX)

	MOVL	DX, 0(SP)
	MOVQ	CX, 8(SP)
	MOVQ	R8, 16(SP)
	MOVQ	R10, 24(SP)

	MOVQ	R8, 32(SP)	// save ucontext
	MOVQ	SI, 40(SP)	// save infostyle
	CALL	DI

	// restore g
	get_tls(BX)
	MOVQ	48(SP), R10
	MOVQ	R10, g(BX)

	// call sigreturn
	MOVL	$(0x2000000+184), AX	// sigreturn(ucontext, infostyle)
	MOVQ	32(SP), DI	// saved ucontext
	MOVQ	40(SP), SI	// saved infostyle
	SYSCALL
	INT $3	// not reached

TEXT runtime·mmap(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 addr
	MOVQ	16(SP), SI		// arg 2 len
	MOVL	24(SP), DX		// arg 3 prot
	MOVL	28(SP), R10		// arg 4 flags
	MOVL	32(SP), R8		// arg 5 fid
	MOVL	36(SP), R9		// arg 6 offset
	MOVL	$(0x2000000+197), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·munmap(SB),7,$0
	MOVQ	8(SP), DI		// arg 1 addr
	MOVQ	16(SP), SI		// arg 2 len
	MOVL	$(0x2000000+73), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack(SB),7,$0
	MOVQ	new+8(SP), DI
	MOVQ	old+16(SP), SI
	MOVQ	$(0x2000000+53), AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep(SB),7,$16
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVQ	AX, 0(SP)  // sec
	MOVL	DX, 8(SP)  // usec

	// select(0, 0, 0, 0, &tv)
	MOVL	$0, DI
	MOVL	$0, SI
	MOVL	$0, DX
	MOVL	$0, R10
	MOVQ	SP, R8
	MOVL	$(0x2000000+93), AX
	SYSCALL
	RET

// void bsdthread_create(void *stk, M *mp, G *gp, void (*fn)(void))
TEXT runtime·bsdthread_create(SB),7,$0
	// Set up arguments to bsdthread_create system call.
	// The ones in quotes pass through to the thread callback
	// uninterpreted, so we can put whatever we want there.
	MOVQ	fn+32(SP), DI	// "func"
	MOVQ	mm+16(SP), SI	// "arg"
	MOVQ	stk+8(SP), DX	// stack
	MOVQ	gg+24(SP), R10	// "pthread"
	MOVQ	$0x01000000, R8	// flags = PTHREAD_START_CUSTOM
	MOVQ	$0, R9	// paranoia
	MOVQ	$(0x2000000+360), AX	// bsdthread_create
	SYSCALL
	JCC 3(PC)
	NEGQ	AX
	RET
	MOVL	$0, AX
	RET

// The thread that bsdthread_create creates starts executing here,
// because we registered this function using bsdthread_register
// at startup.
//	DI = "pthread"
//	SI = mach thread port
//	DX = "func" (= fn)
//	CX = "arg" (= m)
//	R8 = stack
//	R9 = flags (= 0)
//	SP = stack - C_64_REDZONE_LEN (= stack - 128)
TEXT runtime·bsdthread_start(SB),7,$0
	MOVQ	R8, SP		// empirically, SP is very wrong but R8 is right

	PUSHQ	DX
	PUSHQ	CX
	PUSHQ	SI

	// set up thread local storage pointing at m->tls.
	LEAQ	m_tls(CX), DI
	CALL	runtime·settls(SB)

	POPQ	SI
	POPQ	CX
	POPQ	DX

	get_tls(BX)
	MOVQ	CX, m(BX)
	MOVQ	SI, m_procid(CX)	// thread port is m->procid
	MOVQ	m_g0(CX), AX
	MOVQ	AX, g(BX)
	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	CALL	DX	// fn
	CALL	runtime·exit1(SB)
	RET

// void bsdthread_register(void)
// registers callbacks for threadstart (see bsdthread_create above
// and wqthread and pthsize (not used).  returns 0 on success.
TEXT runtime·bsdthread_register(SB),7,$0
	MOVQ	$runtime·bsdthread_start(SB), DI	// threadstart
	MOVQ	$0, SI	// wqthread, not used by us
	MOVQ	$0, DX	// pthsize, not used by us
	MOVQ	$0, R10	// dummy_value [sic]
	MOVQ	$0, R8	// targetconc_ptr
	MOVQ	$0, R9	// dispatchqueue_offset
	MOVQ	$(0x2000000+366), AX	// bsdthread_register
	SYSCALL
	JCC 3(PC)
	NEGQ	AX
	RET
	MOVL	$0, AX
	RET

// Mach system calls use 0x1000000 instead of the BSD's 0x2000000.

// uint32 mach_msg_trap(void*, uint32, uint32, uint32, uint32, uint32, uint32)
TEXT runtime·mach_msg_trap(SB),7,$0
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

TEXT runtime·mach_task_self(SB),7,$0
	MOVL	$(0x1000000+28), AX	// task_self_trap
	SYSCALL
	RET

TEXT runtime·mach_thread_self(SB),7,$0
	MOVL	$(0x1000000+27), AX	// thread_self_trap
	SYSCALL
	RET

TEXT runtime·mach_reply_port(SB),7,$0
	MOVL	$(0x1000000+26), AX	// mach_reply_port
	SYSCALL
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// uint32 mach_semaphore_wait(uint32)
TEXT runtime·mach_semaphore_wait(SB),7,$0
	MOVL	8(SP), DI
	MOVL	$(0x1000000+36), AX	// semaphore_wait_trap
	SYSCALL
	RET

// uint32 mach_semaphore_timedwait(uint32, uint32, uint32)
TEXT runtime·mach_semaphore_timedwait(SB),7,$0
	MOVL	8(SP), DI
	MOVL	12(SP), SI
	MOVL	16(SP), DX
	MOVL	$(0x1000000+38), AX	// semaphore_timedwait_trap
	SYSCALL
	RET

// uint32 mach_semaphore_signal(uint32)
TEXT runtime·mach_semaphore_signal(SB),7,$0
	MOVL	8(SP), DI
	MOVL	$(0x1000000+33), AX	// semaphore_signal_trap
	SYSCALL
	RET

// uint32 mach_semaphore_signal_all(uint32)
TEXT runtime·mach_semaphore_signal_all(SB),7,$0
	MOVL	8(SP), DI
	MOVL	$(0x1000000+34), AX	// semaphore_signal_all_trap
	SYSCALL
	RET

// set tls base to DI
TEXT runtime·settls(SB),7,$32
	/*
	* Same as in sys_darwin_386.s:/ugliness, different constant.
	* See cgo/gcc_darwin_amd64.c for the derivation
	* of the constant.
	*/
	SUBQ $0x8a0, DI

	MOVL	$(0x3000000+3), AX	// thread_fast_set_cthread_self - machdep call #3
	SYSCALL
	RET

TEXT runtime·sysctl(SB),7,$0
	MOVQ	8(SP), DI
	MOVL	16(SP), SI
	MOVQ	24(SP), DX
	MOVQ	32(SP), R10
	MOVQ	40(SP), R8
	MOVQ	48(SP), R9
	MOVL	$(0x2000000+202), AX	// syscall entry
	SYSCALL
	JCC 3(PC)
	NEGQ	AX
	RET
	MOVL	$0, AX
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),7,$0
	MOVQ    $0, DI
	MOVQ    $0, SI
	MOVQ    $0, DX
	MOVL	$(0x2000000+362), AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),7,$0
	MOVL    8(SP), DI
	MOVQ    16(SP), SI
	MOVL    24(SP), DX
	MOVQ    32(SP), R10
	MOVL    40(SP), R8
	MOVQ    48(SP), R9
	MOVL	$(0x2000000+363), AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),7,$0
	MOVL    8(SP), DI  // fd
	MOVQ    $2, SI  // F_SETFD
	MOVQ    $1, DX  // FD_CLOEXEC
	MOVL	$(0x2000000+92), AX  // fcntl
	SYSCALL
	RET
