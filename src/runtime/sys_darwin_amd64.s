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

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$0
	MOVL	code+0(FP), DI		// arg 1 exit status
	MOVL	$(0x2000000+1), AX	// syscall entry
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

// Exit this OS thread (like pthread_exit, which eventually
// calls __bsdthread_terminate).
TEXT runtime·exit1(SB),NOSPLIT,$0
	MOVL	code+0(FP), DI		// arg 1 exit status
	MOVL	$(0x2000000+361), AX	// syscall entry
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·open(SB),NOSPLIT,$0
	MOVQ	name+0(FP), DI		// arg 1 pathname
	MOVL	mode+8(FP), SI		// arg 2 flags
	MOVL	perm+12(FP), DX		// arg 3 mode
	MOVL	$(0x2000000+5), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVL	fd+0(FP), DI		// arg 1 fd
	MOVL	$(0x2000000+6), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVL	fd+0(FP), DI		// arg 1 fd
	MOVQ	p+8(FP), SI		// arg 2 buf
	MOVL	n+16(FP), DX		// arg 3 count
	MOVL	$(0x2000000+3), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$0
	MOVQ	fd+0(FP), DI		// arg 1 fd
	MOVQ	p+8(FP), SI		// arg 2 buf
	MOVL	n+16(FP), DX		// arg 3 count
	MOVL	$(0x2000000+4), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$0
	// Ideally we'd send the signal to the current thread,
	// not the whole process, but that's too hard on OS X.
	JMP	runtime·raiseproc(SB)

TEXT runtime·raiseproc(SB),NOSPLIT,$24
	MOVL	$(0x2000000+20), AX // getpid
	SYSCALL
	MOVQ	AX, DI	// arg 1 - pid
	MOVL	sig+0(FP), SI	// arg 2 - signal
	MOVL	$1, DX	// arg 3 - posix
	MOVL	$(0x2000000+37), AX // kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $0
	MOVL	mode+0(FP), DI
	MOVQ	new+8(FP), SI
	MOVQ	old+16(FP), DX
	MOVL	$(0x2000000+83), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·madvise(SB), NOSPLIT, $0
	MOVQ	addr+0(FP), DI		// arg 1 addr
	MOVQ	n+8(FP), SI		// arg 2 len
	MOVL	flags+16(FP), DX		// arg 3 advice
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

TEXT runtime·nanotime(SB),NOSPLIT,$0-8
	MOVQ	$0x7fffffe00000, BP	/* comm page base */
	// Loop trying to take a consistent snapshot
	// of the time parameters.
timeloop:
	MOVL	nt_generation(BP), R9
	TESTL	R9, R9
	JZ	timeloop
	RDTSC
	MOVQ	nt_tsc_base(BP), R10
	MOVL	nt_scale(BP), R11
	MOVQ	nt_ns_base(BP), R12
	CMPL	nt_generation(BP), R9
	JNE	timeloop

	// Gathered all the data we need. Compute monotonic time:
	//	((tsc - nt_tsc_base) * nt_scale) >> 32 + nt_ns_base
	// The multiply and shift extracts the top 64 bits of the 96-bit product.
	SHLQ	$32, DX
	ADDQ	DX, AX
	SUBQ	R10, AX
	MULQ	R11
	SHRQ	$32, AX:DX
	ADDQ	R12, AX
	MOVQ	runtime·startNano(SB), CX
	SUBQ	CX, AX
	MOVQ	AX, ret+0(FP)
	RET

TEXT time·now(SB), NOSPLIT, $32-24
	// Note: The 32 bytes of stack frame requested on the TEXT line
	// are used in the systime fallback, as the timeval address
	// filled in by the system call.
	MOVQ	$0x7fffffe00000, BP	/* comm page base */
	// Loop trying to take a consistent snapshot
	// of the time parameters.
timeloop:
	MOVL	gtod_generation(BP), R8
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

	// Gathered all the data we need. Compute:
	//	monotonic_time = ((tsc - nt_tsc_base) * nt_scale) >> 32 + nt_ns_base
	// The multiply and shift extracts the top 64 bits of the 96-bit product.
	SHLQ	$32, DX
	ADDQ	DX, AX
	SUBQ	R10, AX
	MULQ	R11
	SHRQ	$32, AX:DX
	ADDQ	R12, AX
	MOVQ	AX, BX
	MOVQ	runtime·startNano(SB), CX
	SUBQ	CX, BX
	MOVQ	BX, monotonic+16(FP)

	// Compute:
	//	wall_time = monotonic time - gtod_ns_base + gtod_sec_base*1e9
	// or, if gtod_generation==0, invoke the system call.
	TESTL	R8, R8
	JZ	systime
	SUBQ	R13, AX
	IMULQ	$1000000000, R14
	ADDQ	R14, AX

	// Split wall time into sec, nsec.
	// generated code for
	//	func f(x uint64) (uint64, uint64) { return x/1e9, x%1e9 }
	// adapted to reduce duplication
	MOVQ	AX, CX
	SHRQ	$9, AX
	MOVQ	$19342813113834067, DX
	MULQ	DX
	SHRQ	$11, DX
	MOVQ	DX, sec+0(FP)
	IMULQ	$1000000000, DX
	SUBQ	DX, CX
	MOVL	CX, nsec+8(FP)
	RET

systime:
	// Fall back to system call (usually first call in this thread).
	MOVQ	SP, DI
	MOVQ	$0, SI
	MOVQ	$0, DX  // required as of Sierra; Issue 16570
	MOVL	$(0x2000000+116), AX // gettimeofday
	SYSCALL
	CMPQ	AX, $0
	JNE	inreg
	MOVQ	0(SP), AX
	MOVL	8(SP), DX
inreg:
	// sec is in AX, usec in DX
	IMULQ	$1000, DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	how+0(FP), DI
	MOVQ	new+8(FP), SI
	MOVQ	old+16(FP), DX
	MOVL	$(0x2000000+329), AX  // pthread_sigmask (on OS X, sigprocmask==entire process)
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0-24
	MOVL	mode+0(FP), DI		// arg 1 sig
	MOVQ	new+8(FP), SI		// arg 2 act
	MOVQ	old+16(FP), DX		// arg 3 oact
	MOVQ	old+16(FP), CX		// arg 3 oact
	MOVQ	old+16(FP), R10		// arg 3 oact
	MOVL	$(0x2000000+46), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVQ	fn+0(FP),    AX
	MOVL	sig+8(FP),   DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP),  DX
	PUSHQ	BP
	MOVQ	SP, BP
	ANDQ	$~15, SP     // alignment for x86_64 ABI
	CALL	AX
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$40
	MOVL SI, 24(SP) // save infostyle for sigreturn below
	MOVQ R8, 32(SP) // save ctx
	MOVL DX, 0(SP)  // sig
	MOVQ CX, 8(SP)  // info
	MOVQ R8, 16(SP) // ctx
	MOVQ $runtime·sigtrampgo(SB), AX
	CALL AX
	MOVQ 32(SP), DI // ctx
	MOVL 24(SP), SI // infostyle
	MOVL $(0x2000000+184), AX
	SYSCALL
	INT $3 // not reached

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 addr
	MOVQ	n+8(FP), SI		// arg 2 len
	MOVL	prot+16(FP), DX		// arg 3 prot
	MOVL	flags+20(FP), R10		// arg 4 flags
	MOVL	fd+24(FP), R8		// arg 5 fid
	MOVL	off+28(FP), R9		// arg 6 offset
	MOVL	$(0x2000000+197), AX	// syscall entry
	SYSCALL
	MOVQ	AX, ret+32(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 addr
	MOVQ	n+8(FP), SI		// arg 2 len
	MOVL	$(0x2000000+73), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVQ	new+0(FP), DI
	MOVQ	old+8(FP), SI
	MOVQ	$(0x2000000+53), AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16
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

// func bsdthread_create(stk, arg unsafe.Pointer, fn uintptr) int32
TEXT runtime·bsdthread_create(SB),NOSPLIT,$0
	// Set up arguments to bsdthread_create system call.
	// The ones in quotes pass through to the thread callback
	// uninterpreted, so we can put whatever we want there.
	MOVQ	fn+16(FP),   DI
	MOVQ	arg+8(FP),  SI
	MOVQ	stk+0(FP),   DX
	MOVQ	$0x01000000, R8  // flags = PTHREAD_START_CUSTOM
	MOVQ	$0,          R9  // paranoia
	MOVQ	$0,          R10 // paranoia, "pthread"
	MOVQ	$(0x2000000+360), AX	// bsdthread_create
	SYSCALL
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+24(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+24(FP)
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
TEXT runtime·bsdthread_start(SB),NOSPLIT,$0
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
	MOVQ	SI, m_procid(CX)	// thread port is m->procid
	MOVQ	m_g0(CX), AX
	MOVQ	AX, g(BX)
	MOVQ	CX, g_m(AX)
	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	CALL	DX	// fn
	CALL	runtime·exit1(SB)
	RET

// func bsdthread_register() int32
// registers callbacks for threadstart (see bsdthread_create above
// and wqthread and pthsize (not used).  returns 0 on success.
TEXT runtime·bsdthread_register(SB),NOSPLIT,$0
	MOVQ	$runtime·bsdthread_start(SB), DI	// threadstart
	MOVQ	$0, SI	// wqthread, not used by us
	MOVQ	$0, DX	// pthsize, not used by us
	MOVQ	$0, R10	// dummy_value [sic]
	MOVQ	$0, R8	// targetconc_ptr
	MOVQ	$0, R9	// dispatchqueue_offset
	MOVQ	$(0x2000000+366), AX	// bsdthread_register
	SYSCALL
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+0(FP)
	RET

// Mach system calls use 0x1000000 instead of the BSD's 0x2000000.

// func mach_msg_trap(h unsafe.Pointer, op int32, send_size, rcv_size, rcv_name, timeout, notify uint32) int32
TEXT runtime·mach_msg_trap(SB),NOSPLIT,$0
	MOVQ	h+0(FP), DI
	MOVL	op+8(FP), SI
	MOVL	send_size+12(FP), DX
	MOVL	rcv_size+16(FP), R10
	MOVL	rcv_name+20(FP), R8
	MOVL	timeout+24(FP), R9
	MOVL	notify+28(FP), R11
	PUSHQ	R11	// seventh arg, on stack
	MOVL	$(0x1000000+31), AX	// mach_msg_trap
	SYSCALL
	POPQ	R11
	MOVL	AX, ret+32(FP)
	RET

TEXT runtime·mach_task_self(SB),NOSPLIT,$0
	MOVL	$(0x1000000+28), AX	// task_self_trap
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·mach_thread_self(SB),NOSPLIT,$0
	MOVL	$(0x1000000+27), AX	// thread_self_trap
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·mach_reply_port(SB),NOSPLIT,$0
	MOVL	$(0x1000000+26), AX	// mach_reply_port
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// func mach_semaphore_wait(sema uint32) int32
TEXT runtime·mach_semaphore_wait(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	$(0x1000000+36), AX	// semaphore_wait_trap
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// func mach_semaphore_timedwait(sema, sec, nsec uint32) int32
TEXT runtime·mach_semaphore_timedwait(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	sec+4(FP), SI
	MOVL	nsec+8(FP), DX
	MOVL	$(0x1000000+38), AX	// semaphore_timedwait_trap
	SYSCALL
	MOVL	AX, ret+16(FP)
	RET

// func mach_semaphore_signal(sema uint32) int32
TEXT runtime·mach_semaphore_signal(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	$(0x1000000+33), AX	// semaphore_signal_trap
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// func mach_semaphore_signal_all(sema uint32) int32
TEXT runtime·mach_semaphore_signal_all(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	$(0x1000000+34), AX	// semaphore_signal_all_trap
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$32
	/*
	* Same as in sys_darwin_386.s:/ugliness, different constant.
	* See cgo/gcc_darwin_amd64.c for the derivation
	* of the constant.
	*/
	SUBQ $0x8a0, DI

	MOVL	$(0x3000000+3), AX	// thread_fast_set_cthread_self - machdep call #3
	SYSCALL
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVQ	mib+0(FP), DI
	MOVL	miblen+8(FP), SI
	MOVQ	out+16(FP), DX
	MOVQ	size+24(FP), R10
	MOVQ	dst+32(FP), R8
	MOVQ	ndst+40(FP), R9
	MOVL	$(0x2000000+202), AX	// syscall entry
	SYSCALL
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+48(FP)
	RET

// func kqueue() int32
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVQ    $0, DI
	MOVQ    $0, SI
	MOVQ    $0, DX
	MOVL	$(0x2000000+362), AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET

// func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL    kq+0(FP), DI
	MOVQ    ch+8(FP), SI
	MOVL    nch+16(FP), DX
	MOVQ    ev+24(FP), R10
	MOVL    nev+32(FP), R8
	MOVQ    ts+40(FP), R9
	MOVL	$(0x2000000+363), AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET

// func closeonexec(fd int32)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL    fd+0(FP), DI  // fd
	MOVQ    $2, SI  // F_SETFD
	MOVQ    $1, DX  // FD_CLOEXEC
	MOVL	$(0x2000000+92), AX  // fcntl
	SYSCALL
	RET
