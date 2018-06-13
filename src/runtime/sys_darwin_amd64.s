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
TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI		// arg 1 exit status
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	8(DI), SI		// arg 2 flags
	MOVL	12(DI), DX		// arg 3 mode
	MOVQ	0(DI), DI		// arg 1 pathname
	CALL	libc_open(SB)
	POPQ	BP
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_close(SB)
	POPQ	BP
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_read(SB)
	POPQ	BP
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVQ	0(DI), DI		// arg 1 fd
	CALL	libc_write(SB)
	POPQ	BP
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 new
	MOVQ	16(DI), DX		// arg 3 old
	MOVL	0(DI), DI		// arg 1 which
	CALL	libc_setitimer(SB)
	POPQ	BP
	RET

TEXT runtime·madvise_trampoline(SB), NOSPLIT, $0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 len
	MOVL	16(DI), DX	// arg 3 advice
	MOVQ	0(DI), DI	// arg 1 addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	POPQ	BP
	RET

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	DI, BX
	CALL	libc_mach_absolute_time(SB)
	MOVQ	AX, 0(BX)
	MOVL	timebase<>+machTimebaseInfo_numer(SB), SI
	MOVL	timebase<>+machTimebaseInfo_denom(SB), DI // atomic read
	TESTL	DI, DI
	JNE	initialized

	SUBQ	$(machTimebaseInfo__size+15)/16*16, SP
	MOVQ	SP, DI
	CALL	libc_mach_timebase_info(SB)
	MOVL	machTimebaseInfo_numer(SP), SI
	MOVL	machTimebaseInfo_denom(SP), DI
	ADDQ	$(machTimebaseInfo__size+15)/16*16, SP

	MOVL	SI, timebase<>+machTimebaseInfo_numer(SB)
	MOVL	DI, AX
	XCHGL	AX, timebase<>+machTimebaseInfo_denom(SB) // atomic write

initialized:
	MOVL	SI, 8(BX)
	MOVL	DI, 12(BX)
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·walltime_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP			// make a frame; keep stack aligned
	MOVQ	SP, BP
	// DI already has *timeval
	XORL	SI, SI // no timezone needed
	CALL	libc_gettimeofday(SB)
	POPQ	BP
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 new
	MOVQ	16(DI), DX		// arg 3 old
	MOVL	0(DI), DI		// arg 1 sig
	CALL	libc_sigaction(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 new
	MOVQ	16(DI), DX	// arg 3 old
	MOVL	0(DI), DI	// arg 1 how
	CALL	libc_pthread_sigmask(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 old
	MOVQ	0(DI), DI		// arg 1 new
	CALL	libc_sigaltstack(SB)
	TESTQ	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), BX	// signal
	CALL	libc_getpid(SB)
	MOVL	AX, DI		// arg 1 pid
	MOVL	BX, SI		// arg 2 signal
	CALL	libc_kill(SB)
	POPQ	BP
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

// This is the function registered during sigaction and is invoked when
// a signal is received. It just redirects to the Go function sigtrampgo.
TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// This runs on the signal stack, so we have lots of stack available.
	// We allocate our own stack space, because if we tell the linker
	// how much we're using, the NOSPLIT check fails.
	PUSHQ	BP
	MOVQ	SP, BP
	SUBQ	$64, SP

	// Save callee-save registers.
	MOVQ	BX, 24(SP)
	MOVQ	R12, 32(SP)
	MOVQ	R13, 40(SP)
	MOVQ	R14, 48(SP)
	MOVQ	R15, 56(SP)

	// Call into the Go signal handler
	MOVL	DI, 0(SP)  // sig
	MOVQ	SI, 8(SP)  // info
	MOVQ	DX, 16(SP) // ctx
	CALL runtime·sigtrampgo(SB)

	// Restore callee-save registers.
	MOVQ	24(SP), BX
	MOVQ	32(SP), R12
	MOVQ	40(SP), R13
	MOVQ	48(SP), R14
	MOVQ	56(SP), R15

	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP			// make a frame; keep stack aligned
	MOVQ	SP, BP
	MOVQ	DI, BX
	MOVQ	0(BX), DI		// arg 1 addr
	MOVQ	8(BX), SI		// arg 2 len
	MOVL	16(BX), DX		// arg 3 prot
	MOVL	20(BX), CX		// arg 4 flags
	MOVL	24(BX), R8		// arg 5 fid
	MOVL	28(BX), R9		// arg 6 offset
	CALL	libc_mmap(SB)
	XORL	DX, DX
	CMPQ	AX, $-1
	JNE	ok
	CALL	libc_error(SB)
	MOVQ	(AX), DX		// errno
	XORL	AX, AX
ok:
	MOVQ	AX, 32(BX)
	MOVQ	DX, 40(BX)
	POPQ	BP
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 len
	MOVQ	0(DI), DI		// arg 1 addr
	CALL	libc_munmap(SB)
	TESTQ	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI	// arg 1 usec
	CALL	libc_usleep(SB)
	POPQ	BP
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

TEXT runtime·settls(SB),NOSPLIT,$32
	// Nothing to do on Darwin, pthread already set thread-local storage up.
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	8(DI), SI		// arg 2 miblen
	MOVQ	16(DI), DX		// arg 3 out
	MOVQ	24(DI), CX		// arg 4 size
	MOVQ	32(DI), R8		// arg 5 dst
	MOVQ	40(DI), R9		// arg 6 ndst
	MOVQ	0(DI), DI		// arg 1 mib
	CALL	libc_sysctl(SB)
	POPQ	BP
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	CALL	libc_kqueue(SB)
	POPQ	BP
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 keventt
	MOVL	16(DI), DX		// arg 3 nch
	MOVQ	24(DI), CX		// arg 4 ev
	MOVL	32(DI), R8		// arg 5 nev
	MOVQ	40(DI), R9		// arg 6 ts
	MOVL	0(DI), DI		// arg 1 kq
	CALL	libc_kevent(SB)
	CMPQ	AX, $-1
	JNE	ok
	CALL	libc_error(SB)
	MOVQ	(AX), AX		// errno
	NEGQ	AX			// caller wants it as a negative error code
ok:
	POPQ	BP
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	4(DI), SI		// arg 2 cmd
	MOVL	8(DI), DX		// arg 3 arg
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_fcntl(SB)
	POPQ	BP
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// DI points to the m.
	// We are already on m's g0 stack.

	// Save callee-save registers.
	SUBQ	$40, SP
	MOVQ	BX, 0(SP)
	MOVQ	R12, 8(SP)
	MOVQ	R13, 16(SP)
	MOVQ	R14, 24(SP)
	MOVQ	R15, 32(SP)

	MOVQ	m_g0(DI), DX // g

	// Initialize TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVQ	DX, 0x30(GS)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOVQ	0(SP), BX
	MOVQ	8(SP), R12
	MOVQ	16(SP), R13
	MOVQ	24(SP), R14
	MOVQ	32(SP), R15

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX

	ADDQ	$40, SP
	RET

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in DI.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP	// make frame, keep stack 16-byte aligned.
	MOVQ	SP, BP
	MOVQ	0(DI), DI // arg 1 attr
	CALL	libc_pthread_attr_init(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_setstacksize_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 size
	MOVQ	0(DI), DI	// arg 1 attr
	CALL	libc_pthread_attr_setstacksize(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 state
	MOVQ	0(DI), DI	// arg 1 attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	SUBQ	$16, SP
	MOVQ	0(DI), SI	// arg 2 attr
	MOVQ	8(DI), DX	// arg 3 start
	MOVQ	16(DI), CX	// arg 4 arg
	MOVQ	SP, DI		// arg 1 &threadid (which we throw away)
	CALL	libc_pthread_create(SB)
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·raise_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI	// arg 1 signal
	CALL	libc_raise(SB)
	POPQ	BP
	RET
