// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for 386, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Exit the entire program (like C exit)
TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP   	// allocate space for callee args (must be 8 mod 16)
	MOVL	16(SP), CX	// arg ptr
	MOVL	0(CX), AX	// arg 1 exit status
	MOVL	AX, 0(SP)
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 name
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 mode
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 perm
	MOVL	AX, 8(SP)
	CALL	libc_open(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	CALL	libc_close(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 buf
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 count
	MOVL	AX, 8(SP)
	CALL	libc_read(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 fd
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 buf
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 count
	MOVL	AX, 8(SP)
	CALL	libc_write(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	MOVL	$20, AX // getpid
	INT	$0x80
	MOVL	AX, 4(SP)	// pid
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)	// signal
	MOVL	$1, 12(SP)	// posix
	MOVL	$37, AX // kill
	INT	$0x80
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 addr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 len
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 prot
	MOVL	AX, 8(SP)
	MOVL	12(CX), AX		// arg 4 flags
	MOVL	AX, 12(SP)
	MOVL	16(CX), AX		// arg 5 fid
	MOVL	AX, 16(SP)
	MOVL	20(CX), AX		// arg 6 offset
	MOVL	AX, 20(SP)
	CALL	libc_mmap(SB)
	XORL	DX, DX
	CMPL	AX, $-1
	JNE	ok
	CALL	libc_error(SB)
	MOVL	(AX), DX		// errno
	XORL	AX, AX
ok:
	MOVL	32(SP), CX
	MOVL	AX, 24(CX)		// result pointer
	MOVL	DX, 28(CX)		// errno
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	MOVL	0(CX), AX		// arg 1 addr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 len
	MOVL	AX, 4(SP)
	MOVL	8(CX), AX		// arg 3 advice
	MOVL	AX, 8(SP)
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX		// arg 1 addr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX		// arg 2 len
	MOVL	AX, 4(SP)
	CALL	libc_munmap(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVL	$83, AX
	INT	$0x80
	RET

TEXT runtime·walltime_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), AX
	MOVL	AX, 0(SP)	// *timeval
	MOVL	$0, 4(SP)	// no timezone needed
	CALL	libc_gettimeofday(SB)
	MOVL	BP, SP
	POPL	BP
	RET

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8+(machTimebaseInfo__size+15)/16*16, SP
	CALL	libc_mach_absolute_time(SB)
	MOVL	16+(machTimebaseInfo__size+15)/16*16(SP), CX
	MOVL	AX, 0(CX)
	MOVL	DX, 4(CX)
	MOVL	timebase<>+machTimebaseInfo_denom(SB), DI // atomic read
	MOVL	timebase<>+machTimebaseInfo_numer(SB), SI
	TESTL	DI, DI
	JNE	initialized

	LEAL	4(SP), AX
	MOVL	AX, 0(SP)
	CALL	libc_mach_timebase_info(SB)
	MOVL	4+machTimebaseInfo_numer(SP), SI
	MOVL	4+machTimebaseInfo_denom(SP), DI

	MOVL	SI, timebase<>+machTimebaseInfo_numer(SB)
	MOVL	DI, AX
	XCHGL	AX, timebase<>+machTimebaseInfo_denom(SB) // atomic write
	MOVL	16+(machTimebaseInfo__size+15)/16*16(SP), CX

initialized:
	MOVL	SI, 8(CX)
	MOVL	DI, 12(CX)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	$329, AX  // pthread_sigmask (on OS X, sigprocmask==entire process)
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0
	MOVL	$46, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-16
	MOVL	fn+0(FP), AX
	MOVL	sig+4(FP), BX
	MOVL	info+8(FP), CX
	MOVL	ctx+12(FP), DX
	MOVL	SP, SI
	SUBL	$32, SP
	ANDL	$~15, SP	// align stack: handler might be a C function
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)	// save SI: handler might be a Go function
	CALL	AX
	MOVL	12(SP), AX
	MOVL	AX, SP
	RET

// Sigtramp's job is to call the actual signal handler.
// It is called with the following arguments on the stack:
//	0(SP)	"return address" - ignored
//	4(SP)	actual handler
//	8(SP)	siginfo style
//	12(SP)	signal number
//	16(SP)	siginfo
//	20(SP)	context
TEXT runtime·sigtramp(SB),NOSPLIT,$20
	MOVL	sig+8(FP), BX
	MOVL	BX, 0(SP)
	MOVL	info+12(FP), BX
	MOVL	BX, 4(SP)
	MOVL	ctx+16(FP), BX
	MOVL	BX, 8(SP)
	CALL	runtime·sigtrampgo(SB)

	// call sigreturn
	MOVL	ctx+16(FP), CX
	MOVL	infostyle+4(FP), BX
	MOVL	$0, 0(SP)	// "caller PC" - ignored
	MOVL	CX, 4(SP)
	MOVL	BX, 8(SP)
	MOVL	$184, AX	// sigreturn(ucontext, infostyle)
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 usec
	MOVL	AX, 0(SP)
	CALL	libc_usleep(SB)
	MOVL	BP, SP
	POPL	BP
	RET

// Invoke Mach system call.
// Assumes system call number in AX,
// caller PC on stack, caller's caller PC next,
// and then the system call arguments.
//
// Can be used for BSD too, but we don't,
// because if you use this interface the BSD
// system call numbers need an extra field
// in the high 16 bits that seems to be the
// argument count in bytes but is not always.
// INT $0x80 works fine for those.
TEXT runtime·sysenter(SB),NOSPLIT,$0
	POPL	DX
	MOVL	SP, CX
	SYSENTER
	// returns to DX with SP set to CX

TEXT runtime·mach_msg_trap(SB),NOSPLIT,$0
	MOVL	$-31, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+28(FP)
	RET

TEXT runtime·mach_reply_port(SB),NOSPLIT,$0
	MOVL	$-26, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·mach_task_self(SB),NOSPLIT,$0
	MOVL	$-28, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+0(FP)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// func mach_semaphore_wait(sema uint32) int32
TEXT runtime·mach_semaphore_wait(SB),NOSPLIT,$0
	MOVL	$-36, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+4(FP)
	RET

// func mach_semaphore_timedwait(sema, sec, nsec uint32) int32
TEXT runtime·mach_semaphore_timedwait(SB),NOSPLIT,$0
	MOVL	$-38, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+12(FP)
	RET

// func mach_semaphore_signal(sema uint32) int32
TEXT runtime·mach_semaphore_signal(SB),NOSPLIT,$0
	MOVL	$-33, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+4(FP)
	RET

// func mach_semaphore_signal_all(sema uint32) int32
TEXT runtime·mach_semaphore_signal_all(SB),NOSPLIT,$0
	MOVL	$-34, AX
	CALL	runtime·sysenter(SB)
	MOVL	AX, ret+4(FP)
	RET

// func setldt(entry int, address int, limit int)
TEXT runtime·setldt(SB),NOSPLIT,$32
	// Nothing to do on Darwin, pthread already set thread-local storage up.
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVL	$202, AX
	INT	$0x80
	JAE	4(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+24(FP)
	RET

// func kqueue() int32
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$362, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+0(FP)
	RET

// func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	$363, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

// func closeonexec(fd int32)
TEXT runtime·closeonexec(SB),NOSPLIT,$32
	MOVL	$92, AX  // fcntl
	// 0(SP) is where the caller PC would be; kernel skips it
	MOVL	fd+0(FP), BX
	MOVL	BX, 4(SP)  // fd
	MOVL	$2, 8(SP)  // F_SETFD
	MOVL	$1, 12(SP)  // FD_CLOEXEC
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// The value at SP+4 points to the m.
	// We are already on m's g0 stack.

	MOVL	SP, AX       // hide argument read from vet (vet thinks this function is using the Go calling convention)
	MOVL	4(AX), DI    // m
	MOVL	m_g0(DI), DX // g

	// Initialize TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVL	DX, 0x18(GS)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·mstart(SB)

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX
	RET

TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 0(SP)
	CALL	libc_pthread_attr_init(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_setstacksize_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 size
	MOVL	AX, 4(SP)
	CALL	libc_pthread_attr_setstacksize(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 0(SP)
	MOVL	4(CX), AX	// arg 2 state
	MOVL	AX, 4(SP)
	CALL	libc_pthread_attr_setdetachstate(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	PUSHL	BP
	MOVL	SP, BP
	SUBL	$24, SP
	MOVL	32(SP), CX
	LEAL	16(SP), AX	// arg "0" &threadid (which we throw away)
	MOVL	AX, 0(SP)
	MOVL	0(CX), AX	// arg 1 attr
	MOVL	AX, 4(SP)
	MOVL	4(CX), AX	// arg 2 start
	MOVL	AX, 8(SP)
	MOVL	8(CX), AX	// arg 3 arg
	MOVL	AX, 12(SP)
	CALL	libc_pthread_create(SB)
	MOVL	BP, SP
	POPL	BP
	RET

TEXT runtime·raise_trampoline(SB),NOSPLIT,$0
	PUSHL   BP
	MOVL    SP, BP
	SUBL	$8, SP
	MOVL	16(SP), CX
	MOVL    0(CX), AX	// arg 1 sig
	MOVL	AX, 0(SP)
	CALL    libc_raise(SB)
	MOVL    BP, SP
	POPL    BP
	RET
