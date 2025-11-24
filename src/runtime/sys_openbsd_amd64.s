// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, OpenBSD.
// System calls are implemented in libc/libpthread, this file
// contains trampolines that convert from Go to C calling convention.
// Some direct system call implementations currently remain.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "cgo/abi_amd64.h"

#define CLOCK_MONOTONIC	$3

TEXT runtime·settls(SB),NOSPLIT,$0
	// Nothing to do, pthread already set thread-local storage up.
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// DI points to the m.
	// We are already on m's g0 stack.

	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Load g and save to TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVQ	m_g0(DI), DX // g
	MOVQ	DX, -8(FS)

	CALL	runtime·mstart(SB)

	POP_REGS_HOST_TO_ABI0()

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVQ	fn+0(FP),    AX
	MOVL	sig+8(FP),   DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP),  DX
	MOVQ	SP, BX		// callee-saved
	ANDQ	$~15, SP	// alignment for x86_64 ABI
	CALL	AX
	MOVQ	BX, SP
	RET

// Called using C ABI.
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME|NOFRAME,$0
	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Set up ABIInternal environment: g in R14, cleared X15.
	get_tls(R12)
	MOVQ	g(R12), R14
	PXOR	X15, X15
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JNE	2(PC)
	VXORPS	X15, X15, X15

	// Reserve space for spill slots.
	NOP	SP		// disable vet stack checking
	ADJSP   $24

	// Call into the Go signal handler
	MOVQ	DI, AX	// sig
	MOVQ	SI, BX	// info
	MOVQ	DX, CX	// ctx
	CALL	·sigtrampgo<ABIInternal>(SB)

	ADJSP	$-24

	POP_REGS_HOST_TO_ABI0()
	RET

//
// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in DI.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_init(SB)
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_destroy(SB)
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 - stacksize
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_getstacksize(SB)
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 - detachstate
	MOVQ	0(DI), DI		// arg 1 - attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$16
	MOVQ	0(DI), SI		// arg 2 - attr
	MOVQ	8(DI), DX		// arg 3 - start
	MOVQ	16(DI), CX		// arg 4 - arg
	MOVQ	SP, DI			// arg 1 - &thread (discarded)
	CALL	libc_pthread_create(SB)
	RET

TEXT runtime·thrkill_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 - signal
	MOVQ	$0, DX			// arg 3 - tcb
	MOVL	0(DI), DI		// arg 1 - tid
	CALL	libc_thrkill(SB)
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 - clock_id
	MOVQ	16(DI), DX		// arg 3 - abstime
	MOVQ	24(DI), CX		// arg 4 - lock
	MOVQ	32(DI), R8		// arg 5 - abort
	MOVQ	0(DI), DI		// arg 1 - id
	CALL	libc_thrsleep(SB)
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 - count
	MOVQ	0(DI), DI		// arg 1 - id
	CALL	libc_thrwakeup(SB)
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI		// arg 1 exit status
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·getthrid_trampoline(SB),NOSPLIT,$0
	MOVQ	DI, BX			// BX is caller-save
	CALL	libc_getthrid(SB)
	MOVL	AX, 0(BX)		// return value
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), BX	// signal
	CALL	libc_getpid(SB)
	MOVL	AX, DI		// arg 1 pid
	MOVL	BX, SI		// arg 2 signal
	CALL	libc_kill(SB)
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$0
	CALL	libc_sched_yield(SB)
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
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
	CALL	libc_errno(SB)
	MOVLQSX	(AX), DX		// errno
	XORQ	AX, AX
ok:
	MOVQ	AX, 32(BX)
	MOVQ	DX, 40(BX)
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 len
	MOVQ	0(DI), DI		// arg 1 addr
	CALL	libc_munmap(SB)
	TESTQ	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise_trampoline(SB), NOSPLIT, $0
	MOVQ	8(DI), SI	// arg 2 len
	MOVL	16(DI), DX	// arg 3 advice
	MOVQ	0(DI), DI	// arg 1 addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 - flags
	MOVL	12(DI), DX		// arg 3 - mode
	MOVQ	0(DI), DI		// arg 1 - path
	XORL	AX, AX			// vararg: say "no float args"
	CALL	libc_open(SB)
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI		// arg 1 - fd
	CALL	libc_close(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 - buf
	MOVL	16(DI), DX		// arg 3 - count
	MOVL	0(DI), DI		// arg 1 - fd
	CALL	libc_read(SB)
	TESTL	AX, AX
	JGE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX		// errno
	NEGL	AX			// caller expects negative errno value
noerr:
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_write(SB)
	TESTL	AX, AX
	JGE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX		// errno
	NEGL	AX			// caller expects negative errno value
noerr:
	RET

TEXT runtime·pipe2_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 flags
	MOVQ	0(DI), DI		// arg 1 filedes
	CALL	libc_pipe2(SB)
	TESTL	AX, AX
	JEQ	3(PC)
	CALL	libc_errno(SB)
	MOVL	(AX), AX		// errno
	NEGL	AX			// caller expects negative errno value
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 new
	MOVQ	16(DI), DX		// arg 3 old
	MOVL	0(DI), DI		// arg 1 which
	CALL	libc_setitimer(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI		// arg 1 usec
	CALL	libc_usleep(SB)
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 miblen
	MOVQ	16(DI), DX		// arg 3 out
	MOVQ	24(DI), CX		// arg 4 size
	MOVQ	32(DI), R8		// arg 5 dst
	MOVQ	40(DI), R9		// arg 6 ndst
	MOVQ	0(DI), DI		// arg 1 mib
	CALL	libc_sysctl(SB)
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	CALL	libc_kqueue(SB)
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 keventt
	MOVL	16(DI), DX		// arg 3 nch
	MOVQ	24(DI), CX		// arg 4 ev
	MOVL	32(DI), R8		// arg 5 nev
	MOVQ	40(DI), R9		// arg 6 ts
	MOVL	0(DI), DI		// arg 1 kq
	CALL	libc_kevent(SB)
	CMPL	AX, $-1
	JNE	ok
	CALL	libc_errno(SB)
	MOVL	(AX), AX		// errno
	NEGL	AX			// caller expects negative errno value
ok:
	RET

TEXT runtime·clock_gettime_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 tp
	MOVL	0(DI), DI		// arg 1 clock_id
	CALL	libc_clock_gettime(SB)
	TESTL	AX, AX
	JEQ	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), AX		// errno
	NEGL	AX			// caller expects negative errno value
noerr:
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	MOVQ	DI, BX
	MOVL	0(BX), DI		// arg 1 fd
	MOVL	4(BX), SI		// arg 2 cmd
	MOVL	8(BX), DX		// arg 3 arg
	XORL	AX, AX			// vararg: say "no float args"
	CALL	libc_fcntl(SB)
	XORL	DX, DX
	CMPL	AX, $-1
	JNE	noerr
	CALL	libc_errno(SB)
	MOVL	(AX), DX
	MOVL	$-1, AX
noerr:
	MOVL	AX, 12(BX)
	MOVL	DX, 16(BX)
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 new
	MOVQ	16(DI), DX		// arg 3 old
	MOVL	0(DI), DI		// arg 1 sig
	CALL	libc_sigaction(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 new
	MOVQ	16(DI), DX	// arg 3 old
	MOVL	0(DI), DI	// arg 1 how
	CALL	libc_pthread_sigmask(SB)
	TESTL	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 old
	MOVQ	0(DI), DI		// arg 1 new
	CALL	libc_sigaltstack(SB)
	TESTQ	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

// syscall calls a function in libc on behalf of the syscall package.
// syscall takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall(SB),NOSPLIT,$16
	MOVQ	(0*8)(DI), CX // fn
	MOVQ	(2*8)(DI), SI // a2
	MOVQ	(3*8)(DI), DX // a3
	MOVQ	DI, (SP)
	MOVQ	(1*8)(DI), DI // a1
	XORL	AX, AX	      // vararg: say "no float args"

	CALL	CX

	MOVQ	(SP), DI
	MOVQ	AX, (4*8)(DI) // r1
	MOVQ	DX, (5*8)(DI) // r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPL	AX, $-1	      // Note: high 32 bits are junk
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (6*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscallX calls a function in libc on behalf of the syscall package.
// syscallX takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscallX must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscallX is like syscall but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscallX(SB),NOSPLIT,$16
	MOVQ	(0*8)(DI), CX // fn
	MOVQ	(2*8)(DI), SI // a2
	MOVQ	(3*8)(DI), DX // a3
	MOVQ	DI, (SP)
	MOVQ	(1*8)(DI), DI // a1
	XORL	AX, AX	      // vararg: say "no float args"

	CALL	CX

	MOVQ	(SP), DI
	MOVQ	AX, (4*8)(DI) // r1
	MOVQ	DX, (5*8)(DI) // r2

	// Standard libc functions return -1 on error
	// and set errno.
	CMPQ	AX, $-1
	JNE	ok

	// Get error code from libc.
	CALL	libc_errno(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (6*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscall6 calls a function in libc on behalf of the syscall package.
// syscall6 takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall6 must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall6 expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall6(SB),NOSPLIT,$16
	MOVQ	(0*8)(DI), R11// fn
	MOVQ	(2*8)(DI), SI // a2
	MOVQ	(3*8)(DI), DX // a3
	MOVQ	(4*8)(DI), CX // a4
	MOVQ	(5*8)(DI), R8 // a5
	MOVQ	(6*8)(DI), R9 // a6
	MOVQ	DI, (SP)
	MOVQ	(1*8)(DI), DI // a1
	XORL	AX, AX	      // vararg: say "no float args"

	CALL	R11

	MOVQ	(SP), DI
	MOVQ	AX, (7*8)(DI) // r1
	MOVQ	DX, (8*8)(DI) // r2

	CMPL	AX, $-1
	JNE	ok

	CALL	libc_errno(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (9*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscall6X calls a function in libc on behalf of the syscall package.
// syscall6X takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall6X must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall6X is like syscall6 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall6X(SB),NOSPLIT,$16
	MOVQ	(0*8)(DI), R11// fn
	MOVQ	(2*8)(DI), SI // a2
	MOVQ	(3*8)(DI), DX // a3
	MOVQ	(4*8)(DI), CX // a4
	MOVQ	(5*8)(DI), R8 // a5
	MOVQ	(6*8)(DI), R9 // a6
	MOVQ	DI, (SP)
	MOVQ	(1*8)(DI), DI // a1
	XORL	AX, AX	      // vararg: say "no float args"

	CALL	R11

	MOVQ	(SP), DI
	MOVQ	AX, (7*8)(DI) // r1
	MOVQ	DX, (8*8)(DI) // r2

	CMPQ	AX, $-1
	JNE	ok

	CALL	libc_errno(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (9*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscall10 calls a function in libc on behalf of the syscall package.
// syscall10 takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	a7    uintptr
//	a8    uintptr
//	a9    uintptr
//	a10   uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall10 must be called on the g0 stack with the
// C calling convention (use libcCall).
TEXT runtime·syscall10(SB),NOSPLIT,$48
	// Arguments a1 to a6 get passed in registers, with a7 onwards being
	// passed via the stack per the x86-64 System V ABI
	// (https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf).
	MOVQ	(7*8)(DI), R10	// a7
	MOVQ	(8*8)(DI), R11	// a8
	MOVQ	(9*8)(DI), R12	// a9
	MOVQ	(10*8)(DI), R13	// a10
	MOVQ	R10, (0*8)(SP)	// a7
	MOVQ	R11, (1*8)(SP)	// a8
	MOVQ	R12, (2*8)(SP)	// a9
	MOVQ	R13, (3*8)(SP)	// a10
	MOVQ	(0*8)(DI), R11	// fn
	MOVQ	(2*8)(DI), SI	// a2
	MOVQ	(3*8)(DI), DX	// a3
	MOVQ	(4*8)(DI), CX	// a4
	MOVQ	(5*8)(DI), R8	// a5
	MOVQ	(6*8)(DI), R9	// a6
	MOVQ	DI, (4*8)(SP)
	MOVQ	(1*8)(DI), DI	// a1
	XORL	AX, AX	     	// vararg: say "no float args"

	CALL	R11

	MOVQ	(4*8)(SP), DI
	MOVQ	AX, (11*8)(DI) // r1
	MOVQ	DX, (12*8)(DI) // r2

	CMPL	AX, $-1
	JNE	ok

	CALL	libc_errno(SB)
	MOVLQSX	(AX), AX
	MOVQ	(4*8)(SP), DI
	MOVQ	AX, (13*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscall10X calls a function in libc on behalf of the syscall package.
// syscall10X takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	a7    uintptr
//	a8    uintptr
//	a9    uintptr
//	a10   uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall10X must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall10X is like syscall10 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall10X(SB),NOSPLIT,$48
	// Arguments a1 to a6 get passed in registers, with a7 onwards being
	// passed via the stack per the x86-64 System V ABI
	// (https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf).
	MOVQ	(7*8)(DI), R10	// a7
	MOVQ	(8*8)(DI), R11	// a8
	MOVQ	(9*8)(DI), R12	// a9
	MOVQ	(10*8)(DI), R13	// a10
	MOVQ	R10, (0*8)(SP)	// a7
	MOVQ	R11, (1*8)(SP)	// a8
	MOVQ	R12, (2*8)(SP)	// a9
	MOVQ	R13, (3*8)(SP)	// a10
	MOVQ	(0*8)(DI), R11	// fn
	MOVQ	(2*8)(DI), SI	// a2
	MOVQ	(3*8)(DI), DX	// a3
	MOVQ	(4*8)(DI), CX	// a4
	MOVQ	(5*8)(DI), R8	// a5
	MOVQ	(6*8)(DI), R9	// a6
	MOVQ	DI, (4*8)(SP)
	MOVQ	(1*8)(DI), DI	// a1
	XORL	AX, AX	     	// vararg: say "no float args"

	CALL	R11

	MOVQ	(4*8)(SP), DI
	MOVQ	AX, (11*8)(DI) // r1
	MOVQ	DX, (12*8)(DI) // r2

	CMPQ	AX, $-1
	JNE	ok

	CALL	libc_errno(SB)
	MOVLQSX	(AX), AX
	MOVQ	(4*8)(SP), DI
	MOVQ	AX, (13*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

TEXT runtime·issetugid_trampoline(SB),NOSPLIT,$0
	MOVQ	DI, BX			// BX is caller-save
	CALL	libc_issetugid(SB)
	MOVL	AX, 0(BX)		// return value
	RET
