// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for AMD64, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "cgo/abi_amd64.h"

#define CLOCK_REALTIME		0

// Exit the entire program (like C exit)
TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI		// arg 1 exit status
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 flags
	MOVL	12(DI), DX		// arg 3 mode
	MOVQ	0(DI), DI		// arg 1 pathname
	XORL	AX, AX			// vararg: say "no float args"
	CALL	libc_open(SB)
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_close(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_read(SB)
	TESTL	AX, AX
	JGE	noerr
	CALL	libc_error(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno value
noerr:
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVQ	0(DI), DI		// arg 1 fd
	CALL	libc_write(SB)
	TESTL	AX, AX
	JGE	noerr
	CALL	libc_error(SB)
	MOVL	(AX), AX
	NEGL	AX			// caller expects negative errno value
noerr:
	RET

TEXT runtime·pipe_trampoline(SB),NOSPLIT,$0
	CALL	libc_pipe(SB)		// pointer already in DI
	TESTL	AX, AX
	JEQ	3(PC)
	CALL	libc_error(SB)		// return negative errno value
	NEGL	AX
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 new
	MOVQ	16(DI), DX		// arg 3 old
	MOVL	0(DI), DI		// arg 1 which
	CALL	libc_setitimer(SB)
	RET

TEXT runtime·madvise_trampoline(SB), NOSPLIT, $0
	MOVQ	8(DI), SI	// arg 2 len
	MOVL	16(DI), DX	// arg 3 advice
	MOVQ	0(DI), DI	// arg 1 addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·mlock_trampoline(SB), NOSPLIT, $0
	UNDEF // unimplemented

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$0
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
	RET

TEXT runtime·walltime_trampoline(SB),NOSPLIT,$0
	MOVQ	DI, SI			// arg 2 timespec
	MOVL	$CLOCK_REALTIME, DI	// arg 1 clock_id
	CALL	libc_clock_gettime(SB)
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

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), BX	// signal
	CALL	libc_getpid(SB)
	MOVL	AX, DI		// arg 1 pid
	MOVL	BX, SI		// arg 2 signal
	CALL	libc_kill(SB)
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

// This is the function registered during sigaction and is invoked when
// a signal is received. It just redirects to the Go function sigtrampgo.
// Called using C ABI.
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME|NOFRAME,$0
	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Set up ABIInternal environment: g in R14, cleared X15.
	get_tls(R12)
	MOVQ	g(R12), R14
	PXOR	X15, X15

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

// Called using C ABI.
TEXT runtime·sigprofNonGoWrapper<>(SB),NOSPLIT|NOFRAME,$0
	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Call into the Go signal handler
	NOP	SP		// disable vet stack checking
	ADJSP	$24
	MOVL	DI, 0(SP)	// sig
	MOVQ	SI, 8(SP)	// info
	MOVQ	DX, 16(SP)	// ctx
	CALL	·sigprofNonGo(SB)
	ADJSP	$-24

	POP_REGS_HOST_TO_ABI0()
	RET

// Used instead of sigtramp in programs that use cgo.
// Arguments from kernel are in DI, SI, DX.
TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	// If no traceback function, do usual sigtramp.
	MOVQ	runtime·cgoTraceback(SB), AX
	TESTQ	AX, AX
	JZ	sigtramp

	// If no traceback support function, which means that
	// runtime/cgo was not linked in, do usual sigtramp.
	MOVQ	_cgo_callers(SB), AX
	TESTQ	AX, AX
	JZ	sigtramp

	// Figure out if we are currently in a cgo call.
	// If not, just do usual sigtramp.
	get_tls(CX)
	MOVQ	g(CX),AX
	TESTQ	AX, AX
	JZ	sigtrampnog     // g == nil
	MOVQ	g_m(AX), AX
	TESTQ	AX, AX
	JZ	sigtramp        // g.m == nil
	MOVL	m_ncgo(AX), CX
	TESTL	CX, CX
	JZ	sigtramp        // g.m.ncgo == 0
	MOVQ	m_curg(AX), CX
	TESTQ	CX, CX
	JZ	sigtramp        // g.m.curg == nil
	MOVQ	g_syscallsp(CX), CX
	TESTQ	CX, CX
	JZ	sigtramp        // g.m.curg.syscallsp == 0
	MOVQ	m_cgoCallers(AX), R8
	TESTQ	R8, R8
	JZ	sigtramp        // g.m.cgoCallers == nil
	MOVL	m_cgoCallersUse(AX), CX
	TESTL	CX, CX
	JNZ	sigtramp	// g.m.cgoCallersUse != 0

	// Jump to a function in runtime/cgo.
	// That function, written in C, will call the user's traceback
	// function with proper unwind info, and will then call back here.
	// The first three arguments, and the fifth, are already in registers.
	// Set the two remaining arguments now.
	MOVQ	runtime·cgoTraceback(SB), CX
	MOVQ	$runtime·sigtramp(SB), R9
	MOVQ	_cgo_callers(SB), AX
	JMP	AX

sigtramp:
	JMP	runtime·sigtramp(SB)

sigtrampnog:
	// Signal arrived on a non-Go thread. If this is SIGPROF, get a
	// stack trace.
	CMPL	DI, $27 // 27 == SIGPROF
	JNZ	sigtramp

	// Lock sigprofCallersUse.
	MOVL	$0, AX
	MOVL	$1, CX
	MOVQ	$runtime·sigprofCallersUse(SB), R11
	LOCK
	CMPXCHGL	CX, 0(R11)
	JNZ	sigtramp  // Skip stack trace if already locked.

	// Jump to the traceback function in runtime/cgo.
	// It will call back to sigprofNonGo, via sigprofNonGoWrapper, to convert
	// the arguments to the Go calling convention.
	// First three arguments to traceback function are in registers already.
	MOVQ	runtime·cgoTraceback(SB), CX
	MOVQ	$runtime·sigprofCallers(SB), R8
	MOVQ	$runtime·sigprofNonGoWrapper<>(SB), R9
	MOVQ	_cgo_callers(SB), AX
	JMP	AX

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
	CALL	libc_error(SB)
	MOVLQSX	(AX), DX		// errno
	XORL	AX, AX
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

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI	// arg 1 usec
	CALL	libc_usleep(SB)
	RET

TEXT runtime·settls(SB),NOSPLIT,$32
	// Nothing to do on Darwin, pthread already set thread-local storage up.
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	MOVL	8(DI), SI		// arg 2 miblen
	MOVQ	16(DI), DX		// arg 3 oldp
	MOVQ	24(DI), CX		// arg 4 oldlenp
	MOVQ	32(DI), R8		// arg 5 newp
	MOVQ	40(DI), R9		// arg 6 newlen
	MOVQ	0(DI), DI		// arg 1 mib
	CALL	libc_sysctl(SB)
	RET

TEXT runtime·sysctlbyname_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 oldp
	MOVQ	16(DI), DX		// arg 3 oldlenp
	MOVQ	24(DI), CX		// arg 4 newp
	MOVQ	32(DI), R8		// arg 5 newlen
	MOVQ	0(DI), DI		// arg 1 name
	CALL	libc_sysctlbyname(SB)
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
	CALL	libc_error(SB)
	MOVLQSX	(AX), AX		// errno
	NEGQ	AX			// caller wants it as a negative error code
ok:
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	MOVQ	DI, BX
	MOVL	0(BX), DI		// arg 1 fd
	MOVL	4(BX), SI		// arg 2 cmd
	MOVL	8(BX), DX		// arg 3 arg
	XORL	AX, AX			// vararg: say "no float args"
	CALL	libc_fcntl(SB)
	XORL	DX, DX
	CMPQ	AX, $-1
	JNE	noerr
	CALL	libc_error(SB)
	MOVL	(AX), DX
	MOVL	$-1, AX
noerr:
	MOVL	AX, 12(BX)
	MOVL	DX, 16(BX)
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT|NOFRAME,$0
	// DI points to the m.
	// We are already on m's g0 stack.

	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	MOVQ	m_g0(DI), DX // g

	// Initialize TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVQ	DX, 0x30(GS)

	CALL	runtime·mstart(SB)

	POP_REGS_HOST_TO_ABI0()

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX
	RET

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in DI.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), DI // arg 1 attr
	CALL	libc_pthread_attr_init(SB)
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 size
	MOVQ	0(DI), DI	// arg 1 attr
	CALL	libc_pthread_attr_getstacksize(SB)
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 state
	MOVQ	0(DI), DI	// arg 1 attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$16
	MOVQ	0(DI), SI	// arg 2 attr
	MOVQ	8(DI), DX	// arg 3 start
	MOVQ	16(DI), CX	// arg 4 arg
	MOVQ	SP, DI		// arg 1 &threadid (which we throw away)
	CALL	libc_pthread_create(SB)
	RET

TEXT runtime·raise_trampoline(SB),NOSPLIT,$0
	MOVL	0(DI), DI	// arg 1 signal
	CALL	libc_raise(SB)
	RET

TEXT runtime·pthread_mutex_init_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 attr
	MOVQ	0(DI), DI	// arg 1 mutex
	CALL	libc_pthread_mutex_init(SB)
	RET

TEXT runtime·pthread_mutex_lock_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), DI	// arg 1 mutex
	CALL	libc_pthread_mutex_lock(SB)
	RET

TEXT runtime·pthread_mutex_unlock_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), DI	// arg 1 mutex
	CALL	libc_pthread_mutex_unlock(SB)
	RET

TEXT runtime·pthread_cond_init_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 attr
	MOVQ	0(DI), DI	// arg 1 cond
	CALL	libc_pthread_cond_init(SB)
	RET

TEXT runtime·pthread_cond_wait_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 mutex
	MOVQ	0(DI), DI	// arg 1 cond
	CALL	libc_pthread_cond_wait(SB)
	RET

TEXT runtime·pthread_cond_timedwait_relative_np_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 mutex
	MOVQ	16(DI), DX	// arg 3 timeout
	MOVQ	0(DI), DI	// arg 1 cond
	CALL	libc_pthread_cond_timedwait_relative_np(SB)
	RET

TEXT runtime·pthread_cond_signal_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), DI	// arg 1 cond
	CALL	libc_pthread_cond_signal(SB)
	RET

TEXT runtime·pthread_self_trampoline(SB),NOSPLIT,$0
	MOVQ	DI, BX		// BX is caller-save
	CALL	libc_pthread_self(SB)
	MOVQ	AX, 0(BX)	// return value
	RET

TEXT runtime·pthread_kill_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI	// arg 2 sig
	MOVQ	0(DI), DI	// arg 1 thread
	CALL	libc_pthread_kill(SB)
	RET

TEXT runtime·osinit_hack_trampoline(SB),NOSPLIT,$0
	MOVQ	$0, DI	// arg 1 val
	CALL	libc_notify_is_valid_token(SB)
	CALL	libc_xpc_date_create_from_current(SB)
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
	CALL	libc_error(SB)
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
	CALL	libc_error(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (6*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscallPtr is like syscallX except that the libc function reports an
// error by returning NULL and setting errno.
TEXT runtime·syscallPtr(SB),NOSPLIT,$16
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

	// syscallPtr libc functions return NULL on error
	// and set errno.
	TESTQ	AX, AX
	JNE	ok

	// Get error code from libc.
	CALL	libc_error(SB)
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

	CALL	libc_error(SB)
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

	CALL	libc_error(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (9*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscall9 calls a function in libc on behalf of the syscall package.
// syscall9 takes a pointer to a struct like:
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
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall9 must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall9 expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall9(SB),NOSPLIT,$16
	MOVQ	(0*8)(DI), R13// fn
	MOVQ	(2*8)(DI), SI // a2
	MOVQ	(3*8)(DI), DX // a3
	MOVQ	(4*8)(DI), CX // a4
	MOVQ	(5*8)(DI), R8 // a5
	MOVQ	(6*8)(DI), R9 // a6
	MOVQ	(7*8)(DI), R10 // a7
	MOVQ	(8*8)(DI), R11 // a8
	MOVQ	(9*8)(DI), R12 // a9
	MOVQ	DI, (SP)
	MOVQ	(1*8)(DI), DI // a1
	XORL	AX, AX	      // vararg: say "no float args"

	CALL	R13

	MOVQ	(SP), DI
	MOVQ	AX, (10*8)(DI) // r1
	MOVQ	DX, (11*8)(DI) // r2

	CMPL	AX, $-1
	JNE	ok

	CALL	libc_error(SB)
	MOVLQSX	(AX), AX
	MOVQ	(SP), DI
	MOVQ	AX, (12*8)(DI) // err

ok:
	XORL	AX, AX        // no error (it's ignored anyway)
	RET

// syscall_x509 is for crypto/x509. It is like syscall6 but does not check for errors,
// takes 5 uintptrs and 1 float64, and only returns one value,
// for use with standard C ABI functions.
TEXT runtime·syscall_x509(SB),NOSPLIT,$16
	MOVQ	(0*8)(DI), R11// fn
	MOVQ	(2*8)(DI), SI // a2
	MOVQ	(3*8)(DI), DX // a3
	MOVQ	(4*8)(DI), CX // a4
	MOVQ	(5*8)(DI), R8 // a5
	MOVQ	(6*8)(DI), X0 // f1
	MOVQ	DI, (SP)
	MOVQ	(1*8)(DI), DI // a1
	XORL	AX, AX	      // vararg: say "no float args"

	CALL	R11

	MOVQ	(SP), DI
	MOVQ	AX, (7*8)(DI) // r1

	XORL	AX, AX        // no error (it's ignored anyway)
	RET

TEXT runtime·issetugid_trampoline(SB),NOSPLIT,$0
	CALL	libc_issetugid(SB)
	RET

// mach_vm_region_trampoline calls mach_vm_region from libc.
TEXT runtime·mach_vm_region_trampoline(SB),NOSPLIT,$0
	MOVQ	0(DI), SI // address
	MOVQ	8(DI), DX // size
	MOVL	16(DI), CX // flavor
	MOVQ	24(DI), R8 // info
	MOVQ	32(DI), R9 // count
	MOVQ	40(DI), R10 // object_name
	MOVQ	$libc_mach_task_self_(SB), DI
	MOVL	0(DI), DI
	CALL	libc_mach_vm_region(SB)
	RET

// proc_regionfilename_trampoline calls proc_regionfilename.
TEXT runtime·proc_regionfilename_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI // address
	MOVQ	16(DI), DX // buffer
	MOVQ	24(DI), CX // buffer_size
	MOVQ	0(DI), DI // pid
	CALL	libc_proc_regionfilename(SB)
	RET
