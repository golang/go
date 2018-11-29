// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM64, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT notok<>(SB),NOSPLIT,$0
	MOVD	$0, R8
	MOVD	R8, (R8)
	B	0(PC)

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	SUB	$16, RSP
	MOVW	8(R0), R1	// arg 2 flags
	MOVW	12(R0), R2	// arg 3 mode
	MOVW	R2, (RSP)	// arg 3 is variadic, pass on stack
	MOVD	0(R0), R0	// arg 1 pathname
	BL	libc_open(SB)
	ADD	$16, RSP
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_close(SB)
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 buf
	MOVW	16(R0), R2	// arg 3 count
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_write(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 buf
	MOVW	16(R0), R2	// arg 3 count
	MOVW	0(R0), R0	// arg 1 fd
	BL libc_read(SB)
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT|NOFRAME,$0
	MOVW	0(R0), R0
	BL	libc_exit(SB)
	MOVD	$1234, R0
	MOVD	$1002, R1
	MOVD	R0, (R1)	// fail hard

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R19	// signal
	BL	libc_getpid(SB)
	// arg 1 pid already in R0 from getpid
	MOVD	R19, R1	// arg 2 signal
	BL	libc_kill(SB)
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	MOVD	0(R19), R0	// arg 1 addr
	MOVD	8(R19), R1	// arg 2 len
	MOVW	16(R19), R2	// arg 3 prot
	MOVW	20(R19), R3	// arg 4 flags
	MOVW	24(R19), R4	// arg 5 fd
	MOVW	28(R19), R5	// arg 6 off
	BL	libc_mmap(SB)
	MOVD	$0, R1
	MOVD	$-1, R2
	CMP	R0, R2
	BNE	ok
	BL libc_error(SB)
	MOVW	(R0), R1
	MOVD	$0, R0
ok:
	MOVD	R0, 32(R19) // ret 1 p
	MOVD	R1, 40(R19)	// ret 2 err
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 len
	MOVD	0(R0), R0	// arg 1 addr
	BL	libc_munmap(SB)
	CMP $0, R0
	BEQ 2(PC)
	BL	notok<>(SB)
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 len
	MOVW	16(R0), R2	// arg 3 advice
	MOVD	0(R0), R0	// arg 1 addr
	BL	libc_madvise(SB)
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 new
	MOVD	16(R0), R2	// arg 3 old
	MOVW	0(R0), R0	// arg 1 which
	BL	libc_setitimer(SB)
	RET

TEXT runtime·walltime_trampoline(SB),NOSPLIT,$0
	// R0 already has *timeval
	MOVD	$0, R1 // no timezone needed
	BL	libc_gettimeofday(SB)
	RET

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$40
	MOVD	R0, R19
	BL	libc_mach_absolute_time(SB)
	MOVD	R0, 0(R19)
	MOVW	timebase<>+machTimebaseInfo_numer(SB), R20
	MOVD	$timebase<>+machTimebaseInfo_denom(SB), R21
	LDARW	(R21), R21	// atomic read
	CMP	$0, R21
	BNE	initialized

	SUB	$(machTimebaseInfo__size+15)/16*16, RSP
	MOVD	RSP, R0
	BL	libc_mach_timebase_info(SB)
	MOVW	machTimebaseInfo_numer(RSP), R20
	MOVW	machTimebaseInfo_denom(RSP), R21
	ADD	$(machTimebaseInfo__size+15)/16*16, RSP

	MOVW	R20, timebase<>+machTimebaseInfo_numer(SB)
	MOVD	$timebase<>+machTimebaseInfo_denom(SB), R22
	STLRW	R21, (R22)	// atomic write

initialized:
	MOVW	R20, 8(R19)
	MOVW	R21, 12(R19)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// Reserve space for callee-save registers and arguments.
	SUB	$(8*16), RSP

	// Save callee-save registers.
	MOVD	R19, (8*4)(RSP)
	MOVD	R20, (8*5)(RSP)
	MOVD	R21, (8*6)(RSP)
	MOVD	R22, (8*7)(RSP)
	MOVD	R23, (8*8)(RSP)
	MOVD	R24, (8*9)(RSP)
	MOVD	R25, (8*10)(RSP)
	MOVD	R26, (8*11)(RSP)
	MOVD	R27, (8*12)(RSP)
	MOVD	g, (8*13)(RSP)
	MOVD	R29, (8*14)(RSP)

	// Save arguments.
	MOVW	R0, (8*1)(RSP)	// sig
	MOVD	R1, (8*2)(RSP)	// info
	MOVD	R2, (8*3)(RSP)	// ctx

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	MOVD RSP, R6
	CMP	 $0, g
	BEQ	 nog
	// iOS always use the main stack to run the signal handler.
	// We need to switch to gsignal ourselves.
	MOVD	g_m(g), R11
	MOVD	m_gsignal(R11), R5
	MOVD	(g_stack+stack_hi)(R5), R6

nog:
	// Restore arguments.
	MOVW	(8*1)(RSP), R0
	MOVD	(8*2)(RSP), R1
	MOVD	(8*3)(RSP), R2

	// Reserve space for args and the stack pointer on the
	// gsignal stack.
	SUB	$48, R6
	// Save stack pointer.
	MOVD	RSP, R4
	MOVD	R4, (8*4)(R6)
	// Switch to gsignal stack.
	MOVD	R6, RSP

	// Call sigtrampgo.
	MOVW	R0, (8*1)(RSP)
	MOVD	R1, (8*2)(RSP)
	MOVD	R2, (8*3)(RSP)
	MOVD	$runtime·sigtrampgo(SB), R11
	BL	(R11)

	// Switch to old stack.
	MOVD	(8*4)(RSP), R5
	MOVD	R5, RSP

	// Restore callee-save registers.
	MOVD	(8*4)(RSP), R19
	MOVD	(8*5)(RSP), R20
	MOVD	(8*6)(RSP), R21
	MOVD	(8*7)(RSP), R22
	MOVD	(8*8)(RSP), R23
	MOVD	(8*9)(RSP), R24
	MOVD	(8*10)(RSP), R25
	MOVD	(8*11)(RSP), R26
	MOVD	(8*12)(RSP), R27
	MOVD	(8*13)(RSP), g
	MOVD	(8*14)(RSP), R29

	ADD $(8*16), RSP

	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 new
	MOVD	16(R0), R2	// arg 3 old
	MOVW	0(R0), R0	// arg 1 how
	BL	libc_pthread_sigmask(SB)
	CMP $0, R0
	BEQ	2(PC)
	BL	notok<>(SB)
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 new
	MOVD	16(R0), R2	// arg 3 old
	MOVW	0(R0), R0	// arg 1 how
	BL	libc_sigaction(SB)
	CMP	$0, R0
	BEQ	2(PC)
	BL	notok<>(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 usec
	BL	libc_usleep(SB)
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1	// arg 2 miblen
	MOVD	16(R0), R2	// arg 3 out
	MOVD	24(R0), R3	// arg 4 size
	MOVD	32(R0), R4	// arg 5 dst
	MOVD	40(R0), R5	// arg 6 ndst
	MOVD	0(R0), R0	// arg 1 mib
	BL	libc_sysctl(SB)
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	BL	libc_kqueue(SB)
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 keventt
	MOVW	16(R0), R2	// arg 3 nch
	MOVD	24(R0), R3	// arg 4 ev
	MOVW	32(R0), R4	// arg 5 nev
	MOVD	40(R0), R5	// arg 6 ts
	MOVW	0(R0), R0	// arg 1 kq
	BL	libc_kevent(SB)
	MOVD	$-1, R2
	CMP	R0, R2
	BNE	ok
	BL libc_error(SB)
	MOVW	(R0), R0	// errno
	NEG	R0, R0	// caller wants it as a negative error code
ok:
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	SUB	$16, RSP
	MOVW	4(R0), R1	// arg 2 cmd
	MOVW	8(R0), R2	// arg 3 arg
	MOVW	R2, (RSP)	// arg 3 is variadic, pass on stack
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_fcntl(SB)
	ADD	$16, RSP
	RET

// sigaltstack on iOS is not supported and will always
// run the signal handler on the main stack, so our sigtramp has
// to do the stack switch ourselves.
TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	MOVW	$43, R0
	BL	libc_exit(SB)
	RET

// Thread related functions
// Note: On darwin/arm64, the runtime always use runtime/cgo to
// create threads, so all thread related functions will just exit with a
// unique status.

TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	MOVW	$44, R0
	BL	libc_exit(SB)
	RET

TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	MOVW	$45, R0
	BL	libc_exit(SB)
	RET

TEXT runtime·pthread_attr_setstacksize_trampoline(SB),NOSPLIT,$0
	MOVW	$46, R0
	BL	libc_exit(SB)
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	MOVW	$47, R0
	BL	libc_exit(SB)
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	MOVW	$48, R0
	BL	libc_exit(SB)
	RET

TEXT runtime·raise_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 sig
	BL	libc_raise(SB)
	RET

TEXT runtime·pthread_mutex_init_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 attr
	MOVD	0(R0), R0	// arg 1 mutex
	BL	libc_pthread_mutex_init(SB)
	RET

TEXT runtime·pthread_mutex_lock_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0	// arg 1 mutex
	BL	libc_pthread_mutex_lock(SB)
	RET

TEXT runtime·pthread_mutex_unlock_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0	// arg 1 mutex
	BL	libc_pthread_mutex_unlock(SB)
	RET

TEXT runtime·pthread_cond_init_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 attr
	MOVD	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_init(SB)
	RET

TEXT runtime·pthread_cond_wait_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 mutex
	MOVD	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_wait(SB)
	RET

TEXT runtime·pthread_cond_timedwait_relative_np_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1	// arg 2 mutex
	MOVD	16(R0), R2	// arg 3 timeout
	MOVD	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_timedwait_relative_np(SB)
	RET

TEXT runtime·pthread_cond_signal_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_signal(SB)
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
TEXT runtime·syscall(SB),NOSPLIT,$0
	SUB	$16, RSP	// push structure pointer
	MOVD	R0, 8(RSP)

	MOVD	0(R0), R12	// fn
	MOVD	16(R0), R1	// a2
	MOVD	24(R0), R2	// a3
	MOVD	8(R0), R0	// a1

	// If fn is declared as vararg, we have to pass the vararg arguments on the stack.
	// (Because ios decided not to adhere to the standard arm64 calling convention, sigh...)
	// The only libSystem calls we support that are vararg are open, fcntl, and ioctl,
	// which are all of the form fn(x, y, ...). So we just need to put the 3rd arg
	// on the stack as well.
	// If we ever have other vararg libSystem calls, we might need to handle more cases.
	MOVD	R2, (RSP)

	BL	(R12)

	MOVD	8(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 32(R2)	// save r1
	MOVD	R1, 40(R2)	// save r2
	CMPW	$-1, R0
	BNE	ok
	SUB	$16, RSP	// push structure pointer
	MOVD	R2, 8(RSP)
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVD	8(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 48(R2)	// save err
ok:
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
TEXT runtime·syscallX(SB),NOSPLIT,$0
	SUB	$16, RSP	// push structure pointer
	MOVD	R0, (RSP)

	MOVD	0(R0), R12	// fn
	MOVD	16(R0), R1	// a2
	MOVD	24(R0), R2	// a3
	MOVD	8(R0), R0	// a1
	BL	(R12)

	MOVD	(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 32(R2)	// save r1
	MOVD	R1, 40(R2)	// save r2
	CMP	$-1, R0
	BNE	ok
	SUB	$16, RSP	// push structure pointer
	MOVD	R2, (RSP)
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVD	(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 48(R2)	// save err
ok:
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
TEXT runtime·syscall6(SB),NOSPLIT,$0
	SUB	$16, RSP	// push structure pointer
	MOVD	R0, 8(RSP)

	MOVD	0(R0), R12	// fn
	MOVD	16(R0), R1	// a2
	MOVD	24(R0), R2	// a3
	MOVD	32(R0), R3	// a4
	MOVD	40(R0), R4	// a5
	MOVD	48(R0), R5	// a6
	MOVD	8(R0), R0	// a1

	// If fn is declared as vararg, we have to pass the vararg arguments on the stack.
	// See syscall above. The only function this applies to is openat, for which the 4th
	// arg must be on the stack.
	MOVD	R3, (RSP)

	BL	(R12)

	MOVD	8(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 56(R2)	// save r1
	MOVD	R1, 64(R2)	// save r2
	CMPW	$-1, R0
	BNE	ok
	SUB	$16, RSP	// push structure pointer
	MOVD	R2, 8(RSP)
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVD	8(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 72(R2)	// save err
ok:
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
TEXT runtime·syscall6X(SB),NOSPLIT,$0
	SUB	$16, RSP	// push structure pointer
	MOVD	R0, (RSP)

	MOVD	0(R0), R12	// fn
	MOVD	16(R0), R1	// a2
	MOVD	24(R0), R2	// a3
	MOVD	32(R0), R3	// a4
	MOVD	40(R0), R4	// a5
	MOVD	48(R0), R5	// a6
	MOVD	8(R0), R0	// a1
	BL	(R12)

	MOVD	(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 56(R2)	// save r1
	MOVD	R1, 64(R2)	// save r2
	CMP	$-1, R0
	BNE	ok
	SUB	$16, RSP	// push structure pointer
	MOVD	R2, (RSP)
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVD	(RSP), R2	// pop structure pointer
	ADD	$16, RSP
	MOVD	R0, 72(R2)	// save err
ok:
	RET
