// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT notok<>(SB),NOSPLIT,$0
	MOVW	$0, R8
	MOVW	R8, (R8)
	B		0(PC)

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 mode
	MOVW	8(R0), R2	// arg 3 perm
	MOVW	0(R0), R0	// arg 1 name
	BL	libc_open(SB)
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_close(SB)
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 buf
	MOVW	8(R0), R2	// arg 3 count
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_write(SB)
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 buf
	MOVW	8(R0), R2	// arg 3 count
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_read(SB)
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT|NOFRAME,$0
	MOVW	0(R0), R0	// arg 0 code
	BL libc_exit(SB)
	MOVW	$1234, R0
	MOVW	$1002, R1
	MOVW	R0, (R1)	// fail hard

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R8	// signal
	BL	libc_getpid(SB)
	// arg 1 pid already in R0 from getpid
	MOVW	R8, R1	// arg 2 signal
	BL	libc_kill(SB)
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	MOVW	R0, R8
	MOVW	0(R8), R0	// arg 1 addr
	MOVW	4(R8), R1	// arg 2 len
	MOVW	8(R8), R2	// arg 3 prot
	MOVW	12(R8), R3	// arg 4 flags
	MOVW	16(R8), R4	// arg 5 fid
	MOVW	20(R8), R5	// arg 6 offset
	MOVW	$0, R6	// off_t is uint64_t
	// Only R0-R3 are used for arguments, the rest
	// go on the stack.
	MOVM.DB.W [R4-R6], (R13)
	BL	libc_mmap(SB)
	ADD $12, R13
	MOVW	$0, R1
	MOVW	$-1, R2
	CMP	R0, R2
	BNE ok
	BL	libc_error(SB)
	MOVW	(R0), R1
	MOVW	$0, R0
ok:
	MOVW	R0, 24(R8)	// ret 1 addr
	MOVW	R1, 28(R8)	// ret 2 err
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 len
	MOVW	0(R0), R0	// arg 1 addr
	BL libc_munmap(SB)
	MOVW	$-1, R2
	CMP	R0, R2
	BL.EQ	notok<>(SB)
	RET

TEXT runtime·madvise_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 len
	MOVW	8(R0), R2	// arg 3 advice
	MOVW	0(R0), R0	// arg 1 addr
	BL	libc_madvise(SB)
	MOVW	$-1, R2
	CMP	R0, R2
	BL.EQ	notok<>(SB)
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 new
	MOVW	8(R0), R2	// arg 3 old
	MOVW	0(R0), R0	// arg 1 which
	BL	libc_setitimer(SB)
	RET

TEXT runtime·walltime_trampoline(SB),NOSPLIT,$0
	// R0 already has *timeval
	MOVW	$0, R1 // no timezone needed
	BL	libc_gettimeofday(SB)
	RET

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$0
	MOVW	R0, R8
	BL	libc_mach_absolute_time(SB)
	MOVW	R0, 0(R8)
	MOVW	R1, 4(R8)
	MOVW	timebase<>+machTimebaseInfo_numer(SB), R6
	MOVW	$timebase<>+machTimebaseInfo_denom(SB), R5
	MOVW	(R5), R7
	DMB	MB_ISH	// memory barrier for atomic read
	CMP	$0, R7
	BNE	initialized

	SUB	$(machTimebaseInfo__size+7)/8*8, R13
	MOVW	R13, R0
	BL	libc_mach_timebase_info(SB)
	MOVW	machTimebaseInfo_numer(R13), R6
	MOVW	machTimebaseInfo_denom(R13), R7
	ADD	$(machTimebaseInfo__size+7)/8*8, R13

	MOVW	R6, timebase<>+machTimebaseInfo_numer(SB)
	MOVW	$timebase<>+machTimebaseInfo_denom(SB), R5
	DMB	MB_ISH	// memory barrier for atomic write
	MOVW	R7, (R5)
	DMB	MB_ISH

initialized:
	MOVW	R6, 8(R8)
	MOVW	R7, 12(R8)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-16
	MOVW	sig+4(FP), R0
	MOVW	info+8(FP), R1
	MOVW	ctx+12(FP), R2
	MOVW	fn+0(FP), R11
	MOVW	R13, R4
	SUB	$24, R13
	BIC	$0x7, R13 // alignment for ELF ABI
	BL	(R11)
	MOVW	R4, R13
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// Reserve space for callee-save registers and arguments.
	SUB	$36, R13

	MOVW	R4, 12(R13)
	MOVW	R5, 16(R13)
	MOVW	R6, 20(R13)
	MOVW	R7, 24(R13)
	MOVW	R8, 28(R13)
	MOVW	R11, 32(R13)

	// Save arguments.
	MOVW	R0, 4(R13)	// sig
	MOVW	R1, 8(R13)	// info
	MOVW	R2, 12(R13)	// ctx

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	BL.NE	runtime·load_g(SB)

	MOVW	R13, R6
	CMP	$0, g
	BEQ nog

	// iOS always use the main stack to run the signal handler.
	// We need to switch to gsignal ourselves.
	MOVW	g_m(g), R11
	MOVW	m_gsignal(R11), R5
	MOVW	(g_stack+stack_hi)(R5), R6

nog:
	// Restore arguments.
	MOVW	4(R13), R0
	MOVW	8(R13), R1
	MOVW	12(R13), R2

	// Reserve space for args and the stack pointer on the
	// gsignal stack.
	SUB $24, R6
	// Save stack pointer.
	MOVW	R13, R4
	MOVW	R4, 16(R6)
	// Switch to gsignal stack.
	MOVW	R6, R13

	// Call sigtrampgo
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·sigtrampgo(SB)

	// Switch to old stack.
	MOVW	16(R13), R5
	MOVW	R5, R13

	// Restore callee-save registers.
	MOVW	12(R13), R4
	MOVW	16(R13), R5
	MOVW	20(R13), R6
	MOVW	24(R13), R7
	MOVW	28(R13), R8
	MOVW	32(R13), R11

	ADD $36, R13

	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 new
	MOVW	8(R0), R2	// arg 3 old
	MOVW	0(R0), R0	// arg 1 how
	BL	libc_pthread_sigmask(SB)
	CMP	$0, R0
	BL.NE	notok<>(SB)
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 new
	MOVW	8(R0), R2	// arg 3 old
	MOVW	0(R0), R0	// arg 1 how
	BL	libc_sigaction(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 usec
	BL libc_usleep(SB)
	RET

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 miblen
	MOVW	8(R0), R2	// arg 3 out
	MOVW	12(R0), R3	// arg 4 size
	MOVW	16(R0), R4	// arg 5 dst
	MOVW	20(R0), R5	// arg 6 ndst
	MOVW	0(R0), R0	// arg 1 mib
	// Only R0-R3 are used for arguments, the rest
	// go on the stack.
	MOVM.DB.W [R4-R5], (R13)
	BL	libc_sysctl(SB)
	ADD $(2*4), R13
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	BL	libc_kqueue(SB)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int events, Timespec *timeout)
TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 keventss
	MOVW	8(R0), R2	// arg 3 nch
	MOVW	12(R0), R3	// arg 4 ev
	MOVW	16(R0), R4	// arg 5 nev
	MOVW	20(R0), R5	// arg 6 ts
	MOVW	0(R0), R0	// arg 1 kq
	// Only R0-R3 are used for arguments, the rest
	// go on the stack.
	MOVM.DB.W [R4-R5], (R13)
	BL	libc_kevent(SB)
	ADD	$(2*4), R13
	MOVW	$-1, R2
	CMP	R0, R2
	BNE	ok
	BL	libc_error(SB)
	MOVW	(R0), R0	// errno
	RSB	$0, R0, R0	// caller wants it as a negative error code
ok:
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 cmd
	MOVW	8(R0), R2	// arg 3 arg
	MOVW	0(R0), R0	// arg 1 fd
	BL	libc_fcntl(SB)
	RET

// sigaltstack is not supported on iOS, so our sigtramp has
// to do the stack switch ourselves.
TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	MOVW	$43, R0
	BL	libc_exit(SB)
	RET

// Thread related functions
// Note: On darwin/arm, the runtime always use runtime/cgo to
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
	MOVW	4(R0), R1	// arg 2 attr
	MOVW	0(R0), R0	// arg 1 mutex
	BL	libc_pthread_mutex_init(SB)
	RET

TEXT runtime·pthread_mutex_lock_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 mutex
	BL	libc_pthread_mutex_lock(SB)
	RET

TEXT runtime·pthread_mutex_unlock_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 mutex
	BL	libc_pthread_mutex_unlock(SB)
	RET

TEXT runtime·pthread_cond_init_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 attr
	MOVW	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_init(SB)
	RET

TEXT runtime·pthread_cond_wait_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 mutex
	MOVW	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_wait(SB)
	RET

TEXT runtime·pthread_cond_timedwait_relative_np_trampoline(SB),NOSPLIT,$0
	MOVW	4(R0), R1	// arg 2 mutex
	MOVW	8(R0), R2	// arg 3 timeout
	MOVW	0(R0), R0	// arg 1 cond
	BL	libc_pthread_cond_timedwait_relative_np(SB)
	RET

TEXT runtime·pthread_cond_signal_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 cond
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
	MOVW.W	R0, -4(R13)	// push structure pointer
	MOVW	0(R0), R12	// fn
	MOVW	8(R0), R1	// a2
	MOVW	12(R0), R2	// a3
	MOVW	4(R0), R0	// a1
	BL	(R12)
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 16(R2)	// save r1
	MOVW	R1, 20(R2)	// save r2
	MOVW	$-1, R3
	CMP	R0, R3
	BNE	ok
	MOVW.W	R2, -4(R13)	// push structure pointer
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 24(R2)	// save err
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
	MOVW.W	R0, -4(R13)	// push structure pointer
	MOVW	0(R0), R12	// fn
	MOVW	24(R0), R1	// a6
	MOVW.W	R1, -4(R13)
	MOVW	20(R0), R1	// a5
	MOVW.W	R1, -4(R13)
	MOVW	8(R0), R1	// a2
	MOVW	12(R0), R2	// a3
	MOVW	16(R0), R3	// a4
	MOVW	4(R0), R0	// a1
	BL	(R12)
	ADD	$8, R13
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 28(R2)	// save r1
	MOVW	R1, 32(R2)	// save r2
	MOVW	$-1, R3
	CMP	R0, R3
	BNE	ok
	MOVW.W	R2, -4(R13)	// push structure pointer
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 36(R2)	// save err
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
	MOVW.W	R0, -4(R13)	// push structure pointer
	MOVW	0(R0), R12	// fn
	MOVW	24(R0), R1	// a6
	MOVW.W	R1, -4(R13)
	MOVW	20(R0), R1	// a5
	MOVW.W	R1, -4(R13)
	MOVW	8(R0), R1	// a2
	MOVW	12(R0), R2	// a3
	MOVW	16(R0), R3	// a4
	MOVW	4(R0), R0	// a1
	BL	(R12)
	ADD	$8, R13
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 28(R2)	// save r1
	MOVW	R1, 32(R2)	// save r2
	MOVW	$-1, R3
	CMP	R0, R3
	BNE	ok
	CMP	R1, R3
	BNE	ok
	MOVW.W	R2, -4(R13)	// push structure pointer
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 36(R2)	// save err
ok:
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
TEXT runtime·syscall9(SB),NOSPLIT,$0
	MOVW.W	R0, -4(R13)	// push structure pointer
	MOVW	0(R0), R12	// fn
	MOVW	36(R0), R1	// a9
	MOVW.W	R1, -4(R13)
	MOVW	32(R0), R1	// a8
	MOVW.W	R1, -4(R13)
	MOVW	28(R0), R1	// a7
	MOVW.W	R1, -4(R13)
	MOVW	24(R0), R1	// a6
	MOVW.W	R1, -4(R13)
	MOVW	20(R0), R1	// a5
	MOVW.W	R1, -4(R13)
	MOVW	8(R0), R1	// a2
	MOVW	12(R0), R2	// a3
	MOVW	16(R0), R3	// a4
	MOVW	4(R0), R0	// a1
	BL	(R12)
	ADD	$20, R13
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 40(R2)	// save r1
	MOVW	R1, 44(R2)	// save r2
	MOVW	$-1, R3
	CMP	R0, R3
	BNE	ok
	MOVW.W	R2, -4(R13)	// push structure pointer
	BL	libc_error(SB)
	MOVW	(R0), R0
	MOVW.P	4(R13), R2	// pop structure pointer
	MOVW	R0, 48(R2)	// save err
ok:
	RET
