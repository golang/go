// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.

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

// uint32 mach_msg_trap(void*, uint32, uint32, uint32, uint32, uint32, uint32)
TEXT runtime·mach_msg_trap(SB),NOSPLIT,$0
	MOVW    h+0(FP), R0
	MOVW    op+4(FP), R1
	MOVW    send_size+8(FP), R2
	MOVW    rcv_size+12(FP), R3
	MOVW    rcv_name+16(FP), R4
	MOVW    timeout+20(FP), R5
	MOVW    notify+24(FP), R6
	MVN     $30, R12
	SWI	$0x80
	MOVW	R0, ret+28(FP)
	RET

TEXT runtime·mach_task_self(SB),NOSPLIT,$0
	MVN     $27, R12 // task_self_trap
	SWI	$0x80
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·mach_thread_self(SB),NOSPLIT,$0
	MVN 	$26, R12 // thread_self_trap
	SWI	$0x80
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·mach_reply_port(SB),NOSPLIT,$0
	MVN 	$25, R12	// mach_reply_port
	SWI	$0x80
	MOVW	R0, ret+0(FP)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// uint32 mach_semaphore_wait(uint32)
TEXT runtime·mach_semaphore_wait(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MVN 	$35, R12	// semaphore_wait_trap
	SWI	$0x80
	MOVW	R0, ret+4(FP)
	RET

// uint32 mach_semaphore_timedwait(uint32, uint32, uint32)
TEXT runtime·mach_semaphore_timedwait(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MOVW	sec+4(FP), R1
	MOVW	nsec+8(FP), R2
	MVN 	$37, R12	// semaphore_timedwait_trap
	SWI	$0x80
	MOVW	R0, ret+12(FP)
	RET

// uint32 mach_semaphore_signal(uint32)
TEXT runtime·mach_semaphore_signal(SB),NOSPLIT,$0
	MOVW    sema+0(FP), R0
	MVN 	$32, R12	// semaphore_signal_trap
	SWI	$0x80
	MOVW	R0, ret+4(FP)
	RET

// uint32 mach_semaphore_signal_all(uint32)
TEXT runtime·mach_semaphore_signal_all(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MVN 	$33, R12	// semaphore_signal_all_trap
	SWI	$0x80
	MOVW	R0, ret+4(FP)
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
