// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Copied from /usr/include/sys/syscall.h
#define	SYS_gettimeofday   116
#define	SYS_kill           37
#define	SYS_getpid         20
#define	SYS_pthread_sigmask 329
#define	SYS_setitimer      83
#define	SYS___sysctl       202
#define	SYS_sigaction      46
#define	SYS_sigreturn      184
#define	SYS_kqueue         362
#define	SYS_kevent         363
#define	SYS_fcntl          92

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

TEXT runtime·raiseproc(SB),NOSPLIT,$24
	MOVW	$SYS_getpid, R12
	SWI	$0x80
	// arg 1 pid already in R0 from getpid
	MOVW	sig+0(FP), R1	// arg 2 - signal
	MOVW	$1, R2	// arg 3 - posix
	MOVW	$SYS_kill, R12
	SWI $0x80
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

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	$SYS_setitimer, R12
	SWI	$0x80
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

// Sigtramp's job is to call the actual signal handler.
// It is called with the following arguments on the stack:
//	 LR  	"return address" - ignored
//	 R0  	actual handler
//	 R1  	siginfo style - ignored
//	 R2   	signal number
//	 R3   	siginfo
//	 -4(FP)	context, beware that 0(FP) is the saved LR
TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVM.DB.W [R0], (R13)
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	BL.NE	runtime·load_g(SB)

	CMP 	$0, g
	BNE 	cont
	// fake function call stack frame for badsignal
	// we only need to pass R2 (signal number), but
	// badsignal will expect R2 at 4(R13), so we also
	// push R1 onto stack. turns out we do need R1
	// to do sigreturn.
	MOVM.DB.W [R1,R2], (R13)
	MOVW  	$runtime·badsignal(SB), R11
	BL	(R11)
	MOVM.IA.W [R1], (R13) // saved infostype
	ADD		$(4+4), R13 // +4: also need to remove the pushed R0.
	MOVW    ucontext-4(FP), R0 // load ucontext
	B	ret

cont:
	// Restore R0
	MOVM.IA.W (R13), [R0]

	// NOTE: some Darwin/ARM kernels always use the main stack to run the
	// signal handler. We need to switch to gsignal ourselves.
	MOVW	g_m(g), R11
	MOVW	m_gsignal(R11), R5
	MOVW	(g_stack+stack_hi)(R5), R6
	SUB		$28, R6

	// copy arguments for call to sighandler
	MOVW	R2, 4(R6) // signal num
	MOVW	R3, 8(R6) // signal info
	MOVW	g, 16(R6) // old_g
	MOVW	context-4(FP), R4
	MOVW	R4, 12(R6) // context

	// Backup ucontext and infostyle
	MOVW    R4, 20(R6)
	MOVW    R1, 24(R6)

	// switch stack and g
	MOVW	R6, R13 // sigtramp is not re-entrant, so no need to back up R13.
	MOVW	R5, g

	BL	(R0)

	// call sigreturn
	MOVW	20(R13), R0	// saved ucontext
	MOVW	24(R13), R1	// saved infostyle
ret:
	MOVW	$SYS_sigreturn, R12 // sigreturn(ucontext, infostyle)
	SWI	$0x80

	// if sigreturn fails, we can do nothing but exit
	B	runtime·exit(SB)

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW	how+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	$SYS_pthread_sigmask, R12
	SWI	$0x80
	BL.CS	notok<>(SB)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	$SYS_sigaction, R12
	SWI	$0x80
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 usec
	BL libc_usleep(SB)
	RET

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVW	mib+0(FP), R0
	MOVW	miblen+4(FP), R1
	MOVW	out+8(FP), R2
	MOVW	size+12(FP), R3
	MOVW	dst+16(FP), R4
	MOVW	ndst+20(FP), R5
	MOVW	$SYS___sysctl, R12 // syscall entry
	SWI	$0x80
	BCC     sysctl_ret
	RSB     $0, R0, R0
	MOVW	R0, ret+24(FP)
	RET
sysctl_ret:
	MOVW	$0, R0
	MOVW	R0, ret+24(FP)
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

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVW	$SYS_kqueue, R12
	SWI	$0x80
	RSB.CS	$0, R0, R0
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int events, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW	$SYS_kevent, R12
	MOVW	kq+0(FP), R0
	MOVW	ch+4(FP), R1
	MOVW	nch+8(FP), R2
	MOVW	ev+12(FP), R3
	MOVW	nev+16(FP), R4
	MOVW	ts+20(FP), R5
	SWI	$0x80
	RSB.CS	$0, R0, R0
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW	$SYS_fcntl, R12
	MOVW	fd+0(FP), R0
	MOVW	$2, R1	// F_SETFD
	MOVW	$1, R2	// FD_CLOEXEC
	SWI	$0x80
	RET

// sigaltstack on some darwin/arm version is buggy and will always
// run the signal handler on the main stack, so our sigtramp has
// to do the stack switch ourselves.
TEXT runtime·sigaltstack(SB),NOSPLIT,$0
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
