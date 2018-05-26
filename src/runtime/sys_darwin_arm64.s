// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM64, Darwin
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
	MOVD	$0, R8
	MOVD	R8, (R8)
	B	0(PC)

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	MOVW	8(R0), R1	// arg 2 flags
	MOVW	12(R0), R2	// arg 3 mode
	MOVD	0(R0), R0	// arg 1 pathname
	BL libc_open(SB)
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

TEXT runtime·raiseproc(SB),NOSPLIT,$0
	MOVW	$SYS_getpid, R16
	SVC	$0x80
	// arg 1 pid already in R0 from getpid
	MOVW	sig+0(FP), R1	// arg 2 - signal
	MOVW	$1, R2	// arg 3 - posix
	MOVW	$SYS_kill, R16
	SVC	$0x80
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
	MOVD	(R0), R1
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

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVW	$SYS_setitimer, R16
	SVC	$0x80
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

// Sigtramp's job is to call the actual signal handler.
// It is called with the following arguments on the stack:
//	LR	"return address" - ignored
//	R0	actual handler
//	R1	siginfo style - ignored
//	R2	signal number
//	R3	siginfo
//	R4	context
TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVD.W	R0, -16(RSP)	// note: stack must be 16-byte aligned
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	CMP	$0, g
	BNE	cont
	// fake function call stack frame for badsignal
	// we only need to pass R2 (signal number), but
	// badsignal will expect R2 at 8(RSP), so we also
	// push R1 onto stack. turns out we do need R1
	// to do sigreturn.
	MOVD.W	R1, -16(RSP)
	MOVD	R2, 8(RSP)
	MOVD	R4, 24(RSP)	// save ucontext, badsignal might clobber R4
	MOVD	$runtime·badsignal(SB), R26
	BL	(R26)
	MOVD	0(RSP), R1	// saved infostype
	MOVD	24(RSP), R0	// the ucontext
	ADD	$(16+16), RSP
	B	ret

cont:
	// Restore R0
	MOVD.P	16(RSP), R0

	// NOTE: some Darwin/ARM kernels always use the main stack to run the
	// signal handler. We need to switch to gsignal ourselves.
	MOVD	g_m(g), R11
	MOVD	m_gsignal(R11), R5
	MOVD	(g_stack+stack_hi)(R5), R6
	SUB	$64, R6

	// copy arguments for call to sighandler
	MOVD	R2, 8(R6)	// signal num
	MOVD	R3, 16(R6)	// signal info
	MOVD	R4, 24(R6)	// context
	MOVD	g, 32(R6)	// old_g

	// Backup ucontext and infostyle
	MOVD	R4, 40(R6)
	MOVD	R1, 48(R6)

	// switch stack and g
	MOVD	R6, RSP	// sigtramp is not re-entrant, so no need to back up RSP.
	MOVD	R5, g

	BL	(R0)

	// call sigreturn
	MOVD	40(RSP), R0	// saved ucontext
	MOVD	48(RSP), R1	// saved infostyle
ret:
	MOVW	$SYS_sigreturn, R16 // sigreturn(ucontext, infostyle)
	SVC	$0x80

	// if sigreturn fails, we can do nothing but exit
	B	runtime·exit(SB)

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW	how+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVW	$SYS_pthread_sigmask, R16
	SVC	$0x80
	BCC	2(PC)
	BL	notok<>(SB)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVW	$SYS_sigaction, R16
	SVC	$0x80
	BCC	2(PC)
	BL	notok<>(SB)
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVW	0(R0), R0	// arg 1 usec
	BL	libc_usleep(SB)
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVD	mib+0(FP), R0
	MOVW	miblen+8(FP), R1
	MOVD	out+16(FP), R2
	MOVD	size+24(FP), R3
	MOVD	dst+32(FP), R4
	MOVD	ndst+40(FP), R5
	MOVW	$SYS___sysctl, R16
	SVC	$0x80
	BCC	ok
	NEG	R0, R0
	MOVW	R0, ret+48(FP)
	RET
ok:
	MOVW	$0, R0
	MOVW	R0, ret+48(FP)
	RET

// uint32 mach_msg_trap(void*, uint32, uint32, uint32, uint32, uint32, uint32)
TEXT runtime·mach_msg_trap(SB),NOSPLIT,$0
	MOVD	h+0(FP), R0
	MOVW	op+8(FP), R1
	MOVW	send_size+12(FP), R2
	MOVW	rcv_size+16(FP), R3
	MOVW	rcv_name+20(FP), R4
	MOVW	timeout+24(FP), R5
	MOVW	notify+28(FP), R6
	MOVN	$30, R16
	SVC	$0x80
	MOVW	R0, ret+32(FP)
	RET

TEXT runtime·mach_task_self(SB),NOSPLIT,$0
	MOVN	$27, R16 // task_self_trap
	SVC	$0x80
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·mach_thread_self(SB),NOSPLIT,$0
	MOVN	$26, R16 // thread_self_trap
	SVC	$0x80
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·mach_reply_port(SB),NOSPLIT,$0
	MOVN	$25, R16	// mach_reply_port
	SVC	$0x80
	MOVW	R0, ret+0(FP)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// uint32 mach_semaphore_wait(uint32)
TEXT runtime·mach_semaphore_wait(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MOVN	$35, R16	// semaphore_wait_trap
	SVC	$0x80
	MOVW	R0, ret+8(FP)
	RET

// uint32 mach_semaphore_timedwait(uint32, uint32, uint32)
TEXT runtime·mach_semaphore_timedwait(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MOVW	sec+4(FP), R1
	MOVW	nsec+8(FP), R2
	MOVN	$37, R16	// semaphore_timedwait_trap
	SVC	$0x80
	MOVW	R0, ret+16(FP)
	RET

// uint32 mach_semaphore_signal(uint32)
TEXT runtime·mach_semaphore_signal(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MOVN	$32, R16	// semaphore_signal_trap
	SVC	$0x80
	MOVW	R0, ret+8(FP)
	RET

// uint32 mach_semaphore_signal_all(uint32)
TEXT runtime·mach_semaphore_signal_all(SB),NOSPLIT,$0
	MOVW	sema+0(FP), R0
	MOVN	$33, R16	// semaphore_signal_all_trap
	SVC	$0x80
	MOVW	R0, ret+8(FP)
	RET

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVW	$SYS_kqueue, R16
	SVC	$0x80
	BCC	2(PC)
	NEG	R0, R0
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *ch, int nch, Kevent *ev, int nev, Timespec *ts)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW	kq+0(FP), R0
	MOVD	ch+8(FP), R1
	MOVW	nch+16(FP), R2
	MOVD	ev+24(FP), R3
	MOVW	nev+32(FP), R4
	MOVD	ts+40(FP), R5
	MOVW	$SYS_kevent, R16
	SVC	$0x80
	BCC	2(PC)
	NEG	R0, R0
	MOVW	R0, ret+48(FP)
	RET

// int32 runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	$2, R1	// F_SETFD
	MOVW	$1, R2	// FD_CLOEXEC
	MOVW	$SYS_fcntl, R16
	SVC	$0x80
	RET

// sigaltstack on some darwin/arm version is buggy and will always
// run the signal handler on the main stack, so our sigtramp has
// to do the stack switch ourselves.
TEXT runtime·sigaltstack(SB),NOSPLIT,$0
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
