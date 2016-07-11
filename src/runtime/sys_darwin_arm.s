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
#define	SYS_exit           1
#define	SYS_read           3
#define	SYS_write          4
#define	SYS_open           5
#define	SYS_close          6
#define	SYS_mmap           197
#define	SYS_munmap         73
#define	SYS_madvise        75
#define	SYS_mincore        78
#define	SYS_gettimeofday   116
#define	SYS_kill           37
#define	SYS_getpid         20
#define	SYS___pthread_kill 328
#define	SYS_pthread_sigmask 329
#define	SYS_setitimer      83
#define	SYS___sysctl       202
#define	SYS_sigaction      46
#define	SYS_sigreturn      184
#define	SYS_select         93
#define	SYS_bsdthread_register 366
#define	SYS_bsdthread_create 360
#define	SYS_bsdthread_terminate 361
#define	SYS_kqueue         362
#define	SYS_kevent         363
#define	SYS_fcntl          92

TEXT notok<>(SB),NOSPLIT,$0
	MOVW	$0, R8
	MOVW	R8, (R8)
	B		0(PC)

TEXT runtime·open(SB),NOSPLIT,$0
	MOVW	name+0(FP), R0
	MOVW	mode+4(FP), R1
	MOVW	perm+8(FP), R2
	MOVW	$SYS_open, R12
	SWI	$0x80
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	$SYS_close, R12
	SWI	$0x80
	MOVW.CS	$-1, R0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	MOVW	$SYS_write, R12
	SWI	$0x80
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	MOVW	$SYS_read, R12
	SWI	$0x80
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVW	code+0(FP), R0
	MOVW	$SYS_exit, R12
	SWI	$0x80
	MOVW	$1234, R0
	MOVW	$1002, R1
	MOVW	R0, (R1)	// fail hard

// Exit this OS thread (like pthread_exit, which eventually
// calls __bsdthread_terminate).
TEXT runtime·exit1(SB),NOSPLIT,$0
	MOVW	$SYS_bsdthread_terminate, R12
	SWI	$0x80
	MOVW	$1234, R0
	MOVW	$1003, R1
	MOVW	R0, (R1)	// fail hard

TEXT runtime·raise(SB),NOSPLIT,$0
	// Ideally we'd send the signal to the current thread,
	// not the whole process, but that's too hard on OS X.
	JMP	runtime·raiseproc(SB)

TEXT runtime·raiseproc(SB),NOSPLIT,$24
	MOVW	$SYS_getpid, R12
	SWI	$0x80
	// arg 1 pid already in R0 from getpid
	MOVW	sig+0(FP), R1	// arg 2 - signal
	MOVW	$1, R2	// arg 3 - posix
	MOVW	$SYS_kill, R12
	SWI $0x80
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	prot+8(FP), R2
	MOVW	flags+12(FP), R3
	MOVW	fd+16(FP), R4
	MOVW	off+20(FP), R5
	MOVW	$0, R6 // off_t is uint64_t
	MOVW	$SYS_mmap, R12
	SWI	$0x80
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	$SYS_munmap, R12
	SWI	$0x80
	BL.CS	notok<>(SB)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	flags+8(FP), R2
	MOVW	$SYS_madvise, R12
	SWI	$0x80
	BL.CS	notok<>(SB)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	$SYS_setitimer, R12
	SWI	$0x80
	RET

TEXT runtime·mincore(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	dst+8(FP), R2
	MOVW	$SYS_mincore, R12
	SWI	$0x80
	MOVW	R0, ret+12(FP)
	RET

TEXT time·now(SB), 7, $32
	MOVW	$8(R13), R0  // timeval
	MOVW	$0, R1  // zone
	MOVW	$0, R2	// see issue 16570
	MOVW	$SYS_gettimeofday, R12
	SWI	$0x80 // Note: R0 is tv_sec, R1 is tv_usec
	CMP	$0, R0
	BNE	inreg
	MOVW	8(R13), R0
	MOVW	12(R13), R1
inreg:
	MOVW    R1, R2  // usec
	MOVW	R0, sec+0(FP)
	MOVW	$0, R1
	MOVW	R1, loc+4(FP)
	MOVW	$1000, R3
	MUL	R3, R2
	MOVW	R2, nsec+8(FP)
	RET

TEXT runtime·nanotime(SB),NOSPLIT,$32
	MOVW	$8(R13), R0  // timeval
	MOVW	$0, R1  // zone
	MOVW	$0, R2	// see issue 16570
	MOVW	$SYS_gettimeofday, R12
	SWI	$0x80 // Note: R0 is tv_sec, R1 is tv_usec
	CMP	$0, R0
	BNE	inreg
	MOVW	8(R13), R0
	MOVW	12(R13), R1
inreg:
	MOVW    R1, R2
	MOVW	$1000000000, R3
	MULLU	R0, R3, (R1, R0)
	MOVW	$1000, R3
	MOVW	$0, R4
	MUL	R3, R2
	ADD.S	R2, R0
	ADC	R4, R1

	MOVW	R0, ret_lo+0(FP)
	MOVW	R1, ret_hi+4(FP)
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

TEXT runtime·usleep(SB),NOSPLIT,$12
	MOVW	usec+0(FP), R0
	CALL	runtime·usplitR0(SB)
	MOVW	R0, a-12(SP)
	MOVW	R1, b-8(SP)

	// select(0, 0, 0, 0, &tv)
	MOVW	$0, R0
	MOVW	$0, R1
	MOVW	$0, R2
	MOVW	$0, R3
	MOVW	$a-12(SP), R4
	MOVW	$SYS_select, R12
	SWI	$0x80
	RET

TEXT ·publicationBarrier(SB),NOSPLIT,$-4-0
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

// Thread related functions
// func bsdthread_create(stk, arg unsafe.Pointer, fn uintptr) int32
TEXT runtime·bsdthread_create(SB),NOSPLIT,$0
	// Set up arguments to bsdthread_create system call.
	// The ones in quotes pass through to the thread callback
	// uninterpreted, so we can put whatever we want there.
	MOVW    fn+8(FP),    R0 // "func"
	MOVW    arg+4(FP),   R1 // "arg"
	MOVW    stk+0(FP),   R2 // stack
	MOVW	$0x01000000, R4	// flags = PTHREAD_START_CUSTOM
	MOVW	$0,          R5 // paranoia
	MOVW	$SYS_bsdthread_create, R12
	SWI	$0x80
	BCC		create_ret
	RSB 	$0, R0, R0
	MOVW	R0, ret+12(FP)
	RET
create_ret:
	MOVW	$0, R0
	MOVW	R0, ret+12(FP)
	RET

// The thread that bsdthread_create creates starts executing here,
// because we registered this function using bsdthread_register
// at startup.
//	R0 = "pthread"
//	R1 = mach thread port
//	R2 = "func" (= fn)
//	R3 = "arg" (= m)
//	R4 = stack
//	R5 = flags (= 0)
// XXX: how to deal with R4/SP? ref: Libc-594.9.1/arm/pthreads/thread_start.s
TEXT runtime·bsdthread_start(SB),NOSPLIT,$0
	MOVW    R1, m_procid(R3) // thread port is m->procid
	MOVW	m_g0(R3), g
	MOVW	R3, g_m(g)
	// ARM don't have runtime·stackcheck(SB)
	// disable runfast mode of vfp
	EOR     R12, R12
	WORD    $0xeee1ca10 // fmxr	fpscr, ip
	BL      (R2) // fn
	BL      runtime·exit1(SB)
	RET

// int32 bsdthread_register(void)
// registers callbacks for threadstart (see bsdthread_create above
// and wqthread and pthsize (not used).  returns 0 on success.
TEXT runtime·bsdthread_register(SB),NOSPLIT,$0
	MOVW	$runtime·bsdthread_start(SB), R0	// threadstart
	MOVW	$0, R1	// wqthread, not used by us
	MOVW	$0, R2	// pthsize, not used by us
	MOVW	$0, R3 	// dummy_value [sic]
	MOVW	$0, R4	// targetconc_ptr
	MOVW	$0, R5	// dispatchqueue_offset
	MOVW	$SYS_bsdthread_register, R12	// bsdthread_register
	SWI	$0x80
	MOVW	R0, ret+0(FP)
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
