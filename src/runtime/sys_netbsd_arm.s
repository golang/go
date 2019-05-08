// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for ARM, NetBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME		0
#define CLOCK_MONOTONIC		3
#define FD_CLOEXEC		1
#define F_SETFD			2

#define SWI_OS_NETBSD			0xa00000
#define SYS_exit			SWI_OS_NETBSD | 1
#define SYS_read			SWI_OS_NETBSD | 3
#define SYS_write			SWI_OS_NETBSD | 4
#define SYS_open			SWI_OS_NETBSD | 5
#define SYS_close			SWI_OS_NETBSD | 6
#define SYS_getpid			SWI_OS_NETBSD | 20
#define SYS_kill			SWI_OS_NETBSD | 37
#define SYS_munmap			SWI_OS_NETBSD | 73
#define SYS_madvise			SWI_OS_NETBSD | 75
#define SYS_fcntl			SWI_OS_NETBSD | 92
#define SYS_mmap			SWI_OS_NETBSD | 197
#define SYS___sysctl			SWI_OS_NETBSD | 202
#define SYS___sigaltstack14		SWI_OS_NETBSD | 281
#define SYS___sigprocmask14		SWI_OS_NETBSD | 293
#define SYS_getcontext			SWI_OS_NETBSD | 307
#define SYS_setcontext			SWI_OS_NETBSD | 308
#define SYS__lwp_create			SWI_OS_NETBSD | 309
#define SYS__lwp_exit			SWI_OS_NETBSD | 310
#define SYS__lwp_self			SWI_OS_NETBSD | 311
#define SYS__lwp_getprivate		SWI_OS_NETBSD | 316
#define SYS__lwp_setprivate		SWI_OS_NETBSD | 317
#define SYS__lwp_kill			SWI_OS_NETBSD | 318
#define SYS__lwp_unpark			SWI_OS_NETBSD | 321
#define SYS___sigaction_sigtramp	SWI_OS_NETBSD | 340
#define SYS_kqueue			SWI_OS_NETBSD | 344
#define SYS_sched_yield			SWI_OS_NETBSD | 350
#define SYS___setitimer50		SWI_OS_NETBSD | 425
#define SYS___clock_gettime50		SWI_OS_NETBSD | 427
#define SYS___nanosleep50		SWI_OS_NETBSD | 430
#define SYS___kevent50			SWI_OS_NETBSD | 435
#define SYS____lwp_park60		SWI_OS_NETBSD | 478

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0
	MOVW code+0(FP), R0	// arg 1 exit status
	SWI $SYS_exit
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVW wait+0(FP), R0
	// We're done using the stack.
	MOVW $0, R2
storeloop:
	LDREX (R0), R4          // loads R4
	STREX R2, (R0), R1      // stores R2
	CMP $0, R1
	BNE storeloop
	SWI $SYS__lwp_exit
	MOVW $1, R8	// crash
	MOVW R8, (R8)
	JMP 0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0
	MOVW name+0(FP), R0
	MOVW mode+4(FP), R1
	MOVW perm+8(FP), R2
	SWI $SYS_open
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0
	MOVW fd+0(FP), R0
	SWI $SYS_close
	MOVW.CS	$-1, R0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0
	MOVW fd+0(FP), R0
	MOVW p+4(FP), R1
	MOVW n+8(FP), R2
	SWI $SYS_read
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·write(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0	// arg 1 - fd
	MOVW	p+4(FP), R1	// arg 2 - buf
	MOVW	n+8(FP), R2	// arg 3 - nbyte
	SWI $SYS_write
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

// int32 lwp_create(void *context, uintptr flags, void *lwpid)
TEXT runtime·lwp_create(SB),NOSPLIT,$0
	MOVW ctxt+0(FP), R0
	MOVW flags+4(FP), R1
	MOVW lwpid+8(FP), R2
	SWI $SYS__lwp_create
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	SWI $SYS_sched_yield
	RET

TEXT runtime·lwp_park(SB),NOSPLIT,$8
	MOVW clockid+0(FP), R0		// arg 1 - clock_id
	MOVW flags+4(FP), R1		// arg 2 - flags
	MOVW ts+8(FP), R2		// arg 3 - ts
	MOVW unpark+12(FP), R3		// arg 4 - unpark
	MOVW hint+16(FP), R4		// arg 5 - hint
	MOVW R4, 4(R13)
	MOVW unparkhint+20(FP), R5	// arg 6 - unparkhint
	MOVW R5, 8(R13)
	SWI $SYS____lwp_park60
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·lwp_unpark(SB),NOSPLIT,$0
	MOVW	lwp+0(FP), R0	// arg 1 - lwp
	MOVW	hint+4(FP), R1	// arg 2 - hint
	SWI	$SYS__lwp_unpark
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·lwp_self(SB),NOSPLIT,$0
	SWI	$SYS__lwp_self
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·lwp_tramp(SB),NOSPLIT,$0
	MOVW R0, g_m(R1)
	MOVW R1, g

	BL runtime·emptyfunc(SB) // fault if stack check is wrong
	BL (R2)
	MOVW $2, R8  // crash (not reached)
	MOVW R8, (R8)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16
	MOVW usec+0(FP), R0
	CALL runtime·usplitR0(SB)
	// 0(R13) is the saved LR, don't use it
	MOVW R0, 4(R13) // tv_sec.low
	MOVW $0, R0
	MOVW R0, 8(R13) // tv_sec.high
	MOVW $1000, R2
	MUL R1, R2
	MOVW R2, 12(R13) // tv_nsec

	MOVW $4(R13), R0 // arg 1 - rqtp
	MOVW $0, R1      // arg 2 - rmtp
	SWI $SYS___nanosleep50
	RET

TEXT runtime·raise(SB),NOSPLIT,$16
	SWI	$SYS__lwp_self	// the returned R0 is arg 1
	MOVW	sig+0(FP), R1	// arg 2 - signal
	SWI	$SYS__lwp_kill
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	SWI	$SYS_getpid	// the returned R0 is arg 1
	MOVW	sig+0(FP), R1	// arg 2 - signal
	SWI	$SYS_kill
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0
	MOVW mode+0(FP), R0	// arg 1 - which
	MOVW new+4(FP), R1	// arg 2 - itv
	MOVW old+8(FP), R2	// arg 3 - oitv
	SWI $SYS___setitimer50
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	MOVW $0, R0	// CLOCK_REALTIME
	MOVW $8(R13), R1
	SWI $SYS___clock_gettime50

	MOVW 8(R13), R0	// sec.low
	MOVW 12(R13), R1 // sec.high
	MOVW 16(R13), R2 // nsec

	MOVW R0, sec_lo+0(FP)
	MOVW R1, sec_hi+4(FP)
	MOVW R2, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB), NOSPLIT, $32
	MOVW $3, R0 // CLOCK_MONOTONIC
	MOVW $8(R13), R1
	SWI $SYS___clock_gettime50

	MOVW 8(R13), R0 // sec.low
	MOVW 12(R13), R4 // sec.high
	MOVW 16(R13), R2 // nsec

	MOVW $1000000000, R3
	MULLU R0, R3, (R1, R0)
	MUL R3, R4
	ADD.S R2, R0
	ADC R4, R1

	MOVW R0, ret_lo+0(FP)
	MOVW R1, ret_hi+4(FP)
	RET

TEXT runtime·getcontext(SB),NOSPLIT|NOFRAME,$0
	MOVW ctxt+0(FP), R0	// arg 1 - context
	SWI $SYS_getcontext
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW how+0(FP), R0	// arg 1 - how
	MOVW new+4(FP), R1	// arg 2 - set
	MOVW old+8(FP), R2	// arg 3 - oset
	SWI $SYS___sigprocmask14
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT sigreturn_tramp<>(SB),NOSPLIT|NOFRAME,$0
	// on entry, SP points to siginfo, we add sizeof(ucontext)
	// to SP to get a pointer to ucontext.
	ADD $0x80, R13, R0 // 0x80 == sizeof(UcontextT)
	SWI $SYS_setcontext
	// something failed, we have to exit
	MOVW $0x4242, R0 // magic return number
	SWI $SYS_exit
	B -2(PC)	// continue exit

TEXT runtime·sigaction(SB),NOSPLIT,$4
	MOVW sig+0(FP), R0	// arg 1 - signum
	MOVW new+4(FP), R1	// arg 2 - nsa
	MOVW old+8(FP), R2	// arg 3 - osa
	MOVW $sigreturn_tramp<>(SB), R3	// arg 4 - tramp
	MOVW $2, R4	// arg 5 - vers
	MOVW R4, 4(R13)
	ADD $4, R13	// pass arg 5 on stack
	SWI $SYS___sigaction_sigtramp
	SUB $4, R13
	MOVW.CS $3, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
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

TEXT runtime·sigtramp(SB),NOSPLIT,$12
	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVW	R0, 4(R13) // signum
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	BL.NE	runtime·load_g(SB)

	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·sigtrampgo(SB)
	RET

TEXT runtime·mmap(SB),NOSPLIT,$12
	MOVW addr+0(FP), R0	// arg 1 - addr
	MOVW n+4(FP), R1	// arg 2 - len
	MOVW prot+8(FP), R2	// arg 3 - prot
	MOVW flags+12(FP), R3	// arg 4 - flags
	// arg 5 (fid) and arg6 (offset_lo, offset_hi) are passed on stack
	// note the C runtime only passes the 32-bit offset_lo to us
	MOVW fd+16(FP), R4		// arg 5
	MOVW R4, 4(R13)
	MOVW off+20(FP), R5		// arg 6 lower 32-bit
	MOVW R5, 8(R13)
	MOVW $0, R6 // higher 32-bit for arg 6
	MOVW R6, 12(R13)
	ADD $4, R13 // pass arg 5 and arg 6 on stack
	SWI $SYS_mmap
	SUB $4, R13
	MOVW	$0, R1
	MOVW.CS R0, R1	// if error, move to R1
	MOVW.CS $0, R0
	MOVW	R0, p+24(FP)
	MOVW	R1, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW addr+0(FP), R0	// arg 1 - addr
	MOVW n+4(FP), R1	// arg 2 - len
	SWI $SYS_munmap
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0	// arg 1 - addr
	MOVW	n+4(FP), R1	// arg 2 - len
	MOVW	flags+8(FP), R2	// arg 3 - behav
	SWI	$SYS_madvise
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVW new+0(FP), R0	// arg 1 - nss
	MOVW old+4(FP), R1	// arg 2 - oss
	SWI $SYS___sigaltstack14
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$8
	MOVW mib+0(FP), R0	// arg 1 - name
	MOVW miblen+4(FP), R1	// arg 2 - namelen
	MOVW out+8(FP), R2	// arg 3 - oldp
	MOVW size+12(FP), R3	// arg 4 - oldlenp
	MOVW dst+16(FP), R4	// arg 5 - newp
	MOVW R4, 4(R13)
	MOVW ndst+20(FP), R4	// arg 6 - newlen
	MOVW R4, 8(R13)
	ADD $4, R13	// pass arg 5 and 6 on stack
	SWI $SYS___sysctl
	SUB $4, R13
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	SWI	$SYS_kqueue
	RSB.CS	$0, R0
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$8
	MOVW kq+0(FP), R0	// kq
	MOVW ch+4(FP), R1	// changelist
	MOVW nch+8(FP), R2	// nchanges
	MOVW ev+12(FP), R3	// eventlist
	MOVW nev+16(FP), R4	// nevents
	MOVW R4, 4(R13)
	MOVW ts+20(FP), R4	// timeout
	MOVW R4, 8(R13)
	ADD $4, R13	// pass arg 5 and 6 on stack
	SWI $SYS___kevent50
	RSB.CS $0, R0
	SUB $4, R13
	MOVW	R0, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW fd+0(FP), R0	// fd
	MOVW $F_SETFD, R1	// F_SETFD
	MOVW $FD_CLOEXEC, R2	// FD_CLOEXEC
	SWI $SYS_fcntl
	RET

// TODO: this is only valid for ARMv7+
TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	MOVM.WP [R1, R2, R3, R12], (R13)
	SWI $SYS__lwp_getprivate
	MOVM.IAW    (R13), [R1, R2, R3, R12]
	RET
