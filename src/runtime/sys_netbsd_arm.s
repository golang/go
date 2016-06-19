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

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVW code+0(FP), R0	// arg 1 exit status
	SWI $0xa00001
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-4
	SWI $0xa00136	// sys__lwp_exit
	MOVW $1, R8	// crash
	MOVW R8, (R8)
	RET
	
TEXT runtime·open(SB),NOSPLIT,$-8
	MOVW name+0(FP), R0
	MOVW mode+4(FP), R1
	MOVW perm+8(FP), R2
	SWI $0xa00005
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-8
	MOVW fd+0(FP), R0
	SWI $0xa00006
	MOVW.CS	$-1, R0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-8
	MOVW fd+0(FP), R0
	MOVW p+4(FP), R1
	MOVW n+8(FP), R2
	SWI $0xa00003
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-4
	MOVW	fd+0(FP), R0	// arg 1 - fd
	MOVW	p+4(FP), R1	// arg 2 - buf
	MOVW	n+8(FP), R2	// arg 3 - nbyte
	SWI $0xa00004	// sys_write
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

// int32 lwp_create(void *context, uintptr flags, void *lwpid)
TEXT runtime·lwp_create(SB),NOSPLIT,$0
	MOVW ctxt+0(FP), R0
	MOVW flags+4(FP), R1
	MOVW lwpid+8(FP), R2
	SWI $0xa00135	// sys__lwp_create
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	SWI $0xa0015e	// sys_sched_yield
	RET

TEXT runtime·lwp_park(SB),NOSPLIT,$0
	MOVW abstime+0(FP), R0	// arg 1 - abstime
	MOVW unpark+4(FP), R1	// arg 2 - unpark
	MOVW hint+8(FP), R2	// arg 3 - hint
	MOVW unparkhint+12(FP), R3	// arg 4 - unparkhint
	SWI $0xa001b2	// sys__lwp_park
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·lwp_unpark(SB),NOSPLIT,$0
	MOVW	lwp+0(FP), R0	// arg 1 - lwp
	MOVW	hint+4(FP), R1	// arg 2 - hint
	SWI $0xa00141 // sys__lwp_unpark
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·lwp_self(SB),NOSPLIT,$0
	SWI $0xa00137	// sys__lwp_self
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
	SWI $0xa001ae	// sys_nanosleep
	RET

TEXT runtime·raise(SB),NOSPLIT,$16
	SWI $0xa00137	// sys__lwp_self, the returned R0 is arg 1
	MOVW	sig+0(FP), R1	// arg 2 - signal
	SWI $0xa0013e	// sys__lwp_kill
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	SWI $0xa00014	// sys_getpid, the returned R0 is arg 1
	MOVW	sig+0(FP), R1	// arg 2 - signal
	SWI $0xa00025	// sys_kill
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$-4
	MOVW mode+0(FP), R0	// arg 1 - which
	MOVW new+4(FP), R1	// arg 2 - itv
	MOVW old+8(FP), R2	// arg 3 - oitv
	SWI $0xa001a9	// sys_setitimer
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVW $0, R0	// CLOCK_REALTIME
	MOVW $8(R13), R1
	SWI $0xa001ab	// clock_gettime

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
	MOVW $0, R0 // CLOCK_REALTIME
	MOVW $8(R13), R1
	SWI $0xa001ab	// clock_gettime

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

TEXT runtime·getcontext(SB),NOSPLIT,$-4
	MOVW ctxt+0(FP), R0	// arg 1 - context
	SWI $0xa00133	// sys_getcontext
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW mode+0(FP), R0	// arg 1 - how
	MOVW new+4(FP), R1	// arg 2 - set
	MOVW old+8(FP), R2	// arg 3 - oset
	SWI $0xa00125	// sys_sigprocmask
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·sigreturn_tramp(SB),NOSPLIT,$-4
	// on entry, SP points to siginfo, we add sizeof(ucontext)
	// to SP to get a pointer to ucontext.
	ADD $0x80, R13, R0 // 0x80 == sizeof(UcontextT)
	SWI $0xa00134	// sys_setcontext
	// something failed, we have to exit
	MOVW $0x4242, R0 // magic return number
	SWI $0xa00001	// sys_exit
	B -2(PC)	// continue exit

TEXT runtime·sigaction(SB),NOSPLIT,$4
	MOVW sig+0(FP), R0	// arg 1 - signum
	MOVW new+4(FP), R1	// arg 2 - nsa
	MOVW old+8(FP), R2	// arg 3 - osa
	MOVW $runtime·sigreturn_tramp(SB), R3	// arg 4 - tramp
	MOVW $2, R4	// arg 5 - vers
	MOVW R4, 4(R13)
	ADD $4, R13	// pass arg 5 on stack
	SWI $0xa00154	// sys___sigaction_sigtramp
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
	SWI $0xa000c5	// sys_mmap
	SUB $4, R13
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW addr+0(FP), R0	// arg 1 - addr
	MOVW n+4(FP), R1	// arg 2 - len
	SWI $0xa00049	// sys_munmap
	MOVW.CS $0, R8	// crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW addr+0(FP), R0	// arg 1 - addr
	MOVW n+4(FP), R1	// arg 2 - len
	MOVW flags+8(FP), R2	// arg 3 - behav
	SWI $0xa0004b	// sys_madvise
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-4
	MOVW new+0(FP), R0	// arg 1 - nss
	MOVW old+4(FP), R1	// arg 2 - oss
	SWI $0xa00119	// sys___sigaltstack14
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
	SWI $0xa000ca	// sys___sysctl
	SUB $4, R13
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	SWI $0xa00158	// sys_kqueue
	RSB.CS $0, R0
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
	SWI $0xa001b3	// sys___kevent50
	RSB.CS $0, R0
	SUB $4, R13
	MOVW	R0, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW fd+0(FP), R0	// fd
	MOVW $2, R1	// F_SETFD
	MOVW $1, R2	// FD_CLOEXEC
	SWI $0xa0005c	// sys_fcntl
	RET

// TODO: this is only valid for ARMv7+
TEXT ·publicationBarrier(SB),NOSPLIT,$-4-0
	B	runtime·armPublicationBarrier(SB)

TEXT runtime·read_tls_fallback(SB),NOSPLIT,$-4
	MOVM.WP [R1, R2, R3, R12], (R13)
	SWI $0x00a0013c // _lwp_getprivate
	MOVM.IAW    (R13), [R1, R2, R3, R12]
	RET
