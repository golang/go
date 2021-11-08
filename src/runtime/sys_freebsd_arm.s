// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for ARM, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// for EABI, as we don't support OABI
#define SYS_BASE 0x0

#define SYS_exit (SYS_BASE + 1)
#define SYS_read (SYS_BASE + 3)
#define SYS_write (SYS_BASE + 4)
#define SYS_open (SYS_BASE + 5)
#define SYS_close (SYS_BASE + 6)
#define SYS_getpid (SYS_BASE + 20)
#define SYS_kill (SYS_BASE + 37)
#define SYS_pipe (SYS_BASE + 42)
#define SYS_sigaltstack (SYS_BASE + 53)
#define SYS_munmap (SYS_BASE + 73)
#define SYS_madvise (SYS_BASE + 75)
#define SYS_setitimer (SYS_BASE + 83)
#define SYS_fcntl (SYS_BASE + 92)
#define SYS___sysctl (SYS_BASE + 202)
#define SYS_nanosleep (SYS_BASE + 240)
#define SYS_clock_gettime (SYS_BASE + 232)
#define SYS_sched_yield (SYS_BASE + 331)
#define SYS_sigprocmask (SYS_BASE + 340)
#define SYS_kqueue (SYS_BASE + 362)
#define SYS_kevent (SYS_BASE + 363)
#define SYS_sigaction (SYS_BASE + 416)
#define SYS_thr_exit (SYS_BASE + 431)
#define SYS_thr_self (SYS_BASE + 432)
#define SYS_thr_kill (SYS_BASE + 433)
#define SYS__umtx_op (SYS_BASE + 454)
#define SYS_thr_new (SYS_BASE + 455)
#define SYS_mmap (SYS_BASE + 477)
#define SYS_cpuset_getaffinity (SYS_BASE + 487)
#define SYS_pipe2 (SYS_BASE + 542)

TEXT runtime·sys_umtx_op(SB),NOSPLIT,$0
	MOVW addr+0(FP), R0
	MOVW mode+4(FP), R1
	MOVW val+8(FP), R2
	MOVW uaddr1+12(FP), R3
	ADD $20, R13 // arg 5 is passed on stack
	MOVW $SYS__umtx_op, R7
	SWI $0
	RSB.CS $0, R0
	SUB $20, R13
	// BCS error
	MOVW	R0, ret+20(FP)
	RET

TEXT runtime·thr_new(SB),NOSPLIT,$0
	MOVW param+0(FP), R0
	MOVW size+4(FP), R1
	MOVW $SYS_thr_new, R7
	SWI $0
	RSB.CS $0, R0
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·thr_start(SB),NOSPLIT,$0
	// set up g
	MOVW m_g0(R0), g
	MOVW R0, g_m(g)
	BL runtime·emptyfunc(SB) // fault if stack check is wrong
	BL runtime·mstart(SB)

	MOVW $2, R8  // crash (not reached)
	MOVW R8, (R8)
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0
	MOVW code+0(FP), R0	// arg 1 exit status
	MOVW $SYS_exit, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVW	wait+0(FP), R0
	// We're done using the stack.
	MOVW	$0, R2
storeloop:
	LDREX	(R0), R4          // loads R4
	STREX	R2, (R0), R1      // stores R2
	CMP	$0, R1
	BNE	storeloop
	MOVW	$0, R0		// arg 1 long *state
	MOVW	$SYS_thr_exit, R7
	SWI	$0
	MOVW.CS	$0, R8 // crash on syscall failure
	MOVW.CS	R8, (R8)
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0
	MOVW name+0(FP), R0	// arg 1 name
	MOVW mode+4(FP), R1	// arg 2 mode
	MOVW perm+8(FP), R2	// arg 3 perm
	MOVW $SYS_open, R7
	SWI $0
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0
	MOVW fd+0(FP), R0	// arg 1 fd
	MOVW p+4(FP), R1	// arg 2 buf
	MOVW n+8(FP), R2	// arg 3 count
	MOVW $SYS_read, R7
	SWI $0
	RSB.CS	$0, R0		// caller expects negative errno
	MOVW	R0, ret+12(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT,$0-12
	MOVW	$SYS_pipe, R7
	SWI	$0
	BCC	ok
	MOVW	$0, R1
	MOVW	R1, r+0(FP)
	MOVW	R1, w+4(FP)
	MOVW	R0, errno+8(FP)
	RET
ok:
	MOVW	R0, r+0(FP)
	MOVW	R1, w+4(FP)
	MOVW	$0, R1
	MOVW	R1, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$0-16
	MOVW	$r+4(FP), R0
	MOVW	flags+0(FP), R1
	MOVW	$SYS_pipe2, R7
	SWI	$0
	RSB.CS $0, R0
	MOVW	R0, errno+12(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0
	MOVW fd+0(FP), R0	// arg 1 fd
	MOVW p+4(FP), R1	// arg 2 buf
	MOVW n+8(FP), R2	// arg 3 count
	MOVW $SYS_write, R7
	SWI $0
	RSB.CS	$0, R0		// caller expects negative errno
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0
	MOVW fd+0(FP), R0	// arg 1 fd
	MOVW $SYS_close, R7
	SWI $0
	MOVW.CS	$-1, R0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·thr_self(SB),NOSPLIT,$0-4
	// thr_self(&0(FP))
	MOVW $ret+0(FP), R0 // arg 1
	MOVW $SYS_thr_self, R7
	SWI $0
	RET

TEXT runtime·thr_kill(SB),NOSPLIT,$0-8
	// thr_kill(tid, sig)
	MOVW tid+0(FP), R0	// arg 1 id
	MOVW sig+4(FP), R1	// arg 2 signal
	MOVW $SYS_thr_kill, R7
	SWI $0
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$0
	// getpid
	MOVW $SYS_getpid, R7
	SWI $0
	// kill(self, sig)
				// arg 1 - pid, now in R0
	MOVW sig+0(FP), R1	// arg 2 - signal
	MOVW $SYS_kill, R7
	SWI $0
	RET

TEXT runtime·setitimer(SB), NOSPLIT|NOFRAME, $0
	MOVW mode+0(FP), R0
	MOVW new+4(FP), R1
	MOVW old+8(FP), R2
	MOVW $SYS_setitimer, R7
	SWI $0
	RET

// func fallback_walltime() (sec int64, nsec int32)
TEXT runtime·fallback_walltime(SB), NOSPLIT, $32-12
	MOVW $0, R0 // CLOCK_REALTIME
	MOVW $8(R13), R1
	MOVW $SYS_clock_gettime, R7
	SWI $0

	MOVW 8(R13), R0 // sec.low
	MOVW 12(R13), R1 // sec.high
	MOVW 16(R13), R2 // nsec

	MOVW R0, sec_lo+0(FP)
	MOVW R1, sec_hi+4(FP)
	MOVW R2, nsec+8(FP)
	RET

// func fallback_nanotime() int64
TEXT runtime·fallback_nanotime(SB), NOSPLIT, $32
	MOVW $4, R0 // CLOCK_MONOTONIC
	MOVW $8(R13), R1
	MOVW $SYS_clock_gettime, R7
	SWI $0

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

TEXT runtime·asmSigaction(SB),NOSPLIT|NOFRAME,$0
	MOVW sig+0(FP), R0		// arg 1 sig
	MOVW new+4(FP), R1		// arg 2 act
	MOVW old+8(FP), R2		// arg 3 oact
	MOVW $SYS_sigaction, R7
	SWI $0
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// Reserve space for callee-save registers and arguments.
	MOVM.DB.W [R4-R11], (R13)
	SUB	$16, R13

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

	// Restore callee-save registers.
	ADD	$16, R13
	MOVM.IA.W (R13), [R4-R11]

	RET

TEXT runtime·mmap(SB),NOSPLIT,$16
	MOVW addr+0(FP), R0		// arg 1 addr
	MOVW n+4(FP), R1		// arg 2 len
	MOVW prot+8(FP), R2		// arg 3 prot
	MOVW flags+12(FP), R3		// arg 4 flags
	// arg 5 (fid) and arg6 (offset_lo, offset_hi) are passed on stack
	// note the C runtime only passes the 32-bit offset_lo to us
	MOVW fd+16(FP), R4		// arg 5
	MOVW R4, 4(R13)
	MOVW off+20(FP), R5		// arg 6 lower 32-bit
	// the word at 8(R13) is skipped due to 64-bit argument alignment.
	MOVW R5, 12(R13)
	MOVW $0, R6 		// higher 32-bit for arg 6
	MOVW R6, 16(R13)
	ADD $4, R13
	MOVW $SYS_mmap, R7
	SWI $0
	SUB $4, R13
	MOVW $0, R1
	MOVW.CS R0, R1		// if failed, put in R1
	MOVW.CS $0, R0
	MOVW	R0, p+24(FP)
	MOVW	R1, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW addr+0(FP), R0		// arg 1 addr
	MOVW n+4(FP), R1		// arg 2 len
	MOVW $SYS_munmap, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0		// arg 1 addr
	MOVW	n+4(FP), R1		// arg 2 len
	MOVW	flags+8(FP), R2		// arg 3 flags
	MOVW	$SYS_madvise, R7
	SWI	$0
	MOVW.CS $-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVW new+0(FP), R0
	MOVW old+4(FP), R1
	MOVW $SYS_sigaltstack, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
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
	MOVW $SYS_nanosleep, R7
	SWI $0
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVW mib+0(FP), R0	// arg 1 - name
	MOVW miblen+4(FP), R1	// arg 2 - namelen
	MOVW out+8(FP), R2	// arg 3 - old
	MOVW size+12(FP), R3	// arg 4 - oldlenp
	// arg 5 (newp) and arg 6 (newlen) are passed on stack
	ADD $20, R13
	MOVW $SYS___sysctl, R7
	SWI $0
	SUB.CS $0, R0, R0
	SUB $20, R13
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVW $SYS_sched_yield, R7
	SWI $0
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW how+0(FP), R0	// arg 1 - how
	MOVW new+4(FP), R1	// arg 2 - set
	MOVW old+8(FP), R2	// arg 3 - oset
	MOVW $SYS_sigprocmask, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

// int32 runtime·kqueue(void)
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVW $SYS_kqueue, R7
	SWI $0
	RSB.CS $0, R0
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW kq+0(FP), R0	// kq
	MOVW ch+4(FP), R1	// changelist
	MOVW nch+8(FP), R2	// nchanges
	MOVW ev+12(FP), R3	// eventlist
	ADD $20, R13	// pass arg 5 and 6 on stack
	MOVW $SYS_kevent, R7
	SWI $0
	RSB.CS $0, R0
	SUB $20, R13
	MOVW	R0, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW fd+0(FP), R0	// fd
	MOVW $2, R1	// F_SETFD
	MOVW $1, R2	// FD_CLOEXEC
	MOVW $SYS_fcntl, R7
	SWI $0
	RET

// func runtime·setNonblock(fd int32)
TEXT runtime·setNonblock(SB),NOSPLIT,$0-4
	MOVW	fd+0(FP), R0	// fd
	MOVW	$3, R1	// F_GETFL
	MOVW	$0, R2
	MOVW	$SYS_fcntl, R7
	SWI	$0
	ORR	$0x4, R0, R2	// O_NONBLOCK
	MOVW	fd+0(FP), R0	// fd
	MOVW	$4, R1	// F_SETFL
	MOVW	$SYS_fcntl, R7
	SWI	$0
	RET

// TODO: this is only valid for ARMv7+
TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

// TODO(minux): this only supports ARMv6K+.
TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	WORD $0xee1d0f70 // mrc p15, 0, r0, c13, c0, 3
	RET

// func cpuset_getaffinity(level int, which int, id int64, size int, mask *byte) int32
TEXT runtime·cpuset_getaffinity(SB), NOSPLIT, $0-28
	MOVW	level+0(FP), R0
	MOVW	which+4(FP), R1
	MOVW	id_lo+8(FP), R2
	MOVW	id_hi+12(FP), R3
	ADD	$20, R13	// Pass size and mask on stack.
	MOVW	$SYS_cpuset_getaffinity, R7
	SWI	$0
	RSB.CS	$0, R0
	SUB	$20, R13
	MOVW	R0, ret+24(FP)
	RET

// func getCntxct(physical bool) uint32
TEXT runtime·getCntxct(SB),NOSPLIT|NOFRAME,$0-8
	MOVB	runtime·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	DMB

	MOVB	physical+0(FP), R0
	CMP	$1, R0
	B.NE	3(PC)

	// get CNTPCT (Physical Count Register) into R0(low) R1(high)
	// mrrc    15, 0, r0, r1, cr14
	WORD	$0xec510f0e
	B	2(PC)

	// get CNTVCT (Virtual Count Register) into R0(low) R1(high)
	// mrrc    15, 1, r0, r1, cr14
	WORD	$0xec510f1e

	MOVW	R0, ret+4(FP)
	RET
