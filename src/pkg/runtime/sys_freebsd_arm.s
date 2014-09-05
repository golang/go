// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for ARM, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"
#include "textflag.h"

// for EABI, as we don't support OABI
#define SYS_BASE 0x0

#define SYS_exit (SYS_BASE + 1)
#define SYS_read (SYS_BASE + 3)
#define SYS_write (SYS_BASE + 4)
#define SYS_open (SYS_BASE + 5)
#define SYS_close (SYS_BASE + 6)
#define SYS_sigaltstack (SYS_BASE + 53)
#define SYS_munmap (SYS_BASE + 73)
#define SYS_madvise (SYS_BASE + 75)
#define SYS_setitimer (SYS_BASE + 83)
#define SYS_fcntl (SYS_BASE + 92)
#define SYS_getrlimit (SYS_BASE + 194)
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
	
TEXT runtime·sys_umtx_op(SB),NOSPLIT,$0
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW 8(FP), R2
	MOVW 12(FP), R3
	ADD $20, R13 // arg 5 is passed on stack
	MOVW $SYS__umtx_op, R7
	SWI $0
	SUB $20, R13
	// BCS error
	MOVW	R0, ret+20(FP)
	RET

TEXT runtime·thr_new(SB),NOSPLIT,$0
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW $SYS_thr_new, R7
	SWI $0
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
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVW 0(FP), R0	// arg 1 exit status
	MOVW $SYS_exit, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·exit1(SB),NOSPLIT,$-8
	MOVW 0(FP), R0	// arg 1 exit status
	MOVW $SYS_thr_exit, R7	
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·open(SB),NOSPLIT,$-8
	MOVW 0(FP), R0	// arg 1 name
	MOVW 4(FP), R1	// arg 2 mode
	MOVW 8(FP), R2	// arg 3 perm
	MOVW $SYS_open, R7
	SWI $0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-8
	MOVW 0(FP), R0	// arg 1 fd
	MOVW 4(FP), R1	// arg 2 buf
	MOVW 8(FP), R2	// arg 3 count
	MOVW $SYS_read, R7
	SWI $0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$-8
	MOVW 0(FP), R0	// arg 1 fd
	MOVW 4(FP), R1	// arg 2 buf
	MOVW 8(FP), R2	// arg 3 count
	MOVW $SYS_write, R7
	SWI $0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·close(SB),NOSPLIT,$-8
	MOVW 0(FP), R0	// arg 1 fd
	MOVW $SYS_close, R7
	SWI $0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$-8
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW $SYS_getrlimit, R7
	SWI $0
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$8
	// thr_self(&4(R13))
	MOVW $4(R13), R0 // arg 1 &4(R13)
	MOVW $SYS_thr_self, R7
	SWI $0
	// thr_kill(self, SIGPIPE)
	MOVW 4(R13), R0	// arg 1 id
	MOVW sig+0(FP), R1	// arg 2 - signal
	MOVW $SYS_thr_kill, R7
	SWI $0
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $-8
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW 8(FP), R2
	MOVW $SYS_setitimer, R7
	SWI $0
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVW $0, R0 // CLOCK_REALTIME
	MOVW $8(R13), R1
	MOVW $SYS_clock_gettime, R7
	SWI $0

	MOVW 8(R13), R0 // sec.low
	MOVW 12(R13), R1 // sec.high
	MOVW 16(R13), R2 // nsec

	MOVW R0, 0(FP)
	MOVW R1, 4(FP)
	MOVW R2, 8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB), NOSPLIT, $32
	// We can use CLOCK_MONOTONIC_FAST here when we drop
	// support for FreeBSD 8-STABLE.
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

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVW 0(FP), R0		// arg 1 sig
	MOVW 4(FP), R1		// arg 2 act
	MOVW 8(FP), R2		// arg 3 oact
	MOVW $SYS_sigaction, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$24
	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVW	R0, 4(R13) // signum
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	BL.NE	runtime·load_g(SB)

	CMP $0, g
	BNE 4(PC)
	// signal number is already prepared in 4(R13)
	MOVW $runtime·badsignal(SB), R11
	BL (R11)
	RET

	// save g
	MOVW g, R4
	MOVW g, 20(R13)

	// g = m->signal
	MOVW g_m(g), R8
	MOVW m_gsignal(R8), g

	// R0 is already saved
	MOVW R1, 8(R13) // info
	MOVW R2, 12(R13) // context
	MOVW R4, 16(R13) // oldg

	BL runtime·sighandler(SB)

	// restore g
	MOVW 20(R13), g
	RET

TEXT runtime·mmap(SB),NOSPLIT,$16
	MOVW 0(FP), R0		// arg 1 addr
	MOVW 4(FP), R1		// arg 2 len
	MOVW 8(FP), R2		// arg 3 prot
	MOVW 12(FP), R3		// arg 4 flags
	// arg 5 (fid) and arg6 (offset_lo, offset_hi) are passed on stack
	// note the C runtime only passes the 32-bit offset_lo to us
	MOVW 16(FP), R4		// arg 5
	MOVW R4, 4(R13)
	MOVW 20(FP), R5		// arg 6 lower 32-bit
	// the word at 8(R13) is skipped due to 64-bit argument alignment.
	MOVW R5, 12(R13)
	MOVW $0, R6 		// higher 32-bit for arg 6
	MOVW R6, 16(R13)
	ADD $4, R13
	MOVW $SYS_mmap, R7
	SWI $0
	SUB $4, R13
	// TODO(dfc) error checking ?
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW 0(FP), R0		// arg 1 addr
	MOVW 4(FP), R1		// arg 2 len
	MOVW $SYS_munmap, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW 0(FP), R0		// arg 1 addr
	MOVW 4(FP), R1		// arg 2 len
	MOVW 8(FP), R2		// arg 3 flags
	MOVW $SYS_madvise, R7
	SWI $0
	// ignore failure - maybe pages are locked
	RET
	
TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVW new+0(FP), R0
	MOVW old+4(FP), R1
	MOVW $SYS_sigaltstack, R7
	SWI $0
	MOVW.CS $0, R8 // crash on syscall failure
	MOVW.CS R8, (R8)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16
	MOVW usec+0(FP), R0
	MOVW R0, R2
	MOVW $1000000, R1
	DIV R1, R0
	// 0(R13) is the saved LR, don't use it
	MOVW R0, 4(R13) // tv_sec.low
	MOVW $0, R0
	MOVW R0, 8(R13) // tv_sec.high
	MOD R1, R2
	MOVW $1000, R1
	MUL R1, R2
	MOVW R2, 12(R13) // tv_nsec

	MOVW $4(R13), R0 // arg 1 - rqtp
	MOVW $0, R1      // arg 2 - rmtp
	MOVW $SYS_nanosleep, R7
	SWI $0
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVW 0(FP), R0	// arg 1 - name
	MOVW 4(FP), R1	// arg 2 - namelen
	MOVW 8(FP), R2	// arg 3 - old
	MOVW 12(FP), R3	// arg 4 - oldlenp
	// arg 5 (newp) and arg 6 (newlen) are passed on stack
	ADD $20, R13
	MOVW $SYS___sysctl, R7
	SWI $0
	SUB.CS $0, R0, R0
	SUB $20, R13
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVW $SYS_sched_yield, R7
	SWI $0
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVW $3, R0	// arg 1 - how (SIG_SETMASK)
	MOVW 0(FP), R1	// arg 2 - set
	MOVW 4(FP), R2	// arg 3 - oset
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
	MOVW 0(FP), R0	// kq
	MOVW 4(FP), R1	// changelist
	MOVW 8(FP), R2	// nchanges
	MOVW 12(FP), R3	// eventlist
	ADD $20, R13	// pass arg 5 and 6 on stack
	MOVW $SYS_kevent, R7
	SWI $0
	RSB.CS $0, R0
	SUB $20, R13
	MOVW	R0, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW 0(FP), R0	// fd
	MOVW $2, R1	// F_SETFD
	MOVW $1, R2	// FD_CLOEXEC
	MOVW $SYS_fcntl, R7
	SWI $0
	RET

TEXT runtime·casp(SB),NOSPLIT,$0
	B	runtime·cas(SB)

// TODO(minux): this is only valid for ARMv6+
// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime·cas(SB),NOSPLIT,$0
	B runtime·armcas(SB)

// TODO(minux): this only supports ARMv6K+.
TEXT runtime·read_tls_fallback(SB),NOSPLIT,$-4
	WORD $0xee1d0f70 // mrc p15, 0, r0, c13, c0, 3
	RET
