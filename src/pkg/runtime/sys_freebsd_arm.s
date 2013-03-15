// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for ARM, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "zasm_GOOS_GOARCH.h"
	
TEXT runtime·sys_umtx_op(SB),7,$0
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW 8(FP), R2
	MOVW 12(FP), R3
	ADD $20, R13 // arg 5 is passed on stack
	SWI $454
	SUB $20, R13
	// BCS error
	RET

TEXT runtime·thr_new(SB),7,$0
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	SWI $455
	RET

TEXT runtime·thr_start(SB),7,$0
	MOVW R0, R9 // m

	// TODO(minux): set up TLS?

	// set up g
	MOVW m_g0(R9), R10
	BL runtime·emptyfunc(SB) // fault if stack check is wrong
	BL runtime·mstart(SB)

	MOVW $2, R9  // crash (not reached)
	MOVW R9, (R9)
	RET

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),7,$-8
	MOVW 0(FP), R0	// arg 1 exit status
	SWI $1
	MOVW.CS $0, R9 // crash on syscall failure
	MOVW.CS R9, (R9)
	RET

TEXT runtime·exit1(SB),7,$-8
	MOVW 0(FP), R0	// arg 1 exit status
	SWI $431
	MOVW.CS $0, R9 // crash on syscall failure
	MOVW.CS R9, (R9)
	RET

TEXT runtime·open(SB),7,$-8
	MOVW 0(FP), R0	// arg 1 name
	MOVW 4(FP), R1	// arg 2 mode
	MOVW 8(FP), R2	// arg 3 perm
	SWI $5
	RET

TEXT runtime·read(SB),7,$-8
	MOVW 0(FP), R0	// arg 1 fd
	MOVW 4(FP), R1	// arg 2 buf
	MOVW 8(FP), R2	// arg 3 count
	SWI $3
	RET

TEXT runtime·write(SB),7,$-8
	MOVW 0(FP), R0	// arg 1 fd
	MOVW 4(FP), R1	// arg 2 buf
	MOVW 8(FP), R2	// arg 3 count
	SWI $4
	RET

TEXT runtime·close(SB),7,$-8
	MOVW 0(FP), R0	// arg 1 fd
	SWI $6
	RET

TEXT runtime·getrlimit(SB),7,$-8
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW 8(FP), R2
	SWI $194
	RET

TEXT runtime·raise(SB),7,$8
	// thr_self(&4(R13))
	MOVW $4(R13), R0 // arg 1 &4(R13)
	SWI $432
	// thr_kill(self, SIGPIPE)
	MOVW 4(R13), R0	// arg 1 id
	MOVW sig+0(FP), R1	// arg 2 - signal
	SWI $433
	RET

TEXT runtime·setitimer(SB), 7, $-8
	MOVW 0(FP), R0
	MOVW 4(FP), R1
	MOVW 8(FP), R2
	SWI $83
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), 7, $32
	MOVW $0, R0 // CLOCK_REALTIME
	MOVW $8(R13), R1
	SWI $232 // clock_gettime

	MOVW 8(R13), R0 // sec.low
	MOVW 12(R13), R1 // sec.high
	MOVW 16(R13), R2 // nsec

	MOVW R0, 0(FP)
	MOVW R1, 4(FP)
	MOVW R2, 8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB), 7, $32
	MOVW $0, R0 // CLOCK_REALTIME
	MOVW $8(R13), R1
	SWI $232 // clock_gettime

	MOVW 8(R13), R0 // sec.low
	MOVW 12(R13), R4 // sec.high
	MOVW 16(R13), R2 // nsec

	MOVW $1000000000, R3
	MULLU R0, R3, (R1, R0)
	MUL R3, R4
	ADD.S R2, R0
	ADC R4, R1

	MOVW 0(FP), R3
	MOVW R0, 0(R3)
	MOVW R1, 4(R3)
	RET

TEXT runtime·sigaction(SB),7,$-8
	MOVW 0(FP), R0		// arg 1 sig
	MOVW 4(FP), R1		// arg 2 act
	MOVW 8(FP), R2		// arg 3 oact
	SWI $416
	MOVW.CS $0, R9 // crash on syscall failure
	MOVW.CS R9, (R9)
	RET

TEXT runtime·sigtramp(SB),7,$24
	// this might be called in external code context,
	// where g and m are not set.
	// first save R0, because _cgo_load_gm will clobber it
	// TODO(adonovan): call runtime·badsignal if m=0, like other platforms?
	MOVW	R0, 4(R13) // signum
	MOVW	_cgo_load_gm(SB), R0
	CMP 	$0, R0
	BL.NE	(R0)

	// save g
	MOVW R10, R4
	MOVW R10, 20(R13)

	// g = m->signal
	MOVW m_gsignal(R9), R10

	// R0 is already saved
	MOVW R1, 8(R13) // info
	MOVW R2, 12(R13) // context
	MOVW R4, 16(R13) // oldg

	BL runtime·sighandler(SB)

	// restore g
	MOVW 20(R13), R10
	RET

TEXT runtime·mmap(SB),7,$12
	MOVW 0(FP), R0		// arg 1 addr
	MOVW 4(FP), R1		// arg 2 len
	MOVW 8(FP), R2		// arg 3 prot
	MOVW 12(FP), R3		// arg 4 flags
	// arg 5 (fid) and arg6 (offset_lo, offset_hi) are passed on stack
	// note the C runtime only passes the 32-bit offset_lo to us
	MOVW 16(FP), R4		// arg 5
	MOVW R4, 4(R13)
	MOVW 20(FP), R5		// arg 6 lower 32-bit
	MOVW R5, 8(R13)
	MOVW $0, R6 // higher 32-bit for arg 6
	MOVW R6, 12(R13)
	ADD $4, R13 // pass arg 5 and arg 6 on stack
	SWI $477
	SUB $4, R13
	RET

TEXT runtime·munmap(SB),7,$0
	MOVW 0(FP), R0		// arg 1 addr
	MOVW 4(FP), R1		// arg 2 len
	SWI $73
	MOVW.CS $0, R9 // crash on syscall failure
	MOVW.CS R9, (R9)
	RET

TEXT runtime·madvise(SB),7,$0
	MOVW 0(FP), R0		// arg 1 addr
	MOVW 4(FP), R1		// arg 2 len
	MOVW 8(FP), R2		// arg 3 flags
	SWI $75
	// ignore failure - maybe pages are locked
	RET
	
TEXT runtime·sigaltstack(SB),7,$-8
	MOVW new+0(FP), R0
	MOVW old+4(FP), R1
	SWI $53
	MOVW.CS $0, R9 // crash on syscall failure
	MOVW.CS R9, (R9)
	RET

TEXT runtime·usleep(SB),7,$16
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
	SWI $240 // sys_nanosleep
	RET

TEXT runtime·sysctl(SB),7,$0
	MOVW 0(FP), R0	// arg 1 - name
	MOVW 4(FP), R1	// arg 2 - namelen
	MOVW 8(FP), R2	// arg 3 - oldp
	MOVW 12(FP), R3	// arg 4 - oldlenp
	// arg 5 (newp) and arg 6 (newlen) are passed on stack
	ADD $20, R13
	SWI $202 // sys___sysctl
	SUB.CS $0, R0, R0
	SUB $20, R13
	RET

TEXT runtime·osyield(SB),7,$-4
	SWI $331	// sys_sched_yield
	RET

TEXT runtime·sigprocmask(SB),7,$0
	MOVW $3, R0	// arg 1 - how (SIG_SETMASK)
	MOVW 0(FP), R1	// arg 2 - set
	MOVW 4(FP), R2	// arg 3 - oset
	SWI $340	// sys_sigprocmask
	MOVW.CS $0, R9 // crash on syscall failure
	MOVW.CS R9, (R9)
	RET

TEXT runtime·casp(SB),7,$0
	B	runtime·cas(SB)

// TODO(minux): this is only valid for ARMv6+
// bool armcas(int32 *val, int32 old, int32 new)
// Atomically:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime·cas(SB),7,$0
	B runtime·armcas(SB)
