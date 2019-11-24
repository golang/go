// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for ARM, OpenBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME	$0
#define	CLOCK_MONOTONIC	$3

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0
	MOVW	code+0(FP), R0	// arg 1 - status
	MOVW	$1, R12			// sys_exit
	SWI	$0
	MOVW.CS	$0, R8			// crash on syscall failure
	MOVW.CS	R8, (R8)
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVW	wait+0(FP), R0		// arg 1 - notdead
	MOVW	$302, R12		// sys___threxit
	SWI	$0
	MOVW.CS	$1, R8			// crash on syscall failure
	MOVW.CS	R8, (R8)
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0
	MOVW	name+0(FP), R0		// arg 1 - path
	MOVW	mode+4(FP), R1		// arg 2 - mode
	MOVW	perm+8(FP), R2		// arg 3 - perm
	MOVW	$5, R12			// sys_open
	SWI	$0
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVW	$6, R12			// sys_close
	SWI	$0
	MOVW.CS	$-1, R0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVW	p+4(FP), R1		// arg 2 - buf
	MOVW	n+8(FP), R2		// arg 3 - nbyte
	MOVW	$3, R12			// sys_read
	SWI	$0
	RSB.CS	$0, R0		// caller expects negative errno
	MOVW	R0, ret+12(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT,$0-12
	MOVW	$r+0(FP), R0
	MOVW	$263, R12
	SWI	$0
	MOVW	R0, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$0-16
	MOVW	$r+4(FP), R0
	MOVW	flags+0(FP), R1
	MOVW	$101, R12
	SWI	$0
	MOVW	R0, errno+12(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVW	p+4(FP), R1		// arg 2 - buf
	MOVW	n+8(FP), R2		// arg 3 - nbyte
	MOVW	$4, R12			// sys_write
	SWI	$0
	RSB.CS	$0, R0		// caller expects negative errno
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16
	MOVW	usec+0(FP), R0
	CALL	runtime·usplitR0(SB)
	MOVW	R0, 4(R13)		// tv_sec - l32
	MOVW	$0, R0
	MOVW	R0, 8(R13)		// tv_sec - h32
	MOVW	$1000, R2
	MUL	R1, R2
	MOVW	R2, 12(R13)		// tv_nsec

	MOVW	$4(R13), R0		// arg 1 - rqtp
	MOVW	$0, R1			// arg 2 - rmtp
	MOVW	$91, R12		// sys_nanosleep
	SWI	$0
	RET

TEXT runtime·getthrid(SB),NOSPLIT,$0-4
	MOVW	$299, R12		// sys_getthrid
	SWI	$0
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·thrkill(SB),NOSPLIT,$0-8
	MOVW	tid+0(FP), R0		// arg 1 - tid
	MOVW	sig+4(FP), R1		// arg 2 - signum
	MOVW	$0, R2			// arg 3 - tcb
	MOVW	$119, R12		// sys_thrkill
	SWI	$0
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVW	$20, R12
	SWI	$0			// sys_getpid
					// arg 1 - pid, already in R0
	MOVW	sig+0(FP), R1		// arg 2 - signum
	MOVW	$122, R12		// sys_kill
	SWI	$0
	RET

TEXT runtime·mmap(SB),NOSPLIT,$16
	MOVW	addr+0(FP), R0		// arg 1 - addr
	MOVW	n+4(FP), R1		// arg 2 - len
	MOVW	prot+8(FP), R2		// arg 3 - prot
	MOVW	flags+12(FP), R3	// arg 4 - flags
	MOVW	fd+16(FP), R4		// arg 5 - fd (on stack)
	MOVW	R4, 4(R13)
	MOVW	$0, R5			// arg 6 - pad (on stack)
	MOVW	R5, 8(R13)
	MOVW	off+20(FP), R6		// arg 7 - offset (on stack)
	MOVW	R6, 12(R13)		// lower 32 bits (from Go runtime)
	MOVW	$0, R7
	MOVW	R7, 16(R13)		// high 32 bits
	ADD	$4, R13
	MOVW	$197, R12		// sys_mmap
	SWI	$0
	SUB	$4, R13
	MOVW	$0, R1
	MOVW.CS	R0, R1			// if error, move to R1
	MOVW.CS $0, R0
	MOVW	R0, p+24(FP)
	MOVW	R1, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0		// arg 1 - addr
	MOVW	n+4(FP), R1		// arg 2 - len
	MOVW	$73, R12		// sys_munmap
	SWI	$0
	MOVW.CS	$0, R8			// crash on syscall failure
	MOVW.CS	R8, (R8)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0		// arg 1 - addr
	MOVW	n+4(FP), R1		// arg 2 - len
	MOVW	flags+8(FP), R2		// arg 2 - flags
	MOVW	$75, R12		// sys_madvise
	SWI	$0
	MOVW.CS	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0		// arg 1 - mode
	MOVW	new+4(FP), R1		// arg 2 - new value
	MOVW	old+8(FP), R2		// arg 3 - old value
	MOVW	$69, R12		// sys_setitimer
	SWI	$0
	RET

// func walltime1() (sec int64, nsec int32)
TEXT runtime·walltime1(SB), NOSPLIT, $32
	MOVW	CLOCK_REALTIME, R0	// arg 1 - clock_id
	MOVW	$8(R13), R1		// arg 2 - tp
	MOVW	$87, R12		// sys_clock_gettime
	SWI	$0

	MOVW	8(R13), R0		// sec - l32
	MOVW	12(R13), R1		// sec - h32
	MOVW	16(R13), R2		// nsec

	MOVW	R0, sec_lo+0(FP)
	MOVW	R1, sec_hi+4(FP)
	MOVW	R2, nsec+8(FP)

	RET

// int64 nanotime1(void) so really
// void nanotime1(int64 *nsec)
TEXT runtime·nanotime1(SB),NOSPLIT,$32
	MOVW	CLOCK_MONOTONIC, R0	// arg 1 - clock_id
	MOVW	$8(R13), R1		// arg 2 - tp
	MOVW	$87, R12		// sys_clock_gettime
	SWI	$0

	MOVW	8(R13), R0		// sec - l32
	MOVW	12(R13), R4		// sec - h32
	MOVW	16(R13), R2		// nsec

	MOVW	$1000000000, R3
	MULLU	R0, R3, (R1, R0)
	MUL	R3, R4
	ADD.S	R2, R0
	ADC	R4, R1

	MOVW	R0, ret_lo+0(FP)
	MOVW	R1, ret_hi+4(FP)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0
	MOVW	sig+0(FP), R0		// arg 1 - signum
	MOVW	new+4(FP), R1		// arg 2 - new sigaction
	MOVW	old+8(FP), R2		// arg 3 - old sigaction
	MOVW	$46, R12		// sys_sigaction
	SWI	$0
	MOVW.CS	$3, R8			// crash on syscall failure
	MOVW.CS	R8, (R8)
	RET

TEXT runtime·obsdsigprocmask(SB),NOSPLIT,$0
	MOVW	how+0(FP), R0		// arg 1 - mode
	MOVW	new+4(FP), R1		// arg 2 - new
	MOVW	$48, R12		// sys_sigprocmask
	SWI	$0
	MOVW.CS	$3, R8			// crash on syscall failure
	MOVW.CS	R8, (R8)
	MOVW	R0, ret+8(FP)
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
	MOVM.DB.W [R4-R11], (R13)
	SUB	$16, R13

	// If called from an external code context, g will not be set.
	// Save R0, since runtime·load_g will clobber it.
	MOVW	R0, 4(R13)		// signum
	MOVB	runtime·iscgo(SB), R0
	CMP	$0, R0
	BL.NE	runtime·load_g(SB)

	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·sigtrampgo(SB)

	// Restore callee-save registers.
	ADD	$16, R13
	MOVM.IA.W (R13), [R4-R11]

	RET

// int32 tfork(void *param, uintptr psize, M *mp, G *gp, void (*fn)(void));
TEXT runtime·tfork(SB),NOSPLIT,$0

	// Copy mp, gp and fn off parent stack for use by child.
	MOVW	mm+8(FP), R4
	MOVW	gg+12(FP), R5
	MOVW	fn+16(FP), R6

	MOVW	param+0(FP), R0		// arg 1 - param
	MOVW	psize+4(FP), R1		// arg 2 - psize
	MOVW	$8, R12			// sys___tfork
	SWI	$0

	// Return if syscall failed.
	B.CC	4(PC)
	RSB	$0, R0
	MOVW	R0, ret+20(FP)
	RET

	// In parent, return.
	CMP	$0, R0
	BEQ	3(PC)
	MOVW	R0, ret+20(FP)
	RET

	// Initialise m, g.
	MOVW	R5, g
	MOVW	R4, g_m(g)

	// Paranoia; check that stack splitting code works.
	BL	runtime·emptyfunc(SB)

	// Call fn.
	BL	(R6)

	// fn should never return.
	MOVW	$2, R8			// crash if reached
	MOVW	R8, (R8)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVW	new+0(FP), R0		// arg 1 - new sigaltstack
	MOVW	old+4(FP), R1		// arg 2 - old sigaltstack
	MOVW	$288, R12		// sys_sigaltstack
	SWI	$0
	MOVW.CS	$0, R8			// crash on syscall failure
	MOVW.CS	R8, (R8)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVW	$298, R12		// sys_sched_yield
	SWI	$0
	RET

TEXT runtime·thrsleep(SB),NOSPLIT,$4
	MOVW	ident+0(FP), R0		// arg 1 - ident
	MOVW	clock_id+4(FP), R1	// arg 2 - clock_id
	MOVW	tsp+8(FP), R2		// arg 3 - tsp
	MOVW	lock+12(FP), R3		// arg 4 - lock
	MOVW	abort+16(FP), R4	// arg 5 - abort (on stack)
	MOVW	R4, 4(R13)
	ADD	$4, R13
	MOVW	$94, R12		// sys___thrsleep
	SWI	$0
	SUB	$4, R13
	MOVW	R0, ret+20(FP)
	RET

TEXT runtime·thrwakeup(SB),NOSPLIT,$0
	MOVW	ident+0(FP), R0		// arg 1 - ident
	MOVW	n+4(FP), R1		// arg 2 - n
	MOVW	$301, R12		// sys___thrwakeup
	SWI	$0
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$8
	MOVW	mib+0(FP), R0		// arg 1 - mib
	MOVW	miblen+4(FP), R1	// arg 2 - miblen
	MOVW	out+8(FP), R2		// arg 3 - out
	MOVW	size+12(FP), R3		// arg 4 - size
	MOVW	dst+16(FP), R4		// arg 5 - dest (on stack)
	MOVW	R4, 4(R13)
	MOVW	ndst+20(FP), R5		// arg 6 - newlen (on stack)
	MOVW	R5, 8(R13)
	ADD	$4, R13
	MOVW	$202, R12		// sys___sysctl
	SWI	$0
	SUB	$4, R13
	MOVW.CC	$0, R0
	RSB.CS	$0, R0
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVW	$269, R12		// sys_kqueue
	SWI	$0
	RSB.CS	$0, R0
	MOVW	R0, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$8
	MOVW	kq+0(FP), R0		// arg 1 - kq
	MOVW	ch+4(FP), R1		// arg 2 - changelist
	MOVW	nch+8(FP), R2		// arg 3 - nchanges
	MOVW	ev+12(FP), R3		// arg 4 - eventlist
	MOVW	nev+16(FP), R4		// arg 5 - nevents (on stack)
	MOVW	R4, 4(R13)
	MOVW	ts+20(FP), R5		// arg 6 - timeout (on stack)
	MOVW	R5, 8(R13)
	ADD	$4, R13
	MOVW	$72, R12		// sys_kevent
	SWI	$0
	RSB.CS	$0, R0
	SUB	$4, R13
	MOVW	R0, ret+24(FP)
	RET

// func closeonexec(fd int32)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0		// arg 1 - fd
	MOVW	$2, R1			// arg 2 - cmd (F_SETFD)
	MOVW	$1, R2			// arg 3 - arg (FD_CLOEXEC)
	MOVW	$92, R12		// sys_fcntl
	SWI	$0
	RET

// func runtime·setNonblock(fd int32)
TEXT runtime·setNonblock(SB),NOSPLIT,$0-4
	MOVW	fd+0(FP), R0	// fd
	MOVW	$3, R1	// F_GETFL
	MOVW	$0, R2
	MOVW	$92, R12
	SWI	$0
	ORR	$0x4, R0, R2	// O_NONBLOCK
	MOVW	fd+0(FP), R0	// fd
	MOVW	$4, R1	// F_SETFL
	MOVW	$92, R12
	SWI	$0
	RET

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	MOVM.WP	[R1, R2, R3, R12], (R13)
	MOVW	$330, R12		// sys___get_tcb
	SWI	$0
	MOVM.IAW (R13), [R1, R2, R3, R12]
	RET
