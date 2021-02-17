// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm64, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME		0
#define CLOCK_MONOTONIC		4
#define FD_CLOEXEC		1
#define F_SETFD			2
#define F_GETFL			3
#define F_SETFL			4
#define O_NONBLOCK		4

#define SYS_exit		1
#define SYS_read		3
#define SYS_write		4
#define SYS_open		5
#define SYS_close		6
#define SYS_getpid		20
#define SYS_kill		37
#define SYS_sigaltstack		53
#define SYS_munmap		73
#define SYS_madvise		75
#define SYS_setitimer		83
#define SYS_fcntl		92
#define SYS___sysctl		202
#define SYS_nanosleep		240
#define SYS_clock_gettime	232
#define SYS_sched_yield		331
#define SYS_sigprocmask		340
#define SYS_kqueue		362
#define SYS_kevent		363
#define SYS_sigaction		416
#define SYS_thr_exit		431
#define SYS_thr_self		432
#define SYS_thr_kill		433
#define SYS__umtx_op		454
#define SYS_thr_new		455
#define SYS_mmap		477
#define SYS_cpuset_getaffinity	487
#define SYS_pipe2 		542

TEXT emptyfunc<>(SB),0,$0-0
	RET

// func sys_umtx_op(addr *uint32, mode int32, val uint32, uaddr1 uintptr, ut *umtx_time) int32
TEXT runtime·sys_umtx_op(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0
	MOVW	mode+8(FP), R1
	MOVW	val+12(FP), R2
	MOVD	uaddr1+16(FP), R3
	MOVD	ut+24(FP), R4
	MOVD	$SYS__umtx_op, R8
	SVC
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+32(FP)
	RET

// func thr_new(param *thrparam, size int32) int32
TEXT runtime·thr_new(SB),NOSPLIT,$0
	MOVD	param+0(FP), R0
	MOVW	size+8(FP), R1
	MOVD	$SYS_thr_new, R8
	SVC
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+16(FP)
	RET

// func thr_start()
TEXT runtime·thr_start(SB),NOSPLIT,$0
	// set up g
	MOVD	m_g0(R0), g
	MOVD	R0, g_m(g)
	BL	emptyfunc<>(SB)	 // fault if stack check is wrong
	BL	runtime·mstart(SB)

	MOVD	$2, R8	// crash (not reached)
	MOVD	R8, (R8)
	RET

// func exit(code int32)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R0
	MOVD	$SYS_exit, R8
	SVC
	MOVD	$0, R0
	MOVD	R0, (R0)

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-8
	MOVD	wait+0(FP), R0
	// We're done using the stack.
	MOVW	$0, R1
	STLRW	R1, (R0)
	MOVW	$0, R0
	MOVD	$SYS_thr_exit, R8
	SVC
	JMP	0(PC)

// func open(name *byte, mode, perm int32) int32
TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	name+0(FP), R0
	MOVW	mode+8(FP), R1
	MOVW	perm+12(FP), R2
	MOVD	$SYS_open, R8
	SVC
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+16(FP)
	RET

// func closefd(fd int32) int32
TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R0
	MOVD	$SYS_close, R8
	SVC
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+8(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT|NOFRAME,$0-12
	MOVD	$r+0(FP), R0
	MOVW	$0, R1
	MOVD	$SYS_pipe2, R8
	SVC
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	$r+8(FP), R0
	MOVW	flags+0(FP), R1
	MOVD	$SYS_pipe2, R8
	SVC
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, errno+16(FP)
	RET

// func write1(fd uintptr, p unsafe.Pointer, n int32) int32
TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	fd+0(FP), R0
	MOVD	p+8(FP), R1
	MOVW	n+16(FP), R2
	MOVD	$SYS_write, R8
	SVC
	BCC	ok
	NEG	R0, R0		// caller expects negative errno
ok:
	MOVW	R0, ret+24(FP)
	RET

// func read(fd int32, p unsafe.Pointer, n int32) int32
TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R0
	MOVD	p+8(FP), R1
	MOVW	n+16(FP), R2
	MOVD	$SYS_read, R8
	SVC
	BCC	ok
	NEG	R0, R0		// caller expects negative errno
ok:
	MOVW	R0, ret+24(FP)
	RET

// func usleep(usec uint32)
TEXT runtime·usleep(SB),NOSPLIT,$24-4
	MOVWU	usec+0(FP), R3
	MOVD	R3, R5
	MOVW	$1000000, R4
	UDIV	R4, R3
	MOVD	R3, 8(RSP)
	MUL	R3, R4
	SUB	R4, R5
	MOVW	$1000, R4
	MUL	R4, R5
	MOVD	R5, 16(RSP)

	// nanosleep(&ts, 0)
	ADD	$8, RSP, R0
	MOVD	$0, R1
	MOVD	$SYS_nanosleep, R8
	SVC
	RET

// func thr_self() thread
TEXT runtime·thr_self(SB),NOSPLIT,$8-8
	MOVD	$ptr-8(SP), R0	// arg 1 &8(SP)
	MOVD	$SYS_thr_self, R8
	SVC
	MOVD	ptr-8(SP), R0
	MOVD	R0, ret+0(FP)
	RET

// func thr_kill(t thread, sig int)
TEXT runtime·thr_kill(SB),NOSPLIT,$0-16
	MOVD	tid+0(FP), R0	// arg 1 pid
	MOVD	sig+8(FP), R1	// arg 2 sig
	MOVD	$SYS_thr_kill, R8
	SVC
	RET

// func raiseproc(sig uint32)
TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVD	$SYS_getpid, R8
	SVC
	MOVW	sig+0(FP), R1
	MOVD	$SYS_kill, R8
	SVC
	RET

// func setitimer(mode int32, new, old *itimerval)
TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	$SYS_setitimer, R8
	SVC
	RET

// func fallback_walltime() (sec int64, nsec int32)
TEXT runtime·fallback_walltime(SB),NOSPLIT,$24-12
	MOVW	$CLOCK_REALTIME, R0
	MOVD	$8(RSP), R1
	MOVD	$SYS_clock_gettime, R8
	SVC
	MOVD	8(RSP), R0	// sec
	MOVW	16(RSP), R1	// nsec
	MOVD	R0, sec+0(FP)
	MOVW	R1, nsec+8(FP)
	RET

// func fallback_nanotime() int64
TEXT runtime·fallback_nanotime(SB),NOSPLIT,$24-8
	MOVD	$CLOCK_MONOTONIC, R0
	MOVD	$8(RSP), R1
	MOVD	$SYS_clock_gettime, R8
	SVC
	MOVD	8(RSP), R0	// sec
	MOVW	16(RSP), R2	// nsec

	// sec is in R0, nsec in R2
	// return nsec in R2
	MOVD	$1000000000, R3
	MUL	R3, R0
	ADD	R2, R0

	MOVD	R0, ret+0(FP)
	RET

// func asmSigaction(sig uintptr, new, old *sigactiont) int32
TEXT runtime·asmSigaction(SB),NOSPLIT|NOFRAME,$0
	MOVD	sig+0(FP), R0		// arg 1 sig
	MOVD	new+8(FP), R1		// arg 2 act
	MOVD	old+16(FP), R2		// arg 3 oact
	MOVD	$SYS_sigaction, R8
	SVC
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+24(FP)
	RET

// func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)
TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)
	RET

// func sigtramp()
TEXT runtime·sigtramp(SB),NOSPLIT,$192
	// Save callee-save registers in the case of signal forwarding.
	// Please refer to https://golang.org/issue/31827 .
	MOVD	R19, 8*4(RSP)
	MOVD	R20, 8*5(RSP)
	MOVD	R21, 8*6(RSP)
	MOVD	R22, 8*7(RSP)
	MOVD	R23, 8*8(RSP)
	MOVD	R24, 8*9(RSP)
	MOVD	R25, 8*10(RSP)
	MOVD	R26, 8*11(RSP)
	MOVD	R27, 8*12(RSP)
	MOVD	g, 8*13(RSP)
	MOVD	R29, 8*14(RSP)
	FMOVD	F8, 8*15(RSP)
	FMOVD	F9, 8*16(RSP)
	FMOVD	F10, 8*17(RSP)
	FMOVD	F11, 8*18(RSP)
	FMOVD	F12, 8*19(RSP)
	FMOVD	F13, 8*20(RSP)
	FMOVD	F14, 8*21(RSP)
	FMOVD	F15, 8*22(RSP)

	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVW	R0, 8(RSP)
	MOVBU	runtime·iscgo(SB), R0
	CMP	$0, R0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	MOVD	R1, 16(RSP)
	MOVD	R2, 24(RSP)
	MOVD	$runtime·sigtrampgo(SB), R0
	BL	(R0)

	// Restore callee-save registers.
	MOVD	8*4(RSP), R19
	MOVD	8*5(RSP), R20
	MOVD	8*6(RSP), R21
	MOVD	8*7(RSP), R22
	MOVD	8*8(RSP), R23
	MOVD	8*9(RSP), R24
	MOVD	8*10(RSP), R25
	MOVD	8*11(RSP), R26
	MOVD	8*12(RSP), R27
	MOVD	8*13(RSP), g
	MOVD	8*14(RSP), R29
	FMOVD	8*15(RSP), F8
	FMOVD	8*16(RSP), F9
	FMOVD	8*17(RSP), F10
	FMOVD	8*18(RSP), F11
	FMOVD	8*19(RSP), F12
	FMOVD	8*20(RSP), F13
	FMOVD	8*21(RSP), F14
	FMOVD	8*22(RSP), F15

	RET

// func mmap(addr uintptr, n uintptr, prot int, flags int, fd int, off int64) (ret uintptr, err error)
TEXT runtime·mmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	prot+16(FP), R2
	MOVW	flags+20(FP), R3
	MOVW	fd+24(FP), R4
	MOVW	off+28(FP), R5
	MOVD	$SYS_mmap, R8
	SVC
	BCS	fail
	MOVD	R0, p+32(FP)
	MOVD	$0, err+40(FP)
	RET
fail:
	MOVD	$0, p+32(FP)
	MOVD	R0, err+40(FP)
	RET

// func munmap(addr uintptr, n uintptr) (err error)
TEXT runtime·munmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	$SYS_munmap, R8
	SVC
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash

// func madvise(addr unsafe.Pointer, n uintptr, flags int32) int32
TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	flags+16(FP), R2
	MOVD	$SYS_madvise, R8
	SVC
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+24(FP)
	RET

// func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVD	mib+0(FP), R0
	MOVD	miblen+8(FP), R1
	MOVD	out+16(FP), R2
	MOVD	size+24(FP), R3
	MOVD	dst+32(FP), R4
	MOVD	ndst+40(FP), R5
	MOVD	$SYS___sysctl, R8
	SVC
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+48(FP)
	RET

// func sigaltstack(new, old *stackt)
TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVD	new+0(FP), R0
	MOVD	old+8(FP), R1
	MOVD	$SYS_sigaltstack, R8
	SVC
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash

// func osyield()
TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVD	$SYS_sched_yield, R8
	SVC
	RET

// func sigprocmask(how int32, new, old *sigset)
TEXT runtime·sigprocmask(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	how+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	$SYS_sigprocmask, R8
	SVC
	BCS	fail
	RET
fail:
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash

// func cpuset_getaffinity(level int, which int, id int64, size int, mask *byte) int32
TEXT runtime·cpuset_getaffinity(SB),NOSPLIT|NOFRAME,$0-44
	MOVD	level+0(FP), R0
	MOVD	which+8(FP), R1
	MOVD	id+16(FP), R2
	MOVD	size+24(FP), R3
	MOVD	mask+32(FP), R4
	MOVD	$SYS_cpuset_getaffinity, R8
	SVC
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+40(FP)
	RET

// func kqueue() int32
TEXT runtime·kqueue(SB),NOSPLIT|NOFRAME,$0
	MOVD $SYS_kqueue, R8
	SVC
	BCC	ok
	MOVW	$-1, R0
ok:
	MOVW	R0, ret+0(FP)
	RET

// func kevent(kq int, ch unsafe.Pointer, nch int, ev unsafe.Pointer, nev int, ts *Timespec) (n int, err error)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW	kq+0(FP), R0
	MOVD	ch+8(FP), R1
	MOVW	nch+16(FP), R2
	MOVD	ev+24(FP), R3
	MOVW	nev+32(FP), R4
	MOVD	ts+40(FP), R5
	MOVD	$SYS_kevent, R8
	SVC
	BCC	ok
	NEG	R0, R0
ok:
	MOVW	R0, ret+48(FP)
	RET

// func closeonexec(fd int32)
TEXT runtime·closeonexec(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0
	MOVD	$F_SETFD, R1
	MOVD	$FD_CLOEXEC, R2
	MOVD	$SYS_fcntl, R8
	SVC
	RET

// func runtime·setNonblock(fd int32)
TEXT runtime·setNonblock(SB),NOSPLIT,$0-4
	MOVW	fd+0(FP), R0
	MOVD	$F_GETFL, R1
	MOVD	$0, R2
	MOVD	$SYS_fcntl, R8
	SVC
	ORR	$O_NONBLOCK, R0, R2
	MOVW	fd+0(FP), R0
	MOVW	$F_SETFL, R1
	MOVW	$SYS_fcntl, R7
	SVC
	RET

// func getCntxct(physical bool) uint32
TEXT runtime·getCntxct(SB),NOSPLIT,$0
	MOVB	physical+0(FP), R0
	CMP	$0, R0
	BEQ	3(PC)

	// get CNTPCT (Physical Count Register) into R0
	MRS	CNTPCT_EL0, R0 // SIGILL
	B	2(PC)

	// get CNTVCT (Virtual Count Register) into R0
	MRS	CNTVCT_EL0, R0

	MOVW	R0, ret+8(FP)
	RET
