// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define AT_FDCWD -100

#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1

#define SYS_exit		93
#define SYS_read		63
#define SYS_write		64
#define SYS_openat		56
#define SYS_close		57
#define SYS_pipe2		59
#define SYS_fcntl		25
#define SYS_nanosleep		101
#define SYS_mmap		222
#define SYS_munmap		215
#define SYS_setitimer		103
#define SYS_clone		220
#define SYS_sched_yield		124
#define SYS_rt_sigreturn	139
#define SYS_rt_sigaction	134
#define SYS_rt_sigprocmask	135
#define SYS_sigaltstack		132
#define SYS_madvise		233
#define SYS_mincore		232
#define SYS_getpid		172
#define SYS_gettid		178
#define SYS_kill		129
#define SYS_tgkill		131
#define SYS_futex		98
#define SYS_sched_getaffinity	123
#define SYS_exit_group		94
#define SYS_epoll_create1	20
#define SYS_epoll_ctl		21
#define SYS_epoll_pwait		22
#define SYS_clock_gettime	113
#define SYS_faccessat		48
#define SYS_socket		198
#define SYS_connect		203
#define SYS_brk			214

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R0
	MOVD	$SYS_exit_group, R8
	SVC
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-8
	MOVD	wait+0(FP), R0
	// We're done using the stack.
	MOVW	$0, R1
	STLRW	R1, (R0)
	MOVW	$0, R0	// exit code
	MOVD	$SYS_exit, R8
	SVC
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	$AT_FDCWD, R0
	MOVD	name+0(FP), R1
	MOVW	mode+8(FP), R2
	MOVW	perm+12(FP), R3
	MOVD	$SYS_openat, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVW	$-1, R0
done:
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R0
	MOVD	$SYS_close, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVW	$-1, R0
done:
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	fd+0(FP), R0
	MOVD	p+8(FP), R1
	MOVW	n+16(FP), R2
	MOVD	$SYS_write, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R0
	MOVD	p+8(FP), R1
	MOVW	n+16(FP), R2
	MOVD	$SYS_read, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT|NOFRAME,$0-12
	MOVD	$r+0(FP), R0
	MOVW	$0, R1
	MOVW	$SYS_pipe2, R8
	SVC
	MOVW	R0, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	$r+8(FP), R0
	MOVW	flags+0(FP), R1
	MOVW	$SYS_pipe2, R8
	SVC
	MOVW	R0, errno+16(FP)
	RET

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

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVD	$SYS_gettid, R8
	SVC
	MOVW	R0, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT|NOFRAME,$0
	MOVD	$SYS_getpid, R8
	SVC
	MOVW	R0, R19
	MOVD	$SYS_gettid, R8
	SVC
	MOVW	R0, R1	// arg 2 tid
	MOVW	R19, R0	// arg 1 pid
	MOVW	sig+0(FP), R2	// arg 3
	MOVD	$SYS_tgkill, R8
	SVC
	RET

TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVD	$SYS_getpid, R8
	SVC
	MOVW	R0, R0		// arg 1 pid
	MOVW	sig+0(FP), R1	// arg 2
	MOVD	$SYS_kill, R8
	SVC
	RET

TEXT ·getpid(SB),NOSPLIT|NOFRAME,$0-8
	MOVD	$SYS_getpid, R8
	SVC
	MOVD	R0, ret+0(FP)
	RET

TEXT ·tgkill(SB),NOSPLIT,$0-24
	MOVD	tgid+0(FP), R0
	MOVD	tid+8(FP), R1
	MOVD	sig+16(FP), R2
	MOVD	$SYS_tgkill, R8
	SVC
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	$SYS_setitimer, R8
	SVC
	RET

TEXT runtime·mincore(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	dst+16(FP), R2
	MOVD	$SYS_mincore, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// func walltime1() (sec int64, nsec int32)
TEXT runtime·walltime1(SB),NOSPLIT,$24-12
	MOVD	RSP, R20	// R20 is unchanged by C code
	MOVD	RSP, R1

	MOVD	g_m(g), R21	// R21 = m

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVD	m_vdsoPC(R21), R2
	MOVD	m_vdsoSP(R21), R3
	MOVD	R2, 8(RSP)
	MOVD	R3, 16(RSP)

	MOVD	LR, m_vdsoPC(R21)
	MOVD	R20, m_vdsoSP(R21)

	MOVD	m_curg(R21), R0
	CMP	g, R0
	BNE	noswitch

	MOVD	m_g0(R21), R3
	MOVD	(g_sched+gobuf_sp)(R3), R1	// Set RSP to g0 stack

noswitch:
	SUB	$16, R1
	BIC	$15, R1	// Align for C code
	MOVD	R1, RSP

	MOVW	$CLOCK_REALTIME, R0
	MOVD	runtime·vdsoClockgettimeSym(SB), R2
	CBZ	R2, fallback

	// Store g on gsignal's stack, so if we receive a signal
	// during VDSO code we can find the g.
	// If we don't have a signal stack, we won't receive signal,
	// so don't bother saving g.
	// When using cgo, we already saved g on TLS, also don't save
	// g here.
	// Also don't save g if we are already on the signal stack.
	// We won't get a nested signal.
	MOVBU	runtime·iscgo(SB), R22
	CBNZ	R22, nosaveg
	MOVD	m_gsignal(R21), R22          // g.m.gsignal
	CBZ	R22, nosaveg
	CMP	g, R22
	BEQ	nosaveg
	MOVD	(g_stack+stack_lo)(R22), R22 // g.m.gsignal.stack.lo
	MOVD	g, (R22)

	BL	(R2)

	MOVD	ZR, (R22)  // clear g slot, R22 is unchanged by C code

	B	finish

nosaveg:
	BL	(R2)
	B	finish

fallback:
	MOVD	$SYS_clock_gettime, R8
	SVC

finish:
	MOVD	0(RSP), R3	// sec
	MOVD	8(RSP), R5	// nsec

	MOVD	R20, RSP	// restore SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVD	16(RSP), R1
	MOVD	R1, m_vdsoSP(R21)
	MOVD	8(RSP), R1
	MOVD	R1, m_vdsoPC(R21)

	MOVD	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$24-8
	MOVD	RSP, R20	// R20 is unchanged by C code
	MOVD	RSP, R1

	MOVD	g_m(g), R21	// R21 = m

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVD	m_vdsoPC(R21), R2
	MOVD	m_vdsoSP(R21), R3
	MOVD	R2, 8(RSP)
	MOVD	R3, 16(RSP)

	MOVD	LR, m_vdsoPC(R21)
	MOVD	R20, m_vdsoSP(R21)

	MOVD	m_curg(R21), R0
	CMP	g, R0
	BNE	noswitch

	MOVD	m_g0(R21), R3
	MOVD	(g_sched+gobuf_sp)(R3), R1	// Set RSP to g0 stack

noswitch:
	SUB	$32, R1
	BIC	$15, R1
	MOVD	R1, RSP

	MOVW	$CLOCK_MONOTONIC, R0
	MOVD	runtime·vdsoClockgettimeSym(SB), R2
	CBZ	R2, fallback

	// Store g on gsignal's stack, so if we receive a signal
	// during VDSO code we can find the g.
	// If we don't have a signal stack, we won't receive signal,
	// so don't bother saving g.
	// When using cgo, we already saved g on TLS, also don't save
	// g here.
	// Also don't save g if we are already on the signal stack.
	// We won't get a nested signal.
	MOVBU	runtime·iscgo(SB), R22
	CBNZ	R22, nosaveg
	MOVD	m_gsignal(R21), R22          // g.m.gsignal
	CBZ	R22, nosaveg
	CMP	g, R22
	BEQ	nosaveg
	MOVD	(g_stack+stack_lo)(R22), R22 // g.m.gsignal.stack.lo
	MOVD	g, (R22)

	BL	(R2)

	MOVD	ZR, (R22)  // clear g slot, R22 is unchanged by C code

	B	finish

nosaveg:
	BL	(R2)
	B	finish

fallback:
	MOVD	$SYS_clock_gettime, R8
	SVC

finish:
	MOVD	0(RSP), R3	// sec
	MOVD	8(RSP), R5	// nsec

	MOVD	R20, RSP	// restore SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVD	16(RSP), R1
	MOVD	R1, m_vdsoSP(R21)
	MOVD	8(RSP), R1
	MOVD	R1, m_vdsoPC(R21)

	// sec is in R3, nsec in R5
	// return nsec in R3
	MOVD	$1000000000, R4
	MUL	R4, R3
	ADD	R5, R3
	MOVD	R3, ret+0(FP)
	RET

TEXT runtime·rtsigprocmask(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	how+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVW	size+24(FP), R3
	MOVD	$SYS_rt_sigprocmask, R8
	SVC
	CMN	$4095, R0
	BCC	done
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash
done:
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT|NOFRAME,$0-36
	MOVD	sig+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	size+24(FP), R3
	MOVD	$SYS_rt_sigaction, R8
	SVC
	MOVW	R0, ret+32(FP)
	RET

// Call the function stored in _cgo_sigaction using the GCC calling convention.
TEXT runtime·callCgoSigaction(SB),NOSPLIT,$0
	MOVD	sig+0(FP), R0
	MOVD	new+8(FP), R1
	MOVD	old+16(FP), R2
	MOVD	 _cgo_sigaction(SB), R3
	SUB	$16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL	R3
	ADD	$16, RSP
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R0
	MOVD	info+16(FP), R1
	MOVD	ctx+24(FP), R2
	MOVD	fn+0(FP), R11
	BL	(R11)
	RET

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
	CBZ	R0, 2(PC)
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

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	MOVD	$runtime·sigtramp(SB), R3
	B	(R3)

TEXT runtime·sysMmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	prot+16(FP), R2
	MOVW	flags+20(FP), R3
	MOVW	fd+24(FP), R4
	MOVW	off+28(FP), R5

	MOVD	$SYS_mmap, R8
	SVC
	CMN	$4095, R0
	BCC	ok
	NEG	R0,R0
	MOVD	$0, p+32(FP)
	MOVD	R0, err+40(FP)
	RET
ok:
	MOVD	R0, p+32(FP)
	MOVD	$0, err+40(FP)
	RET

// Call the function stored in _cgo_mmap using the GCC calling convention.
// This must be called on the system stack.
TEXT runtime·callCgoMmap(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	prot+16(FP), R2
	MOVW	flags+20(FP), R3
	MOVW	fd+24(FP), R4
	MOVW	off+28(FP), R5
	MOVD	_cgo_mmap(SB), R9
	SUB	$16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL	R9
	ADD	$16, RSP
	MOVD	R0, ret+32(FP)
	RET

TEXT runtime·sysMunmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	$SYS_munmap, R8
	SVC
	CMN	$4095, R0
	BCC	cool
	MOVD	R0, 0xf0(R0)
cool:
	RET

// Call the function stored in _cgo_munmap using the GCC calling convention.
// This must be called on the system stack.
TEXT runtime·callCgoMunmap(SB),NOSPLIT,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVD	_cgo_munmap(SB), R9
	SUB	$16, RSP		// reserve 16 bytes for sp-8 where fp may be saved.
	BL	R9
	ADD	$16, RSP
	RET

TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVD	n+8(FP), R1
	MOVW	flags+16(FP), R2
	MOVD	$SYS_madvise, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R0
	MOVW	op+8(FP), R1
	MOVW	val+12(FP), R2
	MOVD	ts+16(FP), R3
	MOVD	addr2+24(FP), R4
	MOVW	val3+32(FP), R5
	MOVD	$SYS_futex, R8
	SVC
	MOVW	R0, ret+40(FP)
	RET

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R0
	MOVD	stk+8(FP), R1

	// Copy mp, gp, fn off parent stack for use by child.
	MOVD	mp+16(FP), R10
	MOVD	gp+24(FP), R11
	MOVD	fn+32(FP), R12

	MOVD	R10, -8(R1)
	MOVD	R11, -16(R1)
	MOVD	R12, -24(R1)
	MOVD	$1234, R10
	MOVD	R10, -32(R1)

	MOVD	$SYS_clone, R8
	SVC

	// In parent, return.
	CMP	ZR, R0
	BEQ	child
	MOVW	R0, ret+40(FP)
	RET
child:

	// In child, on new stack.
	MOVD	-32(RSP), R10
	MOVD	$1234, R0
	CMP	R0, R10
	BEQ	good
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash

good:
	// Initialize m->procid to Linux tid
	MOVD	$SYS_gettid, R8
	SVC

	MOVD	-24(RSP), R12     // fn
	MOVD	-16(RSP), R11     // g
	MOVD	-8(RSP), R10      // m

	CMP	$0, R10
	BEQ	nog
	CMP	$0, R11
	BEQ	nog

	MOVD	R0, m_procid(R10)

	// TODO: setup TLS.

	// In child, set up new stack
	MOVD	R10, g_m(R11)
	MOVD	R11, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	MOVD	R12, R0
	BL	(R0)

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R0
again:
	MOVD	$SYS_exit, R8
	SVC
	B	again	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVD	new+0(FP), R0
	MOVD	old+8(FP), R1
	MOVD	$SYS_sigaltstack, R8
	SVC
	CMN	$4095, R0
	BCC	ok
	MOVD	$0, R0
	MOVD	R0, (R0)	// crash
ok:
	RET

TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVD	$SYS_sched_yield, R8
	SVC
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT|NOFRAME,$0
	MOVD	pid+0(FP), R0
	MOVD	len+8(FP), R1
	MOVD	buf+16(FP), R2
	MOVD	$SYS_sched_getaffinity, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT|NOFRAME,$0
	MOVW	$0, R0
	MOVD	$SYS_epoll_create1, R8
	SVC
	MOVW	R0, ret+8(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R0
	MOVD	$SYS_epoll_create1, R8
	SVC
	MOVW	R0, ret+8(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R0
	MOVW	op+4(FP), R1
	MOVW	fd+8(FP), R2
	MOVD	ev+16(FP), R3
	MOVD	$SYS_epoll_ctl, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R0
	MOVD	ev+8(FP), R1
	MOVW	nev+16(FP), R2
	MOVW	timeout+20(FP), R3
	MOVD	$0, R4
	MOVD	$SYS_epoll_pwait, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R0  // fd
	MOVD	$2, R1	// F_SETFD
	MOVD	$1, R2	// FD_CLOEXEC
	MOVD	$SYS_fcntl, R8
	SVC
	RET

// func runtime·setNonblock(int32 fd)
TEXT runtime·setNonblock(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	fd+0(FP), R0 // fd
	MOVD	$3, R1	// F_GETFL
	MOVD	$0, R2
	MOVD	$SYS_fcntl, R8
	SVC
	MOVD	$0x800, R2 // O_NONBLOCK
	ORR	R0, R2
	MOVW	fd+0(FP), R0 // fd
	MOVD	$4, R1	// F_SETFL
	MOVD	$SYS_fcntl, R8
	SVC
	RET

// int access(const char *name, int mode)
TEXT runtime·access(SB),NOSPLIT,$0-20
	MOVD	$AT_FDCWD, R0
	MOVD	name+0(FP), R1
	MOVW	mode+8(FP), R2
	MOVD	$SYS_faccessat, R8
	SVC
	MOVW	R0, ret+16(FP)
	RET

// int connect(int fd, const struct sockaddr *addr, socklen_t len)
TEXT runtime·connect(SB),NOSPLIT,$0-28
	MOVW	fd+0(FP), R0
	MOVD	addr+8(FP), R1
	MOVW	len+16(FP), R2
	MOVD	$SYS_connect, R8
	SVC
	MOVW	R0, ret+24(FP)
	RET

// int socket(int domain, int typ, int prot)
TEXT runtime·socket(SB),NOSPLIT,$0-20
	MOVW	domain+0(FP), R0
	MOVW	typ+4(FP), R1
	MOVW	prot+8(FP), R2
	MOVD	$SYS_socket, R8
	SVC
	MOVW	R0, ret+16(FP)
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT,$0-8
	// Implemented as brk(NULL).
	MOVD	$0, R0
	MOVD	$SYS_brk, R8
	SVC
	MOVD	R0, ret+0(FP)
	RET

TEXT runtime·sigreturn(SB),NOSPLIT,$0-0
	RET
