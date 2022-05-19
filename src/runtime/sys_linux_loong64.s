// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for loong64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define AT_FDCWD -100

#define SYS_exit		93
#define SYS_read		63
#define SYS_write		64
#define SYS_close		57
#define SYS_getpid		172
#define SYS_kill		129
#define SYS_fcntl		25
#define SYS_mmap		222
#define SYS_munmap		215
#define SYS_setitimer		103
#define SYS_clone		220
#define SYS_nanosleep		101
#define SYS_sched_yield		124
#define SYS_rt_sigreturn	139
#define SYS_rt_sigaction	134
#define SYS_rt_sigprocmask	135
#define SYS_sigaltstack		132
#define SYS_madvise		233
#define SYS_mincore		232
#define SYS_gettid		178
#define SYS_futex		98
#define SYS_sched_getaffinity	123
#define SYS_exit_group		94
#define SYS_epoll_ctl		21
#define SYS_tgkill		131
#define SYS_openat		56
#define SYS_epoll_pwait		22
#define SYS_clock_gettime	113
#define SYS_epoll_create1	20
#define SYS_brk			214
#define SYS_pipe2		59
#define SYS_timer_create	107
#define SYS_timer_settime	110
#define SYS_timer_delete	111

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R4
	MOVV	$SYS_exit_group, R11
	SYSCALL
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-8
	MOVV	wait+0(FP), R19
	// We're done using the stack.
	MOVW	$0, R11
	DBAR
	MOVW	R11, (R19)
	DBAR
	MOVW	$0, R4	// exit code
	MOVV	$SYS_exit, R11
	SYSCALL
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOVW	$AT_FDCWD, R4 // AT_FDCWD, so this acts like open
	MOVV	name+0(FP), R5
	MOVW	mode+8(FP), R6
	MOVW	perm+12(FP), R7
	MOVV	$SYS_openat, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVW	$-1, R4
	MOVW	R4, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R4
	MOVV	$SYS_close, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVW	$-1, R4
	MOVW	R4, ret+8(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOVV	fd+0(FP), R4
	MOVV	p+8(FP), R5
	MOVW	n+16(FP), R6
	MOVV	$SYS_write, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R4
	MOVV	p+8(FP), R5
	MOVW	n+16(FP), R6
	MOVV	$SYS_read, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT|NOFRAME,$0-12
	MOVV	$r+0(FP), R4
	MOVV	R0, R5
	MOVV	$SYS_pipe2, R11
	SYSCALL
	MOVW	R4, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	MOVV	$r+8(FP), R4
	MOVW	flags+0(FP), R5
	MOVV	$SYS_pipe2, R11
	SYSCALL
	MOVW	R4, errno+16(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVWU	usec+0(FP), R6
	MOVV	R6, R5
	MOVW	$1000000, R4
	DIVVU	R4, R6, R6
	MOVV	R6, 8(R3)
	MOVW	$1000, R4
	MULVU	R6, R4, R4
	SUBVU	R4, R5
	MOVV	R5, 16(R3)

	// nanosleep(&ts, 0)
	ADDV	$8, R3, R4
	MOVW	$0, R5
	MOVV	$SYS_nanosleep, R11
	SYSCALL
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVV	$SYS_gettid, R11
	SYSCALL
	MOVW	R4, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT|NOFRAME,$0
	MOVV	$SYS_getpid, R11
	SYSCALL
	MOVW	R4, R23
	MOVV	$SYS_gettid, R11
	SYSCALL
	MOVW	R4, R5	// arg 2 tid
	MOVW	R23, R4	// arg 1 pid
	MOVW	sig+0(FP), R6	// arg 3
	MOVV	$SYS_tgkill, R11
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVV	$SYS_getpid, R11
	SYSCALL
	//MOVW	R4, R4	// arg 1 pid
	MOVW	sig+0(FP), R5	// arg 2
	MOVV	$SYS_kill, R11
	SYSCALL
	RET

TEXT ·getpid(SB),NOSPLIT|NOFRAME,$0-8
	MOVV	$SYS_getpid, R11
	SYSCALL
	MOVV	R4, ret+0(FP)
	RET

TEXT ·tgkill(SB),NOSPLIT|NOFRAME,$0-24
	MOVV	tgid+0(FP), R4
	MOVV	tid+8(FP), R5
	MOVV	sig+16(FP), R6
	MOVV	$SYS_tgkill, R11
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVV	$SYS_setitimer, R11
	SYSCALL
	RET

TEXT runtime·timer_create(SB),NOSPLIT,$0-28
	MOVW	clockid+0(FP), R4
	MOVV	sevp+8(FP), R5
	MOVV	timerid+16(FP), R6
	MOVV	$SYS_timer_create, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

TEXT runtime·timer_settime(SB),NOSPLIT,$0-28
	MOVW	timerid+0(FP), R4
	MOVW	flags+4(FP), R5
	MOVV	new+8(FP), R6
	MOVV	old+16(FP), R7
	MOVV	$SYS_timer_settime, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

TEXT runtime·timer_delete(SB),NOSPLIT,$0-12
	MOVW	timerid+0(FP), R4
	MOVV	$SYS_timer_delete, R11
	SYSCALL
	MOVW	R4, ret+8(FP)
	RET

TEXT runtime·mincore(SB),NOSPLIT|NOFRAME,$0-28
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	dst+16(FP), R6
	MOVV	$SYS_mincore, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB),NOSPLIT,$16-12
	MOVV	R3, R23	// R23 is unchanged by C code
	MOVV	R3, R25

	MOVV	g_m(g), R24	// R24 = m

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVV	m_vdsoPC(R24), R11
	MOVV	m_vdsoSP(R24), R7
	MOVV	R11, 8(R3)
	MOVV	R7, 16(R3)

	MOVV    $ret-8(FP), R11 // caller's SP
	MOVV	R1, m_vdsoPC(R24)
	MOVV	R11, m_vdsoSP(R24)

	MOVV	m_curg(R24), R4
	MOVV	g, R5
	BNE	R4, R5, noswitch

	MOVV	m_g0(R24), R4
	MOVV	(g_sched+gobuf_sp)(R4), R25	// Set SP to g0 stack

noswitch:
	SUBV	$16, R25
	AND	$~15, R25	// Align for C code
	MOVV	R25, R3

	MOVW	$0, R4 // CLOCK_REALTIME=0
	MOVV	$0(R3), R5

	MOVV	runtime·vdsoClockgettimeSym(SB), R20
	BEQ	R20, fallback

	JAL	(R20)

finish:
	MOVV	0(R3), R7	// sec
	MOVV	8(R3), R5	// nsec

	MOVV	R23, R3	// restore SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVV	16(R3), R25
	MOVV	R25, m_vdsoSP(R24)
	MOVV	8(R3), R25
	MOVV	R25, m_vdsoPC(R24)

	MOVV	R7, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

fallback:
	MOVV	$SYS_clock_gettime, R11
	SYSCALL
	JMP finish

TEXT runtime·nanotime1(SB),NOSPLIT,$16-8
	MOVV	R3, R23	// R23 is unchanged by C code
	MOVV	R3, R25

	MOVV	g_m(g), R24	// R24 = m

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVV	m_vdsoPC(R24), R11
	MOVV	m_vdsoSP(R24), R7
	MOVV	R11, 8(R3)
	MOVV	R7, 16(R3)

	MOVV    $ret-8(FP), R11 // caller's SP
	MOVV	R1, m_vdsoPC(R24)
	MOVV	R11, m_vdsoSP(R24)

	MOVV	m_curg(R24), R4
	MOVV	g, R5
	BNE	R4, R5, noswitch

	MOVV	m_g0(R24), R4
	MOVV	(g_sched+gobuf_sp)(R4), R25	// Set SP to g0 stack

noswitch:
	SUBV	$16, R25
	AND	$~15, R25	// Align for C code
	MOVV	R25, R3

	MOVW	$1, R4 // CLOCK_MONOTONIC=1
	MOVV	$0(R3), R5

	MOVV	runtime·vdsoClockgettimeSym(SB), R20
	BEQ	R20, fallback

	JAL	(R20)

finish:
	MOVV	0(R3), R7	// sec
	MOVV	8(R3), R5	// nsec

	MOVV	R23, R3	// restore SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVV	16(R3), R25
	MOVV	R25, m_vdsoSP(R24)
	MOVV	8(R3), R25
	MOVV	R25, m_vdsoPC(R24)

	// sec is in R7, nsec in R5
	// return nsec in R7
	MOVV	$1000000000, R4
	MULVU	R4, R7, R7
	ADDVU	R5, R7
	MOVV	R7, ret+0(FP)
	RET

fallback:
	MOVV	$SYS_clock_gettime, R11
	SYSCALL
	JMP	finish

TEXT runtime·rtsigprocmask(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	how+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVW	size+24(FP), R7
	MOVV	$SYS_rt_sigprocmask, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT|NOFRAME,$0-36
	MOVV	sig+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVV	size+24(FP), R7
	MOVV	$SYS_rt_sigaction, R11
	SYSCALL
	MOVW	R4, ret+32(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R4
	MOVV	info+16(FP), R5
	MOVV	ctx+24(FP), R6
	MOVV	fn+0(FP), R20
	JAL	(R20)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$64
	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R19
	BEQ	R19, 2(PC)
	JAL	runtime·load_g(SB)

	MOVW	R4, 8(R3)
	MOVV	R5, 16(R3)
	MOVV	R6, 24(R3)
	MOVV	$runtime·sigtrampgo(SB), R19
	JAL	(R19)
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

TEXT runtime·mmap(SB),NOSPLIT|NOFRAME,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVW	prot+16(FP), R6
	MOVW	flags+20(FP), R7
	MOVW	fd+24(FP), R8
	MOVW	off+28(FP), R9

	MOVV	$SYS_mmap, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, ok
	MOVV	$0, p+32(FP)
	SUBVU	R4, R0, R4
	MOVV	R4, err+40(FP)
	RET
ok:
	MOVV	R4, p+32(FP)
	MOVV	$0, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT|NOFRAME,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	$SYS_munmap, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf3(R0)	// crash
	RET

TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVW	flags+16(FP), R6
	MOVV	$SYS_madvise, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT|NOFRAME,$0
	MOVV	addr+0(FP), R4
	MOVW	op+8(FP), R5
	MOVW	val+12(FP), R6
	MOVV	ts+16(FP), R7
	MOVV	addr2+24(FP), R8
	MOVW	val3+32(FP), R9
	MOVV	$SYS_futex, R11
	SYSCALL
	MOVW	R4, ret+40(FP)
	RET

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R4
	MOVV	stk+8(FP), R5

	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVV	mp+16(FP), R23
	MOVV	gp+24(FP), R24
	MOVV	fn+32(FP), R25

	MOVV	R23, -8(R5)
	MOVV	R24, -16(R5)
	MOVV	R25, -24(R5)
	MOVV	$1234, R23
	MOVV	R23, -32(R5)

	MOVV	$SYS_clone, R11
	SYSCALL

	// In parent, return.
	BEQ	R4, 3(PC)
	MOVW	R4, ret+40(FP)
	RET

	// In child, on new stack.
	MOVV	-32(R3), R23
	MOVV	$1234, R19
	BEQ	R23, R19, 2(PC)
	MOVV	R0, 0(R0)

	// Initialize m->procid to Linux tid
	MOVV	$SYS_gettid, R11
	SYSCALL

	MOVV	-24(R3), R25		// fn
	MOVV	-16(R3), R24		// g
	MOVV	-8(R3), R23		// m

	BEQ	R23, nog
	BEQ	R24, nog

	MOVV	R4, m_procid(R23)

	// TODO: setup TLS.

	// In child, set up new stack
	MOVV	R23, g_m(R24)
	MOVV	R24, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	JAL	(R25)

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R4
	MOVV	$SYS_exit, R11
	SYSCALL
	JMP	-3(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVV	new+0(FP), R4
	MOVV	old+8(FP), R5
	MOVV	$SYS_sigaltstack, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVV	$SYS_sched_yield, R11
	SYSCALL
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT|NOFRAME,$0
	MOVV	pid+0(FP), R4
	MOVV	len+8(FP), R5
	MOVV	buf+16(FP), R6
	MOVV	$SYS_sched_getaffinity, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT|NOFRAME,$0
	MOVW	size+0(FP), R4
	MOVV	$SYS_epoll_create1, R11
	SYSCALL
	MOVW	R4, ret+8(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R4
	MOVV	$SYS_epoll_create1, R11
	SYSCALL
	MOVW	R4, ret+8(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R4
	MOVW	op+4(FP), R5
	MOVW	fd+8(FP), R6
	MOVV	ev+16(FP), R7
	MOVV	$SYS_epoll_ctl, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R4
	MOVV	ev+8(FP), R5
	MOVW	nev+16(FP), R6
	MOVW	timeout+20(FP), R7
	MOVV	$0, R8
	MOVV	$SYS_epoll_pwait, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT|NOFRAME,$0
	MOVW	fd+0(FP), R4  // fd
	MOVV	$2, R5	// F_SETFD
	MOVV	$1, R6	// FD_CLOEXEC
	MOVV	$SYS_fcntl, R11
	SYSCALL
	RET

// func runtime·setNonblock(int32 fd)
TEXT runtime·setNonblock(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	fd+0(FP), R4 // fd
	MOVV	$3, R5	// F_GETFL
	MOVV	$0, R6
	MOVV	$SYS_fcntl, R11
	SYSCALL
	MOVW	$0x800, R6 // O_NONBLOCK
	OR	R4, R6
	MOVW	fd+0(FP), R4 // fd
	MOVV	$4, R5	// F_SETFL
	MOVV	$SYS_fcntl, R11
	SYSCALL
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT|NOFRAME,$0-8
	// Implemented as brk(NULL).
	MOVV	$0, R4
	MOVV	$SYS_brk, R11
	SYSCALL
	MOVV	R4, ret+0(FP)
	RET

TEXT runtime·access(SB),$0-20
	MOVV	R0, 2(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+16(FP) // for vet
	RET

TEXT runtime·connect(SB),$0-28
	MOVV	R0, 2(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+24(FP) // for vet
	RET

TEXT runtime·socket(SB),$0-20
	MOVV	R0, 2(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+16(FP) // for vet
	RET
