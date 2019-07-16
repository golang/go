// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build ppc64 ppc64le

//
// System calls and other sys.stuff for ppc64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "asm_ppc64x.h"

#define SYS_exit		  1
#define SYS_read		  3
#define SYS_write		  4
#define SYS_open		  5
#define SYS_close		  6
#define SYS_getpid		 20
#define SYS_kill		 37
#define SYS_brk			 45
#define SYS_fcntl		 55
#define SYS_mmap		 90
#define SYS_munmap		 91
#define SYS_setitimer		104
#define SYS_clone		120
#define SYS_sched_yield		158
#define SYS_nanosleep		162
#define SYS_rt_sigreturn	172
#define SYS_rt_sigaction	173
#define SYS_rt_sigprocmask	174
#define SYS_sigaltstack		185
#define SYS_madvise		205
#define SYS_mincore		206
#define SYS_gettid		207
#define SYS_futex		221
#define SYS_sched_getaffinity	223
#define SYS_exit_group		234
#define SYS_epoll_create	236
#define SYS_epoll_ctl		237
#define SYS_epoll_wait		238
#define SYS_clock_gettime	246
#define SYS_tgkill		250
#define SYS_epoll_create1	315

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R3
	SYSCALL	$SYS_exit_group
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-8
	MOVD	wait+0(FP), R1
	// We're done using the stack.
	MOVW	$0, R2
	SYNC
	MOVW	R2, (R1)
	MOVW	$0, R3	// exit code
	SYSCALL	$SYS_exit
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	name+0(FP), R3
	MOVW	mode+8(FP), R4
	MOVW	perm+12(FP), R5
	SYSCALL	$SYS_open
	BVC	2(PC)
	MOVW	$-1, R3
	MOVW	R3, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R3
	SYSCALL	$SYS_close
	BVC	2(PC)
	MOVW	$-1, R3
	MOVW	R3, ret+8(FP)
	RET

TEXT runtime·write(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	SYSCALL	$SYS_write
	BVC	2(PC)
	MOVW	$-1, R3
	MOVW	R3, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	SYSCALL	$SYS_read
	BVC	2(PC)
	MOVW	$-1, R3
	MOVW	R3, ret+24(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVW	usec+0(FP), R3
	MOVD	R3, R5
	MOVW	$1000000, R4
	DIVD	R4, R3
	MOVD	R3, 8(R1)
	MOVW	$1000, R4
	MULLD	R3, R4
	SUB	R4, R5
	MOVD	R5, 16(R1)

	// nanosleep(&ts, 0)
	ADD	$8, R1, R3
	MOVW	$0, R4
	SYSCALL	$SYS_nanosleep
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	SYSCALL	$SYS_gettid
	MOVW	R3, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT|NOFRAME,$0
	SYSCALL	$SYS_getpid
	MOVW	R3, R14
	SYSCALL	$SYS_gettid
	MOVW	R3, R4	// arg 2 tid
	MOVW	R14, R3	// arg 1 pid
	MOVW	sig+0(FP), R5	// arg 3
	SYSCALL	$SYS_tgkill
	RET

TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	SYSCALL	$SYS_getpid
	MOVW	R3, R3	// arg 1 pid
	MOVW	sig+0(FP), R4	// arg 2
	SYSCALL	$SYS_kill
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	SYSCALL	$SYS_setitimer
	RET

TEXT runtime·mincore(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	MOVD	dst+16(FP), R5
	SYSCALL	$SYS_mincore
	NEG	R3		// caller expects negative errno
	MOVW	R3, ret+24(FP)
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB),NOSPLIT,$16
	MOVD	R1, R15		// R15 is unchanged by C code
	MOVD	g_m(g), R21	// R21 = m

	MOVD	$0, R3		// CLOCK_REALTIME

	MOVD	runtime·vdsoClockgettimeSym(SB), R12	// Check for VDSO availability
	CMP	R12, R0
	BEQ	fallback

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	MOVD	LR, R14
	MOVD	R14, m_vdsoPC(R21)
	MOVD	R15, m_vdsoSP(R21)

	MOVD	m_curg(R21), R6
	CMP	g, R6
	BNE	noswitch

	MOVD	m_g0(R21), R7
	MOVD	(g_sched+gobuf_sp)(R7), R1	// Set SP to g0 stack

noswitch:
	SUB	$16, R1			// Space for results
	RLDICR	$0, R1, $59, R1		// Align for C code
	MOVD	R12, CTR
	MOVD	R1, R4
	BL	(CTR)			// Call from VDSO
	MOVD	$0, R0			// Restore R0
	MOVD	R0, m_vdsoSP(R21)	// Clear vdsoSP
	MOVD	0(R1), R3		// sec
	MOVD	8(R1), R5		// nsec
	MOVD	R15, R1			// Restore SP

finish:
	MOVD	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

	// Syscall fallback
fallback:
	ADD	$32, R1, R4
	SYSCALL $SYS_clock_gettime
	MOVD	32(R1), R3
	MOVD	40(R1), R5
	JMP	finish

TEXT runtime·nanotime(SB),NOSPLIT,$16
	MOVD	$1, R3		// CLOCK_MONOTONIC

	MOVD	R1, R15		// R15 is unchanged by C code
	MOVD	g_m(g), R21	// R21 = m

	MOVD	runtime·vdsoClockgettimeSym(SB), R12	// Check for VDSO availability
	CMP	R12, R0
	BEQ	fallback

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	MOVD	LR, R14		// R14 is unchanged by C code
	MOVD	R14, m_vdsoPC(R21)
	MOVD	R15, m_vdsoSP(R21)

	MOVD	m_curg(R21), R6
	CMP	g, R6
	BNE	noswitch

	MOVD	m_g0(R21), R7
	MOVD	(g_sched+gobuf_sp)(R7), R1	// Set SP to g0 stack

noswitch:
	SUB	$16, R1			// Space for results
	RLDICR	$0, R1, $59, R1		// Align for C code
	MOVD	R12, CTR
	MOVD	R1, R4
	BL	(CTR)			// Call from VDSO
	MOVD	$0, R0			// Restore R0
	MOVD	$0, m_vdsoSP(R21)	// Clear vdsoSP
	MOVD	0(R1), R3		// sec
	MOVD	8(R1), R5		// nsec
	MOVD	R15, R1			// Restore SP

finish:
	// sec is in R3, nsec in R5
	// return nsec in R3
	MOVD	$1000000000, R4
	MULLD	R4, R3
	ADD	R5, R3
	MOVD	R3, ret+0(FP)
	RET

	// Syscall fallback
fallback:
	ADD	$32, R1, R4
	SYSCALL $SYS_clock_gettime
	MOVD	32(R1), R3
	MOVD	48(R1), R5
	JMP	finish

TEXT runtime·rtsigprocmask(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	how+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVW	size+24(FP), R6
	SYSCALL	$SYS_rt_sigprocmask
	BVC	2(PC)
	MOVD	R0, 0xf0(R0)	// crash
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT|NOFRAME,$0-36
	MOVD	sig+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVD	size+24(FP), R6
	SYSCALL	$SYS_rt_sigaction
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+32(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R3
	MOVD	info+16(FP), R4
	MOVD	ctx+24(FP), R5
	MOVD	fn+0(FP), R12
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	24(R1), R2
	RET

TEXT runtime·sigreturn(SB),NOSPLIT,$0-0
	RET

#ifdef GOARCH_ppc64le
// ppc64le doesn't need function descriptors
TEXT runtime·sigtramp(SB),NOSPLIT,$64
#else
// function descriptor for the real sigtramp
TEXT runtime·sigtramp(SB),NOSPLIT|NOFRAME,$0
	DWORD	$sigtramp<>(SB)
	DWORD	$0
	DWORD	$0
TEXT sigtramp<>(SB),NOSPLIT,$64
#endif
	// initialize essential registers (just in case)
	BL	runtime·reginit(SB)

	// this might be called in external code context,
	// where g is not set.
	MOVBZ	runtime·iscgo(SB), R6
	CMP 	R6, $0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	MOVW	R3, FIXED_FRAME+0(R1)
	MOVD	R4, FIXED_FRAME+8(R1)
	MOVD	R5, FIXED_FRAME+16(R1)
	MOVD	$runtime·sigtrampgo(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	24(R1), R2
	RET

#ifdef GOARCH_ppc64le
// ppc64le doesn't need function descriptors
TEXT runtime·cgoSigtramp(SB),NOSPLIT|NOFRAME,$0
	// The stack unwinder, presumably written in C, may not be able to
	// handle Go frame correctly. So, this function is NOFRAME, and we
	// save/restore LR manually.
	MOVD	LR, R10

	// We're coming from C code, initialize essential registers.
	CALL	runtime·reginit(SB)

	// If no traceback function, do usual sigtramp.
	MOVD	runtime·cgoTraceback(SB), R6
	CMP	$0, R6
	BEQ	sigtramp

	// If no traceback support function, which means that
	// runtime/cgo was not linked in, do usual sigtramp.
	MOVD	_cgo_callers(SB), R6
	CMP	$0, R6
	BEQ	sigtramp

	// Set up g register.
	CALL	runtime·load_g(SB)

	// Figure out if we are currently in a cgo call.
	// If not, just do usual sigtramp.
	CMP	$0, g
	BEQ	sigtrampnog // g == nil
	MOVD	g_m(g), R6
	CMP	$0, R6
	BEQ	sigtramp    // g.m == nil
	MOVW	m_ncgo(R6), R7
	CMPW	$0, R7
	BEQ	sigtramp    // g.m.ncgo = 0
	MOVD	m_curg(R6), R7
	CMP	$0, R7
	BEQ	sigtramp    // g.m.curg == nil
	MOVD	g_syscallsp(R7), R7
	CMP	$0, R7
	BEQ	sigtramp    // g.m.curg.syscallsp == 0
	MOVD	m_cgoCallers(R6), R7 // R7 is the fifth arg in C calling convention.
	CMP	$0, R7
	BEQ	sigtramp    // g.m.cgoCallers == nil
	MOVW	m_cgoCallersUse(R6), R8
	CMPW	$0, R8
	BNE	sigtramp    // g.m.cgoCallersUse != 0

	// Jump to a function in runtime/cgo.
	// That function, written in C, will call the user's traceback
	// function with proper unwind info, and will then call back here.
	// The first three arguments, and the fifth, are already in registers.
	// Set the two remaining arguments now.
	MOVD	runtime·cgoTraceback(SB), R6
	MOVD	$runtime·sigtramp(SB), R8
	MOVD	_cgo_callers(SB), R12
	MOVD	R12, CTR
	MOVD	R10, LR // restore LR
	JMP	(CTR)

sigtramp:
	MOVD	R10, LR // restore LR
	JMP	runtime·sigtramp(SB)

sigtrampnog:
	// Signal arrived on a non-Go thread. If this is SIGPROF, get a
	// stack trace.
	CMPW	R3, $27 // 27 == SIGPROF
	BNE	sigtramp

	// Lock sigprofCallersUse (cas from 0 to 1).
	MOVW	$1, R7
	MOVD	$runtime·sigprofCallersUse(SB), R8
	SYNC
	LWAR    (R8), R6
	CMPW    $0, R6
	BNE     sigtramp
	STWCCC  R7, (R8)
	BNE     -4(PC)
	ISYNC

	// Jump to the traceback function in runtime/cgo.
	// It will call back to sigprofNonGo, which will ignore the
	// arguments passed in registers.
	// First three arguments to traceback function are in registers already.
	MOVD	runtime·cgoTraceback(SB), R6
	MOVD	$runtime·sigprofCallers(SB), R7
	MOVD	$runtime·sigprofNonGoWrapper<>(SB), R8
	MOVD	_cgo_callers(SB), R12
	MOVD	R12, CTR
	MOVD	R10, LR // restore LR
	JMP	(CTR)
#else
// function descriptor for the real sigtramp
TEXT runtime·cgoSigtramp(SB),NOSPLIT|NOFRAME,$0
	DWORD	$cgoSigtramp<>(SB)
	DWORD	$0
	DWORD	$0
TEXT cgoSigtramp<>(SB),NOSPLIT,$0
	JMP	sigtramp<>(SB)
#endif

TEXT runtime·sigprofNonGoWrapper<>(SB),NOSPLIT,$0
	// We're coming from C code, set up essential register, then call sigprofNonGo.
	CALL	runtime·reginit(SB)
	CALL	runtime·sigprofNonGo(SB)
	RET

TEXT runtime·mmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	MOVW	prot+16(FP), R5
	MOVW	flags+20(FP), R6
	MOVW	fd+24(FP), R7
	MOVW	off+28(FP), R8

	SYSCALL	$SYS_mmap
	BVC	ok
	MOVD	$0, p+32(FP)
	MOVD	R3, err+40(FP)
	RET
ok:
	MOVD	R3, p+32(FP)
	MOVD	$0, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	SYSCALL	$SYS_munmap
	BVC	2(PC)
	MOVD	R0, 0xf0(R0)
	RET

TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R3
	MOVD	n+8(FP), R4
	MOVW	flags+16(FP), R5
	SYSCALL	$SYS_madvise
	MOVW	R3, ret+24(FP)
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R3
	MOVW	op+8(FP), R4
	MOVW	val+12(FP), R5
	MOVD	ts+16(FP), R6
	MOVD	addr2+24(FP), R7
	MOVW	val3+32(FP), R8
	SYSCALL	$SYS_futex
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+40(FP)
	RET

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R3
	MOVD	stk+8(FP), R4

	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVD	mp+16(FP), R7
	MOVD	gp+24(FP), R8
	MOVD	fn+32(FP), R12

	MOVD	R7, -8(R4)
	MOVD	R8, -16(R4)
	MOVD	R12, -24(R4)
	MOVD	$1234, R7
	MOVD	R7, -32(R4)

	SYSCALL $SYS_clone
	BVC	2(PC)
	NEG	R3	// caller expects negative errno

	// In parent, return.
	CMP	R3, $0
	BEQ	3(PC)
	MOVW	R3, ret+40(FP)
	RET

	// In child, on new stack.
	// initialize essential registers
	BL	runtime·reginit(SB)
	MOVD	-32(R1), R7
	CMP	R7, $1234
	BEQ	2(PC)
	MOVD	R0, 0(R0)

	// Initialize m->procid to Linux tid
	SYSCALL $SYS_gettid

	MOVD	-24(R1), R12       // fn
	MOVD	-16(R1), R8        // g
	MOVD	-8(R1), R7         // m

	CMP	R7, $0
	BEQ	nog
	CMP	R8, $0
	BEQ	nog

	MOVD	R3, m_procid(R7)

	// TODO: setup TLS.

	// In child, set up new stack
	MOVD	R7, g_m(R8)
	MOVD	R8, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	MOVD	R12, CTR
	BL	(CTR)

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R3
	SYSCALL	$SYS_exit
	BR	-2(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVD	new+0(FP), R3
	MOVD	old+8(FP), R4
	SYSCALL	$SYS_sigaltstack
	BVC	2(PC)
	MOVD	R0, 0xf0(R0)  // crash
	RET

TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	SYSCALL	$SYS_sched_yield
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT|NOFRAME,$0
	MOVD	pid+0(FP), R3
	MOVD	len+8(FP), R4
	MOVD	buf+16(FP), R5
	SYSCALL	$SYS_sched_getaffinity
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+24(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT|NOFRAME,$0
	MOVW    size+0(FP), R3
	SYSCALL	$SYS_epoll_create
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+8(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R3
	SYSCALL	$SYS_epoll_create1
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+8(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R3
	MOVW	op+4(FP), R4
	MOVW	fd+8(FP), R5
	MOVD	ev+16(FP), R6
	SYSCALL	$SYS_epoll_ctl
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+24(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT|NOFRAME,$0
	MOVW	epfd+0(FP), R3
	MOVD	ev+8(FP), R4
	MOVW	nev+16(FP), R5
	MOVW	timeout+20(FP), R6
	SYSCALL	$SYS_epoll_wait
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+24(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT|NOFRAME,$0
	MOVW    fd+0(FP), R3  // fd
	MOVD    $2, R4  // F_SETFD
	MOVD    $1, R5  // FD_CLOEXEC
	SYSCALL	$SYS_fcntl
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT|NOFRAME,$0
	// Implemented as brk(NULL).
	MOVD	$0, R3
	SYSCALL	$SYS_brk
	MOVD	R3, ret+0(FP)
	RET

TEXT runtime·access(SB),$0-20
	MOVD	R0, 0(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+16(FP) // for vet
	RET

TEXT runtime·connect(SB),$0-28
	MOVD	R0, 0(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+24(FP) // for vet
	RET

TEXT runtime·socket(SB),$0-20
	MOVD	R0, 0(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+16(FP) // for vet
	RET
