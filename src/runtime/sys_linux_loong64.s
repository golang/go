// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for loong64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "cgo/abi_loong64.h"

#define AT_FDCWD	-100
#define CLOCK_REALTIME	0
#define CLOCK_MONOTONIC	1

#define SYS_exit		93
#define SYS_read		63
#define SYS_write		64
#define SYS_close		57
#define SYS_getpid		172
#define SYS_kill		129
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
#define SYS_tgkill		131
#define SYS_openat		56
#define SYS_clock_gettime	113
#define SYS_brk			214
#define SYS_pipe2		59
#define SYS_timer_create	107
#define SYS_timer_settime	110
#define SYS_timer_delete	111

// func exit(code int32)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R4
	MOVV	$SYS_exit_group, R11
	SYSCALL
	RET

// func exitThread(wait *atomic.Uint32)
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

// func open(name *byte, mode, perm int32) int32
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

// func closefd(fd int32) int32
TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R4
	MOVV	$SYS_close, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVW	$-1, R4
	MOVW	R4, ret+8(FP)
	RET

// func write1(fd uintptr, p unsafe.Pointer, n int32) int32
TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOVV	fd+0(FP), R4
	MOVV	p+8(FP), R5
	MOVW	n+16(FP), R6
	MOVV	$SYS_write, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func read(fd int32, p unsafe.Pointer, n int32) int32
TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R4
	MOVV	p+8(FP), R5
	MOVW	n+16(FP), R6
	MOVV	$SYS_read, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	MOVV	$r+8(FP), R4
	MOVW	flags+0(FP), R5
	MOVV	$SYS_pipe2, R11
	SYSCALL
	MOVW	R4, errno+16(FP)
	RET

// func usleep(usec uint32)
TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVWU	usec+0(FP), R7
	MOVV	$1000, R6
	MULVU	R6, R7, R7
	MOVV	$1000000000, R6

	DIVVU	R6, R7, R5	// ts->tv_sec
	REMVU	R6, R7, R4	// ts->tv_nsec
	MOVV	R5, 8(R3)
	MOVV	R4, 16(R3)

	// nanosleep(&ts, 0)
	ADDV	$8, R3, R4
	MOVV	R0, R5
	MOVV	$SYS_nanosleep, R11
	SYSCALL
	RET

// func gettid() uint32
TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVV	$SYS_gettid, R11
	SYSCALL
	MOVW	R4, ret+0(FP)
	RET

// func raise(sig uint32)
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

// func raiseproc(sig uint32)
TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVV	$SYS_getpid, R11
	SYSCALL
	//MOVW	R4, R4	// arg 1 pid
	MOVW	sig+0(FP), R5	// arg 2
	MOVV	$SYS_kill, R11
	SYSCALL
	RET

// func getpid() int
TEXT ·getpid(SB),NOSPLIT|NOFRAME,$0-8
	MOVV	$SYS_getpid, R11
	SYSCALL
	MOVV	R4, ret+0(FP)
	RET

// func tgkill(tgid, tid, sig int)
TEXT ·tgkill(SB),NOSPLIT|NOFRAME,$0-24
	MOVV	tgid+0(FP), R4
	MOVV	tid+8(FP), R5
	MOVV	sig+16(FP), R6
	MOVV	$SYS_tgkill, R11
	SYSCALL
	RET

// func setitimer(mode int32, new, old *itimerval)
TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVV	$SYS_setitimer, R11
	SYSCALL
	RET

// func timer_create(clockid int32, sevp *sigevent, timerid *int32) int32
TEXT runtime·timer_create(SB),NOSPLIT,$0-28
	MOVW	clockid+0(FP), R4
	MOVV	sevp+8(FP), R5
	MOVV	timerid+16(FP), R6
	MOVV	$SYS_timer_create, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32
TEXT runtime·timer_settime(SB),NOSPLIT,$0-28
	MOVW	timerid+0(FP), R4
	MOVW	flags+4(FP), R5
	MOVV	new+8(FP), R6
	MOVV	old+16(FP), R7
	MOVV	$SYS_timer_settime, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func timer_delete(timerid int32) int32
TEXT runtime·timer_delete(SB),NOSPLIT,$0-12
	MOVW	timerid+0(FP), R4
	MOVV	$SYS_timer_delete, R11
	SYSCALL
	MOVW	R4, ret+8(FP)
	RET

// func mincore(addr unsafe.Pointer, n uintptr, dst *byte) int32
TEXT runtime·mincore(SB),NOSPLIT|NOFRAME,$0-28
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	dst+16(FP), R6
	MOVV	$SYS_mincore, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB),NOSPLIT,$24-12
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

	MOVW	$CLOCK_REALTIME, R4
	MOVV	$0(R3), R5

	MOVV	runtime·vdsoClockgettimeSym(SB), R20
	BEQ	R20, fallback

	// Store g on gsignal's stack, see sys_linux_arm64.s for detail
	MOVBU	runtime·iscgo(SB), R25
	BNE	R25, nosaveg

	MOVV	m_gsignal(R24), R25	// g.m.gsignal
	BEQ	R25, nosaveg
	BEQ	g, R25, nosaveg

	MOVV	(g_stack+stack_lo)(R25), R25	// g.m.gsignal.stack.lo
	MOVV	g, (R25)

	JAL	(R20)

	MOVV	R0, (R25)
	JMP	finish

nosaveg:
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

// func nanotime1() int64
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

	MOVW	$CLOCK_MONOTONIC, R4
	MOVV	$0(R3), R5

	MOVV	runtime·vdsoClockgettimeSym(SB), R20
	BEQ	R20, fallback

	// Store g on gsignal's stack, see sys_linux_arm64.s for detail
	MOVBU	runtime·iscgo(SB), R25
	BNE	R25, nosaveg

	MOVV	m_gsignal(R24), R25	// g.m.gsignal
	BEQ	R25, nosaveg
	BEQ	g, R25, nosaveg

	MOVV	(g_stack+stack_lo)(R25), R25	// g.m.gsignal.stack.lo
	MOVV	g, (R25)

	JAL	(R20)

	MOVV	R0, (R25)
	JMP	finish

nosaveg:
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

// func rtsigprocmask(how int32, new, old *sigset, size int32)
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

// func rt_sigaction(sig uintptr, new, old *sigactiont, size uintptr) int32
TEXT runtime·rt_sigaction(SB),NOSPLIT|NOFRAME,$0-36
	MOVV	sig+0(FP), R4
	MOVV	new+8(FP), R5
	MOVV	old+16(FP), R6
	MOVV	size+24(FP), R7
	MOVV	$SYS_rt_sigaction, R11
	SYSCALL
	MOVW	R4, ret+32(FP)
	RET

// func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)
TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R4
	MOVV	info+16(FP), R5
	MOVV	ctx+24(FP), R6
	MOVV	fn+0(FP), R20
	JAL	(R20)
	RET

// func sigtramp(signo, ureg, ctxt unsafe.Pointer)
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$168
	MOVW	R4, (1*8)(R3)
	MOVV	R5, (2*8)(R3)
	MOVV	R6, (3*8)(R3)

	// Save callee-save registers in the case of signal forwarding.
	// Please refer to https://golang.org/issue/31827 .
	SAVE_R22_TO_R31((4*8))
	SAVE_F24_TO_F31((14*8))

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R4
	BEQ	R4, 2(PC)
	JAL	runtime·load_g(SB)

	MOVV	$runtime·sigtrampgo(SB), R4
	JAL	(R4)

	// Restore callee-save registers.
	RESTORE_R22_TO_R31((4*8))
	RESTORE_F24_TO_F31((14*8))

	RET

// func cgoSigtramp()
TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

// func sysMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (p unsafe.Pointer, err int)
TEXT runtime·sysMmap(SB),NOSPLIT|NOFRAME,$0
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

// Call the function stored in _cgo_mmap using the GCC calling convention.
// This must be called on the system stack.
// func callCgoMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) uintptr
TEXT runtime·callCgoMmap(SB),NOSPLIT,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVW	prot+16(FP), R6
	MOVW	flags+20(FP), R7
	MOVW	fd+24(FP), R8
	MOVW	off+28(FP), R9
	MOVV	_cgo_mmap(SB), R13
	SUBV	$16, R3		// reserve 16 bytes for sp-8 where fp may be saved.
	JAL	(R13)
	ADDV	$16, R3
	MOVV	R4, ret+32(FP)
	RET

// func sysMunmap(addr unsafe.Pointer, n uintptr)
TEXT runtime·sysMunmap(SB),NOSPLIT|NOFRAME,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	$SYS_munmap, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf3(R0)	// crash
	RET

// Call the function stored in _cgo_munmap using the GCC calling convention.
// This must be called on the system stack.
// func callCgoMunmap(addr unsafe.Pointer, n uintptr)
TEXT runtime·callCgoMunmap(SB),NOSPLIT,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVV	_cgo_munmap(SB), R13
	SUBV	$16, R3		// reserve 16 bytes for sp-8 where fp may be saved.
	JAL	(R13)
	ADDV	$16, R3
	RET

// func madvise(addr unsafe.Pointer, n uintptr, flags int32)
TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVV	addr+0(FP), R4
	MOVV	n+8(FP), R5
	MOVW	flags+16(FP), R6
	MOVV	$SYS_madvise, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
	RET

// func futex(addr unsafe.Pointer, op int32, val uint32, ts, addr2 unsafe.Pointer, val3 uint32) int32
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

// func sigaltstack(new, old *stackt)
TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVV	new+0(FP), R4
	MOVV	old+8(FP), R5
	MOVV	$SYS_sigaltstack, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

// func osyield()
TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVV	$SYS_sched_yield, R11
	SYSCALL
	RET

// func sched_getaffinity(pid, len uintptr, buf *uintptr) int32
TEXT runtime·sched_getaffinity(SB),NOSPLIT|NOFRAME,$0
	MOVV	pid+0(FP), R4
	MOVV	len+8(FP), R5
	MOVV	buf+16(FP), R6
	MOVV	$SYS_sched_getaffinity, R11
	SYSCALL
	MOVW	R4, ret+24(FP)
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
