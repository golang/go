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
TEXT runtime·exit<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_exit_group, R11
	SYSCALL
	RET

// func exitThread(wait *atomic.Uint32)
TEXT runtime·exitThread<ABIInternal>(SB),NOSPLIT|NOFRAME,$0
	// We're done using the stack.
	DBAR	$0x12	// StoreRelease barrier
	MOVW	R0, (R4)
	MOVW	$0, R4	// exit code
	MOVV	$SYS_exit, R11
	SYSCALL
	JMP	0(PC)

// func open(name *byte, mode, perm int32) int32
TEXT runtime·open<ABIInternal>(SB),NOSPLIT,$0
	// before:
	//    R4:  name, R5: mode, R6: perm
	// after:
	//    R4: AT_FDCWD, R5: name, R6: mode, R7: perm
	MOVW	R6, R7
	MOVW	R5, R6
	MOVV	R4, R5
	MOVW	$AT_FDCWD, R4 // AT_FDCWD, so this acts like open

	MOVV	$SYS_openat, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVW	$-1, R4
	RET

// func closefd(fd int32) int32
TEXT runtime·closefd<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_close, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVW	$-1, R4
	RET

// func write1(fd uintptr, p unsafe.Pointer, n int32) int32
TEXT runtime·write1<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_write, R11
	SYSCALL
	RET

// func read(fd int32, p unsafe.Pointer, n int32) int32
TEXT runtime·read<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_read, R11
	SYSCALL
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
TEXT runtime·usleep<ABIInternal>(SB),NOSPLIT,$16
	MOVV	$1000, R6
	MULVU	R6, R4, R4
	MOVV	$1000000000, R6

	DIVVU	R6, R4, R5	// ts->tv_sec
	REMVU	R6, R4, R8	// ts->tv_nsec
	MOVV	R5, 8(R3)
	MOVV	R8, 16(R3)

	// nanosleep(&ts, 0)
	ADDV	$8, R3, R4
	MOVV	R0, R5
	MOVV	$SYS_nanosleep, R11
	SYSCALL
	RET

// func gettid() uint32
TEXT runtime·gettid<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_gettid, R11
	SYSCALL
	RET

// func raise(sig uint32)
TEXT runtime·raise<ABIInternal>(SB),NOSPLIT,$0
	MOVW	R4, R24 // backup sig
	MOVV	$SYS_getpid, R11
	SYSCALL
	MOVW	R4, R23
	MOVV	$SYS_gettid, R11
	SYSCALL
	MOVW	R4, R5	// arg 2 tid
	MOVW	R23, R4	// arg 1 pid
	MOVW	R24, R6	// arg 3
	MOVV	$SYS_tgkill, R11
	SYSCALL
	RET

// func raiseproc(sig uint32)
TEXT runtime·raiseproc<ABIInternal>(SB),NOSPLIT,$0
	MOVW	R4, R24 // backup sig
	MOVV	$SYS_getpid, R11
	SYSCALL
	//MOVW	R4, R4	// arg 1 pid
	MOVW	R24, R5	// arg 2
	MOVV	$SYS_kill, R11
	SYSCALL
	RET

// func getpid() int
TEXT ·getpid<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_getpid, R11
	SYSCALL
	RET

// func tgkill(tgid, tid, sig int)
TEXT ·tgkill<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_tgkill, R11
	SYSCALL
	RET

// func setitimer(mode int32, new, old *itimerval)
TEXT runtime·setitimer<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_setitimer, R11
	SYSCALL
	RET

// func timer_create(clockid int32, sevp *sigevent, timerid *int32) int32
TEXT runtime·timer_create<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_timer_create, R11
	SYSCALL
	RET

// func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32
TEXT runtime·timer_settime<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_timer_settime, R11
	SYSCALL
	RET

// func timer_delete(timerid int32) int32
TEXT runtime·timer_delete<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_timer_delete, R11
	SYSCALL
	RET

// func mincore(addr unsafe.Pointer, n uintptr, dst *byte) int32
TEXT runtime·mincore<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_mincore, R11
	SYSCALL
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime<ABIInternal>(SB),NOSPLIT,$24
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
	MOVV	0(R3), R4	// sec
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

	RET

fallback:
	MOVV	$SYS_clock_gettime, R11
	SYSCALL
	JMP finish

// func nanotime1() int64
TEXT runtime·nanotime1<ABIInternal>(SB),NOSPLIT,$24
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
	ADDVU	R5, R7, R4
	RET

fallback:
	MOVV	$SYS_clock_gettime, R11
	SYSCALL
	JMP	finish

// func rtsigprocmask(how int32, new, old *sigset, size int32)
TEXT runtime·rtsigprocmask<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_rt_sigprocmask, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

// func rt_sigaction(sig uintptr, new, old *sigactiont, size uintptr) int32
TEXT runtime·rt_sigaction<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_rt_sigaction, R11
	SYSCALL
	RET

// Call the function stored in _cgo_sigaction using the GCC calling convention.
TEXT runtime·callCgoSigaction<ABIInternal>(SB),NOSPLIT,$0
	// R4: sig, R5: new, R6: old
	MOVV    _cgo_sigaction(SB), R7
	SUBV    $16, R3 // reserve 16 bytes for sp-8 where fp may be saved.
	JAL     (R7)
	ADDV    $16, R3
	MOVW    R4, R4
	RET

// func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)
TEXT runtime·sigfwd<ABIInternal>(SB),NOSPLIT,$0
	// before:
	//    R4:  fn, R5: sig, R6: info, R7: ctx
	// after:
	//    R20: fn, R4: sig, R5: info, R6: ctx
	MOVV	R4, R20
	MOVV	R5, R4
	MOVV	R6, R5
	MOVV	R7, R6
	JAL	(R20)
	RET

// Called from c-abi, R4: sig, R5: info, R6: cxt
// func sigtramp(signo, ureg, ctxt unsafe.Pointer)
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$168
	// Save callee-save registers in the case of signal forwarding.
	// Please refer to https://golang.org/issue/31827 .
	SAVE_R22_TO_R31((4*8))
	SAVE_F24_TO_F31((14*8))

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R7
	BEQ	R7, 2(PC)
	JAL	runtime·load_g(SB)

	// R5 and R6 already contain info and ctx, respectively.
	MOVV	$runtime·sigtrampgo<ABIInternal>(SB), R7
	JAL	(R7)

	// Restore callee-save registers.
	RESTORE_R22_TO_R31((4*8))
	RESTORE_F24_TO_F31((14*8))

	RET

// Called from c-abi, R4: sig, R5: info, R6: cxt
TEXT runtime·sigprofNonGoWrapper<>(SB),NOSPLIT,$168
	// Save callee-save registers because it's a callback from c code.
	SAVE_R22_TO_R31((4*8))
	SAVE_F24_TO_F31((14*8))

	// R4, R5 and R6 already contain sig, info and ctx, respectively.
	CALL	runtime·sigprofNonGo<ABIInternal>(SB)

	// Restore callee-save registers.
	RESTORE_R22_TO_R31((4*8))
	RESTORE_F24_TO_F31((14*8))
	RET

// Called from c-abi, R4: sig, R5: info, R6: cxt
TEXT runtime·cgoSigtramp(SB),NOSPLIT|NOFRAME,$0
	// The stack unwinder, presumably written in C, may not be able to
	// handle Go frame correctly. So, this function is NOFRAME, and we
	// save/restore LR manually.
	MOVV	R1, R12
	// Save R30, g because they will be clobbered,
	// we need to restore them before jump to sigtramp.
	MOVV	R30, R13
	MOVV	g, R14

	// If no traceback function, do usual sigtramp.
	MOVV	runtime·cgoTraceback(SB), R15
	BEQ	R15, sigtramp

	// If no traceback support function, which means that
	// runtime/cgo was not linked in, do usual sigtramp.
	MOVV	_cgo_callers(SB), R15
	BEQ	R15, sigtramp

	// Figure out if we are currently in a cgo call.
	// If not, just do usual sigtramp.
	CALL	runtime·load_g(SB)
	BEQ	g, sigtrampnog // g == nil

	MOVV	g_m(g), R15
	BEQ	R15, sigtramp    // g.m == nil
	MOVW	m_ncgo(R15), R16
	BEQ	R16, sigtramp    // g.m.ncgo = 0
	MOVV	m_curg(R15), R16
	BEQ	R16, sigtramp    // g.m.curg == nil
	MOVV	g_syscallsp(R16), R17
	BEQ     R17, sigtramp    // g.m.curg.syscallsp == 0
	MOVV	m_cgoCallers(R15), R8 // R8 is the fifth arg in C calling convention.
	BEQ	R8, sigtramp    // g.m.cgoCallers == nil
	MOVW	m_cgoCallersUse(R15), R16
	BNE	R16, sigtramp    // g.m.cgoCallersUse != 0

	// Jump to a function in runtime/cgo.
	// That function, written in C, will call the user's traceback
	// function with proper unwind info, and will then call back here.
	// The first three arguments, and the fifth, are already in registers.
	// Set the two remaining arguments now.
	MOVV	runtime·cgoTraceback(SB), R7
	MOVV	$runtime·sigtramp(SB), R9
	MOVV	_cgo_callers(SB), R15
	MOVV	R12, R1 // restore
	MOVV	R13, R30
	MOVV	R14, g
	JMP	(R15)

sigtramp:
	MOVV	R12, R1 // restore
	MOVV	R13, R30
	MOVV	R14, g
	JMP	runtime·sigtramp(SB)

sigtrampnog:
	// Signal arrived on a non-Go thread. If this is SIGPROF, get a
	// stack trace.
	MOVW    $27, R15 // 27 == SIGPROF
	BNE     R4, R15, sigtramp

	MOVV    $runtime·sigprofCallersUse(SB), R16
	DBAR	$0x14
cas_again:
	MOVV    $1, R15
	LL	(R16), R17
	BNE	R17, fail
	SC	R15, (R16)
	BEQ	R15, cas_again
	DBAR    $0x14

	// Jump to the traceback function in runtime/cgo.
	// It will call back to sigprofNonGo, which will ignore the
	// arguments passed in registers.
	// First three arguments to traceback function are in registers already.
	MOVV	runtime·cgoTraceback(SB), R7
	MOVV	$runtime·sigprofCallers(SB), R8
	MOVV	$runtime·sigprofNonGoWrapper<>(SB), R9
	MOVV	_cgo_callers(SB), R15
	MOVV	R12, R1 // restore
	MOVV	R13, R30
	MOVV	R14, g
	JMP	(R15)

fail:
	DBAR    $0x14
	JMP     sigtramp

// func sysMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (p unsafe.Pointer, err int)
TEXT runtime·sysMmap<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_mmap, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, ok
	SUBVU	R4, R0, R5
	MOVV	$0, R4
	RET
ok:
	MOVV	$0, R5
	RET

// Call the function stored in _cgo_mmap using the GCC calling convention.
// This must be called on the system stack.
// func callCgoMmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) uintptr
TEXT runtime·callCgoMmap<ABIInternal>(SB),NOSPLIT,$0
	MOVV	_cgo_mmap(SB), R13
	SUBV	$16, R3		// reserve 16 bytes for sp-8 where fp may be saved.
	JAL	(R13)
	ADDV	$16, R3
	MOVV	R4, R4
	RET

// func sysMunmap(addr unsafe.Pointer, n uintptr)
TEXT runtime·sysMunmap<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_munmap, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf3(R0)	// crash
	RET

// Call the function stored in _cgo_munmap using the GCC calling convention.
// This must be called on the system stack.
// func callCgoMunmap(addr unsafe.Pointer, n uintptr)
TEXT runtime·callCgoMunmap<ABIInternal>(SB),NOSPLIT,$0
	MOVV	_cgo_munmap(SB), R13
	SUBV	$16, R3		// reserve 16 bytes for sp-8 where fp may be saved.
	JAL	(R13)
	ADDV	$16, R3
	RET

// func madvise(addr unsafe.Pointer, n uintptr, flags int32)
TEXT runtime·madvise<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_madvise, R11
	SYSCALL
	RET

// func futex(addr unsafe.Pointer, op int32, val uint32, ts, addr2 unsafe.Pointer, val3 uint32) int32
TEXT runtime·futex<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_futex, R11
	SYSCALL
	RET

// int64 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone<ABIInternal>(SB),NOSPLIT,$0
	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVV	R6, R23
	MOVV	R7, R24
	MOVV	R8, R25

	MOVV	R23, -8(R5)
	MOVV	R24, -16(R5)
	MOVV	R25, -24(R5)
	MOVV	$1234, R23
	MOVV	R23, -32(R5)

	MOVV	$SYS_clone, R11
	SYSCALL

	// In parent, return.
	BEQ	R4, 2(PC)
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
TEXT runtime·sigaltstack<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_sigaltstack, R11
	SYSCALL
	MOVW	$-4096, R5
	BGEU	R5, R4, 2(PC)
	MOVV	R0, 0xf1(R0)	// crash
	RET

// func osyield()
TEXT runtime·osyield<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_sched_yield, R11
	SYSCALL
	RET

// func sched_getaffinity(pid, len uintptr, buf *uintptr) int32
TEXT runtime·sched_getaffinity<ABIInternal>(SB),NOSPLIT,$0
	MOVV	$SYS_sched_getaffinity, R11
	SYSCALL
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0<ABIInternal>(SB),NOSPLIT,$0
	// Implemented as brk(NULL).
	MOVV	$0, R4
	MOVV	$SYS_brk, R11
	SYSCALL
	RET

// unimplemented, only needed for android; declared in stubs_linux.go
TEXT runtime·access(SB),$0-20
	MOVV	R0, 2(R0)
	MOVW	R0, ret+16(FP) // for vet
	RET

// unimplemented, only needed for android; declared in stubs_linux.go
TEXT runtime·connect(SB),$0-28
	MOVV	R0, 2(R0)
	MOVW	R0, ret+24(FP) // for vet
	RET

// unimplemented, only needed for android; declared in stubs_linux.go
TEXT runtime·socket(SB),$0-20
	MOVV	R0, 2(R0)
	MOVW	R0, ret+16(FP) // for vet
	RET

// func vgetrandom1(buf *byte, length uintptr, flags uint32, state uintptr, stateSize uintptr) int
TEXT runtime·vgetrandom1<ABIInternal>(SB),NOSPLIT,$16
	MOVV	R3, R23

	MOVV	runtime·vdsoGetrandomSym(SB), R12

	MOVV	g_m(g), R24

	MOVV	m_vdsoPC(R24), R13
	MOVV	R13, 8(R3)
	MOVV	m_vdsoSP(R24), R13
	MOVV	R13, 16(R3)
	MOVV	R1, m_vdsoPC(R24)
	MOVV    $buf-8(FP), R13
	MOVV	R13, m_vdsoSP(R24)

	AND	$~15, R3

	MOVBU	runtime·iscgo(SB), R13
	BNE	R13, nosaveg
	MOVV	m_gsignal(R24), R13
	BEQ	R13, nosaveg
	BEQ	g, R13, nosaveg
	MOVV	(g_stack+stack_lo)(R13), R25
	MOVV	g, (R25)

	JAL	(R12)

	MOVV	R0, (R25)
	JMP	restore

nosaveg:
	JAL	(R12)

restore:
	MOVV	R23, R3
	MOVV	16(R3), R25
	MOVV	R25, m_vdsoSP(R24)
	MOVV	8(R3), R25
	MOVV	R25, m_vdsoPC(R24)
	NOP	R4 // Satisfy go vet, since the return value comes from the vDSO function.
	RET
