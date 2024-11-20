// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

//
// System calls and other sys.stuff for ppc64, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "asm_ppc64x.h"
#include "cgo/abi_ppc64x.h"

#define SYS_exit		  1
#define SYS_read		  3
#define SYS_write		  4
#define SYS_open		  5
#define SYS_close		  6
#define SYS_getpid		 20
#define SYS_kill		 37
#define SYS_brk			 45
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
#define SYS_timer_create	240
#define SYS_timer_settime	241
#define SYS_timer_delete	244
#define SYS_clock_gettime	246
#define SYS_tgkill		250
#define SYS_pipe2		317

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R3
	SYSCALL	$SYS_exit_group
	RET

// func exitThread(wait *atomic.Uint32)
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

TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	SYSCALL	$SYS_write
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R3
	MOVD	p+8(FP), R4
	MOVW	n+16(FP), R5
	SYSCALL	$SYS_read
	BVC	2(PC)
	NEG	R3	// caller expects negative errno
	MOVW	R3, ret+24(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	ADD	$FIXED_FRAME+8, R1, R3
	MOVW	flags+0(FP), R4
	SYSCALL	$SYS_pipe2
	MOVW	R3, errno+16(FP)
	RET

// func usleep(usec uint32)
TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVW	usec+0(FP), R3

	// Use magic constant 0x8637bd06 and shift right 51
	// to perform usec/1000000.
	MOVD	$0x8637bd06, R4
	MULLD	R3, R4, R4	// Convert usec to S.
	SRD	$51, R4, R4
	MOVD	R4, 8(R1)	// Store to tv_sec

	MOVD	$1000000, R5
	MULLW	R4, R5, R5	// Convert tv_sec back into uS
	SUB	R5, R3, R5	// Compute remainder uS.
	MULLD	$1000, R5, R5	// Convert to nsec
	MOVD	R5, 16(R1)	// Store to tv_nsec

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

TEXT ·getpid(SB),NOSPLIT|NOFRAME,$0-8
	SYSCALL $SYS_getpid
	MOVD	R3, ret+0(FP)
	RET

TEXT ·tgkill(SB),NOSPLIT|NOFRAME,$0-24
	MOVD	tgid+0(FP), R3
	MOVD	tid+8(FP), R4
	MOVD	sig+16(FP), R5
	SYSCALL $SYS_tgkill
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	SYSCALL	$SYS_setitimer
	RET

TEXT runtime·timer_create(SB),NOSPLIT,$0-28
	MOVW	clockid+0(FP), R3
	MOVD	sevp+8(FP), R4
	MOVD	timerid+16(FP), R5
	SYSCALL	$SYS_timer_create
	MOVW	R3, ret+24(FP)
	RET

TEXT runtime·timer_settime(SB),NOSPLIT,$0-28
	MOVW	timerid+0(FP), R3
	MOVW	flags+4(FP), R4
	MOVD	new+8(FP), R5
	MOVD	old+16(FP), R6
	SYSCALL	$SYS_timer_settime
	MOVW	R3, ret+24(FP)
	RET

TEXT runtime·timer_delete(SB),NOSPLIT,$0-12
	MOVW	timerid+0(FP), R3
	SYSCALL	$SYS_timer_delete
	MOVW	R3, ret+8(FP)
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
TEXT runtime·walltime(SB),NOSPLIT,$16-12
	MOVD	R1, R15		// R15 is unchanged by C code
	MOVD	g_m(g), R21	// R21 = m

	MOVD	$0, R3		// CLOCK_REALTIME

	MOVD	runtime·vdsoClockgettimeSym(SB), R12	// Check for VDSO availability
	CMP	R12, $0
	BEQ	fallback

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVD	m_vdsoPC(R21), R4
	MOVD	m_vdsoSP(R21), R5
	MOVD	R4, 32(R1)
	MOVD	R5, 40(R1)

	MOVD	LR, R14
	MOVD	$ret-FIXED_FRAME(FP), R5 // caller's SP
	MOVD	R14, m_vdsoPC(R21)
	MOVD	R5, m_vdsoSP(R21)

	MOVD	m_curg(R21), R6
	CMP	g, R6
	BNE	noswitch

	MOVD	m_g0(R21), R7
	MOVD	(g_sched+gobuf_sp)(R7), R1	// Set SP to g0 stack

noswitch:
	SUB	$16, R1                 // Space for results
	RLDICR	$0, R1, $59, R1         // Align for C code
	MOVD	R12, CTR
	MOVD	R1, R4

	// Store g on gsignal's stack, so if we receive a signal
	// during VDSO code we can find the g.
	// If we don't have a signal stack, we won't receive signal,
	// so don't bother saving g.
	// When using cgo, we already saved g on TLS, also don't save
	// g here.
	// Also don't save g if we are already on the signal stack.
	// We won't get a nested signal.
	MOVBZ	runtime·iscgo(SB), R22
	CMP	R22, $0
	BNE	nosaveg
	MOVD	m_gsignal(R21), R22	// g.m.gsignal
	CMP	R22, $0
	BEQ	nosaveg

	CMP	g, R22
	BEQ	nosaveg
	MOVD	(g_stack+stack_lo)(R22), R22 // g.m.gsignal.stack.lo
	MOVD	g, (R22)

	BL	(CTR)	// Call from VDSO

	MOVD	$0, (R22)	// clear g slot, R22 is unchanged by C code

	JMP	finish

nosaveg:
	BL	(CTR)	// Call from VDSO

finish:
	MOVD	$0, R0		// Restore R0
	MOVD	0(R1), R3	// sec
	MOVD	8(R1), R5	// nsec
	MOVD	R15, R1		// Restore SP

	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVD	40(R1), R6
	MOVD	R6, m_vdsoSP(R21)
	MOVD	32(R1), R6
	MOVD	R6, m_vdsoPC(R21)

return:
	MOVD	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

	// Syscall fallback
fallback:
	ADD	$32, R1, R4
	SYSCALL $SYS_clock_gettime
	MOVD	32(R1), R3
	MOVD	40(R1), R5
	JMP	return

TEXT runtime·nanotime1(SB),NOSPLIT,$16-8
	MOVD	$1, R3		// CLOCK_MONOTONIC

	MOVD	R1, R15		// R15 is unchanged by C code
	MOVD	g_m(g), R21	// R21 = m

	MOVD	runtime·vdsoClockgettimeSym(SB), R12	// Check for VDSO availability
	CMP	R12, $0
	BEQ	fallback

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVD	m_vdsoPC(R21), R4
	MOVD	m_vdsoSP(R21), R5
	MOVD	R4, 32(R1)
	MOVD	R5, 40(R1)

	MOVD	LR, R14				// R14 is unchanged by C code
	MOVD	$ret-FIXED_FRAME(FP), R5	// caller's SP
	MOVD	R14, m_vdsoPC(R21)
	MOVD	R5, m_vdsoSP(R21)

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

	// Store g on gsignal's stack, so if we receive a signal
	// during VDSO code we can find the g.
	// If we don't have a signal stack, we won't receive signal,
	// so don't bother saving g.
	// When using cgo, we already saved g on TLS, also don't save
	// g here.
	// Also don't save g if we are already on the signal stack.
	// We won't get a nested signal.
	MOVBZ	runtime·iscgo(SB), R22
	CMP	R22, $0
	BNE	nosaveg
	MOVD	m_gsignal(R21), R22	// g.m.gsignal
	CMP	R22, $0
	BEQ	nosaveg

	CMP	g, R22
	BEQ	nosaveg
	MOVD	(g_stack+stack_lo)(R22), R22 // g.m.gsignal.stack.lo
	MOVD	g, (R22)

	BL	(CTR)	// Call from VDSO

	MOVD	$0, (R22)	// clear g slot, R22 is unchanged by C code

	JMP	finish

nosaveg:
	BL	(CTR)	// Call from VDSO

finish:
	MOVD	$0, R0			// Restore R0
	MOVD	0(R1), R3		// sec
	MOVD	8(R1), R5		// nsec
	MOVD	R15, R1			// Restore SP

	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVD	40(R1), R6
	MOVD	R6, m_vdsoSP(R21)
	MOVD	32(R1), R6
	MOVD	R6, m_vdsoPC(R21)

return:
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
	MOVD	40(R1), R5
	JMP	return

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

#ifdef GOARCH_ppc64le
// Call the function stored in _cgo_sigaction using the GCC calling convention.
TEXT runtime·callCgoSigaction(SB),NOSPLIT,$0
	MOVD    sig+0(FP), R3
	MOVD    new+8(FP), R4
	MOVD    old+16(FP), R5
	MOVD     _cgo_sigaction(SB), R12
	MOVD    R12, CTR                // R12 should contain the function address
	MOVD    R1, R15                 // Save R1
	MOVD    R2, 24(R1)              // Save R2
	SUB     $48, R1                 // reserve 32 (frame) + 16 bytes for sp-8 where fp may be saved.
	RLDICR  $0, R1, $59, R1         // Align to 16 bytes for C code
	BL      (CTR)
	XOR     R0, R0, R0              // Clear R0 as Go expects
	MOVD    R15, R1                 // Restore R1
	MOVD    24(R1), R2              // Restore R2
	MOVW    R3, ret+24(FP)          // Return result
	RET
#endif

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R3
	MOVD	info+16(FP), R4
	MOVD	ctx+24(FP), R5
	MOVD	fn+0(FP), R12
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	24(R1), R2
	RET

#ifdef GO_PPC64X_HAS_FUNCDESC
DEFINE_PPC64X_FUNCDESC(runtime·sigtramp, sigtramp<>)
// cgo isn't supported on ppc64, but we need to supply a cgoSigTramp function.
DEFINE_PPC64X_FUNCDESC(runtime·cgoSigtramp, sigtramp<>)
TEXT sigtramp<>(SB),NOSPLIT|NOFRAME|TOPFRAME,$0
#else
// ppc64le doesn't need function descriptors
// Save callee-save registers in the case of signal forwarding.
// Same as on ARM64 https://golang.org/issue/31827 .
//
// Note, it is assumed this is always called indirectly (e.g via
// a function pointer) as R2 may not be preserved when calling this
// function. In those cases, the caller preserves their R2.
TEXT runtime·sigtramp(SB),NOSPLIT|NOFRAME,$0
#endif
	// This is called with ELF calling conventions. Convert to Go.
	// Allocate space for argument storage to call runtime.sigtrampgo.
	STACK_AND_SAVE_HOST_TO_GO_ABI(32)

	// this might be called in external code context,
	// where g is not set.
	MOVBZ	runtime·iscgo(SB), R6
	CMP	R6, $0
	BEQ	2(PC)
	BL	runtime·load_g(SB)

	// R3,R4,R5 already hold the arguments. Forward them on.
	// TODO: Indirectly call runtime.sigtrampgo to avoid the linker's static NOSPLIT stack
	// overflow detection. It thinks this might be called on a small Go stack, but this is only
	// called from a larger pthread or sigaltstack stack. Can the checker be improved to not
	// flag a direct call here?
	MOVD	$runtime·sigtrampgo<ABIInternal>(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
	// Restore R2 (TOC pointer) in the event it might be used later in this function.
	// If this was not compiled as shared code, R2 is undefined, reloading it is harmless.
	MOVD	24(R1), R2

	UNSTACK_AND_RESTORE_GO_TO_HOST_ABI(32)
	RET

#ifdef GOARCH_ppc64le
TEXT runtime·cgoSigtramp(SB),NOSPLIT|NOFRAME,$0
	// The stack unwinder, presumably written in C, may not be able to
	// handle Go frame correctly. So, this function is NOFRAME, and we
	// save/restore LR manually, and obey ELFv2 calling conventions.
	MOVD	LR, R10

	// We're coming from C code, initialize R0
	MOVD	$0, R0

	// If no traceback function, do usual sigtramp.
	MOVD	runtime·cgoTraceback(SB), R6
	CMP	$0, R6
	BEQ	sigtramp

	// If no traceback support function, which means that
	// runtime/cgo was not linked in, do usual sigtramp.
	MOVD	_cgo_callers(SB), R6
	CMP	$0, R6
	BEQ	sigtramp

	// Inspect the g in TLS without clobbering R30/R31 via runtime.load_g.
	MOVD	runtime·tls_g(SB), R9
	MOVD	0(R9), R9

	// Figure out if we are currently in a cgo call.
	// If not, just do usual sigtramp.
	// compared to ARM64 and others.
	CMP	$0, R9
	BEQ	sigtrampnog // g == nil

	// g is not nil. Check further.
	MOVD	g_m(R9), R6
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
#endif

// Used by cgoSigtramp to inspect without clobbering R30/R31 via runtime.load_g.
GLOBL runtime·tls_g+0(SB), TLSBSS+DUPOK, $8

TEXT runtime·sigprofNonGoWrapper<>(SB),NOSPLIT|NOFRAME,$0
	// This is called from C code. Callee save registers must be saved.
	// R3,R4,R5 hold arguments, and allocate argument space to call sigprofNonGo.
	STACK_AND_SAVE_HOST_TO_GO_ABI(32)

	CALL	runtime·sigprofNonGo<ABIInternal>(SB)

	UNSTACK_AND_RESTORE_GO_TO_HOST_ABI(32)
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

// func vgetrandom1(buf *byte, length uintptr, flags uint32, state uintptr, stateSize uintptr) int
TEXT runtime·vgetrandom1<ABIInternal>(SB),NOSPLIT,$16-48
	MOVD	R1, R15

	MOVD	runtime·vdsoGetrandomSym(SB), R12
	MOVD	R12, CTR
	MOVD	g_m(g), R21

	MOVD	m_vdsoPC(R21), R22
	MOVD	R22, 32(R1)
	MOVD	m_vdsoSP(R21), R22
	MOVD	R22, 40(R1)
	MOVD	LR, m_vdsoPC(R21)
	MOVD	$buf-FIXED_FRAME(FP), R22
	MOVD	R22, m_vdsoSP(R21)

	RLDICR  $0, R1, $59, R1

	MOVBZ	runtime·iscgo(SB), R22
	CMP	R22, $0
	BNE	nosaveg
	MOVD	m_gsignal(R21), R22
	CMP	R22, $0
	BEQ	nosaveg
	CMP	R22, g
	BEQ	nosaveg
	MOVD	(g_stack+stack_lo)(R22), R22
	MOVD	g, (R22)

	BL	(CTR)

	MOVD	$0, (R22)
	JMP	restore

nosaveg:
	BL	(CTR)

restore:
	MOVD	$0, R0
	MOVD	R15, R1
	MOVD	40(R1), R22
	MOVD	R22, m_vdsoSP(R21)
	MOVD	32(R1), R22
	MOVD	R22, m_vdsoPC(R21)

	BVC	out
	NEG	R3, R3
out:
	RET
