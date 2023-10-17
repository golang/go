// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME	0
#define CLOCK_MONOTONIC	1

// for EABI, as we don't support OABI
#define SYS_BASE 0x0

#define SYS_exit (SYS_BASE + 1)
#define SYS_read (SYS_BASE + 3)
#define SYS_write (SYS_BASE + 4)
#define SYS_open (SYS_BASE + 5)
#define SYS_close (SYS_BASE + 6)
#define SYS_getpid (SYS_BASE + 20)
#define SYS_kill (SYS_BASE + 37)
#define SYS_clone (SYS_BASE + 120)
#define SYS_rt_sigreturn (SYS_BASE + 173)
#define SYS_rt_sigaction (SYS_BASE + 174)
#define SYS_rt_sigprocmask (SYS_BASE + 175)
#define SYS_sigaltstack (SYS_BASE + 186)
#define SYS_mmap2 (SYS_BASE + 192)
#define SYS_futex (SYS_BASE + 240)
#define SYS_exit_group (SYS_BASE + 248)
#define SYS_munmap (SYS_BASE + 91)
#define SYS_madvise (SYS_BASE + 220)
#define SYS_setitimer (SYS_BASE + 104)
#define SYS_mincore (SYS_BASE + 219)
#define SYS_gettid (SYS_BASE + 224)
#define SYS_tgkill (SYS_BASE + 268)
#define SYS_sched_yield (SYS_BASE + 158)
#define SYS_nanosleep (SYS_BASE + 162)
#define SYS_sched_getaffinity (SYS_BASE + 242)
#define SYS_clock_gettime (SYS_BASE + 263)
#define SYS_timer_create (SYS_BASE + 257)
#define SYS_timer_settime (SYS_BASE + 258)
#define SYS_timer_delete (SYS_BASE + 261)
#define SYS_pipe2 (SYS_BASE + 359)
#define SYS_access (SYS_BASE + 33)
#define SYS_connect (SYS_BASE + 283)
#define SYS_socket (SYS_BASE + 281)
#define SYS_brk (SYS_BASE + 45)

#define ARM_BASE (SYS_BASE + 0x0f0000)

TEXT runtime·open(SB),NOSPLIT,$0
	MOVW	name+0(FP), R0
	MOVW	mode+4(FP), R1
	MOVW	perm+8(FP), R2
	MOVW	$SYS_open, R7
	SWI	$0
	MOVW	$0xfffff001, R1
	CMP	R1, R0
	MOVW.HI	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	$SYS_close, R7
	SWI	$0
	MOVW	$0xfffff001, R1
	CMP	R1, R0
	MOVW.HI	$-1, R0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	MOVW	$SYS_write, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	MOVW	$SYS_read, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$0-16
	MOVW	$r+4(FP), R0
	MOVW	flags+0(FP), R1
	MOVW	$SYS_pipe2, R7
	SWI	$0
	MOVW	R0, errno+12(FP)
	RET

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0
	MOVW	code+0(FP), R0
	MOVW	$SYS_exit_group, R7
	SWI	$0
	MOVW	$1234, R0
	MOVW	$1002, R1
	MOVW	R0, (R1)	// fail hard

TEXT exit1<>(SB),NOSPLIT|NOFRAME,$0
	MOVW	code+0(FP), R0
	MOVW	$SYS_exit, R7
	SWI	$0
	MOVW	$1234, R0
	MOVW	$1003, R1
	MOVW	R0, (R1)	// fail hard

// func exitThread(wait *atomic.Uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	wait+0(FP), R0
	// We're done using the stack.
	// Alas, there's no reliable way to make this write atomic
	// without potentially using the stack. So it goes.
	MOVW	$0, R1
	MOVW	R1, (R0)
	MOVW	$0, R0	// exit code
	MOVW	$SYS_exit, R7
	SWI	$0
	MOVW	$1234, R0
	MOVW	$1004, R1
	MOVW	R0, (R1)	// fail hard
	JMP	0(PC)

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVW	$SYS_gettid, R7
	SWI	$0
	MOVW	R0, ret+0(FP)
	RET

TEXT	runtime·raise(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_getpid, R7
	SWI	$0
	MOVW	R0, R4
	MOVW	$SYS_gettid, R7
	SWI	$0
	MOVW	R0, R1	// arg 2 tid
	MOVW	R4, R0	// arg 1 pid
	MOVW	sig+0(FP), R2	// arg 3
	MOVW	$SYS_tgkill, R7
	SWI	$0
	RET

TEXT	runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_getpid, R7
	SWI	$0
	// arg 1 tid already in R0 from getpid
	MOVW	sig+0(FP), R1	// arg 2 - signal
	MOVW	$SYS_kill, R7
	SWI	$0
	RET

TEXT ·getpid(SB),NOSPLIT,$0-4
	MOVW	$SYS_getpid, R7
	SWI	$0
	MOVW	R0, ret+0(FP)
	RET

TEXT ·tgkill(SB),NOSPLIT,$0-12
	MOVW	tgid+0(FP), R0
	MOVW	tid+4(FP), R1
	MOVW	sig+8(FP), R2
	MOVW	$SYS_tgkill, R7
	SWI	$0
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	prot+8(FP), R2
	MOVW	flags+12(FP), R3
	MOVW	fd+16(FP), R4
	MOVW	off+20(FP), R5
	MOVW	$SYS_mmap2, R7
	SWI	$0
	MOVW	$0xfffff001, R6
	CMP		R6, R0
	MOVW	$0, R1
	RSB.HI	$0, R0
	MOVW.HI	R0, R1		// if error, put in R1
	MOVW.HI	$0, R0
	MOVW	R0, p+24(FP)
	MOVW	R1, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	$SYS_munmap, R7
	SWI	$0
	MOVW	$0xfffff001, R6
	CMP 	R6, R0
	MOVW.HI	$0, R8  // crash on syscall failure
	MOVW.HI	R8, (R8)
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	flags+8(FP), R2
	MOVW	$SYS_madvise, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	$SYS_setitimer, R7
	SWI	$0
	RET

TEXT runtime·timer_create(SB),NOSPLIT,$0-16
	MOVW	clockid+0(FP), R0
	MOVW	sevp+4(FP), R1
	MOVW	timerid+8(FP), R2
	MOVW	$SYS_timer_create, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·timer_settime(SB),NOSPLIT,$0-20
	MOVW	timerid+0(FP), R0
	MOVW	flags+4(FP), R1
	MOVW	new+8(FP), R2
	MOVW	old+12(FP), R3
	MOVW	$SYS_timer_settime, R7
	SWI	$0
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·timer_delete(SB),NOSPLIT,$0-8
	MOVW	timerid+0(FP), R0
	MOVW	$SYS_timer_delete, R7
	SWI	$0
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·mincore(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	dst+8(FP), R2
	MOVW	$SYS_mincore, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

// Call a VDSO function.
//
// R0-R3: arguments to VDSO function (C calling convention)
// R4: uintptr function to call
//
// There is no return value.
TEXT runtime·vdsoCall(SB),NOSPLIT,$8-0
	// R0-R3 may be arguments to fn, do not touch.
	// R4 is function to call.
	// R5-R9 are available as locals. They are unchanged by the C call
	// (callee-save).

	// We don't know how much stack space the VDSO code will need,
	// so switch to g0.

	// Save old SP. Use R13 instead of SP to avoid linker rewriting the offsets.
	MOVW	R13, R5

	MOVW	g_m(g), R6

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVW	m_vdsoPC(R6), R7
	MOVW	m_vdsoSP(R6), R8
	MOVW	R7, 4(R13)
	MOVW	R8, 8(R13)

	MOVW	$sp-4(FP), R7 // caller's SP
	MOVW	LR, m_vdsoPC(R6)
	MOVW	R7, m_vdsoSP(R6)

	MOVW	m_curg(R6), R7

	CMP	g, R7		// Only switch if on curg.
	B.NE	noswitch

	MOVW	m_g0(R6), R7
	MOVW	(g_sched+gobuf_sp)(R7), R13	 // Set SP to g0 stack

noswitch:
	BIC	$0x7, R13	// Align for C code

	// Store g on gsignal's stack, so if we receive a signal
	// during VDSO code we can find the g.

	// When using cgo, we already saved g on TLS, also don't save g here.
	MOVB	runtime·iscgo(SB), R7
	CMP	$0, R7
	BNE	nosaveg
	// If we don't have a signal stack, we won't receive signal, so don't
	// bother saving g.
	MOVW	m_gsignal(R6), R7          // g.m.gsignal
	CMP	$0, R7
	BEQ	nosaveg
	// Don't save g if we are already on the signal stack, as we won't get
	// a nested signal.
	CMP	g, R7
	BEQ	nosaveg
	// If we don't have a signal stack, we won't receive signal, so don't
	// bother saving g.
	MOVW	(g_stack+stack_lo)(R7), R7 // g.m.gsignal.stack.lo
	CMP	$0, R7
	BEQ	nosaveg
	MOVW	g, (R7)

	BL	(R4)

	MOVW	$0, R8
	MOVW	R8, (R7) // clear g slot

	JMP	finish

nosaveg:
	BL	(R4)

finish:
	MOVW	R5, R13		// Restore real SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVW	8(R13), R7
	MOVW	R7, m_vdsoSP(R6)
	MOVW	4(R13), R7
	MOVW	R7, m_vdsoPC(R6)
	RET

TEXT runtime·walltime(SB),NOSPLIT,$12-12
	MOVW	$CLOCK_REALTIME, R0
	MOVW	$spec-12(SP), R1	// timespec

	MOVW	runtime·vdsoClockgettimeSym(SB), R4
	CMP	$0, R4
	B.EQ	fallback

	BL	runtime·vdsoCall(SB)

	JMP	finish

fallback:
	MOVW	$SYS_clock_gettime, R7
	SWI	$0

finish:
	MOVW	sec-12(SP), R0  // sec
	MOVW	nsec-8(SP), R2  // nsec

	MOVW	R0, sec_lo+0(FP)
	MOVW	$0, R1
	MOVW	R1, sec_hi+4(FP)
	MOVW	R2, nsec+8(FP)
	RET

// func nanotime1() int64
TEXT runtime·nanotime1(SB),NOSPLIT,$12-8
	MOVW	$CLOCK_MONOTONIC, R0
	MOVW	$spec-12(SP), R1	// timespec

	MOVW	runtime·vdsoClockgettimeSym(SB), R4
	CMP	$0, R4
	B.EQ	fallback

	BL	runtime·vdsoCall(SB)

	JMP	finish

fallback:
	MOVW	$SYS_clock_gettime, R7
	SWI	$0

finish:
	MOVW	sec-12(SP), R0  // sec
	MOVW	nsec-8(SP), R2  // nsec

	MOVW	$1000000000, R3
	MULLU	R0, R3, (R1, R0)
	ADD.S	R2, R0
	ADC	$0, R1	// Add carry bit to upper half.

	MOVW	R0, ret_lo+0(FP)
	MOVW	R1, ret_hi+4(FP)

	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$0
	MOVW    addr+0(FP), R0
	MOVW    op+4(FP), R1
	MOVW    val+8(FP), R2
	MOVW    ts+12(FP), R3
	MOVW    addr2+16(FP), R4
	MOVW    val3+20(FP), R5
	MOVW	$SYS_futex, R7
	SWI	$0
	MOVW	R0, ret+24(FP)
	RET

// int32 clone(int32 flags, void *stack, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT,$0
	MOVW	flags+0(FP), R0
	MOVW	stk+4(FP), R1
	MOVW	$0, R2	// parent tid ptr
	MOVW	$0, R3	// tls_val
	MOVW	$0, R4	// child tid ptr
	MOVW	$0, R5

	// Copy mp, gp, fn off parent stack for use by child.
	MOVW	$-16(R1), R1
	MOVW	mp+8(FP), R6
	MOVW	R6, 0(R1)
	MOVW	gp+12(FP), R6
	MOVW	R6, 4(R1)
	MOVW	fn+16(FP), R6
	MOVW	R6, 8(R1)
	MOVW	$1234, R6
	MOVW	R6, 12(R1)

	MOVW	$SYS_clone, R7
	SWI	$0

	// In parent, return.
	CMP	$0, R0
	BEQ	3(PC)
	MOVW	R0, ret+20(FP)
	RET

	// Paranoia: check that SP is as we expect. Use R13 to avoid linker 'fixup'
	NOP	R13	// tell vet SP/R13 changed - stop checking offsets
	MOVW	12(R13), R0
	MOVW	$1234, R1
	CMP	R0, R1
	BEQ	2(PC)
	BL	runtime·abort(SB)

	MOVW	0(R13), R8    // m
	MOVW	4(R13), R0    // g

	CMP	$0, R8
	BEQ	nog
	CMP	$0, R0
	BEQ	nog

	MOVW	R0, g
	MOVW	R8, g_m(g)

	// paranoia; check they are not nil
	MOVW	0(R8), R0
	MOVW	0(g), R0

	BL	runtime·emptyfunc(SB)	// fault if stack check is wrong

	// Initialize m->procid to Linux tid
	MOVW	$SYS_gettid, R7
	SWI	$0
	MOVW	g_m(g), R8
	MOVW	R0, m_procid(R8)

nog:
	// Call fn
	MOVW	8(R13), R0
	MOVW	$16(R13), R13
	BL	(R0)

	// It shouldn't return. If it does, exit that thread.
	SUB	$16, R13 // restore the stack pointer to avoid memory corruption
	MOVW	$0, R0
	MOVW	R0, 4(R13)
	BL	exit1<>(SB)

	MOVW	$1234, R0
	MOVW	$1005, R1
	MOVW	R0, (R1)

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVW	new+0(FP), R0
	MOVW	old+4(FP), R1
	MOVW	$SYS_sigaltstack, R7
	SWI	$0
	MOVW	$0xfffff001, R6
	CMP 	R6, R0
	MOVW.HI	$0, R8  // crash on syscall failure
	MOVW.HI	R8, (R8)
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

TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$0
	// Reserve space for callee-save registers and arguments.
	MOVM.DB.W [R4-R11], (R13)
	SUB	$16, R13

	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVW	R0, 4(R13)
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	BL.NE	runtime·load_g(SB)

	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	MOVW  	$runtime·sigtrampgo(SB), R11
	BL	(R11)

	// Restore callee-save registers.
	ADD	$16, R13
	MOVM.IA.W (R13), [R4-R11]

	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	MOVW  	$runtime·sigtramp(SB), R11
	B	(R11)

TEXT runtime·rtsigprocmask(SB),NOSPLIT,$0
	MOVW	how+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	size+12(FP), R3
	MOVW	$SYS_rt_sigprocmask, R7
	SWI	$0
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT,$0
	MOVW	sig+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	size+12(FP), R3
	MOVW	$SYS_rt_sigaction, R7
	SWI	$0
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$12
	MOVW	usec+0(FP), R0
	CALL	runtime·usplitR0(SB)
	MOVW	R0, 4(R13)
	MOVW	$1000, R0	// usec to nsec
	MUL	R0, R1
	MOVW	R1, 8(R13)
	MOVW	$4(R13), R0
	MOVW	$0, R1
	MOVW	$SYS_nanosleep, R7
	SWI	$0
	RET

// As for cas, memory barriers are complicated on ARM, but the kernel
// provides a user helper. ARMv5 does not support SMP and has no
// memory barrier instruction at all. ARMv6 added SMP support and has
// a memory barrier, but it requires writing to a coprocessor
// register. ARMv7 introduced the DMB instruction, but it's expensive
// even on single-core devices. The kernel helper takes care of all of
// this for us.

TEXT kernelPublicationBarrier<>(SB),NOSPLIT,$0
	// void __kuser_memory_barrier(void);
	MOVW	$0xffff0fa0, R11
	CALL	(R11)
	RET

TEXT ·publicationBarrier(SB),NOSPLIT,$0
	MOVB	·goarm(SB), R11
	CMP	$7, R11
	BLT	2(PC)
	JMP	·armPublicationBarrier(SB)
	JMP	kernelPublicationBarrier<>(SB) // extra layer so this function is leaf and no SP adjustment on GOARM=7

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVW	$SYS_sched_yield, R7
	SWI	$0
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$0
	MOVW	pid+0(FP), R0
	MOVW	len+4(FP), R1
	MOVW	buf+8(FP), R2
	MOVW	$SYS_sched_getaffinity, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

// b __kuser_get_tls @ 0xffff0fe0
TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	MOVW	$0xffff0fe0, R0
	B	(R0)

TEXT runtime·access(SB),NOSPLIT,$0
	MOVW	name+0(FP), R0
	MOVW	mode+4(FP), R1
	MOVW	$SYS_access, R7
	SWI	$0
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·connect(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	addr+4(FP), R1
	MOVW	len+8(FP), R2
	MOVW	$SYS_connect, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·socket(SB),NOSPLIT,$0
	MOVW	domain+0(FP), R0
	MOVW	typ+4(FP), R1
	MOVW	prot+8(FP), R2
	MOVW	$SYS_socket, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT,$0-4
	// Implemented as brk(NULL).
	MOVW	$0, R0
	MOVW	$SYS_brk, R7
	SWI	$0
	MOVW	R0, ret+0(FP)
	RET
