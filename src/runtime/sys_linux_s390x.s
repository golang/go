// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other system stuff for Linux s390x; see
// /usr/include/asm/unistd.h for the syscall number definitions.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define SYS_exit                  1
#define SYS_read                  3
#define SYS_write                 4
#define SYS_open                  5
#define SYS_close                 6
#define SYS_getpid               20
#define SYS_kill                 37
#define SYS_brk			 45
#define SYS_mmap                 90
#define SYS_munmap               91
#define SYS_setitimer           104
#define SYS_clone               120
#define SYS_sched_yield         158
#define SYS_nanosleep           162
#define SYS_rt_sigreturn        173
#define SYS_rt_sigaction        174
#define SYS_rt_sigprocmask      175
#define SYS_sigaltstack         186
#define SYS_madvise             219
#define SYS_mincore             218
#define SYS_gettid              236
#define SYS_futex               238
#define SYS_sched_getaffinity   240
#define SYS_tgkill              241
#define SYS_exit_group          248
#define SYS_timer_create        254
#define SYS_timer_settime       255
#define SYS_timer_delete        258
#define SYS_clock_gettime       260
#define SYS_pipe2		325

TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), R2
	MOVW	$SYS_exit_group, R1
	SYSCALL
	RET

// func exitThread(wait *atomic.Uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-8
	MOVD	wait+0(FP), R1
	// We're done using the stack.
	MOVW	$0, R2
	MOVW	R2, (R1)
	MOVW	$0, R2	// exit code
	MOVW	$SYS_exit, R1
	SYSCALL
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	name+0(FP), R2
	MOVW	mode+8(FP), R3
	MOVW	perm+12(FP), R4
	MOVW	$SYS_open, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), R2
	MOVW	$SYS_close, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVW	$-1, R2
	MOVW	R2, ret+8(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	fd+0(FP), R2
	MOVD	p+8(FP), R3
	MOVW	n+16(FP), R4
	MOVW	$SYS_write, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), R2
	MOVD	p+8(FP), R3
	MOVW	n+16(FP), R4
	MOVW	$SYS_read, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// func pipe2() (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	MOVD	$r+8(FP), R2
	MOVW	flags+0(FP), R3
	MOVW	$SYS_pipe2, R1
	SYSCALL
	MOVW	R2, errno+16(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16-4
	MOVW	usec+0(FP), R2
	MOVD	R2, R4
	MOVW	$1000000, R3
	DIVD	R3, R2
	MOVD	R2, 8(R15)
	MULLD	R2, R3		// Convert sec to usec and subtract
	SUB	R3, R4
	MOVW	$1000, R3
	MULLD	R3, R4		// Convert remaining usec into nsec.
	MOVD	R4, 16(R15)

	// nanosleep(&ts, 0)
	ADD	$8, R15, R2
	MOVW	$0, R3
	MOVW	$SYS_nanosleep, R1
	SYSCALL
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVW	$SYS_gettid, R1
	SYSCALL
	MOVW	R2, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_getpid, R1
	SYSCALL
	MOVW	R2, R10
	MOVW	$SYS_gettid, R1
	SYSCALL
	MOVW	R2, R3	// arg 2 tid
	MOVW	R10, R2	// arg 1 pid
	MOVW	sig+0(FP), R4	// arg 2
	MOVW	$SYS_tgkill, R1
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_getpid, R1
	SYSCALL
	MOVW	R2, R2	// arg 1 pid
	MOVW	sig+0(FP), R3	// arg 2
	MOVW	$SYS_kill, R1
	SYSCALL
	RET

TEXT ·getpid(SB),NOSPLIT|NOFRAME,$0-8
	MOVW	$SYS_getpid, R1
	SYSCALL
	MOVD	R2, ret+0(FP)
	RET

TEXT ·tgkill(SB),NOSPLIT|NOFRAME,$0-24
	MOVD	tgid+0(FP), R2
	MOVD	tid+8(FP), R3
	MOVD	sig+16(FP), R4
	MOVW	$SYS_tgkill, R1
	SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), R2
	MOVD	new+8(FP), R3
	MOVD	old+16(FP), R4
	MOVW	$SYS_setitimer, R1
	SYSCALL
	RET

TEXT runtime·timer_create(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	clockid+0(FP), R2
	MOVD	sevp+8(FP), R3
	MOVD	timerid+16(FP), R4
	MOVW	$SYS_timer_create, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·timer_settime(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	timerid+0(FP), R2
	MOVW	flags+4(FP), R3
	MOVD	new+8(FP), R4
	MOVD	old+16(FP), R5
	MOVW	$SYS_timer_settime, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

TEXT runtime·timer_delete(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	timerid+0(FP), R2
	MOVW	$SYS_timer_delete, R1
	SYSCALL
	MOVW	R2, ret+8(FP)
	RET

TEXT runtime·mincore(SB),NOSPLIT|NOFRAME,$0-28
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVD	dst+16(FP), R4
	MOVW	$SYS_mincore, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB),NOSPLIT,$32-12
	MOVW	$0, R2			// CLOCK_REALTIME
	MOVD	R15, R7			// Backup stack pointer

	MOVD	g_m(g), R6		//m

	MOVD	runtime·vdsoClockgettimeSym(SB), R9	// Check for VDSO availability
	CMPBEQ	R9, $0, fallback

	MOVD	m_vdsoPC(R6), R4
	MOVD	R4, 16(R15)
	MOVD	m_vdsoSP(R6), R4
	MOVD	R4, 24(R15)

	MOVD	R14, R8 		// Backup return address
	MOVD	$sec+0(FP), R4 	// return parameter caller

	MOVD	R8, m_vdsoPC(R6)
	MOVD	R4, m_vdsoSP(R6)

	MOVD	m_curg(R6), R5
	CMP		g, R5
	BNE		noswitch

	MOVD	m_g0(R6), R4
	MOVD	(g_sched+gobuf_sp)(R4), R15	// Set SP to g0 stack

noswitch:
	SUB		$16, R15		// reserve 2x 8 bytes for parameters
	MOVD	$~7, R4			// align to 8 bytes because of gcc ABI
	AND		R4, R15
	MOVD	R15, R3			// R15 needs to be in R3 as expected by kernel_clock_gettime

	MOVB	runtime·iscgo(SB),R12
	CMPBNE	R12, $0, nosaveg

	MOVD	m_gsignal(R6), R12	// g.m.gsignal
	CMPBEQ	R12, $0, nosaveg

	CMPBEQ	g, R12, nosaveg
	MOVD	(g_stack+stack_lo)(R12), R12 // g.m.gsignal.stack.lo
	MOVD	g, (R12)

	BL	R9 // to vdso lookup

	MOVD	$0, (R12)

	JMP	finish

nosaveg:
	BL	R9					// to vdso lookup

finish:
	MOVD	0(R15), R3		// sec
	MOVD	8(R15), R5		// nsec
	MOVD	R7, R15			// Restore SP

	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVD	24(R15), R12
	MOVD	R12, m_vdsoSP(R6)
	MOVD	16(R15), R12
	MOVD	R12, m_vdsoPC(R6)

return:
	// sec is in R3, nsec in R5
	// return nsec in R3
	MOVD	R3, sec+0(FP)
	MOVW	R5, nsec+8(FP)
	RET

	// Syscall fallback
fallback:
	MOVD	$tp-16(SP), R3
	MOVW	$SYS_clock_gettime, R1
	SYSCALL
	LMG		tp-16(SP), R2, R3
	// sec is in R2, nsec in R3
	MOVD	R2, sec+0(FP)
	MOVW	R3, nsec+8(FP)
	RET

TEXT runtime·nanotime1(SB),NOSPLIT,$32-8
	MOVW	$1, R2			// CLOCK_MONOTONIC

	MOVD	R15, R7			// Backup stack pointer

	MOVD	g_m(g), R6		//m

	MOVD	runtime·vdsoClockgettimeSym(SB), R9	// Check for VDSO availability
	CMPBEQ	R9, $0, fallback

	MOVD	m_vdsoPC(R6), R4
	MOVD	R4, 16(R15)
	MOVD	m_vdsoSP(R6), R4
	MOVD	R4, 24(R15)

	MOVD	R14, R8			// Backup return address
	MOVD	$ret+0(FP), R4	// caller's SP

	MOVD	R8, m_vdsoPC(R6)
	MOVD	R4, m_vdsoSP(R6)

	MOVD	m_curg(R6), R5
	CMP		g, R5
	BNE		noswitch

	MOVD	m_g0(R6), R4
	MOVD	(g_sched+gobuf_sp)(R4), R15	// Set SP to g0 stack

noswitch:
	SUB		$16, R15		// reserve 2x 8 bytes for parameters
	MOVD	$~7, R4			// align to 8 bytes because of gcc ABI
	AND		R4, R15
	MOVD	R15, R3			// R15 needs to be in R3 as expected by kernel_clock_gettime

	MOVB	runtime·iscgo(SB),R12
	CMPBNE	R12, $0, nosaveg

	MOVD	m_gsignal(R6), R12	// g.m.gsignal
	CMPBEQ	R12, $0, nosaveg

	CMPBEQ	g, R12, nosaveg
	MOVD	(g_stack+stack_lo)(R12), R12	// g.m.gsignal.stack.lo
	MOVD	g, (R12)

	BL	R9 					// to vdso lookup

	MOVD $0, (R12)

	JMP	finish

nosaveg:
	BL	R9					// to vdso lookup

finish:
	MOVD	0(R15), R3		// sec
	MOVD	8(R15), R5		// nsec
	MOVD	R7, R15			// Restore SP

	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.

	MOVD	24(R15), R12
	MOVD	R12, m_vdsoSP(R6)
	MOVD	16(R15), R12
	MOVD	R12, m_vdsoPC(R6)

return:
	// sec is in R3, nsec in R5
	// return nsec in R3
	MULLD	$1000000000, R3
	ADD		R5, R3
	MOVD	R3, ret+0(FP)
	RET

	// Syscall fallback
fallback:
	MOVD	$tp-16(SP), R3
	MOVD	$SYS_clock_gettime, R1
	SYSCALL
	LMG		tp-16(SP), R2, R3
	MOVD	R3, R5
	MOVD	R2, R3
	JMP	return

TEXT runtime·rtsigprocmask(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	how+0(FP), R2
	MOVD	new+8(FP), R3
	MOVD	old+16(FP), R4
	MOVW	size+24(FP), R5
	MOVW	$SYS_rt_sigprocmask, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVD	R0, 0(R0) // crash
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT|NOFRAME,$0-36
	MOVD	sig+0(FP), R2
	MOVD	new+8(FP), R3
	MOVD	old+16(FP), R4
	MOVD	size+24(FP), R5
	MOVW	$SYS_rt_sigaction, R1
	SYSCALL
	MOVW	R2, ret+32(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), R2
	MOVD	info+16(FP), R3
	MOVD	ctx+24(FP), R4
	MOVD	fn+0(FP), R5
	BL	R5
	RET

TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME,$64
	// initialize essential registers (just in case)
	XOR	R0, R0

	// this might be called in external code context,
	// where g is not set.
	MOVB	runtime·iscgo(SB), R6
	CMPBEQ	R6, $0, 2(PC)
	BL	runtime·load_g(SB)

	MOVW	R2, 8(R15)
	MOVD	R3, 16(R15)
	MOVD	R4, 24(R15)
	MOVD	$runtime·sigtrampgo(SB), R5
	BL	R5
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	BR	runtime·sigtramp(SB)

// func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) unsafe.Pointer
TEXT runtime·mmap(SB),NOSPLIT,$48-48
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVW	prot+16(FP), R4
	MOVW	flags+20(FP), R5
	MOVW	fd+24(FP), R6
	MOVWZ	off+28(FP), R7

	// s390x uses old_mmap, so the arguments need to be placed into
	// a struct and a pointer to the struct passed to mmap.
	MOVD	R2, addr-48(SP)
	MOVD	R3, n-40(SP)
	MOVD	R4, prot-32(SP)
	MOVD	R5, flags-24(SP)
	MOVD	R6, fd-16(SP)
	MOVD	R7, off-8(SP)

	MOVD	$addr-48(SP), R2
	MOVW	$SYS_mmap, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, ok
	NEG	R2
	MOVD	$0, p+32(FP)
	MOVD	R2, err+40(FP)
	RET
ok:
	MOVD	R2, p+32(FP)
	MOVD	$0, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVW	$SYS_munmap, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVD	R0, 0(R0) // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R2
	MOVD	n+8(FP), R3
	MOVW	flags+16(FP), R4
	MOVW	$SYS_madvise, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// int64 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT|NOFRAME,$0
	MOVD	addr+0(FP), R2
	MOVW	op+8(FP), R3
	MOVW	val+12(FP), R4
	MOVD	ts+16(FP), R5
	MOVD	addr2+24(FP), R6
	MOVW	val3+32(FP),  R7
	MOVW	$SYS_futex, R1
	SYSCALL
	MOVW	R2, ret+40(FP)
	RET

// int32 clone(int32 flags, void *stk, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT|NOFRAME,$0
	MOVW	flags+0(FP), R3
	MOVD	stk+8(FP), R2

	// Copy mp, gp, fn off parent stack for use by child.
	// Careful: Linux system call clobbers ???.
	MOVD	mp+16(FP), R7
	MOVD	gp+24(FP), R8
	MOVD	fn+32(FP), R9

	MOVD	R7, -8(R2)
	MOVD	R8, -16(R2)
	MOVD	R9, -24(R2)
	MOVD	$1234, R7
	MOVD	R7, -32(R2)

	SYSCALL $SYS_clone

	// In parent, return.
	CMPBEQ	R2, $0, 3(PC)
	MOVW	R2, ret+40(FP)
	RET

	// In child, on new stack.
	// initialize essential registers
	XOR	R0, R0
	MOVD	-32(R15), R7
	CMP	R7, $1234
	BEQ	2(PC)
	MOVD	R0, 0(R0)

	// Initialize m->procid to Linux tid
	SYSCALL $SYS_gettid

	MOVD	-24(R15), R9        // fn
	MOVD	-16(R15), R8        // g
	MOVD	-8(R15), R7         // m

	CMPBEQ	R7, $0, nog
	CMP	R8, $0
	BEQ	nog

	MOVD	R2, m_procid(R7)

	// In child, set up new stack
	MOVD	R7, g_m(R8)
	MOVD	R8, g
	//CALL	runtime·stackcheck(SB)

nog:
	// Call fn
	BL	R9

	// It shouldn't return.	 If it does, exit that thread.
	MOVW	$111, R2
	MOVW	$SYS_exit, R1
	SYSCALL
	BR	-2(PC)	// keep exiting

TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOVD	new+0(FP), R2
	MOVD	old+8(FP), R3
	MOVW	$SYS_sigaltstack, R1
	SYSCALL
	MOVD	$-4095, R3
	CMPUBLT	R2, R3, 2(PC)
	MOVD	R0, 0(R0) // crash
	RET

TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOVW	$SYS_sched_yield, R1
	SYSCALL
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT|NOFRAME,$0
	MOVD	pid+0(FP), R2
	MOVD	len+8(FP), R3
	MOVD	buf+16(FP), R4
	MOVW	$SYS_sched_getaffinity, R1
	SYSCALL
	MOVW	R2, ret+24(FP)
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT|NOFRAME,$0-8
	// Implemented as brk(NULL).
	MOVD	$0, R2
	MOVW	$SYS_brk, R1
	SYSCALL
	MOVD	R2, ret+0(FP)
	RET

TEXT runtime·access(SB),$0-20
	MOVD	$0, 2(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·connect(SB),$0-28
	MOVD	$0, 2(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+24(FP)
	RET

TEXT runtime·socket(SB),$0-20
	MOVD	$0, 2(R0) // unimplemented, only needed for android; declared in stubs_linux.go
	MOVW	R0, ret+16(FP)
	RET
