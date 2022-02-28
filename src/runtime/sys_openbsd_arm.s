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

#define	CLOCK_REALTIME	$0
#define	CLOCK_MONOTONIC	$3

// With OpenBSD 6.7 onwards, an armv7 syscall returns two instructions
// after the SWI instruction, to allow for a speculative execution
// barrier to be placed after the SWI without impacting performance.
// For now use hardware no-ops as this works with both older and newer
// kernels. After OpenBSD 6.8 is released this should be changed to
// speculation barriers.
#define NOOP	MOVW    R0, R0
#define	INVOKE_SYSCALL	\
	SWI	$0;	\
	NOOP;		\
	NOOP

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// R0 points to the m.
	// We are already on m's g0 stack.

	// Save callee-save registers.
	MOVM.DB.W [R4-R11], (R13)

	MOVW	m_g0(R0), g
	BL	runtime·save_g(SB)

	BL	runtime·mstart(SB)

	// Restore callee-save registers.
	MOVM.IA.W (R13), [R4-R11]

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	MOVW	$0, R0
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-16
	MOVW	sig+4(FP), R0
	MOVW	info+8(FP), R1
	MOVW	ctx+12(FP), R2
	MOVW	fn+0(FP), R3
	MOVW	R13, R9
	SUB	$24, R13
	BIC	$0x7, R13 // alignment for ELF ABI
	BL	(R3)
	MOVW	R9, R13
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$0
	// Reserve space for callee-save registers and arguments.
	MOVM.DB.W [R4-R11], (R13)
	SUB	$16, R13

	// If called from an external code context, g will not be set.
	// Save R0, since runtime·load_g will clobber it.
	MOVW	R0, 4(R13)		// signum
	BL	runtime·load_g(SB)

	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	BL	runtime·sigtrampgo(SB)

	// Restore callee-save registers.
	ADD	$16, R13
	MOVM.IA.W (R13), [R4-R11]

	RET

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	B	runtime·armPublicationBarrier(SB)

// TODO(jsing): OpenBSD only supports GOARM=7 machines... this
// should not be needed, however the linker still allows GOARM=5
// on this platform.
TEXT runtime·read_tls_fallback(SB),NOSPLIT|NOFRAME,$0
	MOVM.WP	[R1, R2, R3, R12], (R13)
	MOVW	$330, R12		// sys___get_tcb
	INVOKE_SYSCALL
	MOVM.IAW (R13), [R1, R2, R3, R12]
	RET

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall - note that while asmcgocall does
// stack alignment, creation of a frame undoes it again.
// A pointer to the arguments is passed in R0.
// A single int32 result is returned in R0.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	0(R0), R0		// arg 1 attr
	CALL	libc_pthread_attr_init(SB)
	MOVW	R9, R13
	RET

TEXT runtime·pthread_attr_destroy_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	0(R0), R0		// arg 1 attr
	CALL	libc_pthread_attr_destroy(SB)
	MOVW	R9, R13
	RET

TEXT runtime·pthread_attr_getstacksize_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 size
	MOVW	0(R0), R0		// arg 1 attr
	CALL	libc_pthread_attr_getstacksize(SB)
	MOVW	R9, R13
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 state
	MOVW	0(R0), R0		// arg 1 attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	MOVW	R9, R13
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$16, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	0(R0), R1		// arg 2 attr
	MOVW	4(R0), R2		// arg 3 start
	MOVW	8(R0), R3		// arg 4 arg
	MOVW	R13, R0			// arg 1 &threadid (discarded)
	CALL	libc_pthread_create(SB)
	MOVW	R9, R13
	RET

TEXT runtime·thrkill_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 - signal
	MOVW	$0, R2			// arg 3 - tcb
	MOVW	0(R0), R0		// arg 1 - tid
	CALL	libc_thrkill(SB)
	MOVW	R9, R13
	RET

TEXT runtime·thrsleep_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$16, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 - clock_id
	MOVW	8(R0), R2		// arg 3 - abstime
	MOVW	12(R0), R3		// arg 4 - lock
	MOVW	16(R0), R4		// arg 5 - abort (on stack)
	MOVW	R4, 0(R13)
	MOVW	0(R0), R0		// arg 1 - id
	CALL	libc_thrsleep(SB)
	MOVW	R9, R13
	RET

TEXT runtime·thrwakeup_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 - count
	MOVW	0(R0), R0		// arg 1 - id
	CALL	libc_thrwakeup(SB)
	MOVW	R9, R13
	RET

TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	0(R0), R0		// arg 1 exit status
	BL	libc_exit(SB)
	MOVW	$0, R8			// crash on failure
	MOVW	R8, (R8)
	MOVW	R9, R13
	RET

TEXT runtime·getthrid_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	MOVW	R0, R8
	BIC     $0x7, R13		// align for ELF ABI
	BL	libc_getthrid(SB)
	MOVW	R0, 0(R8)
	MOVW	R9, R13
	RET

TEXT runtime·raiseproc_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	R0, R4
	BL	libc_getpid(SB)		// arg 1 pid
	MOVW	R4, R1			// arg 2 signal
	BL	libc_kill(SB)
	MOVW	R9, R13
	RET

TEXT runtime·sched_yield_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	BL	libc_sched_yield(SB)
	MOVW	R9, R13
	RET

TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$16, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	R0, R8
	MOVW	4(R0), R1		// arg 2 len
	MOVW	8(R0), R2		// arg 3 prot
	MOVW	12(R0), R3		// arg 4 flags
	MOVW	16(R0), R4		// arg 5 fid (on stack)
	MOVW	R4, 0(R13)
	MOVW	$0, R5			// pad (on stack)
	MOVW	R5, 4(R13)
	MOVW	20(R0), R6		// arg 6 offset (on stack)
	MOVW	R6, 8(R13)		// low 32 bits
	MOVW    $0, R7
	MOVW	R7, 12(R13)		// high 32 bits
	MOVW	0(R0), R0		// arg 1 addr
	BL	libc_mmap(SB)
	MOVW	$0, R1
	CMP	$-1, R0
	BNE	ok
	BL	libc_errno(SB)
	MOVW	(R0), R1		// errno
	MOVW	$0, R0
ok:
	MOVW	R0, 24(R8)
	MOVW	R1, 28(R8)
	MOVW	R9, R13
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 len
	MOVW	0(R0), R0		// arg 1 addr
	BL	libc_munmap(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVW	$0, R8			// crash on failure
	MOVW	R8, (R8)
	MOVW	R9, R13
	RET

TEXT runtime·madvise_trampoline(SB), NOSPLIT, $0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 len
	MOVW	8(R0), R2		// arg 3 advice
	MOVW	0(R0), R0		// arg 1 addr
	BL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	MOVW	R9, R13
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$8, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 - flags
	MOVW	8(R0), R2		// arg 3 - mode (vararg, on stack)
	MOVW	R2, 0(R13)
	MOVW	0(R0), R0		// arg 1 - path
	MOVW	R13, R4
	BIC     $0x7, R13		// align for ELF ABI
	BL	libc_open(SB)
	MOVW	R9, R13
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	0(R0), R0		// arg 1 - fd
	BL	libc_close(SB)
	MOVW	R9, R13
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 - buf
	MOVW	8(R0), R2		// arg 3 - count
	MOVW	0(R0), R0		// arg 1 - fd
	BL	libc_read(SB)
	CMP	$-1, R0
	BNE	noerr
	BL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	RSB.CS	$0, R0			// caller expects negative errno
noerr:
	MOVW	R9, R13
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 buf
	MOVW	8(R0), R2		// arg 3 count
	MOVW	0(R0), R0		// arg 1 fd
	BL	libc_write(SB)
	CMP	$-1, R0
	BNE	noerr
	BL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	RSB.CS	$0, R0			// caller expects negative errno
noerr:
	MOVW	R9, R13
	RET

TEXT runtime·pipe2_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 flags
	MOVW	0(R0), R0		// arg 1 filedes
	BL	libc_pipe2(SB)
	CMP	$-1, R0
	BNE	3(PC)
	BL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	RSB.CS	$0, R0			// caller expects negative errno
	MOVW	R9, R13
	RET

TEXT runtime·setitimer_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 new
	MOVW	8(R0), R2		// arg 3 old
	MOVW	0(R0), R0		// arg 1 which
	BL	libc_setitimer(SB)
	MOVW	R9, R13
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	0(R0), R0		// arg 1 usec
	BL	libc_usleep(SB)
	MOVW	R9, R13
	RET

TEXT runtime·sysctl_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$8, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 miblen
	MOVW	8(R0), R2		// arg 3 out
	MOVW	12(R0), R3		// arg 4 size
	MOVW	16(R0), R4		// arg 5 dst (on stack)
	MOVW	R4, 0(R13)
	MOVW	20(R0), R5		// arg 6 ndst (on stack)
	MOVW	R5, 4(R13)
	MOVW	0(R0), R0		// arg 1 mib
	BL	libc_sysctl(SB)
	MOVW	R9, R13
	RET

TEXT runtime·kqueue_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	BL	libc_kqueue(SB)
	MOVW	R9, R13
	RET

TEXT runtime·kevent_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$8, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 keventt
	MOVW	8(R0), R2		// arg 3 nch
	MOVW	12(R0), R3		// arg 4 ev
	MOVW	16(R0), R4		// arg 5 nev (on stack)
	MOVW	R4, 0(R13)
	MOVW	20(R0), R5		// arg 6 ts (on stack)
	MOVW	R5, 4(R13)
	MOVW	0(R0), R0		// arg 1 kq
	BL	libc_kevent(SB)
	CMP	$-1, R0
	BNE	ok
	BL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	RSB.CS	$0, R0			// caller expects negative errno
ok:
	MOVW	R9, R13
	RET

TEXT runtime·clock_gettime_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 tp
	MOVW	0(R0), R0		// arg 1 clock_id
	BL	libc_clock_gettime(SB)
	CMP	$-1, R0
	BNE	noerr
	BL	libc_errno(SB)
	MOVW	(R0), R0		// errno
	RSB.CS	$0, R0			// caller expects negative errno
noerr:
	MOVW	R9, R13
	RET

TEXT runtime·fcntl_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$8, R13
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 cmd
	MOVW	8(R0), R2		// arg 3 arg (vararg, on stack)
	MOVW	R2, 0(R13)
	MOVW	0(R0), R0		// arg 1 fd
	BL	libc_fcntl(SB)
	MOVW	R9, R13
	RET

TEXT runtime·sigaction_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 new
	MOVW	8(R0), R2		// arg 3 old
	MOVW	0(R0), R0		// arg 1 sig
	BL	libc_sigaction(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVW	$0, R8			// crash on failure
	MOVW	R8, (R8)
	MOVW	R9, R13
	RET

TEXT runtime·sigprocmask_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 new
	MOVW	8(R0), R2		// arg 3 old
	MOVW	0(R0), R0		// arg 1 how
	BL	libc_pthread_sigmask(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVW	$0, R8			// crash on failure
	MOVW	R8, (R8)
	MOVW	R9, R13
	RET

TEXT runtime·sigaltstack_trampoline(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI
	MOVW	4(R0), R1		// arg 2 old
	MOVW	0(R0), R0		// arg 1 new
	BL	libc_sigaltstack(SB)
	CMP	$-1, R0
	BNE	3(PC)
	MOVW	$0, R8			// crash on failure
	MOVW	R8, (R8)
	MOVW	R9, R13
	RET

// syscall calls a function in libc on behalf of the syscall package.
// syscall takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI

	MOVW	R0, R8

	MOVW	(0*4)(R8), R7 // fn
	MOVW	(1*4)(R8), R0 // a1
	MOVW	(2*4)(R8), R1 // a2
	MOVW	(3*4)(R8), R2 // a3

	BL	(R7)

	MOVW	R0, (4*4)(R8) // r1
	MOVW	R1, (5*4)(R8) // r2

	// Standard libc functions return -1 on error and set errno.
	CMP	$-1, R0
	BNE	ok

	// Get error code from libc.
	BL	libc_errno(SB)
	MOVW	(R0), R1
	MOVW	R1, (6*4)(R8) // err

ok:
	MOVW	$0, R0		// no error (it's ignored anyway)
	MOVW	R9, R13
	RET

// syscallX calls a function in libc on behalf of the syscall package.
// syscallX takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscallX must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscallX is like syscall but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscallX(SB),NOSPLIT,$0
	MOVW	R13, R9
	BIC     $0x7, R13		// align for ELF ABI

	MOVW	R0, R8

	MOVW	(0*4)(R8), R7 // fn
	MOVW	(1*4)(R8), R0 // a1
	MOVW	(2*4)(R8), R1 // a2
	MOVW	(3*4)(R8), R2 // a3

	BL	(R7)

	MOVW	R0, (4*4)(R8) // r1
	MOVW	R1, (5*4)(R8) // r2

	// Standard libc functions return -1 on error and set errno.
	CMP	$-1, R0
	BNE	ok
	CMP	$-1, R1
	BNE	ok

	// Get error code from libc.
	BL	libc_errno(SB)
	MOVW	(R0), R1
	MOVW	R1, (6*4)(R8) // err

ok:
	MOVW	$0, R0		// no error (it's ignored anyway)
	MOVW	R9, R13
	RET

// syscall6 calls a function in libc on behalf of the syscall package.
// syscall6 takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall6 must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall6 expects a 32-bit result and tests for 32-bit -1
// to decide there was an error.
TEXT runtime·syscall6(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$8, R13
	BIC     $0x7, R13		// align for ELF ABI

	MOVW	R0, R8

	MOVW	(0*4)(R8), R7 // fn
	MOVW	(1*4)(R8), R0 // a1
	MOVW	(2*4)(R8), R1 // a2
	MOVW	(3*4)(R8), R2 // a3
	MOVW	(4*4)(R8), R3 // a4
	MOVW	(5*4)(R8), R4 // a5
	MOVW	R4, 0(R13)
	MOVW	(6*4)(R8), R5 // a6
	MOVW	R5, 4(R13)

	BL	(R7)

	MOVW	R0, (7*4)(R8) // r1
	MOVW	R1, (8*4)(R8) // r2

	// Standard libc functions return -1 on error and set errno.
	CMP	$-1, R0
	BNE	ok

	// Get error code from libc.
	BL	libc_errno(SB)
	MOVW	(R0), R1
	MOVW	R1, (9*4)(R8) // err

ok:
	MOVW	$0, R0		// no error (it's ignored anyway)
	MOVW	R9, R13
	RET

// syscall6X calls a function in libc on behalf of the syscall package.
// syscall6X takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall6X must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall6X is like syscall6 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall6X(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$8, R13
	BIC     $0x7, R13		// align for ELF ABI

	MOVW	R0, R8

	MOVW	(0*4)(R8), R7 // fn
	MOVW	(1*4)(R8), R0 // a1
	MOVW	(2*4)(R8), R1 // a2
	MOVW	(3*4)(R8), R2 // a3
	MOVW	(4*4)(R8), R3 // a4
	MOVW	(5*4)(R8), R4 // a5
	MOVW	R4, 0(R13)
	MOVW	(6*4)(R8), R5 // a6
	MOVW	R5, 4(R13)

	BL	(R7)

	MOVW	R0, (7*4)(R8) // r1
	MOVW	R1, (8*4)(R8) // r2

	// Standard libc functions return -1 on error and set errno.
	CMP	$-1, R0
	BNE	ok
	CMP	$-1, R1
	BNE	ok

	// Get error code from libc.
	BL	libc_errno(SB)
	MOVW	(R0), R1
	MOVW	R1, (9*4)(R8) // err

ok:
	MOVW	$0, R0		// no error (it's ignored anyway)
	MOVW	R9, R13
	RET

// syscall10 calls a function in libc on behalf of the syscall package.
// syscall10 takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	a7    uintptr
//	a8    uintptr
//	a9    uintptr
//	a10   uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall10 must be called on the g0 stack with the
// C calling convention (use libcCall).
TEXT runtime·syscall10(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$24, R13
	BIC     $0x7, R13		// align for ELF ABI

	MOVW	R0, R8

	MOVW	(0*4)(R8), R7 // fn
	MOVW	(1*4)(R8), R0 // a1
	MOVW	(2*4)(R8), R1 // a2
	MOVW	(3*4)(R8), R2 // a3
	MOVW	(4*4)(R8), R3 // a4
	MOVW	(5*4)(R8), R4 // a5
	MOVW	R4, 0(R13)
	MOVW	(6*4)(R8), R5 // a6
	MOVW	R5, 4(R13)
	MOVW	(7*4)(R8), R6 // a7
	MOVW	R6, 8(R13)
	MOVW	(8*4)(R8), R4 // a8
	MOVW	R4, 12(R13)
	MOVW	(9*4)(R8), R5 // a9
	MOVW	R5, 16(R13)
	MOVW	(10*4)(R8), R6 // a10
	MOVW	R6, 20(R13)

	BL	(R7)

	MOVW	R0, (11*4)(R8) // r1
	MOVW	R1, (12*4)(R8) // r2

	// Standard libc functions return -1 on error and set errno.
	CMP	$-1, R0
	BNE	ok

	// Get error code from libc.
	BL	libc_errno(SB)
	MOVW	(R0), R1
	MOVW	R1, (13*4)(R8) // err

ok:
	MOVW	$0, R0		// no error (it's ignored anyway)
	MOVW	R9, R13
	RET

// syscall10X calls a function in libc on behalf of the syscall package.
// syscall10X takes a pointer to a struct like:
// struct {
//	fn    uintptr
//	a1    uintptr
//	a2    uintptr
//	a3    uintptr
//	a4    uintptr
//	a5    uintptr
//	a6    uintptr
//	a7    uintptr
//	a8    uintptr
//	a9    uintptr
//	a10   uintptr
//	r1    uintptr
//	r2    uintptr
//	err   uintptr
// }
// syscall10X must be called on the g0 stack with the
// C calling convention (use libcCall).
//
// syscall10X is like syscall10 but expects a 64-bit result
// and tests for 64-bit -1 to decide there was an error.
TEXT runtime·syscall10X(SB),NOSPLIT,$0
	MOVW	R13, R9
	SUB	$24, R13
	BIC     $0x7, R13		// align for ELF ABI

	MOVW	R0, R8

	MOVW	(0*4)(R8), R7 // fn
	MOVW	(1*4)(R8), R0 // a1
	MOVW	(2*4)(R8), R1 // a2
	MOVW	(3*4)(R8), R2 // a3
	MOVW	(4*4)(R8), R3 // a4
	MOVW	(5*4)(R8), R4 // a5
	MOVW	R4, 0(R13)
	MOVW	(6*4)(R8), R5 // a6
	MOVW	R5, 4(R13)
	MOVW	(7*4)(R8), R6 // a7
	MOVW	R6, 8(R13)
	MOVW	(8*4)(R8), R4 // a8
	MOVW	R4, 12(R13)
	MOVW	(9*4)(R8), R5 // a9
	MOVW	R5, 16(R13)
	MOVW	(10*4)(R8), R6 // a10
	MOVW	R6, 20(R13)

	BL	(R7)

	MOVW	R0, (11*4)(R8) // r1
	MOVW	R1, (12*4)(R8) // r2

	// Standard libc functions return -1 on error and set errno.
	CMP	$-1, R0
	BNE	ok
	CMP	$-1, R1
	BNE	ok

	// Get error code from libc.
	BL	libc_errno(SB)
	MOVW	(R0), R1
	MOVW	R1, (13*4)(R8) // err

ok:
	MOVW	$0, R0		// no error (it's ignored anyway)
	MOVW	R9, R13
	RET
