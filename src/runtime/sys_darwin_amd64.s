// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for AMD64, Darwin
// See http://fxr.watson.org/fxr/source/bsd/kern/syscalls.c?v=xnu-1228
// or /usr/include/sys/syscall.h (on a Mac) for system call numbers.
//
// The low 24 bits are the system call number.
// The high 8 bits specify the kind of system call: 1=Mach, 2=BSD, 3=Machine-Dependent.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Exit the entire program (like C exit)
TEXT runtime·exit_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI		// arg 1 exit status
	CALL	libc_exit(SB)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·open_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	8(DI), SI		// arg 2 flags
	MOVL	12(DI), DX		// arg 3 mode
	MOVQ	0(DI), DI		// arg 1 pathname
	CALL	libc_open(SB)
	POPQ	BP
	RET

TEXT runtime·close_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_close(SB)
	POPQ	BP
	RET

TEXT runtime·read_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVL	0(DI), DI		// arg 1 fd
	CALL	libc_read(SB)
	POPQ	BP
	RET

TEXT runtime·write_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 buf
	MOVL	16(DI), DX		// arg 3 count
	MOVQ	0(DI), DI		// arg 1 fd
	CALL	libc_write(SB)
	POPQ	BP
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$24
	MOVL	$(0x2000000+20), AX // getpid
	SYSCALL
	MOVQ	AX, DI	// arg 1 - pid
	MOVL	sig+0(FP), SI	// arg 2 - signal
	MOVL	$1, DX	// arg 3 - posix
	MOVL	$(0x2000000+37), AX // kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $0
	MOVL	mode+0(FP), DI
	MOVQ	new+8(FP), SI
	MOVQ	old+16(FP), DX
	MOVL	$(0x2000000+83), AX	// syscall entry
	SYSCALL
	RET

TEXT runtime·madvise_trampoline(SB), NOSPLIT, $0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 len
	MOVL	16(DI), DX	// arg 3 advice
	MOVQ	0(DI), DI	// arg 1 addr
	CALL	libc_madvise(SB)
	// ignore failure - maybe pages are locked
	POPQ	BP
	RET

// OS X comm page time offsets
// https://opensource.apple.com/source/xnu/xnu-4570.1.46/osfmk/i386/cpu_capabilities.h

#define	nt_tsc_base	0x50
#define	nt_scale	0x58
#define	nt_shift	0x5c
#define	nt_ns_base	0x60
#define	nt_generation	0x68
#define	gtod_generation	0x6c  // obsolete since Darwin v17 (High Sierra)
#define	gtod_ns_base	0x70  // obsolete since Darwin v17 (High Sierra)
#define	gtod_sec_base	0x78  // obsolete since Darwin v17 (High Sierra)

#define	v17_gtod_ns_base	0xd0
#define	v17_gtod_sec_ofs	0xd8
#define	v17_gtod_frac_ofs	0xe0
#define	v17_gtod_scale		0xe8
#define	v17_gtod_tkspersec	0xf0

GLOBL timebase<>(SB),NOPTR,$(machTimebaseInfo__size)

TEXT runtime·nanotime_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	DI, BX
	CALL	libc_mach_absolute_time(SB)
	MOVQ	AX, 0(BX)
	MOVL	timebase<>+machTimebaseInfo_numer(SB), SI
	MOVL	timebase<>+machTimebaseInfo_denom(SB), DI // atomic read
	TESTL	DI, DI
	JNE	initialized

	SUBQ	$(machTimebaseInfo__size+15)/16*16, SP
	MOVQ	SP, DI
	CALL	libc_mach_timebase_info(SB)
	MOVL	machTimebaseInfo_numer(SP), SI
	MOVL	machTimebaseInfo_denom(SP), DI
	ADDQ	$(machTimebaseInfo__size+15)/16*16, SP

	MOVL	SI, timebase<>+machTimebaseInfo_numer(SB)
	MOVL	DI, AX
	XCHGL	AX, timebase<>+machTimebaseInfo_denom(SB) // atomic write

initialized:
	MOVL	SI, 8(BX)
	MOVL	DI, 12(BX)
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT time·now(SB), NOSPLIT, $32-24
	// Note: The 32 bytes of stack frame requested on the TEXT line
	// are used in the systime fallback, as the timeval address
	// filled in by the system call.
	MOVQ	$0x7fffffe00000, BP	/* comm page base */
	CMPQ	runtime·darwinVersion(SB), $17
	JB		legacy /* sierra and older */

	// This is the new code, for macOS High Sierra (Darwin v17) and newer.
v17:
	// Loop trying to take a consistent snapshot
	// of the time parameters.
timeloop17:
	MOVQ 	v17_gtod_ns_base(BP), R12

	MOVL	nt_generation(BP), CX
	TESTL	CX, CX
	JZ		timeloop17
	RDTSC
	MOVQ	nt_tsc_base(BP), SI
	MOVL	nt_scale(BP), DI
	MOVQ	nt_ns_base(BP), BX
	CMPL	nt_generation(BP), CX
	JNE		timeloop17

	MOVQ 	v17_gtod_sec_ofs(BP), R8
	MOVQ 	v17_gtod_frac_ofs(BP), R9
	MOVQ 	v17_gtod_scale(BP), R10
	MOVQ 	v17_gtod_tkspersec(BP), R11
	CMPQ 	v17_gtod_ns_base(BP), R12
	JNE 	timeloop17

	// Compute monotonic time
	//	mono = ((tsc - nt_tsc_base) * nt_scale) >> 32 + nt_ns_base
	// The multiply and shift extracts the top 64 bits of the 96-bit product.
	SHLQ	$32, DX
	ADDQ	DX, AX
	SUBQ	SI, AX
	MULQ	DI
	SHRQ	$32, AX:DX
	ADDQ	BX, AX

	// Subtract startNano base to return the monotonic runtime timer
	// which is an offset from process boot.
	MOVQ	AX, BX
	MOVQ	runtime·startNano(SB), CX
	SUBQ	CX, BX
	MOVQ	BX, monotonic+16(FP)

	// Now compute the 128-bit wall time:
	//  wall = ((mono - gtod_ns_base) * gtod_scale) + gtod_offs
	// The parameters are updated every second, so if we found them
	// outdated (that is, more than one second is passed from the ns base),
	// fallback to the syscall.
	TESTQ	R12, R12
	JZ		systime
	SUBQ	R12, AX
	CMPQ	R11, AX
	JB		systime
	MULQ 	R10
	ADDQ	R9, AX
	ADCQ	R8, DX

	// Convert the 128-bit wall time into (sec,nsec).
	// High part (seconds) is already good to go, while low part
	// (fraction of seconds) must be converted to nanoseconds.
	MOVQ	DX, sec+0(FP)
	MOVQ 	$1000000000, CX
	MULQ	CX
	MOVQ	DX, nsec+8(FP)
	RET

	// This is the legacy code needed for macOS Sierra (Darwin v16) and older.
legacy:
	// Loop trying to take a consistent snapshot
	// of the time parameters.
timeloop:
	MOVL	gtod_generation(BP), R8
	MOVL	nt_generation(BP), R9
	TESTL	R9, R9
	JZ	timeloop
	RDTSC
	MOVQ	nt_tsc_base(BP), R10
	MOVL	nt_scale(BP), R11
	MOVQ	nt_ns_base(BP), R12
	CMPL	nt_generation(BP), R9
	JNE	timeloop
	MOVQ	gtod_ns_base(BP), R13
	MOVQ	gtod_sec_base(BP), R14
	CMPL	gtod_generation(BP), R8
	JNE	timeloop

	// Gathered all the data we need. Compute:
	//	monotonic_time = ((tsc - nt_tsc_base) * nt_scale) >> 32 + nt_ns_base
	// The multiply and shift extracts the top 64 bits of the 96-bit product.
	SHLQ	$32, DX
	ADDQ	DX, AX
	SUBQ	R10, AX
	MULQ	R11
	SHRQ	$32, AX:DX
	ADDQ	R12, AX
	MOVQ	AX, BX
	MOVQ	runtime·startNano(SB), CX
	SUBQ	CX, BX
	MOVQ	BX, monotonic+16(FP)

	// Compute:
	//	wall_time = monotonic time - gtod_ns_base + gtod_sec_base*1e9
	// or, if gtod_generation==0, invoke the system call.
	TESTL	R8, R8
	JZ	systime
	SUBQ	R13, AX
	IMULQ	$1000000000, R14
	ADDQ	R14, AX

	// Split wall time into sec, nsec.
	// generated code for
	//	func f(x uint64) (uint64, uint64) { return x/1e9, x%1e9 }
	// adapted to reduce duplication
	MOVQ	AX, CX
	SHRQ	$9, AX
	MOVQ	$19342813113834067, DX
	MULQ	DX
	SHRQ	$11, DX
	MOVQ	DX, sec+0(FP)
	IMULQ	$1000000000, DX
	SUBQ	DX, CX
	MOVL	CX, nsec+8(FP)
	RET

systime:
	// Fall back to system call (usually first call in this thread).
	MOVQ	SP, DI
	MOVQ	$0, SI
	MOVQ	$0, DX  // required as of Sierra; Issue 16570
	MOVL	$(0x2000000+116), AX // gettimeofday
	SYSCALL
	CMPQ	AX, $0
	JNE	inreg
	MOVQ	0(SP), AX
	MOVL	8(SP), DX
inreg:
	// sec is in AX, usec in DX
	IMULQ	$1000, DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	how+0(FP), DI
	MOVQ	new+8(FP), SI
	MOVQ	old+16(FP), DX
	MOVL	$(0x2000000+329), AX  // pthread_sigmask (on OS X, sigprocmask==entire process)
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$0-24
	MOVL	mode+0(FP), DI		// arg 1 sig
	MOVQ	new+8(FP), SI		// arg 2 act
	MOVQ	old+16(FP), DX		// arg 3 oact
	MOVQ	old+16(FP), CX		// arg 3 oact
	MOVQ	old+16(FP), R10		// arg 3 oact
	MOVL	$(0x2000000+46), AX	// syscall entry
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVQ	fn+0(FP),    AX
	MOVL	sig+8(FP),   DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP),  DX
	PUSHQ	BP
	MOVQ	SP, BP
	ANDQ	$~15, SP     // alignment for x86_64 ABI
	CALL	AX
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$40
	MOVL SI, 24(SP) // save infostyle for sigreturn below
	MOVQ R8, 32(SP) // save ctx
	MOVL DX, 0(SP)  // sig
	MOVQ CX, 8(SP)  // info
	MOVQ R8, 16(SP) // ctx
	MOVQ $runtime·sigtrampgo(SB), AX
	CALL AX
	MOVQ 32(SP), DI // ctx
	MOVL 24(SP), SI // infostyle
	MOVL $(0x2000000+184), AX
	SYSCALL
	INT $3 // not reached



TEXT runtime·mmap_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP			// make a frame; keep stack aligned
	MOVQ	SP, BP
	MOVQ	DI, BX
	MOVQ	0(BX), DI		// arg 1 addr
	MOVQ	8(BX), SI		// arg 2 len
	MOVL	16(BX), DX		// arg 3 prot
	MOVL	20(BX), CX		// arg 4 flags
	MOVL	24(BX), R8		// arg 5 fid
	MOVL	28(BX), R9		// arg 6 offset
	CALL	libc_mmap(SB)
	XORL	DX, DX
	CMPQ	AX, $-1
	JNE	ok
	CALL	libc_error(SB)
	MOVQ	(AX), DX		// errno
	XORL	AX, AX
ok:
	MOVQ	AX, 32(BX)
	MOVQ	DX, 40(BX)
	POPQ	BP
	RET

TEXT runtime·munmap_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI		// arg 2 len
	MOVQ	0(DI), DI		// arg 1 addr
	CALL	libc_munmap(SB)
	TESTQ	AX, AX
	JEQ	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	POPQ	BP
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVQ	new+0(FP), DI
	MOVQ	old+8(FP), SI
	MOVQ	$(0x2000000+53), AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVL	0(DI), DI	// arg 1 usec
	CALL	libc_usleep(SB)
	POPQ	BP
	RET

// Mach system calls use 0x1000000 instead of the BSD's 0x2000000.

// func mach_msg_trap(h unsafe.Pointer, op int32, send_size, rcv_size, rcv_name, timeout, notify uint32) int32
TEXT runtime·mach_msg_trap(SB),NOSPLIT,$0
	MOVQ	h+0(FP), DI
	MOVL	op+8(FP), SI
	MOVL	send_size+12(FP), DX
	MOVL	rcv_size+16(FP), R10
	MOVL	rcv_name+20(FP), R8
	MOVL	timeout+24(FP), R9
	MOVL	notify+28(FP), R11
	PUSHQ	R11	// seventh arg, on stack
	MOVL	$(0x1000000+31), AX	// mach_msg_trap
	SYSCALL
	POPQ	R11
	MOVL	AX, ret+32(FP)
	RET

TEXT runtime·mach_task_self(SB),NOSPLIT,$0
	MOVL	$(0x1000000+28), AX	// task_self_trap
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·mach_thread_self(SB),NOSPLIT,$0
	MOVL	$(0x1000000+27), AX	// thread_self_trap
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·mach_reply_port(SB),NOSPLIT,$0
	MOVL	$(0x1000000+26), AX	// mach_reply_port
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

// Mach provides trap versions of the semaphore ops,
// instead of requiring the use of RPC.

// func mach_semaphore_wait(sema uint32) int32
TEXT runtime·mach_semaphore_wait(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	$(0x1000000+36), AX	// semaphore_wait_trap
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// func mach_semaphore_timedwait(sema, sec, nsec uint32) int32
TEXT runtime·mach_semaphore_timedwait(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	sec+4(FP), SI
	MOVL	nsec+8(FP), DX
	MOVL	$(0x1000000+38), AX	// semaphore_timedwait_trap
	SYSCALL
	MOVL	AX, ret+16(FP)
	RET

// func mach_semaphore_signal(sema uint32) int32
TEXT runtime·mach_semaphore_signal(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	$(0x1000000+33), AX	// semaphore_signal_trap
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// func mach_semaphore_signal_all(sema uint32) int32
TEXT runtime·mach_semaphore_signal_all(SB),NOSPLIT,$0
	MOVL	sema+0(FP), DI
	MOVL	$(0x1000000+34), AX	// semaphore_signal_all_trap
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·settls(SB),NOSPLIT,$32
	// Nothing to do on Darwin, pthread already set thread-local storage up.
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVQ	mib+0(FP), DI
	MOVL	miblen+8(FP), SI
	MOVQ	out+16(FP), DX
	MOVQ	size+24(FP), R10
	MOVQ	dst+32(FP), R8
	MOVQ	ndst+40(FP), R9
	MOVL	$(0x2000000+202), AX	// syscall entry
	SYSCALL
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+48(FP)
	RET

// func kqueue() int32
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVQ    $0, DI
	MOVQ    $0, SI
	MOVQ    $0, DX
	MOVL	$(0x2000000+362), AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET

// func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL    kq+0(FP), DI
	MOVQ    ch+8(FP), SI
	MOVL    nch+16(FP), DX
	MOVQ    ev+24(FP), R10
	MOVL    nev+32(FP), R8
	MOVQ    ts+40(FP), R9
	MOVL	$(0x2000000+363), AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET

// func closeonexec(fd int32)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL    fd+0(FP), DI  // fd
	MOVQ    $2, SI  // F_SETFD
	MOVQ    $1, DX  // FD_CLOEXEC
	MOVL	$(0x2000000+92), AX  // fcntl
	SYSCALL
	RET

// mstart_stub is the first function executed on a new thread started by pthread_create.
// It just does some low-level setup and then calls mstart.
// Note: called with the C calling convention.
TEXT runtime·mstart_stub(SB),NOSPLIT,$0
	// DI points to the m.
	// We are already on m's g0 stack.

	MOVQ	m_g0(DI), DX // g

	// Initialize TLS entry.
	// See cmd/link/internal/ld/sym.go:computeTLSOffset.
	MOVQ	DX, 0x30(GS)

	// Someday the convention will be D is always cleared.
	CLD

	CALL	runtime·mstart(SB)

	// Go is all done with this OS thread.
	// Tell pthread everything is ok (we never join with this thread, so
	// the value here doesn't really matter).
	XORL	AX, AX
	RET

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in DI.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·pthread_attr_init_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP	// make frame, keep stack 16-byte aligned.
	MOVQ	SP, BP
	MOVQ	0(DI), DI // arg 1 attr
	CALL	libc_pthread_attr_init(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_setstacksize_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 size
	MOVQ	0(DI), DI	// arg 1 attr
	CALL	libc_pthread_attr_setstacksize(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_attr_setdetachstate_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 state
	MOVQ	0(DI), DI	// arg 1 attr
	CALL	libc_pthread_attr_setdetachstate(SB)
	POPQ	BP
	RET

TEXT runtime·pthread_create_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	SUBQ	$16, SP
	MOVQ	0(DI), SI	// arg 2 attr
	MOVQ	8(DI), DX	// arg 3 start
	MOVQ	16(DI), CX	// arg 4 arg
	MOVQ	SP, DI		// arg 1 &threadid (which we throw away)
	CALL	libc_pthread_create(SB)
	MOVQ	BP, SP
	POPQ	BP
	RET

TEXT runtime·pthread_self_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	DI, BX		// Note: asmcgocall doesn't save anything in BX, so it is ok to clobber it here.
	CALL	libc_pthread_self(SB)
	MOVQ	AX, 0(BX)	// Save result.
	POPQ	BP
	RET

TEXT runtime·pthread_kill_trampoline(SB),NOSPLIT,$0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI	// arg 2 signal
	MOVQ	0(DI), DI	// arg 1 thread
	CALL	libc_pthread_kill(SB)
	POPQ	BP
	RET
