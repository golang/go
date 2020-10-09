// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for 386, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// Most linux systems use glibc's dynamic linker, which puts the
// __kernel_vsyscall vdso helper at 0x10(GS) for easy access from position
// independent code and setldt in runtime does the same in the statically
// linked case. However, systems that use alternative libc such as Android's
// bionic and musl, do not save the helper anywhere, and so the only way to
// invoke a syscall from position independent code is boring old int $0x80
// (which is also what syscall wrappers in bionic/musl use).
//
// The benchmarks also showed that using int $0x80 is as fast as calling
// *%gs:0x10 except on AMD Opteron. See https://golang.org/cl/19833
// for the benchmark program and raw data.
//#define INVOKE_SYSCALL	CALL	0x10(GS) // non-portable
#define INVOKE_SYSCALL	INT	$0x80

#define SYS_exit		1
#define SYS_read		3
#define SYS_write		4
#define SYS_open		5
#define SYS_close		6
#define SYS_getpid		20
#define SYS_access		33
#define SYS_kill		37
#define SYS_pipe		42
#define SYS_brk 		45
#define SYS_fcntl		55
#define SYS_munmap		91
#define SYS_socketcall		102
#define SYS_setittimer		104
#define SYS_clone		120
#define SYS_sched_yield 	158
#define SYS_nanosleep		162
#define SYS_rt_sigreturn	173
#define SYS_rt_sigaction	174
#define SYS_rt_sigprocmask	175
#define SYS_sigaltstack 	186
#define SYS_mmap2		192
#define SYS_mincore		218
#define SYS_madvise		219
#define SYS_gettid		224
#define SYS_futex		240
#define SYS_sched_getaffinity	242
#define SYS_set_thread_area	243
#define SYS_exit_group		252
#define SYS_epoll_create	254
#define SYS_epoll_ctl		255
#define SYS_epoll_wait		256
#define SYS_clock_gettime	265
#define SYS_tgkill		270
#define SYS_epoll_create1	329
#define SYS_pipe2		331

TEXT runtime·exit(SB),NOSPLIT,$0
	MOVL	$SYS_exit_group, AX
	MOVL	code+0(FP), BX
	INVOKE_SYSCALL
	INT $3	// not reached
	RET

TEXT exit1<>(SB),NOSPLIT,$0
	MOVL	$SYS_exit, AX
	MOVL	code+0(FP), BX
	INVOKE_SYSCALL
	INT $3	// not reached
	RET

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVL	wait+0(FP), AX
	// We're done using the stack.
	MOVL	$0, (AX)
	MOVL	$1, AX	// exit (just this thread)
	MOVL	$0, BX	// exit code
	INT	$0x80	// no stack; must not use CALL
	// We may not even have a stack any more.
	INT	$3
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT,$0
	MOVL	$SYS_open, AX
	MOVL	name+0(FP), BX
	MOVL	mode+4(FP), CX
	MOVL	perm+8(FP), DX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVL	$SYS_close, AX
	MOVL	fd+0(FP), BX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$0
	MOVL	$SYS_write, AX
	MOVL	fd+0(FP), BX
	MOVL	p+4(FP), CX
	MOVL	n+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVL	$SYS_read, AX
	MOVL	fd+0(FP), BX
	MOVL	p+4(FP), CX
	MOVL	n+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT,$0-12
	MOVL	$SYS_pipe, AX
	LEAL	r+0(FP), BX
	INVOKE_SYSCALL
	MOVL	AX, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$0-16
	MOVL	$SYS_pipe2, AX
	LEAL	r+4(FP), BX
	MOVL	flags+0(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, errno+12(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$8
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 0(SP)
	MOVL	$1000, AX	// usec to nsec
	MULL	DX
	MOVL	AX, 4(SP)

	// nanosleep(&ts, 0)
	MOVL	$SYS_nanosleep, AX
	LEAL	0(SP), BX
	MOVL	$0, CX
	INVOKE_SYSCALL
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVL	$SYS_gettid, AX
	INVOKE_SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$12
	MOVL	$SYS_getpid, AX
	INVOKE_SYSCALL
	MOVL	AX, BX	// arg 1 pid
	MOVL	$SYS_gettid, AX
	INVOKE_SYSCALL
	MOVL	AX, CX	// arg 2 tid
	MOVL	sig+0(FP), DX	// arg 3 signal
	MOVL	$SYS_tgkill, AX
	INVOKE_SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVL	$SYS_getpid, AX
	INVOKE_SYSCALL
	MOVL	AX, BX	// arg 1 pid
	MOVL	sig+0(FP), CX	// arg 2 signal
	MOVL	$SYS_kill, AX
	INVOKE_SYSCALL
	RET

TEXT ·getpid(SB),NOSPLIT,$0-4
	MOVL	$SYS_getpid, AX
	INVOKE_SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT ·tgkill(SB),NOSPLIT,$0
	MOVL	$SYS_tgkill, AX
	MOVL	tgid+0(FP), BX
	MOVL	tid+4(FP), CX
	MOVL	sig+8(FP), DX
	INVOKE_SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0-12
	MOVL	$SYS_setittimer, AX
	MOVL	mode+0(FP), BX
	MOVL	new+4(FP), CX
	MOVL	old+8(FP), DX
	INVOKE_SYSCALL
	RET

TEXT runtime·mincore(SB),NOSPLIT,$0-16
	MOVL	$SYS_mincore, AX
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	dst+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// func walltime1() (sec int64, nsec int32)
TEXT runtime·walltime1(SB), NOSPLIT, $8-12
	// We don't know how much stack space the VDSO code will need,
	// so switch to g0.

	MOVL	SP, BP	// Save old SP; BP unchanged by C code.

	get_tls(CX)
	MOVL	g(CX), AX
	MOVL	g_m(AX), SI // SI unchanged by C code.

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVL	m_vdsoPC(SI), CX
	MOVL	m_vdsoSP(SI), DX
	MOVL	CX, 0(SP)
	MOVL	DX, 4(SP)

	LEAL	sec+0(FP), DX
	MOVL	-4(DX), CX
	MOVL	CX, m_vdsoPC(SI)
	MOVL	DX, m_vdsoSP(SI)

	CMPL	AX, m_curg(SI)	// Only switch if on curg.
	JNE	noswitch

	MOVL	m_g0(SI), DX
	MOVL	(g_sched+gobuf_sp)(DX), SP	// Set SP to g0 stack

noswitch:
	SUBL	$16, SP		// Space for results
	ANDL	$~15, SP	// Align for C code

	// Stack layout, depending on call path:
	//  x(SP)   vDSO            INVOKE_SYSCALL
	//    12    ts.tv_nsec      ts.tv_nsec
	//     8    ts.tv_sec       ts.tv_sec
	//     4    &ts             -
	//     0    CLOCK_<id>      -

	MOVL	runtime·vdsoClockgettimeSym(SB), AX
	CMPL	AX, $0
	JEQ	fallback

	LEAL	8(SP), BX	// &ts (struct timespec)
	MOVL	BX, 4(SP)
	MOVL	$0, 0(SP)	// CLOCK_REALTIME
	CALL	AX
	JMP finish

fallback:
	MOVL	$SYS_clock_gettime, AX
	MOVL	$0, BX		// CLOCK_REALTIME
	LEAL	8(SP), CX
	INVOKE_SYSCALL

finish:
	MOVL	8(SP), AX	// sec
	MOVL	12(SP), BX	// nsec

	MOVL	BP, SP		// Restore real SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVL	4(SP), CX
	MOVL	CX, m_vdsoSP(SI)
	MOVL	0(SP), CX
	MOVL	CX, m_vdsoPC(SI)

	// sec is in AX, nsec in BX
	MOVL	AX, sec_lo+0(FP)
	MOVL	$0, sec_hi+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime1(SB), NOSPLIT, $8-8
	// Switch to g0 stack. See comment above in runtime·walltime.

	MOVL	SP, BP	// Save old SP; BP unchanged by C code.

	get_tls(CX)
	MOVL	g(CX), AX
	MOVL	g_m(AX), SI // SI unchanged by C code.

	// Set vdsoPC and vdsoSP for SIGPROF traceback.
	// Save the old values on stack and restore them on exit,
	// so this function is reentrant.
	MOVL	m_vdsoPC(SI), CX
	MOVL	m_vdsoSP(SI), DX
	MOVL	CX, 0(SP)
	MOVL	DX, 4(SP)

	LEAL	ret+0(FP), DX
	MOVL	-4(DX), CX
	MOVL	CX, m_vdsoPC(SI)
	MOVL	DX, m_vdsoSP(SI)

	CMPL	AX, m_curg(SI)	// Only switch if on curg.
	JNE	noswitch

	MOVL	m_g0(SI), DX
	MOVL	(g_sched+gobuf_sp)(DX), SP	// Set SP to g0 stack

noswitch:
	SUBL	$16, SP		// Space for results
	ANDL	$~15, SP	// Align for C code

	MOVL	runtime·vdsoClockgettimeSym(SB), AX
	CMPL	AX, $0
	JEQ	fallback

	LEAL	8(SP), BX	// &ts (struct timespec)
	MOVL	BX, 4(SP)
	MOVL	$1, 0(SP)	// CLOCK_MONOTONIC
	CALL	AX
	JMP finish

fallback:
	MOVL	$SYS_clock_gettime, AX
	MOVL	$1, BX		// CLOCK_MONOTONIC
	LEAL	8(SP), CX
	INVOKE_SYSCALL

finish:
	MOVL	8(SP), AX	// sec
	MOVL	12(SP), BX	// nsec

	MOVL	BP, SP		// Restore real SP
	// Restore vdsoPC, vdsoSP
	// We don't worry about being signaled between the two stores.
	// If we are not in a signal handler, we'll restore vdsoSP to 0,
	// and no one will care about vdsoPC. If we are in a signal handler,
	// we cannot receive another signal.
	MOVL	4(SP), CX
	MOVL	CX, m_vdsoSP(SI)
	MOVL	0(SP), CX
	MOVL	CX, m_vdsoPC(SI)

	// sec is in AX, nsec in BX
	// convert to DX:AX nsec
	MOVL	$1000000000, CX
	MULL	CX
	ADDL	BX, AX
	ADCL	$0, DX

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET

TEXT runtime·rtsigprocmask(SB),NOSPLIT,$0
	MOVL	$SYS_rt_sigprocmask, AX
	MOVL	how+0(FP), BX
	MOVL	new+4(FP), CX
	MOVL	old+8(FP), DX
	MOVL	size+12(FP), SI
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT $3
	RET

TEXT runtime·rt_sigaction(SB),NOSPLIT,$0
	MOVL	$SYS_rt_sigaction, AX
	MOVL	sig+0(FP), BX
	MOVL	new+4(FP), CX
	MOVL	old+8(FP), DX
	MOVL	size+12(FP), SI
	INVOKE_SYSCALL
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$12-16
	MOVL	fn+0(FP), AX
	MOVL	sig+4(FP), BX
	MOVL	info+8(FP), CX
	MOVL	ctx+12(FP), DX
	MOVL	SP, SI
	SUBL	$32, SP
	ANDL	$-15, SP	// align stack: handler might be a C function
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)	// save SI: handler might be a Go function
	CALL	AX
	MOVL	12(SP), AX
	MOVL	AX, SP
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$28
	// Save callee-saved C registers, since the caller may be a C signal handler.
	MOVL	BX, bx-4(SP)
	MOVL	BP, bp-8(SP)
	MOVL	SI, si-12(SP)
	MOVL	DI, di-16(SP)
	// We don't save mxcsr or the x87 control word because sigtrampgo doesn't
	// modify them.

	MOVL	sig+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	info+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	ctx+8(FP), BX
	MOVL	BX, 8(SP)
	CALL	runtime·sigtrampgo(SB)

	MOVL	di-16(SP), DI
	MOVL	si-12(SP), SI
	MOVL	bp-8(SP),  BP
	MOVL	bx-4(SP),  BX
	RET

TEXT runtime·cgoSigtramp(SB),NOSPLIT,$0
	JMP	runtime·sigtramp(SB)

TEXT runtime·sigreturn(SB),NOSPLIT,$0
	MOVL	$SYS_rt_sigreturn, AX
	// Sigreturn expects same SP as signal handler,
	// so cannot CALL 0x10(GS) here.
	INT	$0x80
	INT	$3	// not reached
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVL	$SYS_mmap2, AX
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	prot+8(FP), DX
	MOVL	flags+12(FP), SI
	MOVL	fd+16(FP), DI
	MOVL	off+20(FP), BP
	SHRL	$12, BP
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	ok
	NOTL	AX
	INCL	AX
	MOVL	$0, p+24(FP)
	MOVL	AX, err+28(FP)
	RET
ok:
	MOVL	AX, p+24(FP)
	MOVL	$0, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVL	$SYS_munmap, AX
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT $3
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVL	$SYS_madvise, AX
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	flags+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$0
	MOVL	$SYS_futex, AX
	MOVL	addr+0(FP), BX
	MOVL	op+4(FP), CX
	MOVL	val+8(FP), DX
	MOVL	ts+12(FP), SI
	MOVL	addr2+16(FP), DI
	MOVL	val3+20(FP), BP
	INVOKE_SYSCALL
	MOVL	AX, ret+24(FP)
	RET

// int32 clone(int32 flags, void *stack, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT,$0
	MOVL	$SYS_clone, AX
	MOVL	flags+0(FP), BX
	MOVL	stk+4(FP), CX
	MOVL	$0, DX	// parent tid ptr
	MOVL	$0, DI	// child tid ptr

	// Copy mp, gp, fn off parent stack for use by child.
	SUBL	$16, CX
	MOVL	mp+8(FP), SI
	MOVL	SI, 0(CX)
	MOVL	gp+12(FP), SI
	MOVL	SI, 4(CX)
	MOVL	fn+16(FP), SI
	MOVL	SI, 8(CX)
	MOVL	$1234, 12(CX)

	// cannot use CALL 0x10(GS) here, because the stack changes during the
	// system call (after CALL 0x10(GS), the child is still using the
	// parent's stack when executing its RET instruction).
	INT	$0x80

	// In parent, return.
	CMPL	AX, $0
	JEQ	3(PC)
	MOVL	AX, ret+20(FP)
	RET

	// Paranoia: check that SP is as we expect.
	NOP	SP // tell vet SP changed - stop checking offsets
	MOVL	12(SP), BP
	CMPL	BP, $1234
	JEQ	2(PC)
	INT	$3

	// Initialize AX to Linux tid
	MOVL	$SYS_gettid, AX
	INVOKE_SYSCALL

	MOVL	0(SP), BX	    // m
	MOVL	4(SP), DX	    // g
	MOVL	8(SP), SI	    // fn

	CMPL	BX, $0
	JEQ	nog
	CMPL	DX, $0
	JEQ	nog

	MOVL	AX, m_procid(BX)	// save tid as m->procid

	// set up ldt 7+id to point at m->tls.
	LEAL	m_tls(BX), BP
	MOVL	m_id(BX), DI
	ADDL	$7, DI	// m0 is LDT#7. count up.
	// setldt(tls#, &tls, sizeof tls)
	PUSHAL	// save registers
	PUSHL	$32	// sizeof tls
	PUSHL	BP	// &tls
	PUSHL	DI	// tls #
	CALL	runtime·setldt(SB)
	POPL	AX
	POPL	AX
	POPL	AX
	POPAL

	// Now segment is established. Initialize m, g.
	get_tls(AX)
	MOVL	DX, g(AX)
	MOVL	BX, g_m(DX)

	CALL	runtime·stackcheck(SB)	// smashes AX, CX
	MOVL	0(DX), DX	// paranoia; check they are not nil
	MOVL	0(BX), BX

	// more paranoia; check that stack splitting code works
	PUSHAL
	CALL	runtime·emptyfunc(SB)
	POPAL

nog:
	CALL	SI	// fn()
	CALL	exit1<>(SB)
	MOVL	$0x1234, 0x1005

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVL	$SYS_sigaltstack, AX
	MOVL	new+0(FP), BX
	MOVL	old+4(FP), CX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT	$3
	RET

// <asm-i386/ldt.h>
// struct user_desc {
//	unsigned int  entry_number;
//	unsigned long base_addr;
//	unsigned int  limit;
//	unsigned int  seg_32bit:1;
//	unsigned int  contents:2;
//	unsigned int  read_exec_only:1;
//	unsigned int  limit_in_pages:1;
//	unsigned int  seg_not_present:1;
//	unsigned int  useable:1;
// };
#define SEG_32BIT 0x01
// contents are the 2 bits 0x02 and 0x04.
#define CONTENTS_DATA 0x00
#define CONTENTS_STACK 0x02
#define CONTENTS_CODE 0x04
#define READ_EXEC_ONLY 0x08
#define LIMIT_IN_PAGES 0x10
#define SEG_NOT_PRESENT 0x20
#define USEABLE 0x40

// `-1` means the kernel will pick a TLS entry on the first setldt call,
// which happens during runtime init, and that we'll store back the saved
// entry and reuse that on subsequent calls when creating new threads.
DATA  runtime·tls_entry_number+0(SB)/4, $-1
GLOBL runtime·tls_entry_number(SB), NOPTR, $4

// setldt(int entry, int address, int limit)
// We use set_thread_area, which mucks with the GDT, instead of modify_ldt,
// which would modify the LDT, but is disabled on some kernels.
// The name, setldt, is a misnomer, although we leave this name as it is for
// the compatibility with other platforms.
TEXT runtime·setldt(SB),NOSPLIT,$32
	MOVL	base+4(FP), DX

#ifdef GOOS_android
	// Android stores the TLS offset in runtime·tls_g.
	SUBL	runtime·tls_g(SB), DX
	MOVL	DX, 0(DX)
#else
	/*
	 * When linking against the system libraries,
	 * we use its pthread_create and let it set up %gs
	 * for us.  When we do that, the private storage
	 * we get is not at 0(GS), but -4(GS).
	 * To insulate the rest of the tool chain from this
	 * ugliness, 8l rewrites 0(TLS) into -4(GS) for us.
	 * To accommodate that rewrite, we translate
	 * the address here and bump the limit to 0xffffffff (no limit)
	 * so that -4(GS) maps to 0(address).
	 * Also, the final 0(GS) (current 4(DX)) has to point
	 * to itself, to mimic ELF.
	 */
	ADDL	$0x4, DX	// address
	MOVL	DX, 0(DX)
#endif

	// get entry number
	MOVL	runtime·tls_entry_number(SB), CX

	// set up user_desc
	LEAL	16(SP), AX	// struct user_desc
	MOVL	CX, 0(AX)	// unsigned int entry_number
	MOVL	DX, 4(AX)	// unsigned long base_addr
	MOVL	$0xfffff, 8(AX)	// unsigned int limit
	MOVL	$(SEG_32BIT|LIMIT_IN_PAGES|USEABLE|CONTENTS_DATA), 12(AX)	// flag bits

	// call set_thread_area
	MOVL	AX, BX	// user_desc
	MOVL	$SYS_set_thread_area, AX
	// We can't call this via 0x10(GS) because this is called from setldt0 to set that up.
	INT     $0x80

	// breakpoint on error
	CMPL AX, $0xfffff001
	JLS 2(PC)
	INT $3

	// read allocated entry number back out of user_desc
	LEAL	16(SP), AX	// get our user_desc back
	MOVL	0(AX), AX

	// store entry number if the kernel allocated it
	CMPL	CX, $-1
	JNE	2(PC)
	MOVL	AX, runtime·tls_entry_number(SB)

	// compute segment selector - (entry*8+3)
	SHLL	$3, AX
	ADDL	$3, AX
	MOVW	AX, GS

	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVL	$SYS_sched_yield, AX
	INVOKE_SYSCALL
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$0
	MOVL	$SYS_sched_getaffinity, AX
	MOVL	pid+0(FP), BX
	MOVL	len+4(FP), CX
	MOVL	buf+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT,$0
	MOVL    $SYS_epoll_create, AX
	MOVL	size+0(FP), BX
	INVOKE_SYSCALL
	MOVL	AX, ret+4(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT,$0
	MOVL    $SYS_epoll_create1, AX
	MOVL	flags+0(FP), BX
	INVOKE_SYSCALL
	MOVL	AX, ret+4(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$0
	MOVL	$SYS_epoll_ctl, AX
	MOVL	epfd+0(FP), BX
	MOVL	op+4(FP), CX
	MOVL	fd+8(FP), DX
	MOVL	ev+12(FP), SI
	INVOKE_SYSCALL
	MOVL	AX, ret+16(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT,$0
	MOVL	$SYS_epoll_wait, AX
	MOVL	epfd+0(FP), BX
	MOVL	ev+4(FP), CX
	MOVL	nev+8(FP), DX
	MOVL	timeout+12(FP), SI
	INVOKE_SYSCALL
	MOVL	AX, ret+16(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL	$SYS_fcntl, AX
	MOVL	fd+0(FP), BX  // fd
	MOVL	$2, CX  // F_SETFD
	MOVL	$1, DX  // FD_CLOEXEC
	INVOKE_SYSCALL
	RET

// func runtime·setNonblock(fd int32)
TEXT runtime·setNonblock(SB),NOSPLIT,$0-4
	MOVL	$SYS_fcntl, AX
	MOVL	fd+0(FP), BX // fd
	MOVL	$3, CX // F_GETFL
	MOVL	$0, DX
	INVOKE_SYSCALL
	MOVL	fd+0(FP), BX // fd
	MOVL	$4, CX // F_SETFL
	MOVL	$0x800, DX // O_NONBLOCK
	ORL	AX, DX
	MOVL	$SYS_fcntl, AX
	INVOKE_SYSCALL
	RET

// int access(const char *name, int mode)
TEXT runtime·access(SB),NOSPLIT,$0
	MOVL	$SYS_access, AX
	MOVL	name+0(FP), BX
	MOVL	mode+4(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// int connect(int fd, const struct sockaddr *addr, socklen_t addrlen)
TEXT runtime·connect(SB),NOSPLIT,$0-16
	// connect is implemented as socketcall(NR_socket, 3, *(rest of args))
	// stack already should have fd, addr, addrlen.
	MOVL	$SYS_socketcall, AX
	MOVL	$3, BX  // connect
	LEAL	fd+0(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// int socket(int domain, int type, int protocol)
TEXT runtime·socket(SB),NOSPLIT,$0-16
	// socket is implemented as socketcall(NR_socket, 1, *(rest of args))
	// stack already should have domain, type, protocol.
	MOVL	$SYS_socketcall, AX
	MOVL	$1, BX  // socket
	LEAL	domain+0(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// func sbrk0() uintptr
TEXT runtime·sbrk0(SB),NOSPLIT,$0-4
	// Implemented as brk(NULL).
	MOVL	$SYS_brk, AX
	MOVL	$0, BX  // NULL
	INVOKE_SYSCALL
	MOVL	AX, ret+0(FP)
	RET
