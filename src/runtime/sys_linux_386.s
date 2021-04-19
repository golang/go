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

TEXT runtime·exit(SB),NOSPLIT,$0
	MOVL	$252, AX	// syscall number
	MOVL	code+0(FP), BX
	INVOKE_SYSCALL
	INT $3	// not reached
	RET

TEXT runtime·exit1(SB),NOSPLIT,$0
	MOVL	$1, AX	// exit - exit the current os thread
	MOVL	code+0(FP), BX
	INVOKE_SYSCALL
	INT $3	// not reached
	RET

TEXT runtime·open(SB),NOSPLIT,$0
	MOVL	$5, AX		// syscall - open
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
	MOVL	$6, AX		// syscall - close
	MOVL	fd+0(FP), BX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$0
	MOVL	$4, AX		// syscall - write
	MOVL	fd+0(FP), BX
	MOVL	p+4(FP), CX
	MOVL	n+8(FP), DX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVL	$3, AX		// syscall - read
	MOVL	fd+0(FP), BX
	MOVL	p+4(FP), CX
	MOVL	n+8(FP), DX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$0
	MOVL	$191, AX		// syscall - ugetrlimit
	MOVL	kind+0(FP), BX
	MOVL	limit+4(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$8
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 0(SP)
	MOVL	$1000, AX	// usec to nsec
	MULL	DX
	MOVL	DX, 4(SP)

	// pselect6(0, 0, 0, 0, &ts, 0)
	MOVL	$308, AX
	MOVL	$0, BX
	MOVL	$0, CX
	MOVL	$0, DX
	MOVL	$0, SI
	LEAL	0(SP), DI
	MOVL	$0, BP
	INVOKE_SYSCALL
	RET

TEXT runtime·gettid(SB),NOSPLIT,$0-4
	MOVL	$224, AX	// syscall - gettid
	INVOKE_SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·raise(SB),NOSPLIT,$12
	MOVL	$224, AX	// syscall - gettid
	INVOKE_SYSCALL
	MOVL	AX, BX	// arg 1 tid
	MOVL	sig+0(FP), CX	// arg 2 signal
	MOVL	$238, AX	// syscall - tkill
	INVOKE_SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVL	$20, AX	// syscall - getpid
	INVOKE_SYSCALL
	MOVL	AX, BX	// arg 1 pid
	MOVL	sig+0(FP), CX	// arg 2 signal
	MOVL	$37, AX	// syscall - kill
	INVOKE_SYSCALL
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0-12
	MOVL	$104, AX			// syscall - setitimer
	MOVL	mode+0(FP), BX
	MOVL	new+4(FP), CX
	MOVL	old+8(FP), DX
	INVOKE_SYSCALL
	RET

TEXT runtime·mincore(SB),NOSPLIT,$0-16
	MOVL	$218, AX			// syscall - mincore
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	dst+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVL	$265, AX			// syscall - clock_gettime
	MOVL	$0, BX		// CLOCK_REALTIME
	LEAL	8(SP), CX
	MOVL	$0, DX
	INVOKE_SYSCALL
	MOVL	8(SP), AX	// sec
	MOVL	12(SP), BX	// nsec

	// sec is in AX, nsec in BX
	MOVL	AX, sec+0(FP)
	MOVL	$0, sec+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB), NOSPLIT, $32
	MOVL	$265, AX			// syscall - clock_gettime
	MOVL	$1, BX		// CLOCK_MONOTONIC
	LEAL	8(SP), CX
	MOVL	$0, DX
	INVOKE_SYSCALL
	MOVL	8(SP), AX	// sec
	MOVL	12(SP), BX	// nsec

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
	MOVL	$175, AX		// syscall entry
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
	MOVL	$174, AX		// syscall - rt_sigaction
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
	MOVL	$173, AX	// rt_sigreturn
	// Sigreturn expects same SP as signal handler,
	// so cannot CALL 0x10(GS) here.
	INT	$0x80
	INT	$3	// not reached
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVL	$192, AX	// mmap2
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	prot+8(FP), DX
	MOVL	flags+12(FP), SI
	MOVL	fd+16(FP), DI
	MOVL	off+20(FP), BP
	SHRL	$12, BP
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	3(PC)
	NOTL	AX
	INCL	AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVL	$91, AX	// munmap
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT $3
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVL	$219, AX	// madvise
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	flags+8(FP), DX
	INVOKE_SYSCALL
	// ignore failure - maybe pages are locked
	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$0
	MOVL	$240, AX	// futex
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
	MOVL	$120, AX	// clone
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
	MOVL	12(SP), BP
	CMPL	BP, $1234
	JEQ	2(PC)
	INT	$3

	// Initialize AX to Linux tid
	MOVL	$224, AX
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
	CALL	runtime·exit1(SB)
	MOVL	$0x1234, 0x1005

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVL	$186, AX	// sigaltstack
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
	MOVL	address+4(FP), DX	// base address

#ifdef GOOS_android
	/*
	 * Same as in sys_darwin_386.s:/ugliness, different constant.
	 * address currently holds m->tls, which must be %gs:0xf8.
	 * See cgo/gcc_android_386.c for the derivation of the constant.
	 */
	SUBL	$0xf8, DX
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
	MOVL	$243, AX	// syscall - set_thread_area
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
	MOVL	$158, AX
	INVOKE_SYSCALL
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$0
	MOVL	$242, AX		// syscall - sched_getaffinity
	MOVL	pid+0(FP), BX
	MOVL	len+4(FP), CX
	MOVL	buf+8(FP), DX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT,$0
	MOVL    $254, AX
	MOVL	size+0(FP), BX
	INVOKE_SYSCALL
	MOVL	AX, ret+4(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT,$0
	MOVL    $329, AX
	MOVL	flags+0(FP), BX
	INVOKE_SYSCALL
	MOVL	AX, ret+4(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$0
	MOVL	$255, AX
	MOVL	epfd+0(FP), BX
	MOVL	op+4(FP), CX
	MOVL	fd+8(FP), DX
	MOVL	ev+12(FP), SI
	INVOKE_SYSCALL
	MOVL	AX, ret+16(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT,$0
	MOVL	$256, AX
	MOVL	epfd+0(FP), BX
	MOVL	ev+4(FP), CX
	MOVL	nev+8(FP), DX
	MOVL	timeout+12(FP), SI
	INVOKE_SYSCALL
	MOVL	AX, ret+16(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL	$55, AX  // fcntl
	MOVL	fd+0(FP), BX  // fd
	MOVL	$2, CX  // F_SETFD
	MOVL	$1, DX  // FD_CLOEXEC
	INVOKE_SYSCALL
	RET

// int access(const char *name, int mode)
TEXT runtime·access(SB),NOSPLIT,$0
	MOVL	$33, AX  // syscall - access
	MOVL	name+0(FP), BX
	MOVL	mode+4(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+8(FP)
	RET

// int connect(int fd, const struct sockaddr *addr, socklen_t addrlen)
TEXT runtime·connect(SB),NOSPLIT,$0-16
	// connect is implemented as socketcall(NR_socket, 3, *(rest of args))
	// stack already should have fd, addr, addrlen.
	MOVL	$102, AX  // syscall - socketcall
	MOVL	$3, BX  // connect
	LEAL	fd+0(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET

// int socket(int domain, int type, int protocol)
TEXT runtime·socket(SB),NOSPLIT,$0-16
	// socket is implemented as socketcall(NR_socket, 1, *(rest of args))
	// stack already should have domain, type, protocol.
	MOVL	$102, AX  // syscall - socketcall
	MOVL	$1, BX  // socket
	LEAL	domain+0(FP), CX
	INVOKE_SYSCALL
	MOVL	AX, ret+12(FP)
	RET
