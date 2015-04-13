// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for 386, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·exit(SB),NOSPLIT,$0
	MOVL	$252, AX	// syscall number
	MOVL	code+0(FP), BX
	CALL	*runtime·_vdso(SB)
	INT $3	// not reached
	RET

TEXT runtime·exit1(SB),NOSPLIT,$0
	MOVL	$1, AX	// exit - exit the current os thread
	MOVL	code+0(FP), BX
	CALL	*runtime·_vdso(SB)
	INT $3	// not reached
	RET

TEXT runtime·open(SB),NOSPLIT,$0
	MOVL	$5, AX		// syscall - open
	MOVL	name+0(FP), BX
	MOVL	mode+4(FP), CX
	MOVL	perm+8(FP), DX
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVL	$6, AX		// syscall - close
	MOVL	fd+0(FP), BX
	CALL	*runtime·_vdso(SB)
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
	CALL	*runtime·_vdso(SB)
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
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$0
	MOVL	$191, AX		// syscall - ugetrlimit
	MOVL	kind+0(FP), BX
	MOVL	limit+4(FP), CX
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·usleep(SB),NOSPLIT,$8
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 0(SP)
	MOVL	DX, 4(SP)

	// select(0, 0, 0, 0, &tv)
	MOVL	$142, AX
	MOVL	$0, BX
	MOVL	$0, CX
	MOVL	$0, DX
	MOVL	$0, SI
	LEAL	0(SP), DI
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·raise(SB),NOSPLIT,$12
	MOVL	$224, AX	// syscall - gettid
	CALL	*runtime·_vdso(SB)
	MOVL	AX, BX	// arg 1 tid
	MOVL	sig+0(FP), CX	// arg 2 signal
	MOVL	$238, AX	// syscall - tkill
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$12
	MOVL	$20, AX	// syscall - getpid
	CALL	*runtime·_vdso(SB)
	MOVL	AX, BX	// arg 1 pid
	MOVL	sig+0(FP), CX	// arg 2 signal
	MOVL	$37, AX	// syscall - kill
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0-12
	MOVL	$104, AX			// syscall - setitimer
	MOVL	mode+0(FP), BX
	MOVL	new+4(FP), CX
	MOVL	old+8(FP), DX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·mincore(SB),NOSPLIT,$0-16
	MOVL	$218, AX			// syscall - mincore
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	dst+8(FP), DX
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+12(FP)
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), NOSPLIT, $32
	MOVL	$265, AX			// syscall - clock_gettime
	MOVL	$0, BX		// CLOCK_REALTIME
	LEAL	8(SP), CX
	MOVL	$0, DX
	CALL	*runtime·_vdso(SB)
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
	CALL	*runtime·_vdso(SB)
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
	MOVL	sig+0(FP), BX
	MOVL	new+4(FP), CX
	MOVL	old+8(FP), DX
	MOVL	size+12(FP), SI
	CALL	*runtime·_vdso(SB)
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
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$44
	get_tls(CX)

	// check that g exists
	MOVL	g(CX), DI
	CMPL	DI, $0
	JNE	6(PC)
	MOVL	sig+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	$runtime·badsignal(SB), AX
	CALL	AX
	RET

	// save g
	MOVL	DI, 20(SP)

	// g = m->gsignal
	MOVL	g_m(DI), BX
	MOVL	m_gsignal(BX), BX
	MOVL	BX, g(CX)

	// copy arguments for call to sighandler
	MOVL	sig+0(FP), BX
	MOVL	BX, 0(SP)
	MOVL	info+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	context+8(FP), BX
	MOVL	BX, 8(SP)
	MOVL	DI, 12(SP)

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(CX)
	MOVL	20(SP), BX
	MOVL	BX, g(CX)

	RET

TEXT runtime·sigreturn(SB),NOSPLIT,$0
	MOVL	$173, AX	// rt_sigreturn
	// Sigreturn expects same SP as signal handler,
	// so cannot CALL *runtime._vsdo(SB) here.
	INT	$0x80
	INT $3	// not reached
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
	CALL	*runtime·_vdso(SB)
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
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT $3
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVL	$219, AX	// madvise
	MOVL	addr+0(FP), BX
	MOVL	n+4(FP), CX
	MOVL	flags+8(FP), DX
	CALL	*runtime·_vdso(SB)
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
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+24(FP)
	RET

// int32 clone(int32 flags, void *stack, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),NOSPLIT,$0
	MOVL	$120, AX	// clone
	MOVL	flags+4(SP), BX
	MOVL	stack+8(SP), CX
	MOVL	$0, DX	// parent tid ptr
	MOVL	$0, DI	// child tid ptr

	// Copy mp, gp, fn off parent stack for use by child.
	SUBL	$16, CX
	MOVL	mm+12(SP), SI
	MOVL	SI, 0(CX)
	MOVL	gg+16(SP), SI
	MOVL	SI, 4(CX)
	MOVL	fn+20(SP), SI
	MOVL	SI, 8(CX)
	MOVL	$1234, 12(CX)

	// cannot use CALL *runtime·_vdso(SB) here, because
	// the stack changes during the system call (after
	// CALL *runtime·_vdso(SB), the child is still using
	// the parent's stack when executing its RET instruction).
	INT	$0x80

	// In parent, return.
	CMPL	AX, $0
	JEQ	3(PC)
	MOVL	AX, ret+20(FP)
	RET

	// Paranoia: check that SP is as we expect.
	MOVL	mm+8(FP), BP
	CMPL	BP, $1234
	JEQ	2(PC)
	INT	$3

	// Initialize AX to Linux tid
	MOVL	$224, AX
	CALL	*runtime·_vdso(SB)

	// In child on new stack.  Reload registers (paranoia).
	MOVL	0(SP), BX	// m
	MOVL	flags+0(FP), DX	// g
	MOVL	stk+4(FP), SI	// fn

	MOVL	AX, m_procid(BX)	// save tid as m->procid

	// set up ldt 7+id to point at m->tls.
	// newosproc left the id in tls[0].
	LEAL	m_tls(BX), BP
	MOVL	0(BP), DI
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

	// Now segment is established.  Initialize m, g.
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

	CALL	SI	// fn()
	CALL	runtime·exit1(SB)
	MOVL	$0x1234, 0x1005

// int32 clone0(int32 flags, void *stack, void* fn, void* fnarg);
TEXT runtime·clone0(SB),NOSPLIT,$0
	// TODO(spetrovic): Implement this method.
	MOVL	$-1, ret+16(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVL	$186, AX	// sigaltstack
	MOVL	new+4(SP), BX
	MOVL	old+8(SP), CX
	CALL	*runtime·_vdso(SB)
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

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),NOSPLIT,$32
	MOVL	entry+0(FP), BX	// entry
	MOVL	address+4(FP), CX	// base address

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
	 * Also, the final 0(GS) (current 4(CX)) has to point
	 * to itself, to mimic ELF.
	 */
	ADDL	$0x4, CX	// address
	MOVL	CX, 0(CX)

	// set up user_desc
	LEAL	16(SP), AX	// struct user_desc
	MOVL	BX, 0(AX)
	MOVL	CX, 4(AX)
	MOVL	$0xfffff, 8(AX)
	MOVL	$(SEG_32BIT|LIMIT_IN_PAGES|USEABLE|CONTENTS_DATA), 12(AX)	// flag bits

	// call modify_ldt
	MOVL	$1, BX	// func = 1 (write)
	MOVL	AX, CX	// user_desc
	MOVL	$16, DX	// sizeof(user_desc)
	MOVL	$123, AX	// syscall - modify_ldt
	CALL	*runtime·_vdso(SB)

	// breakpoint on error
	CMPL AX, $0xfffff001
	JLS 2(PC)
	INT $3

	// compute segment selector - (entry*8+7)
	MOVL	entry+0(FP), AX
	SHLL	$3, AX
	ADDL	$7, AX
	MOVW	AX, GS

	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	MOVL	$158, AX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·sched_getaffinity(SB),NOSPLIT,$0
	MOVL	$242, AX		// syscall - sched_getaffinity
	MOVL	pid+0(FP), BX
	MOVL	len+4(FP), CX
	MOVL	buf+8(FP), DX
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+12(FP)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),NOSPLIT,$0
	MOVL    $254, AX
	MOVL	size+0(FP), BX
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+4(FP)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),NOSPLIT,$0
	MOVL    $329, AX
	MOVL	flags+0(FP), BX
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+4(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$0
	MOVL	$255, AX
	MOVL	epfd+0(FP), BX
	MOVL	op+4(FP), CX
	MOVL	fd+8(FP), DX
	MOVL	ev+12(FP), SI
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+16(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),NOSPLIT,$0
	MOVL	$256, AX
	MOVL	epfd+0(FP), BX
	MOVL	ev+4(FP), CX
	MOVL	nev+8(FP), DX
	MOVL	timeout+12(FP), SI
	CALL	*runtime·_vdso(SB)
	MOVL	AX, ret+16(FP)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVL	$55, AX  // fcntl
	MOVL	fd+0(FP), BX  // fd
	MOVL	$2, CX  // F_SETFD
	MOVL	$1, DX  // FD_CLOEXEC
	CALL	*runtime·_vdso(SB)
	RET
