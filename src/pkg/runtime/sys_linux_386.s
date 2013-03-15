// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for 386, Linux
//

#include "zasm_GOOS_GOARCH.h"

TEXT runtime·exit(SB),7,$0
	MOVL	$252, AX	// syscall number
	MOVL	4(SP), BX
	CALL	*runtime·_vdso(SB)
	INT $3	// not reached
	RET

TEXT runtime·exit1(SB),7,$0
	MOVL	$1, AX	// exit - exit the current os thread
	MOVL	4(SP), BX
	CALL	*runtime·_vdso(SB)
	INT $3	// not reached
	RET

TEXT runtime·open(SB),7,$0
	MOVL	$5, AX		// syscall - open
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·close(SB),7,$0
	MOVL	$6, AX		// syscall - close
	MOVL	4(SP), BX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·write(SB),7,$0
	MOVL	$4, AX		// syscall - write
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·read(SB),7,$0
	MOVL	$3, AX		// syscall - read
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·getrlimit(SB),7,$0
	MOVL	$191, AX		// syscall - ugetrlimit
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·usleep(SB),7,$8
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

TEXT runtime·raise(SB),7,$12
	MOVL	$224, AX	// syscall - gettid
	CALL	*runtime·_vdso(SB)
	MOVL	AX, BX	// arg 1 tid
	MOVL	sig+0(FP), CX	// arg 2 signal
	MOVL	$238, AX	// syscall - tkill
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·setitimer(SB),7,$0-24
	MOVL	$104, AX			// syscall - setitimer
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·mincore(SB),7,$0-24
	MOVL	$218, AX			// syscall - mincore
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	RET

// func now() (sec int64, nsec int32)
TEXT time·now(SB), 7, $32
	MOVL	$265, AX			// syscall - clock_gettime
	MOVL	$0, BX
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
TEXT runtime·nanotime(SB), 7, $32
	MOVL	$265, AX			// syscall - clock_gettime
	MOVL	$0, BX
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

	MOVL	ret+0(FP), DI
	MOVL	AX, 0(DI)
	MOVL	DX, 4(DI)
	RET

TEXT runtime·rtsigprocmask(SB),7,$0
	MOVL	$175, AX		// syscall entry
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT $3
	RET

TEXT runtime·rt_sigaction(SB),7,$0
	MOVL	$174, AX		// syscall - rt_sigaction
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·sigtramp(SB),7,$44
	get_tls(CX)

	// check that m exists
	MOVL	m(CX), BX
	CMPL	BX, $0
	JNE	5(PC)
	MOVL	sig+0(FP), BX
	MOVL	BX, 0(SP)
	CALL	runtime·badsignal(SB)
	RET

	// save g
	MOVL	g(CX), DI
	MOVL	DI, 20(SP)

	// g = m->gsignal
	MOVL	m(CX), BX
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

TEXT runtime·sigreturn(SB),7,$0
	MOVL	$173, AX	// rt_sigreturn
	// Sigreturn expects same SP as signal handler,
	// so cannot CALL *runtime._vsdo(SB) here.
	INT	$0x80
	INT $3	// not reached
	RET

TEXT runtime·mmap(SB),7,$0
	MOVL	$192, AX	// mmap2
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	MOVL	20(SP), DI
	MOVL	24(SP), BP
	SHRL	$12, BP
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	3(PC)
	NOTL	AX
	INCL	AX
	RET

TEXT runtime·munmap(SB),7,$0
	MOVL	$91, AX	// munmap
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	CALL	*runtime·_vdso(SB)
	CMPL	AX, $0xfffff001
	JLS	2(PC)
	INT $3
	RET

TEXT runtime·madvise(SB),7,$0
	MOVL	$219, AX	// madvise
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	// ignore failure - maybe pages are locked
	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),7,$0
	MOVL	$240, AX	// futex
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	MOVL	20(SP), DI
	MOVL	24(SP), BP
	CALL	*runtime·_vdso(SB)
	RET

// int32 clone(int32 flags, void *stack, M *mp, G *gp, void (*fn)(void));
TEXT runtime·clone(SB),7,$0
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
	JEQ	2(PC)
	RET

	// Paranoia: check that SP is as we expect.
	MOVL	12(SP), BP
	CMPL	BP, $1234
	JEQ	2(PC)
	INT	$3

	// Initialize AX to Linux tid
	MOVL	$224, AX
	CALL	*runtime·_vdso(SB)

	// In child on new stack.  Reload registers (paranoia).
	MOVL	0(SP), BX	// m
	MOVL	4(SP), DX	// g
	MOVL	8(SP), SI	// fn

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
	MOVL	BX, m(AX)

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
	RET

TEXT runtime·sigaltstack(SB),7,$-8
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
TEXT runtime·setldt(SB),7,$32
	MOVL	entry+0(FP), BX	// entry
	MOVL	address+4(FP), CX	// base address

	/*
	 * When linking against the system libraries,
	 * we use its pthread_create and let it set up %gs
	 * for us.  When we do that, the private storage
	 * we get is not at 0(GS), 4(GS), but -8(GS), -4(GS).
	 * To insulate the rest of the tool chain from this
	 * ugliness, 8l rewrites 0(GS) into -8(GS) for us.
	 * To accommodate that rewrite, we translate
	 * the address here and bump the limit to 0xffffffff (no limit)
	 * so that -8(GS) maps to 0(address).
	 * Also, the final 0(GS) (current 8(CX)) has to point
	 * to itself, to mimic ELF.
	 */
	ADDL	$0x8, CX	// address
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

TEXT runtime·osyield(SB),7,$0
	MOVL	$158, AX
	CALL	*runtime·_vdso(SB)
	RET

TEXT runtime·sched_getaffinity(SB),7,$0
	MOVL	$242, AX		// syscall - sched_getaffinity
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	CALL	*runtime·_vdso(SB)
	RET

// int32 runtime·epollcreate(int32 size);
TEXT runtime·epollcreate(SB),7,$0
	MOVL    $254, AX
	MOVL	4(SP), BX
	CALL	*runtime·_vdso(SB)
	RET

// int32 runtime·epollcreate1(int32 flags);
TEXT runtime·epollcreate1(SB),7,$0
	MOVL    $329, AX
	MOVL	4(SP), BX
	CALL	*runtime·_vdso(SB)
	RET

// int32 runtime·epollctl(int32 epfd, int32 op, int32 fd, EpollEvent *ev);
TEXT runtime·epollctl(SB),7,$0
	MOVL	$255, AX
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	CALL	*runtime·_vdso(SB)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
TEXT runtime·epollwait(SB),7,$0
	MOVL	$256, AX
	MOVL	4(SP), BX
	MOVL	8(SP), CX
	MOVL	12(SP), DX
	MOVL	16(SP), SI
	CALL	*runtime·_vdso(SB)
	RET

// void runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),7,$0
	MOVL	$55, AX  // fcntl
	MOVL	4(SP), BX  // fd
	MOVL	$2, CX  // F_SETFD
	MOVL	$1, DX  // FD_CLOEXEC
	CALL	*runtime·_vdso(SB)
	RET
