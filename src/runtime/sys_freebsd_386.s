// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for 386, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·sys_umtx_op(SB),NOSPLIT,$-4
	MOVL	$454, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+20(FP)
	RET

TEXT runtime·thr_new(SB),NOSPLIT,$-4
	MOVL	$455, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+8(FP)
	RET

// Called by OS using C ABI.
TEXT runtime·thr_start(SB),NOSPLIT,$0
	NOP	SP	// tell vet SP changed - stop checking offsets
	MOVL	4(SP), AX // m
	MOVL	m_g0(AX), BX
	LEAL	m_tls(AX), BP
	MOVL	m_id(AX), DI
	ADDL	$7, DI
	PUSHAL
	PUSHL	$32
	PUSHL	BP
	PUSHL	DI
	CALL	runtime·setldt(SB)
	POPL	AX
	POPL	AX
	POPL	AX
	POPAL
	get_tls(CX)
	MOVL	BX, g(CX)

	MOVL	AX, g_m(BX)
	CALL	runtime·stackcheck(SB)		// smashes AX
	CALL	runtime·mstart(SB)

	MOVL	0, AX			// crash (not reached)

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVL	$1, AX
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

GLOBL exitStack<>(SB),RODATA,$8
DATA exitStack<>+0x00(SB)/4, $0
DATA exitStack<>+0x04(SB)/4, $0

// func exitThread(wait *uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-4
	MOVL	wait+0(FP), AX
	// We're done using the stack.
	MOVL	$0, (AX)
	// thr_exit takes a single pointer argument, which it expects
	// on the stack. We want to pass 0, so switch over to a fake
	// stack of 0s. It won't write to the stack.
	MOVL	$exitStack<>(SB), SP
	MOVL	$431, AX	// thr_exit
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT,$-4
	MOVL	$5, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-4
	MOVL	$6, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-4
	MOVL	$3, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+12(FP)
	RET

// func pipe() (r, w int32, errno int32)
TEXT runtime·pipe(SB),NOSPLIT,$8-12
	MOVL	$42, AX
	INT	$0x80
	JAE	ok
	MOVL	$0, r+0(FP)
	MOVL	$0, w+4(FP)
	MOVL	AX, errno+8(FP)
	RET
ok:
	MOVL	AX, r+0(FP)
	MOVL	DX, w+4(FP)
	MOVL	$0, errno+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$12-16
	MOVL	$542, AX
	LEAL	r+4(FP), BX
	MOVL	BX, 4(SP)
	MOVL	flags+0(FP), BX
	MOVL	BX, 8(SP)
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, errno+12(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$-4
	MOVL	$4, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·thr_self(SB),NOSPLIT,$8-4
	// thr_self(&0(FP))
	LEAL	ret+0(FP), AX
	MOVL	AX, 4(SP)
	MOVL	$432, AX
	INT	$0x80
	RET

TEXT runtime·thr_kill(SB),NOSPLIT,$-4
	// thr_kill(tid, sig)
	MOVL	$433, AX
	INT	$0x80
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$16
	// getpid
	MOVL	$20, AX
	INT	$0x80
	// kill(self, sig)
	MOVL	AX, 4(SP)
	MOVL	sig+0(FP), AX
	MOVL	AX, 8(SP)
	MOVL	$37, AX
	INT	$0x80
	RET

TEXT runtime·mmap(SB),NOSPLIT,$32
	LEAL addr+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVL	$0, AX	// top 32 bits of file offset
	STOSL
	MOVL	$477, AX
	INT	$0x80
	JAE	ok
	MOVL	$0, p+24(FP)
	MOVL	AX, err+28(FP)
	RET
ok:
	MOVL	AX, p+24(FP)
	MOVL	$0, err+28(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$-4
	MOVL	$73, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$-4
	MOVL	$75, AX	// madvise
	INT	$0x80
	JAE	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $-4
	MOVL	$83, AX
	INT	$0x80
	RET

// func fallback_walltime() (sec int64, nsec int32)
TEXT runtime·fallback_walltime(SB), NOSPLIT, $32-12
	MOVL	$232, AX // clock_gettime
	LEAL	12(SP), BX
	MOVL	$0, 4(SP)	// CLOCK_REALTIME
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	12(SP), AX	// sec
	MOVL	16(SP), BX	// nsec

	// sec is in AX, nsec in BX
	MOVL	AX, sec_lo+0(FP)
	MOVL	$0, sec_hi+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

// func fallback_nanotime() int64
TEXT runtime·fallback_nanotime(SB), NOSPLIT, $32-8
	MOVL	$232, AX
	LEAL	12(SP), BX
	MOVL	$4, 4(SP)	// CLOCK_MONOTONIC
	MOVL	BX, 8(SP)
	INT	$0x80
	MOVL	12(SP), AX	// sec
	MOVL	16(SP), BX	// nsec

	// sec is in AX, nsec in BX
	// convert to DX:AX nsec
	MOVL	$1000000000, CX
	MULL	CX
	ADDL	BX, AX
	ADCL	$0, DX

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET


TEXT runtime·asmSigaction(SB),NOSPLIT,$-4
	MOVL	$416, AX
	INT	$0x80
	MOVL	AX, ret+12(FP)
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$12-16
	MOVL	fn+0(FP), AX
	MOVL	sig+4(FP), BX
	MOVL	info+8(FP), CX
	MOVL	ctx+12(FP), DX
	MOVL	SP, SI
	SUBL	$32, SP
	ANDL	$~15, SP	// align stack: handler might be a C function
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)	// save SI: handler might be a Go function
	CALL	AX
	MOVL	12(SP), AX
	MOVL	AX, SP
	RET

// Called by OS using C ABI.
TEXT runtime·sigtramp(SB),NOSPLIT,$12
	NOP	SP	// tell vet SP changed - stop checking offsets
	MOVL	16(SP), BX	// signo
	MOVL	BX, 0(SP)
	MOVL	20(SP), BX // info
	MOVL	BX, 4(SP)
	MOVL	24(SP), BX // context
	MOVL	BX, 8(SP)
	CALL	runtime·sigtrampgo(SB)

	// call sigreturn
	MOVL	24(SP), AX	// context
	MOVL	$0, 0(SP)	// syscall gap
	MOVL	AX, 4(SP)
	MOVL	$417, AX	// sigreturn(ucontext)
	INT	$0x80
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$0
	MOVL	$53, AX
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep(SB),NOSPLIT,$20
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVL	AX, 12(SP)		// tv_sec
	MOVL	$1000, AX
	MULL	DX
	MOVL	AX, 16(SP)		// tv_nsec

	MOVL	$0, 0(SP)
	LEAL	12(SP), AX
	MOVL	AX, 4(SP)		// arg 1 - rqtp
	MOVL	$0, 8(SP)		// arg 2 - rmtp
	MOVL	$240, AX		// sys_nanosleep
	INT	$0x80
	RET

/*
descriptor entry format for system call
is the native machine format, ugly as it is:

	2-byte limit
	3-byte base
	1-byte: 0x80=present, 0x60=dpl<<5, 0x1F=type
	1-byte: 0x80=limit is *4k, 0x40=32-bit operand size,
		0x0F=4 more bits of limit
	1 byte: 8 more bits of base

int i386_get_ldt(int, union ldt_entry *, int);
int i386_set_ldt(int, const union ldt_entry *, int);

*/

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),NOSPLIT,$32
	MOVL	base+4(FP), BX
	// see comment in sys_linux_386.s; freebsd is similar
	ADDL	$0x4, BX

	// set up data_desc
	LEAL	16(SP), AX	// struct data_desc
	MOVL	$0, 0(AX)
	MOVL	$0, 4(AX)

	MOVW	BX, 2(AX)
	SHRL	$16, BX
	MOVB	BX, 4(AX)
	SHRL	$8, BX
	MOVB	BX, 7(AX)

	MOVW	$0xffff, 0(AX)
	MOVB	$0xCF, 6(AX)	// 32-bit operand, 4k limit unit, 4 more bits of limit

	MOVB	$0xF2, 5(AX)	// r/w data descriptor, dpl=3, present

	// call i386_set_ldt(entry, desc, 1)
	MOVL	$0xffffffff, 0(SP)	// auto-allocate entry and return in AX
	MOVL	AX, 4(SP)
	MOVL	$1, 8(SP)
	CALL	i386_set_ldt<>(SB)

	// compute segment selector - (entry*8+7)
	SHLL	$3, AX
	ADDL	$7, AX
	MOVW	AX, GS
	RET

TEXT i386_set_ldt<>(SB),NOSPLIT,$16
	LEAL	args+0(FP), AX	// 0(FP) == 4(SP) before SP got moved
	MOVL	$0, 0(SP)	// syscall gap
	MOVL	$1, 4(SP)
	MOVL	AX, 8(SP)
	MOVL	$165, AX
	INT	$0x80
	JAE	2(PC)
	INT	$3
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$28
	LEAL	mib+0(FP), SI
	LEAL	4(SP), DI
	CLD
	MOVSL				// arg 1 - name
	MOVSL				// arg 2 - namelen
	MOVSL				// arg 3 - oldp
	MOVSL				// arg 4 - oldlenp
	MOVSL				// arg 5 - newp
	MOVSL				// arg 6 - newlen
	MOVL	$202, AX		// sys___sysctl
	INT	$0x80
	JAE	4(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$331, AX		// sys_sched_yield
	INT	$0x80
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$16
	MOVL	$0, 0(SP)		// syscall gap
	MOVL	how+0(FP), AX		// arg 1 - how
	MOVL	AX, 4(SP)
	MOVL	new+4(FP), AX
	MOVL	AX, 8(SP)		// arg 2 - set
	MOVL	old+8(FP), AX
	MOVL	AX, 12(SP)		// arg 3 - oset
	MOVL	$340, AX		// sys_sigprocmask
	INT	$0x80
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVL	$362, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	$363, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

// int32 runtime·closeonexec(int32 fd);
TEXT runtime·closeonexec(SB),NOSPLIT,$32
	MOVL	$92, AX		// fcntl
	// 0(SP) is where the caller PC would be; kernel skips it
	MOVL	fd+0(FP), BX
	MOVL	BX, 4(SP)	// fd
	MOVL	$2, 8(SP)	// F_SETFD
	MOVL	$1, 12(SP)	// FD_CLOEXEC
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	RET

// func runtime·setNonblock(fd int32)
TEXT runtime·setNonblock(SB),NOSPLIT,$16-4
	MOVL	$92, AX // fcntl
	MOVL	fd+0(FP), BX // fd
	MOVL	BX, 4(SP)
	MOVL	$3, 8(SP) // F_GETFL
	MOVL	$0, 12(SP)
	INT	$0x80
	MOVL	fd+0(FP), BX // fd
	MOVL	BX, 4(SP)
	MOVL	$4, 8(SP) // F_SETFL
	ORL	$4, AX // O_NONBLOCK
	MOVL	AX, 12(SP)
	MOVL	$92, AX // fcntl
	INT	$0x80
	RET

// func cpuset_getaffinity(level int, which int, id int64, size int, mask *byte) int32
TEXT runtime·cpuset_getaffinity(SB), NOSPLIT, $0-28
	MOVL	$487, AX
	INT	$0x80
	JAE	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

GLOBL runtime·tlsoffset(SB),NOPTR,$4
