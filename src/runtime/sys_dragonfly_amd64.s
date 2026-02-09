// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// System calls and other sys.stuff for AMD64, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "cgo/abi_amd64.h"

TEXT runtime·sys_umtx_sleep(SB),NOSPLIT,$0
	MOVQ addr+0(FP), DI		// arg 1 - ptr
	MOVL val+8(FP), SI		// arg 2 - value
	MOVL timeout+12(FP), DX		// arg 3 - timeout
	MOVL $469, AX		// umtx_sleep
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·sys_umtx_wakeup(SB),NOSPLIT,$0
	MOVQ addr+0(FP), DI		// arg 1 - ptr
	MOVL val+8(FP), SI		// arg 2 - count
	MOVL $470, AX		// umtx_wakeup
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·lwp_create(SB),NOSPLIT,$0
	MOVQ param+0(FP), DI		// arg 1 - params
	MOVL $495, AX		// lwp_create
	SYSCALL
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·lwp_start(SB),NOSPLIT,$0
	MOVQ	DI, R13 // m

	// set up FS to point at m->tls
	LEAQ	m_tls(R13), DI
	CALL	runtime·settls(SB)	// smashes DI

	// set up m, g
	get_tls(CX)
	MOVQ	m_g0(R13), DI
	MOVQ	R13, g_m(DI)
	MOVQ	DI, g(CX)

	CALL	runtime·stackcheck(SB)
	CALL	runtime·mstart(SB)

	MOVQ 0, AX			// crash (not reached)

// Exit the entire program (like C exit)
TEXT runtime·exit(SB),NOSPLIT,$-8
	MOVL	code+0(FP), DI		// arg 1 exit status
	MOVL	$1, AX
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	RET

// func exitThread(wait *atomic.Uint32)
TEXT runtime·exitThread(SB),NOSPLIT,$0-8
	MOVQ	wait+0(FP), AX
	// We're done using the stack.
	MOVL	$0, (AX)
	MOVL	$0x10000, DI	// arg 1 how - EXTEXIT_LWP
	MOVL	$0, SI		// arg 2 status
	MOVL	$0, DX		// arg 3 addr
	MOVL	$494, AX	// extexit
	SYSCALL
	MOVL	$0xf1, 0xf1  // crash
	JMP	0(PC)

TEXT runtime·open(SB),NOSPLIT,$-8
	MOVQ	name+0(FP), DI		// arg 1 pathname
	MOVL	mode+8(FP), SI		// arg 2 flags
	MOVL	perm+12(FP), DX		// arg 3 mode
	MOVL	$5, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+16(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$-8
	MOVL	fd+0(FP), DI		// arg 1 fd
	MOVL	$6, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+8(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$-8
	MOVL	fd+0(FP), DI		// arg 1 fd
	MOVQ	p+8(FP), SI		// arg 2 buf
	MOVL	n+16(FP), DX		// arg 3 count
	MOVL	$3, AX
	SYSCALL
	JCC	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+24(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT,$0-20
	MOVL	$0, DI
	// dragonfly expects flags as the 2nd argument
	MOVL	flags+0(FP), SI
	MOVL	$538, AX
	SYSCALL
	JCC	pipe2ok
	MOVL	$-1,r+8(FP)
	MOVL	$-1,w+12(FP)
	MOVL	AX, errno+16(FP)
	RET
pipe2ok:
	MOVL	AX, r+8(FP)
	MOVL	DX, w+12(FP)
	MOVL	$0, errno+16(FP)
	RET

TEXT runtime·write1(SB),NOSPLIT,$-8
	MOVQ	fd+0(FP), DI		// arg 1 fd
	MOVQ	p+8(FP), SI		// arg 2 buf
	MOVL	n+16(FP), DX		// arg 3 count
	MOVL	$4, AX
	SYSCALL
	JCC	2(PC)
	NEGL	AX			// caller expects negative errno
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·lwp_gettid(SB),NOSPLIT,$0-4
	MOVL	$496, AX	// lwp_gettid
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET

TEXT runtime·lwp_kill(SB),NOSPLIT,$0-16
	MOVL	pid+0(FP), DI	// arg 1 - pid
	MOVL	tid+4(FP), SI	// arg 2 - tid
	MOVQ	sig+8(FP), DX	// arg 3 - signum
	MOVL	$497, AX	// lwp_kill
	SYSCALL
	RET

TEXT runtime·raiseproc(SB),NOSPLIT,$0
	MOVL	$20, AX		// getpid
	SYSCALL
	MOVQ	AX, DI		// arg 1 - pid
	MOVL	sig+0(FP), SI	// arg 2 - signum
	MOVL	$37, AX		// kill
	SYSCALL
	RET

TEXT runtime·setitimer(SB), NOSPLIT, $-8
	MOVL	mode+0(FP), DI
	MOVQ	new+8(FP), SI
	MOVQ	old+16(FP), DX
	MOVL	$83, AX
	SYSCALL
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB), NOSPLIT, $32
	MOVL	$232, AX // clock_gettime
	MOVQ	$0, DI  	// CLOCK_REALTIME
	LEAQ	8(SP), SI
	SYSCALL
	MOVQ	8(SP), AX	// sec
	MOVQ	16(SP), DX	// nsec

	// sec is in AX, nsec in DX
	MOVQ	AX, sec+0(FP)
	MOVL	DX, nsec+8(FP)
	RET

TEXT runtime·nanotime1(SB), NOSPLIT, $32
	MOVL	$232, AX
	MOVQ	$4, DI  	// CLOCK_MONOTONIC
	LEAQ	8(SP), SI
	SYSCALL
	MOVQ	8(SP), AX	// sec
	MOVQ	16(SP), DX	// nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET

TEXT runtime·sigaction(SB),NOSPLIT,$-8
	MOVL	sig+0(FP), DI		// arg 1 sig
	MOVQ	new+8(FP), SI		// arg 2 act
	MOVQ	old+16(FP), DX		// arg 3 oact
	MOVL	$342, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVQ	fn+0(FP),    AX
	MOVL	sig+8(FP),   DI
	MOVQ	info+16(FP), SI
	MOVQ	ctx+24(FP),  DX
	MOVQ	SP, BX		// callee-saved
	ANDQ	$~15, SP	// alignment for x86_64 ABI
	CALL	AX
	MOVQ	BX, SP
	RET

// Called using C ABI.
TEXT runtime·sigtramp(SB),NOSPLIT|TOPFRAME|NOFRAME,$0
	// Transition from C ABI to Go ABI.
	PUSH_REGS_HOST_TO_ABI0()

	// Set up ABIInternal environment: g in R14, cleared X15.
	get_tls(R12)
	MOVQ	g(R12), R14
	PXOR	X15, X15

	// Reserve space for spill slots.
	NOP	SP		// disable vet stack checking
	ADJSP   $24

	// Call into the Go signal handler
	MOVQ	DI, AX	// sig
	MOVQ	SI, BX	// info
	MOVQ	DX, CX	// ctx
	CALL	·sigtrampgo<ABIInternal>(SB)

	ADJSP	$-24

	POP_REGS_HOST_TO_ABI0()
	RET

TEXT runtime·mmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 - addr
	MOVQ	n+8(FP), SI		// arg 2 - len
	MOVL	prot+16(FP), DX		// arg 3 - prot
	MOVL	flags+20(FP), R10		// arg 4 - flags
	MOVL	fd+24(FP), R8		// arg 5 - fd
	MOVL	off+28(FP), R9
	SUBQ	$16, SP
	MOVQ	R9, 8(SP)		// arg 7 - offset (passed on stack)
	MOVQ	$0, R9			// arg 6 - pad
	MOVL	$197, AX
	SYSCALL
	JCC	ok
	ADDQ	$16, SP
	MOVQ	$0, p+32(FP)
	MOVQ	AX, err+40(FP)
	RET
ok:
	ADDQ	$16, SP
	MOVQ	AX, p+32(FP)
	MOVQ	$0, err+40(FP)
	RET

TEXT runtime·munmap(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI		// arg 1 addr
	MOVQ	n+8(FP), SI		// arg 2 len
	MOVL	$73, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·madvise(SB),NOSPLIT,$0
	MOVQ	addr+0(FP), DI
	MOVQ	n+8(FP), SI
	MOVL	flags+16(FP), DX
	MOVQ	$75, AX	// madvise
	SYSCALL
	JCC	2(PC)
	MOVL	$-1, AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·sigaltstack(SB),NOSPLIT,$-8
	MOVQ	new+0(FP), DI
	MOVQ	old+8(FP), SI
	MOVQ	$53, AX
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·usleep(SB),NOSPLIT,$16
	MOVL	$0, DX
	MOVL	usec+0(FP), AX
	MOVL	$1000000, CX
	DIVL	CX
	MOVQ	AX, 0(SP)		// tv_sec
	MOVL	$1000, AX
	MULL	DX
	MOVQ	AX, 8(SP)		// tv_nsec

	MOVQ	SP, DI			// arg 1 - rqtp
	MOVQ	$0, SI			// arg 2 - rmtp
	MOVL	$240, AX		// sys_nanosleep
	SYSCALL
	RET

// set tls base to DI
TEXT runtime·settls(SB),NOSPLIT,$16
	ADDQ	$8, DI	// adjust for ELF: wants to use -8(FS) for g
	MOVQ	DI, 0(SP)
	MOVQ	$16, 8(SP)
	MOVQ	$0, DI			// arg 1 - which
	MOVQ	SP, SI			// arg 2 - tls_info
	MOVQ	$16, DX			// arg 3 - infosize
	MOVQ	$472, AX		// set_tls_area
	SYSCALL
	JCC	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOVQ	mib+0(FP), DI		// arg 1 - name
	MOVL	miblen+8(FP), SI		// arg 2 - namelen
	MOVQ	out+16(FP), DX		// arg 3 - oldp
	MOVQ	size+24(FP), R10		// arg 4 - oldlenp
	MOVQ	dst+32(FP), R8		// arg 5 - newp
	MOVQ	ndst+40(FP), R9		// arg 6 - newlen
	MOVQ	$202, AX		// sys___sysctl
	SYSCALL
	JCC 4(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET
	MOVL	$0, AX
	MOVL	AX, ret+48(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$-4
	MOVL	$331, AX		// sys_sched_yield
	SYSCALL
	RET

TEXT runtime·sigprocmask(SB),NOSPLIT,$0
	MOVL	how+0(FP), DI		// arg 1 - how
	MOVQ	new+8(FP), SI		// arg 2 - set
	MOVQ	old+16(FP), DX		// arg 3 - oset
	MOVL	$340, AX		// sys_sigprocmask
	SYSCALL
	JAE	2(PC)
	MOVL	$0xf1, 0xf1  // crash
	RET

// int32 runtime·kqueue(void);
TEXT runtime·kqueue(SB),NOSPLIT,$0
	MOVQ	$0, DI
	MOVQ	$0, SI
	MOVQ	$0, DX
	MOVL	$362, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+0(FP)
	RET

// int32 runtime·kevent(int kq, Kevent *changelist, int nchanges, Kevent *eventlist, int nevents, Timespec *timeout);
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVL	kq+0(FP), DI
	MOVQ	ch+8(FP), SI
	MOVL	nch+16(FP), DX
	MOVQ	ev+24(FP), R10
	MOVL	nev+32(FP), R8
	MOVQ	ts+40(FP), R9
	MOVL	$363, AX
	SYSCALL
	JCC	2(PC)
	NEGQ	AX
	MOVL	AX, ret+48(FP)
	RET

// func fcntl(fd, cmd, arg int32) (ret int32, errno int32)
TEXT runtime·fcntl(SB),NOSPLIT,$0
	MOVL	fd+0(FP), DI	// fd
	MOVL	cmd+4(FP), SI	// cmd
	MOVL	arg+8(FP), DX	// arg
	MOVL	$92, AX		// fcntl
	SYSCALL
	JCC	noerr
	MOVL	$-1, ret+16(FP)
	MOVL	AX, errno+20(FP)
	RET
noerr:
	MOVL	AX, ret+16(FP)
	MOVL	$0, errno+20(FP)
	RET

// func issetugid() int32
TEXT runtime·issetugid(SB),NOSPLIT,$0
	MOVQ	$0, DI
	MOVQ	$0, SI
	MOVQ	$0, DX
	MOVL	$253, AX
	SYSCALL
	MOVL	AX, ret+0(FP)
	RET
