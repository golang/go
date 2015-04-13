// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm, Linux
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// for EABI, as we don't support OABI
#define SYS_BASE 0x0

#define SYS_exit (SYS_BASE + 1)
#define SYS_read (SYS_BASE + 3)
#define SYS_write (SYS_BASE + 4)
#define SYS_open (SYS_BASE + 5)
#define SYS_close (SYS_BASE + 6)
#define SYS_getpid (SYS_BASE + 20)
#define SYS_kill (SYS_BASE + 37)
#define SYS_gettimeofday (SYS_BASE + 78)
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
#define SYS_tkill (SYS_BASE + 238)
#define SYS_sched_yield (SYS_BASE + 158)
#define SYS_select (SYS_BASE + 142) // newselect
#define SYS_ugetrlimit (SYS_BASE + 191)
#define SYS_sched_getaffinity (SYS_BASE + 242)
#define SYS_clock_gettime (SYS_BASE + 263)
#define SYS_epoll_create (SYS_BASE + 250)
#define SYS_epoll_ctl (SYS_BASE + 251)
#define SYS_epoll_wait (SYS_BASE + 252)
#define SYS_epoll_create1 (SYS_BASE + 357)
#define SYS_fcntl (SYS_BASE + 55)
#define SYS_access (SYS_BASE + 33)
#define SYS_connect (SYS_BASE + 283)
#define SYS_socket (SYS_BASE + 281)

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

TEXT runtime·write(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	MOVW	$SYS_write, R7
	SWI	$0
	MOVW	$0xfffff001, R1
	CMP	R1, R0
	MOVW.HI	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	MOVW	$SYS_read, R7
	SWI	$0
	MOVW	$0xfffff001, R1
	CMP	R1, R0
	MOVW.HI	$-1, R0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·getrlimit(SB),NOSPLIT,$0
	MOVW	kind+0(FP), R0
	MOVW	limit+4(FP), R1
	MOVW	$SYS_ugetrlimit, R7
	SWI	$0
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·exit(SB),NOSPLIT,$-4
	MOVW	code+0(FP), R0
	MOVW	$SYS_exit_group, R7
	SWI	$0
	MOVW	$1234, R0
	MOVW	$1002, R1
	MOVW	R0, (R1)	// fail hard

TEXT runtime·exit1(SB),NOSPLIT,$-4
	MOVW	code+0(FP), R0
	MOVW	$SYS_exit, R7
	SWI	$0
	MOVW	$1234, R0
	MOVW	$1003, R1
	MOVW	R0, (R1)	// fail hard

TEXT	runtime·raise(SB),NOSPLIT,$-4
	MOVW	$SYS_gettid, R7
	SWI	$0
	// arg 1 tid already in R0 from gettid
	MOVW	sig+0(FP), R1	// arg 2 - signal
	MOVW	$SYS_tkill, R7
	SWI	$0
	RET

TEXT	runtime·raiseproc(SB),NOSPLIT,$-4
	MOVW	$SYS_getpid, R7
	SWI	$0
	// arg 1 tid already in R0 from getpid
	MOVW	sig+0(FP), R1	// arg 2 - signal
	MOVW	$SYS_kill, R7
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
	RSB.HI	$0, R0
	MOVW	R0, ret+24(FP)
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
	// ignore failure - maybe pages are locked
	RET

TEXT runtime·setitimer(SB),NOSPLIT,$0
	MOVW	mode+0(FP), R0
	MOVW	new+4(FP), R1
	MOVW	old+8(FP), R2
	MOVW	$SYS_setitimer, R7
	SWI	$0
	RET

TEXT runtime·mincore(SB),NOSPLIT,$0
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	dst+8(FP), R2
	MOVW	$SYS_mincore, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

TEXT time·now(SB), NOSPLIT, $32
	MOVW	$0, R0  // CLOCK_REALTIME
	MOVW	$8(R13), R1  // timespec
	MOVW	$SYS_clock_gettime, R7
	SWI	$0
	
	MOVW	8(R13), R0  // sec
	MOVW	12(R13), R2  // nsec
	
	MOVW	R0, sec+0(FP)
	MOVW	$0, R1
	MOVW	R1, loc+4(FP)
	MOVW	R2, nsec+8(FP)
	RET	

// int64 nanotime(void)
TEXT runtime·nanotime(SB),NOSPLIT,$32
	MOVW	$1, R0  // CLOCK_MONOTONIC
	MOVW	$8(R13), R1  // timespec
	MOVW	$SYS_clock_gettime, R7
	SWI	$0
	
	MOVW	8(R13), R0  // sec
	MOVW	12(R13), R2  // nsec
	
	MOVW	$1000000000, R3
	MULLU	R0, R3, (R1, R0)
	MOVW	$0, R4
	ADD.S	R2, R0
	ADC	R4, R1

	MOVW	R0, ret_lo+0(FP)
	MOVW	R1, ret_hi+4(FP)
	RET

// int32 futex(int32 *uaddr, int32 op, int32 val,
//	struct timespec *timeout, int32 *uaddr2, int32 val2);
TEXT runtime·futex(SB),NOSPLIT,$0
	// TODO: Rewrite to use FP references. Vet complains.
	MOVW	4(R13), R0
	MOVW	8(R13), R1
	MOVW	12(R13), R2
	MOVW	16(R13), R3
	MOVW	20(R13), R4
	MOVW	24(R13), R5
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
	// TODO(kaib): figure out which registers are clobbered by clone and avoid stack copying
	MOVW	$-16(R1), R1
	MOVW	mm+8(FP), R6
	MOVW	R6, 0(R1)
	MOVW	gg+12(FP), R6
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
	MOVW	12(R13), R0
	MOVW	$1234, R1
	CMP	R0, R1
	BEQ	2(PC)
	BL	runtime·abort(SB)

	MOVW	4(R13), g
	MOVW	0(R13), R8
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

	// Call fn
	MOVW	8(R13), R0
	MOVW	$16(R13), R13
	BL	(R0)

	MOVW	$0, R0
	MOVW	R0, 4(R13)
	BL	runtime·exit1(SB)

	// It shouldn't return
	MOVW	$1234, R0
	MOVW	$1005, R1
	MOVW	R0, (R1)

// int32 clone0(int32 flags, void *stack, void* fn, void* fnarg);
TEXT runtime·clone0(SB),NOSPLIT,$0-20
	MOVW	flags+0(FP), R0
	MOVW	stack+4(FP), R1
	// Update child's future stack and save fn and fnarg on it.
	MOVW	$-8(R1), R1
	MOVW	fn+8(FP), R6
	MOVW	R6, 0(R1)
	MOVW	fnarg+12(FP), R6
	MOVW	R6, 4(R1)
	MOVW	$0, R2	// parent tid ptr
	MOVW	$0, R3	// tls_val
	MOVW	$0, R4	// child tid ptr
	MOVW	$0, R5
	MOVW	$SYS_clone, R7
	SWI	$0

	// In parent, return.
	CMP	$0, R0
	BEQ	3(PC)
	MOVW	R0, ret+16(FP)
	RET

	// In child.
	MOVW	0(R13), R6   // fn
	MOVW	4(R13), R0   // fnarg
	MOVW	$8(R13), R13
	BL	(R6)

	MOVW	$0, R0
	MOVW	R0, 4(R13)
	BL	runtime·exit1(SB)

	// It shouldn't return
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

TEXT runtime·sigtramp(SB),NOSPLIT,$24
	// this might be called in external code context,
	// where g is not set.
	// first save R0, because runtime·load_g will clobber it
	MOVW	R0, 4(R13)
	MOVB	runtime·iscgo(SB), R0
	CMP 	$0, R0
	BL.NE	runtime·load_g(SB)

	CMP 	$0, g
	BNE 	4(PC)
	// signal number is already prepared in 4(R13)
	MOVW  	$runtime·badsignal(SB), R11
	BL	(R11)
	RET

	// save g
	MOVW	g, R3
	MOVW	g, 20(R13)

	// g = m->gsignal
	MOVW	g_m(g), R8
	MOVW	m_gsignal(R8), g

	// copy arguments for call to sighandler
	// R0 is already saved above
	MOVW	R1, 8(R13)
	MOVW	R2, 12(R13)
	MOVW	R3, 16(R13)

	BL	runtime·sighandler(SB)

	// restore g
	MOVW	20(R13), g

	RET

TEXT runtime·rtsigprocmask(SB),NOSPLIT,$0
	MOVW	sig+0(FP), R0
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
	MOVW	R0, R1
	MOVW	$1000000, R2
	DIV	R2, R0
	MOD	R2, R1
	MOVW	R0, 4(R13)
	MOVW	R1, 8(R13)
	MOVW	$0, R0
	MOVW	$0, R1
	MOVW	$0, R2
	MOVW	$0, R3
	MOVW	$4(R13), R4
	MOVW	$SYS_select, R7
	SWI	$0
	RET

// Use kernel version instead of native armcas in asm_arm.s.
// See ../sync/atomic/asm_linux_arm.s for details.
TEXT cas<>(SB),NOSPLIT,$0
	MOVW	$0xffff0fc0, R15 // R15 is hardware PC.

TEXT runtime·cas(SB),NOSPLIT,$0
	MOVW	ptr+0(FP), R2
	MOVW	old+4(FP), R0
loop:
	MOVW	new+8(FP), R1
	BL	cas<>(SB)
	BCC	check
	MOVW	$1, R0
	MOVB	R0, ret+12(FP)
	RET
check:
	// Kernel lies; double-check.
	MOVW	ptr+0(FP), R2
	MOVW	old+4(FP), R0
	MOVW	0(R2), R3
	CMP	R0, R3
	BEQ	loop
	MOVW	$0, R0
	MOVB	R0, ret+12(FP)
	RET

TEXT runtime·casp1(SB),NOSPLIT,$0
	B	runtime·cas(SB)

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

// int32 runtime·epollcreate(int32 size)
TEXT runtime·epollcreate(SB),NOSPLIT,$0
	MOVW	size+0(FP), R0
	MOVW	$SYS_epoll_create, R7
	SWI	$0
	MOVW	R0, ret+4(FP)
	RET

// int32 runtime·epollcreate1(int32 flags)
TEXT runtime·epollcreate1(SB),NOSPLIT,$0
	MOVW	flags+0(FP), R0
	MOVW	$SYS_epoll_create1, R7
	SWI	$0
	MOVW	R0, ret+4(FP)
	RET

// func epollctl(epfd, op, fd int32, ev *epollEvent) int
TEXT runtime·epollctl(SB),NOSPLIT,$0
	MOVW	epfd+0(FP), R0
	MOVW	op+4(FP), R1
	MOVW	fd+8(FP), R2
	MOVW	ev+12(FP), R3
	MOVW	$SYS_epoll_ctl, R7
	SWI	$0
	MOVW	R0, ret+16(FP)
	RET

// int32 runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout)
TEXT runtime·epollwait(SB),NOSPLIT,$0
	MOVW	epfd+0(FP), R0
	MOVW	ev+4(FP), R1
	MOVW	nev+8(FP), R2
	MOVW	timeout+12(FP), R3
	MOVW	$SYS_epoll_wait, R7
	SWI	$0
	MOVW	R0, ret+16(FP)
	RET

// void runtime·closeonexec(int32 fd)
TEXT runtime·closeonexec(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0	// fd
	MOVW	$2, R1	// F_SETFD
	MOVW	$1, R2	// FD_CLOEXEC
	MOVW	$SYS_fcntl, R7
	SWI	$0
	RET

// b __kuser_get_tls @ 0xffff0fe0
TEXT runtime·read_tls_fallback(SB),NOSPLIT,$-4
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
	MOVW	addrlen+8(FP), R2
	MOVW	$SYS_connect, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·socket(SB),NOSPLIT,$0
	MOVW	domain+0(FP), R0
	MOVW	type+4(FP), R1
	MOVW	protocol+8(FP), R2
	MOVW	$SYS_socket, R7
	SWI	$0
	MOVW	R0, ret+12(FP)
	RET
