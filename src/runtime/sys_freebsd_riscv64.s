// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for riscv64, FreeBSD
// /usr/src/sys/kern/syscalls.master for syscall numbers.
//

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

#define CLOCK_REALTIME		0
#define CLOCK_MONOTONIC		4

#define SYS_exit		1
#define SYS_read		3
#define SYS_write		4
#define SYS_open		5
#define SYS_close		6
#define SYS_getpid		20
#define SYS_kill		37
#define SYS_sigaltstack		53
#define SYS_munmap		73
#define SYS_madvise		75
#define SYS_setitimer		83
#define SYS_fcntl		92
#define SYS___sysctl		202
#define SYS_nanosleep		240
#define SYS_issetugid		253
#define SYS_clock_gettime	232
#define SYS_sched_yield		331
#define SYS_sigprocmask		340
#define SYS_kqueue		362
#define SYS_sigaction		416
#define SYS_thr_exit		431
#define SYS_thr_self		432
#define SYS_thr_kill		433
#define SYS__umtx_op		454
#define SYS_thr_new		455
#define SYS_mmap		477
#define SYS_cpuset_getaffinity	487
#define SYS_pipe2 		542
#define SYS_kevent		560

TEXT emptyfunc<>(SB),0,$0-0
	RET

// func sys_umtx_op(addr *uint32, mode int32, val uint32, uaddr1 uintptr, ut *umtx_time) int32
TEXT runtime·sys_umtx_op(SB),NOSPLIT,$0
	MOV	addr+0(FP), A0
	MOVW	mode+8(FP), A1
	MOVW	val+12(FP), A2
	MOV	uaddr1+16(FP), A3
	MOV	ut+24(FP), A4
	MOV	$SYS__umtx_op, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, ret+32(FP)
	RET

// func thr_new(param *thrparam, size int32) int32
TEXT runtime·thr_new(SB),NOSPLIT,$0
	MOV	param+0(FP), A0
	MOVW	size+8(FP), A1
	MOV	$SYS_thr_new, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, ret+16(FP)
	RET

// func thr_start()
TEXT runtime·thr_start(SB),NOSPLIT,$0
	// set up g
	MOV	m_g0(A0), g
	MOV	A0, g_m(g)
	CALL	emptyfunc<>(SB)	 // fault if stack check is wrong
	CALL	runtime·mstart(SB)

	WORD	$0	// crash
	RET

// func exit(code int32)
TEXT runtime·exit(SB),NOSPLIT|NOFRAME,$0-4
	MOVW	code+0(FP), A0
	MOV	$SYS_exit, T0
	ECALL
	WORD	$0	// crash

// func exitThread(wait *atomic.Uint32)
TEXT runtime·exitThread(SB),NOSPLIT|NOFRAME,$0-8
	MOV	wait+0(FP), A0
	// We're done using the stack.
	FENCE
	MOVW	ZERO, (A0)
	FENCE
	MOV	$0, A0	// exit code
	MOV	$SYS_thr_exit, T0
	ECALL
	JMP	0(PC)

// func open(name *byte, mode, perm int32) int32
TEXT runtime·open(SB),NOSPLIT|NOFRAME,$0-20
	MOV	name+0(FP), A0
	MOVW	mode+8(FP), A1
	MOVW	perm+12(FP), A2
	MOV	$SYS_open, T0
	ECALL
	BEQ	T0, ZERO, ok
	MOV	$-1, A0
ok:
	MOVW	A0, ret+16(FP)
	RET

// func closefd(fd int32) int32
TEXT runtime·closefd(SB),NOSPLIT|NOFRAME,$0-12
	MOVW	fd+0(FP), A0
	MOV	$SYS_close, T0
	ECALL
	BEQ	T0, ZERO, ok
	MOV	$-1, A0
ok:
	MOVW	A0, ret+8(FP)
	RET

// func pipe2(flags int32) (r, w int32, errno int32)
TEXT runtime·pipe2(SB),NOSPLIT|NOFRAME,$0-20
	MOV	$r+8(FP), A0
	MOVW	flags+0(FP), A1
	MOV	$SYS_pipe2, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, errno+16(FP)
	RET

// func write1(fd uintptr, p unsafe.Pointer, n int32) int32
TEXT runtime·write1(SB),NOSPLIT|NOFRAME,$0-28
	MOV	fd+0(FP), A0
	MOV	p+8(FP), A1
	MOVW	n+16(FP), A2
	MOV	$SYS_write, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, ret+24(FP)
	RET

// func read(fd int32, p unsafe.Pointer, n int32) int32
TEXT runtime·read(SB),NOSPLIT|NOFRAME,$0-28
	MOVW	fd+0(FP), A0
	MOV	p+8(FP), A1
	MOVW	n+16(FP), A2
	MOV	$SYS_read, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, ret+24(FP)
	RET

// func usleep(usec uint32)
TEXT runtime·usleep(SB),NOSPLIT,$24-4
	MOVWU	usec+0(FP), A0
	MOV	$1000, A1
	MUL	A1, A0, A0
	MOV	$1000000000, A1
	DIV	A1, A0, A2
	MOV	A2, 8(X2)
	REM	A1, A0, A3
	MOV	A3, 16(X2)
	ADD	$8, X2, A0
	MOV	ZERO, A1
	MOV	$SYS_nanosleep, T0
	ECALL
	RET

// func thr_self() thread
TEXT runtime·thr_self(SB),NOSPLIT,$8-8
	MOV	$ptr-8(SP), A0	// arg 1 &8(SP)
	MOV	$SYS_thr_self, T0
	ECALL
	MOV	ptr-8(SP), A0
	MOV	A0, ret+0(FP)
	RET

// func thr_kill(t thread, sig int)
TEXT runtime·thr_kill(SB),NOSPLIT,$0-16
	MOV	tid+0(FP), A0	// arg 1 pid
	MOV	sig+8(FP), A1	// arg 2 sig
	MOV	$SYS_thr_kill, T0
	ECALL
	RET

// func raiseproc(sig uint32)
TEXT runtime·raiseproc(SB),NOSPLIT|NOFRAME,$0
	MOV	$SYS_getpid, T0
	ECALL
	// arg 1 pid - already in A0
	MOVW	sig+0(FP), A1	// arg 2
	MOV	$SYS_kill, T0
	ECALL
	RET

// func setitimer(mode int32, new, old *itimerval)
TEXT runtime·setitimer(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	mode+0(FP), A0
	MOV	new+8(FP), A1
	MOV	old+16(FP), A2
	MOV	$SYS_setitimer, T0
	ECALL
	RET

// func fallback_walltime() (sec int64, nsec int32)
TEXT runtime·fallback_walltime(SB),NOSPLIT,$24-12
	MOV	$CLOCK_REALTIME, A0
	MOV	$8(X2), A1
	MOV	$SYS_clock_gettime, T0
	ECALL
	MOV	8(X2), T0	// sec
	MOVW	16(X2), T1	// nsec
	MOV	T0, sec+0(FP)
	MOVW	T1, nsec+8(FP)
	RET

// func fallback_nanotime() int64
TEXT runtime·fallback_nanotime(SB),NOSPLIT,$24-8
	MOV	$CLOCK_MONOTONIC, A0
	MOV	$8(X2), A1
	MOV	$SYS_clock_gettime, T0
	ECALL
	MOV	8(X2), T0	// sec
	MOV	16(X2), T1	// nsec

	// sec is in T0, nsec in T1
	// return nsec in T0
	MOV	$1000000000, T2
	MUL	T2, T0
	ADD	T1, T0

	MOV	T0, ret+0(FP)
	RET

// func asmSigaction(sig uintptr, new, old *sigactiont) int32
TEXT runtime·asmSigaction(SB),NOSPLIT|NOFRAME,$0
	MOV	sig+0(FP), A0		// arg 1 sig
	MOV	new+8(FP), A1		// arg 2 act
	MOV	old+16(FP), A2		// arg 3 oact
	MOV	$SYS_sigaction, T0
	ECALL
	BEQ	T0, ZERO, ok
	MOV	$-1, A0
ok:
	MOVW	A0, ret+24(FP)
	RET

// func sigfwd(fn uintptr, sig uint32, info *siginfo, ctx unsafe.Pointer)
TEXT runtime·sigfwd(SB),NOSPLIT,$0-32
	MOVW	sig+8(FP), A0
	MOV	info+16(FP), A1
	MOV	ctx+24(FP), A2
	MOV	fn+0(FP), T1
	JALR	RA, T1
	RET

// func sigtramp(signo, ureg, ctxt unsafe.Pointer)
TEXT runtime·sigtramp(SB),NOSPLIT,$64
	MOVW	A0, 8(X2)
	MOV	A1, 16(X2)
	MOV	A2, 24(X2)

	// this might be called in external code context,
	// where g is not set.
	MOVBU	runtime·iscgo(SB), A0
	BEQ	A0, ZERO, ok
	CALL	runtime·load_g(SB)
ok:
	MOV	$runtime·sigtrampgo(SB), A0
	JALR	RA, A0
	RET

// func mmap(addr uintptr, n uintptr, prot int, flags int, fd int, off int64) (ret uintptr, err error)
TEXT runtime·mmap(SB),NOSPLIT|NOFRAME,$0
	MOV	addr+0(FP), A0
	MOV	n+8(FP), A1
	MOVW	prot+16(FP), A2
	MOVW	flags+20(FP), A3
	MOVW	fd+24(FP), A4
	MOVW	off+28(FP), A5
	MOV	$SYS_mmap, T0
	ECALL
	BNE	T0, ZERO, fail
	MOV	A0, p+32(FP)
	MOV	ZERO, err+40(FP)
	RET
fail:
	MOV	ZERO, p+32(FP)
	MOV	A0, err+40(FP)
	RET

// func munmap(addr uintptr, n uintptr) (err error)
TEXT runtime·munmap(SB),NOSPLIT|NOFRAME,$0
	MOV	addr+0(FP), A0
	MOV	n+8(FP), A1
	MOV	$SYS_munmap, T0
	ECALL
	BNE	T0, ZERO, fail
	RET
fail:
	WORD	$0	// crash

// func madvise(addr unsafe.Pointer, n uintptr, flags int32) int32
TEXT runtime·madvise(SB),NOSPLIT|NOFRAME,$0
	MOV	addr+0(FP), A0
	MOV	n+8(FP), A1
	MOVW	flags+16(FP), A2
	MOV	$SYS_madvise, T0
	ECALL
	BEQ	T0, ZERO, ok
	MOV	$-1, A0
ok:
	MOVW	A0, ret+24(FP)
	RET

// func sysctl(mib *uint32, miblen uint32, out *byte, size *uintptr, dst *byte, ndst uintptr) int32
TEXT runtime·sysctl(SB),NOSPLIT,$0
	MOV	mib+0(FP), A0
	MOV	miblen+8(FP), A1
	MOV	out+16(FP), A2
	MOV	size+24(FP), A3
	MOV	dst+32(FP), A4
	MOV	ndst+40(FP), A5
	MOV	$SYS___sysctl, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, ret+48(FP)
	RET

// func sigaltstack(new, old *stackt)
TEXT runtime·sigaltstack(SB),NOSPLIT|NOFRAME,$0
	MOV	new+0(FP), A0
	MOV	old+8(FP), A1
	MOV	$SYS_sigaltstack, T0
	ECALL
	BNE	T0, ZERO, fail
	RET
fail:
	WORD	$0	// crash

// func osyield()
TEXT runtime·osyield(SB),NOSPLIT|NOFRAME,$0
	MOV	$SYS_sched_yield, T0
	ECALL
	RET

// func sigprocmask(how int32, new, old *sigset)
TEXT runtime·sigprocmask(SB),NOSPLIT|NOFRAME,$0-24
	MOVW	how+0(FP), A0
	MOV	new+8(FP), A1
	MOV	old+16(FP), A2
	MOV	$SYS_sigprocmask, T0
	ECALL
	BNE	T0, ZERO, fail
	RET
fail:
	WORD	$0	// crash


// func cpuset_getaffinity(level int, which int, id int64, size int, mask *byte) int32
TEXT runtime·cpuset_getaffinity(SB),NOSPLIT|NOFRAME,$0-44
	MOV	level+0(FP), A0
	MOV	which+8(FP), A1
	MOV	id+16(FP), A2
	MOV	size+24(FP), A3
	MOV	mask+32(FP), A4
	MOV	$SYS_cpuset_getaffinity, T0
	ECALL
	BEQ	T0, ZERO, ok
	MOV	$-1, A0
ok:
	MOVW	A0, ret+40(FP)
	RET

// func kqueue() int32
TEXT runtime·kqueue(SB),NOSPLIT|NOFRAME,$0
	MOV $SYS_kqueue, T0
	ECALL
	BEQ	T0, ZERO, ok
	MOV	$-1, A0
ok:
	MOVW	A0, ret+0(FP)
	RET

// func kevent(kq int, ch unsafe.Pointer, nch int, ev unsafe.Pointer, nev int, ts *Timespec) (n int, err error)
TEXT runtime·kevent(SB),NOSPLIT,$0
	MOVW	kq+0(FP), A0
	MOV	ch+8(FP), A1
	MOVW	nch+16(FP), A2
	MOV	ev+24(FP), A3
	MOVW	nev+32(FP), A4
	MOV	ts+40(FP), A5
	MOV	$SYS_kevent, T0
	ECALL
	BEQ	T0, ZERO, ok
	NEG	A0, A0
ok:
	MOVW	A0, ret+48(FP)
	RET

// func fcntl(fd, cmd, arg int32) (int32, int32)
TEXT runtime·fcntl(SB),NOSPLIT,$0
	MOVW	fd+0(FP), A0
	MOVW	cmd+4(FP), A1
	MOVW	arg+8(FP), A2
	MOV	$SYS_fcntl, T0
	ECALL
	BEQ	T0, ZERO, noerr
	MOV	$-1, A1
	MOVW	A1, ret+16(FP)
	MOVW	A0, errno+20(FP)
	RET
noerr:
	MOVW	A0, ret+16(FP)
	MOVW	ZERO, errno+20(FP)
	RET

// func getCntxct() uint32
TEXT runtime·getCntxct(SB),NOSPLIT|NOFRAME,$0
	RDTIME	A0
	MOVW	A0, ret+0(FP)
	RET

// func issetugid() int32
TEXT runtime·issetugid(SB),NOSPLIT|NOFRAME,$0
	MOV $SYS_issetugid, T0
	ECALL
	MOVW	A0, ret+0(FP)
	RET

