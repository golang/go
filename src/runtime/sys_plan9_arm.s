// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// from ../syscall/zsysnum_plan9.go

#define SYS_SYSR1       0
#define SYS_BIND        2
#define SYS_CHDIR       3
#define SYS_CLOSE       4
#define SYS_DUP         5
#define SYS_ALARM       6
#define SYS_EXEC        7
#define SYS_EXITS       8
#define SYS_FAUTH       10
#define SYS_SEGBRK      12
#define SYS_OPEN        14
#define SYS_OSEEK       16
#define SYS_SLEEP       17
#define SYS_RFORK       19
#define SYS_PIPE        21
#define SYS_CREATE      22
#define SYS_FD2PATH     23
#define SYS_BRK_        24
#define SYS_REMOVE      25
#define SYS_NOTIFY      28
#define SYS_NOTED       29
#define SYS_SEGATTACH   30
#define SYS_SEGDETACH   31
#define SYS_SEGFREE     32
#define SYS_SEGFLUSH    33
#define SYS_RENDEZVOUS  34
#define SYS_UNMOUNT     35
#define SYS_SEMACQUIRE  37
#define SYS_SEMRELEASE  38
#define SYS_SEEK        39
#define SYS_FVERSION    40
#define SYS_ERRSTR      41
#define SYS_STAT        42
#define SYS_FSTAT       43
#define SYS_WSTAT       44
#define SYS_FWSTAT      45
#define SYS_MOUNT       46
#define SYS_AWAIT       47
#define SYS_PREAD       50
#define SYS_PWRITE      51
#define SYS_TSEMACQUIRE 52
#define SYS_NSEC        53

//func open(name *byte, mode, perm int32) int32
TEXT runtime·open(SB),NOSPLIT,$0-16
	MOVW    $SYS_OPEN, R0
	SWI	0
	MOVW	R0, ret+12(FP)
	RET

//func pread(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32
TEXT runtime·pread(SB),NOSPLIT,$0-24
	MOVW    $SYS_PREAD, R0
	SWI	0
	MOVW	R0, ret+20(FP)
	RET

//func pwrite(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32
TEXT runtime·pwrite(SB),NOSPLIT,$0-24
	MOVW    $SYS_PWRITE, R0
	SWI	0
	MOVW	R0, ret+20(FP)
	RET

//func seek(fd int32, offset int64, whence int32) int64
TEXT runtime·seek(SB),NOSPLIT,$0-24
	MOVW	$ret_lo+16(FP), R0
	MOVW	0(R13), R1
	MOVW	R0, 0(R13)
	MOVW.W	R1, -4(R13)
	MOVW	$SYS_SEEK, R0
	SWI	0
	MOVW.W	R1, 4(R13)
	CMP	$-1, R0
	MOVW.EQ	R0, ret_lo+16(FP)
	MOVW.EQ	R0, ret_hi+20(FP)
	RET

//func closefd(fd int32) int32
TEXT runtime·closefd(SB),NOSPLIT,$0-8
	MOVW	$SYS_CLOSE, R0
	SWI	0
	MOVW	R0, ret+4(FP)
	RET

//func exits(msg *byte)
TEXT runtime·exits(SB),NOSPLIT,$0-4
	MOVW    $SYS_EXITS, R0
	SWI	0
	RET

//func brk_(addr unsafe.Pointer) int32
TEXT runtime·brk_(SB),NOSPLIT,$0-8
	MOVW    $SYS_BRK_, R0
	SWI	0
	MOVW	R0, ret+4(FP)
	RET

//func sleep(ms int32) int32
TEXT runtime·sleep(SB),NOSPLIT,$0-8
	MOVW    $SYS_SLEEP, R0
	SWI	0
	MOVW	R0, ret+4(FP)
	RET

//func plan9_semacquire(addr *uint32, block int32) int32
TEXT runtime·plan9_semacquire(SB),NOSPLIT,$0-12
	MOVW	$SYS_SEMACQUIRE, R0
	SWI	0
	MOVW	R0, ret+8(FP)
	RET

//func plan9_tsemacquire(addr *uint32, ms int32) int32
TEXT runtime·plan9_tsemacquire(SB),NOSPLIT,$0-12
	MOVW	$SYS_TSEMACQUIRE, R0
	SWI	0
	MOVW	R0, ret+8(FP)
	RET

//func nsec(*int64) int64
TEXT runtime·nsec(SB),NOSPLIT,$-4-12
	MOVW	$SYS_NSEC, R0
	SWI	0
	MOVW	arg+0(FP), R1
	MOVW	0(R1), R0
	MOVW	R0, ret_lo+4(FP)
	MOVW	4(R1), R0
	MOVW	R0, ret_hi+8(FP)
	RET

// time.now() (sec int64, nsec int32)
TEXT time·now(SB),NOSPLIT,$12-12
	// use nsec system call to get current time in nanoseconds
	MOVW	$sysnsec_lo-8(SP), R0	// destination addr
	MOVW	R0,res-12(SP)
	MOVW	$SYS_NSEC, R0
	SWI	0
	MOVW	sysnsec_lo-8(SP), R1	// R1:R2 = nsec
	MOVW	sysnsec_hi-4(SP), R2

	// multiply nanoseconds by reciprocal of 10**9 (scaled by 2**61)
	// to get seconds (96 bit scaled result)
	MOVW	$0x89705f41, R3		// 2**61 * 10**-9
	MULLU	R1,R3,(R6,R5)		// R5:R6:R7 = R1:R2 * R3
	MOVW	$0,R7
	MULALU	R2,R3,(R7,R6)

	// unscale by discarding low 32 bits, shifting the rest by 29
	MOVW	R6>>29,R6		// R6:R7 = (R5:R6:R7 >> 61)
	ORR	R7<<3,R6
	MOVW	R7>>29,R7

	// subtract (10**9 * sec) from nsec to get nanosecond remainder
	MOVW	$1000000000, R5		// 10**9
	MULLU	R6,R5,(R9,R8)		// R8:R9 = R6:R7 * R5
	MULA	R7,R5,R9,R9
	SUB.S	R8,R1			// R1:R2 -= R8:R9
	SBC	R9,R2

	// because reciprocal was a truncated repeating fraction, quotient
	// may be slightly too small -- adjust to make remainder < 10**9
	CMP	R5,R1			// if remainder > 10**9
	SUB.HS	R5,R1			//    remainder -= 10**9
	ADD.HS	$1,R6			//    sec += 1

	MOVW	R6,sec_lo+0(FP)
	MOVW	R7,sec_hi+4(FP)
	MOVW	R1,nsec+8(FP)
	RET

//func notify(fn unsafe.Pointer) int32
TEXT runtime·notify(SB),NOSPLIT,$0-8
	MOVW	$SYS_NOTIFY, R0
	SWI	0
	MOVW	R0, ret+4(FP)
	RET

//func noted(mode int32) int32
TEXT runtime·noted(SB),NOSPLIT,$0-8
	MOVW	$SYS_NOTED, R0
	SWI	0
	MOVW	R0, ret+4(FP)
	RET

//func plan9_semrelease(addr *uint32, count int32) int32
TEXT runtime·plan9_semrelease(SB),NOSPLIT,$0-12
	MOVW	$SYS_SEMRELEASE, R0
	SWI	0
	MOVW	R0, ret+8(FP)
	RET

//func rfork(flags int32) int32
TEXT runtime·rfork(SB),NOSPLIT,$0-8
	MOVW	$SYS_RFORK, R0
	SWI	0
	MOVW	R0, ret+4(FP)
	RET

//func tstart_plan9(newm *m)
TEXT runtime·tstart_plan9(SB),NOSPLIT,$0-4
	MOVW	newm+0(FP), R1
	MOVW	m_g0(R1), g

	// Layout new m scheduler stack on os stack.
	MOVW	R13, R0
	MOVW	R0, g_stack+stack_hi(g)
	SUB	$(64*1024), R0
	MOVW	R0, (g_stack+stack_lo)(g)
	MOVW	R0, g_stackguard0(g)
	MOVW	R0, g_stackguard1(g)

	// Initialize procid from TOS struct.
	MOVW	_tos(SB), R0
	MOVW	48(R0), R0
	MOVW	R0, m_procid(R1)	// save pid as m->procid

	BL	runtime·mstart(SB)

	MOVW	$0x1234, R0
	MOVW	R0, 0(R0)		// not reached
	RET

//func sigtramp(ureg, note unsafe.Pointer)
TEXT runtime·sigtramp(SB),NOSPLIT,$0-8
	// check that g and m exist
	CMP	$0, g
	BEQ	4(PC)
	MOVW	g_m(g), R0
	CMP 	$0, R0
	BNE	2(PC)
	BL	runtime·badsignal2(SB)	// will exit

	// save args
	MOVW	ureg+0(FP), R1
	MOVW	note+4(FP), R2

	// change stack
	MOVW	m_gsignal(R0), R3
	MOVW	(g_stack+stack_hi)(R3), R13

	// make room for args, retval and g
	SUB	$24, R13

	// save g
	MOVW	g, R3
	MOVW	R3, 20(R13)

	// g = m->gsignal
	MOVW	m_gsignal(R0), g

	// load args and call sighandler
	ADD	$4,R13,R5
	MOVM.IA	[R1-R3], (R5)
	BL	runtime·sighandler(SB)
	MOVW	16(R13), R0			// retval

	// restore g
	MOVW	20(R13), g

	// call noted(R0)
	MOVW	R0, 4(R13)
	BL	runtime·noted(SB)
	RET

//func sigpanictramp()
TEXT  runtime·sigpanictramp(SB),NOSPLIT,$0-0
	MOVW.W	R0, -4(R13)
	B	runtime·sigpanic(SB)

//func setfpmasks()
// Only used by the 64-bit runtime.
TEXT runtime·setfpmasks(SB),NOSPLIT,$0
	RET

#define ERRMAX 128	/* from os_plan9.h */

// func errstr() string
// Only used by package syscall.
// Grab error string due to a syscall made
// in entersyscall mode, without going
// through the allocator (issue 4994).
// See ../syscall/asm_plan9_arm.s:/·Syscall/
TEXT runtime·errstr(SB),NOSPLIT,$0-8
	MOVW	g_m(g), R0
	MOVW	(m_mOS+mOS_errstr)(R0), R1
	MOVW	R1, ret_base+0(FP)
	MOVW	$ERRMAX, R2
	MOVW	R2, ret_len+4(FP)
	MOVW    $SYS_ERRSTR, R0
	SWI	0
	MOVW	R1, R2
	MOVBU	0(R2), R0
	CMP	$0, R0
	BEQ	3(PC)
	ADD	$1, R2
	B	-4(PC)
	SUB	R1, R2
	MOVW	R2, ret_len+4(FP)
	RET

TEXT ·publicationBarrier(SB),NOSPLIT,$-4-0
	B	runtime·armPublicationBarrier(SB)

// never called (cgo not supported)
TEXT runtime·read_tls_fallback(SB),NOSPLIT,$-4
	MOVW	$0, R0
	MOVW	R0, (R0)
	RET
