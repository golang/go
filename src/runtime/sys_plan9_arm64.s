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
TEXT runtime·open(SB),NOSPLIT,$0-20
	// Kernel expects each int32 arg to be 64-bit-aligned.
	// Copy perm from +12 to +16 (the return value slot) so it's 8-byte aligned.
	MOVWU	perm+12(FP), R1
	MOVW	R1, ret+16(FP)
	MOVD    $SYS_OPEN, R0
	SVC	$0
	MOVWU	R0, ret+16(FP)
	RET

//func pread(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32
TEXT runtime·pread(SB),NOSPLIT,$0-36
	MOVD    $SYS_PREAD, R0
	SVC	$0
	MOVWU	R0, ret+32(FP)
	RET

//func pwrite(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32
TEXT runtime·pwrite(SB),NOSPLIT,$0-36
	MOVD    $SYS_PWRITE, R0
	SVC	$0
	MOVWU	R0, ret+32(FP)
	RET

//func seek(fd int32, offset int64, whence int32) int64
TEXT runtime·seek(SB),NOSPLIT,$0-32
	MOVD	$ret+24(FP), R0
	MOVWU	fd+0(FP), R2
	MOVD	offset+8(FP), R3
	MOVWU	whence+16(FP), R4

	MOVD    $sysargs-0(SP), R1

	MOVD	R0, 8(R1)
	MOVWU	R2, 16(R1)
	MOVD	R3, 24(R1)
	MOVWU	R4, 32(R1)

	MOVD	$SYS_SEEK, R0
	SVC	$0

	CMP	$-1, R0
	BNE	2(PC)
	MOVD	R0, ret+24(FP)
	RET

//func closefd(fd int32) int32
TEXT runtime·closefd(SB),NOSPLIT,$0-12
	MOVD	$SYS_CLOSE, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func dupfd(old, new int32) int32
TEXT runtime·dupfd(SB),NOSPLIT,$0-12
	// Kernel expects each int32 arg to be 64-bit-aligned.
	// The return value slot is where the kernel
	// expects to find the second argument, so copy it there.
	MOVWU	new+4(FP), R1
	MOVW	R1, ret+8(FP)
	MOVD	$SYS_DUP, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func exits(msg *byte)
TEXT runtime·exits(SB),NOSPLIT,$0-8
	MOVD    $SYS_EXITS, R0
	SVC	$0
	RET

//func brk_(addr unsafe.Pointer) int32
TEXT runtime·brk_(SB),NOSPLIT,$0-12
	MOVD    $SYS_BRK_, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func sleep(ms int32) int32
TEXT runtime·sleep(SB),NOSPLIT,$0-12
	MOVD    $SYS_SLEEP, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func plan9_semacquire(addr *uint32, block int32) int32
TEXT runtime·plan9_semacquire(SB),NOSPLIT,$0-20
	MOVD	$SYS_SEMACQUIRE, R0
	SVC	$0
	MOVWU	R0, ret+16(FP)
	RET

//func plan9_tsemacquire(addr *uint32, ms int32) int32
TEXT runtime·plan9_tsemacquire(SB),NOSPLIT,$0-20
	MOVD	$SYS_TSEMACQUIRE, R0
	SVC	$0
	MOVWU	R0, ret+16(FP)
	RET

// func timesplit(u uint64) (sec int64, nsec int32)
TEXT runtime·timesplit(SB), NOSPLIT, $0-20
	MOVD    u+0(FP), R0
	MOVD    R0, R1
	MOVD    $1000000000, R2
	UDIV    R2, R1
	MUL    R1, R2
	SUB    R2, R0
	MOVD    R1,sec+8(FP)
	MOVWU    R0,nsec+16(FP)
	RET

//func nsec(*int64) int64
TEXT runtime·nsec(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	$SYS_NSEC, R0
	SVC	$0
	MOVD	R0, ret+8(FP)
	RET

// func walltime() (sec int64, nsec int32)
TEXT runtime·walltime(SB),NOSPLIT,$16-12
	// use nsec system call to get current time in nanoseconds
	MOVD	$SYS_NSEC, R0
	SVC	$0

	MOVD	R0, R1
	MOVD	$1000000000, R2
	UDIV	R2, R1

	MOVD	R1, R3
	MUL	R3, R2
	SUB	R2, R0

	MOVD	R1,sec+0(FP)
	MOVWU	R0,nsec+8(FP)
	RET

//func notify(fn unsafe.Pointer) int32
TEXT runtime·notify(SB),NOSPLIT,$0-12
	MOVD	$SYS_NOTIFY, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func noted(mode int32) int32
TEXT runtime·noted(SB),NOSPLIT,$0-12
	MOVD	$SYS_NOTED, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func plan9_semrelease(addr *uint32, count int32) int32
TEXT runtime·plan9_semrelease(SB),NOSPLIT,$0-20
	MOVD	$SYS_SEMRELEASE, R0
	SVC	$0
	MOVWU	R0, ret+16(FP)
	RET

//func rfork(flags int32) int32
TEXT runtime·rfork(SB),NOSPLIT,$0-12
	MOVD	$SYS_RFORK, R0
	SVC	$0
	MOVWU	R0, ret+8(FP)
	RET

//func tstart_plan9(newm *m)
TEXT runtime·tstart_plan9(SB),NOSPLIT,$8-8
	MOVD	newm+0(FP), R1
	MOVD	m_g0(R1), g

	// Layout new m scheduler stack on os stack.
	MOVD	RSP, R0
	MOVD	R0, g_stack+stack_hi(g)
	SUB	$(64*1024), R0
	MOVD	R0, (g_stack+stack_lo)(g)
	MOVD	R0, g_stackguard0(g)
	MOVD	R0, g_stackguard1(g)

	// Initialize procid from TOS struct.
	MOVD	_tos(SB), R0
	MOVWU	64(R0), R0
	MOVD	R0, m_procid(R1)	// save pid as m->procid

	BL	runtime·mstart(SB)

	// Exit the thread.
	MOVD	$0, R0
	MOVD	R0, 8(RSP)
	CALL	runtime·exits(SB)
	JMP	0(PC)

//func sigtramp(ureg, note unsafe.Pointer)
TEXT runtime·sigtramp(SB),NOSPLIT,$0-16
	// check that g and m exist
	CMP	$0, g
	BEQ	4(PC)
	MOVD	g_m(g), R0
	CMP 	$0, R0
	BNE	2(PC)
	BL	runtime·badsignal2(SB)	// will exit

	// save args
	MOVD	ureg+0(FP), R1
	MOVD	note+8(FP), R2

	// change stack
	MOVD	m_gsignal(R0), R3
	MOVD	(g_stack+stack_hi)(R3), R4
	MOVD	R4, RSP

	// make room for args, retval and g
	SUB	$48, RSP

	// save g
	MOVD	g, R3
	MOVD	R3, 40(RSP)

	// g = m->gsignal
	MOVD	m_gsignal(R0), g

	// load args and call sighandler
	MOVD R1, 8(RSP)
	MOVD R2, 16(RSP)
	MOVD R3, 24(RSP)

	BL	runtime·sighandler(SB)
	MOVWU	32(RSP), R0			// retval

	// restore g
	MOVD	40(RSP), g

	// call noted(R0)
	MOVD	R0, 8(RSP)
	BL	runtime·noted(SB)
	RET

//func sigpanictramp()
TEXT  runtime·sigpanictramp(SB),NOSPLIT,$0-0
	MOVD.W	R0, -16(RSP)
	B	runtime·sigpanic(SB)

//func setfpmasks()
// Mask all SSE floating-point exceptions (only amd64?)
TEXT runtime·setfpmasks(SB),NOSPLIT,$0
	RET
