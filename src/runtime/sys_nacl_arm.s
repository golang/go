// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "syscall_nacl.h"

#define NACL_SYSCALL(code) \
	MOVW	$(0x10000 + ((code)<<5)), R8; BL (R8)

TEXT runtime·exit(SB),NOSPLIT,$0
	MOVW	code+0(FP), R0
	NACL_SYSCALL(SYS_exit)
	RET

TEXT runtime·exit1(SB),NOSPLIT,$0
	MOVW	code+0(FP), R0
	NACL_SYSCALL(SYS_thread_exit)
	RET

TEXT runtime·open(SB),NOSPLIT,$0
	MOVW	name+0(FP), R0
	MOVW	name+0(FP), R1
	MOVW	name+0(FP), R2
	NACL_SYSCALL(SYS_open)
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	NACL_SYSCALL(SYS_close)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	NACL_SYSCALL(SYS_read)
	MOVW	R0, ret+12(FP)
	RET

// func naclWrite(fd int, b []byte) int
TEXT syscall·naclWrite(SB),NOSPLIT,$0
	MOVW	arg1+0(FP), R0
	MOVW	arg2+4(FP), R1
	MOVW	arg3+8(FP), R2
	NACL_SYSCALL(SYS_write)
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$0
	MOVW	fd+0(FP), R0
	MOVW	p+4(FP), R1
	MOVW	n+8(FP), R2
	NACL_SYSCALL(SYS_write)
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·nacl_exception_stack(SB),NOSPLIT,$0
	MOVW	p+0(FP), R0
	MOVW	size+4(FP), R1
	NACL_SYSCALL(SYS_exception_stack)
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·nacl_exception_handler(SB),NOSPLIT,$0
	MOVW	fn+0(FP), R0
	MOVW	arg+4(FP), R1
	NACL_SYSCALL(SYS_exception_handler)
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·nacl_sem_create(SB),NOSPLIT,$0
	MOVW	flag+0(FP), R0
	NACL_SYSCALL(SYS_sem_create)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_sem_wait(SB),NOSPLIT,$0
	MOVW	sem+0(FP), R0
	NACL_SYSCALL(SYS_sem_wait)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_sem_post(SB),NOSPLIT,$0
	MOVW	sem+0(FP), R0
	NACL_SYSCALL(SYS_sem_post)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_create(SB),NOSPLIT,$0
	MOVW	flag+0(FP), R0
	NACL_SYSCALL(SYS_mutex_create)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_lock(SB),NOSPLIT,$0
	MOVW	mutex+0(FP), R0
	NACL_SYSCALL(SYS_mutex_lock)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_trylock(SB),NOSPLIT,$0
	MOVW	mutex+0(FP), R0
	NACL_SYSCALL(SYS_mutex_trylock)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_unlock(SB),NOSPLIT,$0
	MOVW	mutex+0(FP), R0
	NACL_SYSCALL(SYS_mutex_unlock)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_cond_create(SB),NOSPLIT,$0
	MOVW	flag+0(FP), R0
	NACL_SYSCALL(SYS_cond_create)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_cond_wait(SB),NOSPLIT,$0
	MOVW	cond+0(FP), R0
	MOVW	n+4(FP), R1
	NACL_SYSCALL(SYS_cond_wait)
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·nacl_cond_signal(SB),NOSPLIT,$0
	MOVW	cond+0(FP), R0
	NACL_SYSCALL(SYS_cond_signal)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_cond_broadcast(SB),NOSPLIT,$0
	MOVW	cond+0(FP), R0
	NACL_SYSCALL(SYS_cond_broadcast)
	MOVW	R0, ret+4(FP)
	RET

TEXT runtime·nacl_cond_timed_wait_abs(SB),NOSPLIT,$0
	MOVW	cond+0(FP), R0
	MOVW	lock+4(FP), R1
	MOVW	ts+8(FP), R2
	NACL_SYSCALL(SYS_cond_timed_wait_abs)
	MOVW	R0, ret+12(FP)
	RET

TEXT runtime·nacl_thread_create(SB),NOSPLIT,$0
	MOVW	fn+0(FP), R0
	MOVW	stk+4(FP), R1
	MOVW	tls+8(FP), R2
	MOVW	xx+12(FP), R3
	NACL_SYSCALL(SYS_thread_create)
	MOVW	R0, ret+16(FP)
	RET

TEXT runtime·mstart_nacl(SB),NOSPLIT,$0
	MOVW	0(R9), R0 // TLS
	MOVW	-8(R0), R1 // g
	MOVW	-4(R0), R2 // m
	MOVW	R2, g_m(R1)
	MOVW	R1, g
	B runtime·mstart(SB)

TEXT runtime·nacl_nanosleep(SB),NOSPLIT,$0
	MOVW	ts+0(FP), R0
	MOVW	extra+4(FP), R1
	NACL_SYSCALL(SYS_nanosleep)
	MOVW	R0, ret+8(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	NACL_SYSCALL(SYS_sched_yield)
	RET

TEXT runtime·mmap(SB),NOSPLIT,$8
	MOVW	addr+0(FP), R0
	MOVW	n+4(FP), R1
	MOVW	prot+8(FP), R2
	MOVW	flags+12(FP), R3
	MOVW	fd+16(FP), R4
	// arg6:offset should be passed as a pointer (to int64)
	MOVW	off+20(FP), R5
	MOVW	R5, 4(R13)
	MOVW	$0, R6
	MOVW	R6, 8(R13)
	MOVW	$4(R13), R5
	MOVM.DB.W [R4,R5], (R13) // arg5 and arg6 are passed on stack
	NACL_SYSCALL(SYS_mmap)
	MOVM.IA.W (R13), [R4, R5]
	CMP	$-4095, R0
	RSB.HI	$0, R0
	MOVW	R0, ret+24(FP)
	RET

TEXT time·now(SB),NOSPLIT,$16
	MOVW	$0, R0 // real time clock
	MOVW	$4(R13), R1
	NACL_SYSCALL(SYS_clock_gettime)
	MOVW	4(R13), R0 // low 32-bit sec
	MOVW	8(R13), R1 // high 32-bit sec
	MOVW	12(R13), R2 // nsec
	MOVW	R0, sec+0(FP)
	MOVW	R1, sec+4(FP)
	MOVW	R2, sec+8(FP)
	RET

TEXT syscall·now(SB),NOSPLIT,$0
	B time·now(SB)

TEXT runtime·nacl_clock_gettime(SB),NOSPLIT,$0
	MOVW	arg1+0(FP), R0
	MOVW	arg2+4(FP), R1
	NACL_SYSCALL(SYS_clock_gettime)
	MOVW	R0, ret+8(FP)
	RET

// int64 nanotime(void) so really
// void nanotime(int64 *nsec)
TEXT runtime·nanotime(SB),NOSPLIT,$16
	MOVW	$0, R0 // real time clock
	MOVW	$4(R13), R1
	NACL_SYSCALL(SYS_clock_gettime)
	MOVW	4(R13), R0 // low 32-bit sec
	MOVW	8(R13), R1 // high 32-bit sec (ignored for now)
	MOVW	12(R13), R2 // nsec
	MOVW	$1000000000, R3
	MULLU	R0, R3, (R1, R0)
	MOVW	$0, R4
	ADD.S	R2, R0
	ADC	R4, R1
	MOVW	R0, ret_lo+0(FP)
	MOVW	R1, ret_hi+4(FP)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$80
	// load g from thread context
	MOVW	$ctxt+-4(FP), R0
	MOVW	(16*4+10*4)(R0), g

	// check that g exists
	CMP	$0, g
	BNE 	4(PC)
	MOVW  	$runtime·badsignal2(SB), R11
	BL	(R11)
	RET

	// save g
	MOVW	g, R3
	MOVW	g, 20(R13)

	// g = m->gsignal
	MOVW	g_m(g), R8
	MOVW	m_gsignal(R8), g

	// copy arguments for call to sighandler
	MOVW	$11, R0
	MOVW	R0, 4(R13) // signal
	MOVW	$0, R0
	MOVW	R0, 8(R13) // siginfo
	MOVW	$ctxt+-4(FP), R0
	MOVW	R0, 12(R13) // context
	MOVW	R3, 16(R13) // g

	BL	runtime·sighandler(SB)

	// restore g
	MOVW	20(R13), g

	// Enable exceptions again.
	NACL_SYSCALL(SYS_exception_clear_flag)

	// Restore registers as best we can. Impossible to do perfectly.
	// See comment in sys_nacl_386.s for extended rationale.
	MOVW	$ctxt+-4(FP), R1
	ADD	$64, R1
	MOVW	(0*4)(R1), R0
	MOVW	(2*4)(R1), R2
	MOVW	(3*4)(R1), R3
	MOVW	(4*4)(R1), R4
	MOVW	(5*4)(R1), R5
	MOVW	(6*4)(R1), R6
	MOVW	(7*4)(R1), R7
	MOVW	(8*4)(R1), R8
	// cannot write to R9
	MOVW	(10*4)(R1), g
	MOVW	(11*4)(R1), R11
	MOVW	(12*4)(R1), R12
	MOVW	(13*4)(R1), R13
	MOVW	(14*4)(R1), R14
	MOVW	(15*4)(R1), R1
	B	(R1)

nog:
	MOVW	$0, R0
	RET

TEXT runtime·nacl_sysinfo(SB),NOSPLIT,$16
	RET

// func getRandomData([]byte)
TEXT runtime·getRandomData(SB),NOSPLIT,$0-12
	MOVW buf+0(FP), R0
	MOVW len+4(FP), R1
	NACL_SYSCALL(SYS_get_random_bytes)
	RET

TEXT runtime·casp1(SB),NOSPLIT,$0
	B	runtime·cas(SB)

// This is only valid for ARMv6+, however, NaCl/ARM is only defined
// for ARMv7A anyway.
// bool armcas(int32 *val, int32 old, int32 new)
// AtomiBLy:
//	if(*val == old){
//		*val = new;
//		return 1;
//	}else
//		return 0;
TEXT runtime·cas(SB),NOSPLIT,$0
	B runtime·armcas(SB)

TEXT runtime·read_tls_fallback(SB),NOSPLIT,$-4
	WORD $0xe7fedef0 // NACL_INSTR_ARM_ABORT_NOW (UDF #0xEDE0)
