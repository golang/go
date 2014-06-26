// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"
#include "../../cmd/ld/textflag.h"
#include "syscall_nacl.h"

#define NACL_SYSCALL(code) \
	MOVL $(0x10000 + ((code)<<5)), AX; CALL AX

#define NACL_SYSJMP(code) \
	MOVL $(0x10000 + ((code)<<5)), AX; JMP AX

TEXT runtime·settls(SB),NOSPLIT,$0
	MOVL	DI, TLS // really BP
	RET

TEXT runtime·exit(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_exit)

TEXT runtime·exit1(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_thread_exit)

TEXT runtime·open(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	NACL_SYSJMP(SYS_open)

TEXT runtime·close(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_close)

TEXT runtime·read(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	NACL_SYSJMP(SYS_read)

TEXT syscall·naclWrite(SB), NOSPLIT, $16-20
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL DX, 8(SP)
	CALL runtime·write(SB)
	MOVL AX, ret+16(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$16-12
	// If using fake time and writing to stdout or stderr,
	// emit playback header before actual data.
	MOVQ runtime·timens(SB), AX
	CMPQ AX, $0
	JEQ write
	MOVL arg1+0(FP), DI
	CMPL DI, $1
	JEQ playback
	CMPL DI, $2
	JEQ playback

write:
	// Ordinary write.
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	NACL_SYSCALL(SYS_write)
	RET

	// Write with playback header.
	// First, lock to avoid interleaving writes.
playback:
	MOVL $1, BX
	XCHGL	runtime·writelock(SB), BX
	CMPL BX, $0
	JNE playback

	// Playback header: 0 0 P B <8-byte time> <4-byte data length>
	MOVL $(('B'<<24) | ('P'<<16)), 0(SP)
	BSWAPQ AX
	MOVQ AX, 4(SP)
	MOVL arg3+8(FP), DX
	BSWAPL DX
	MOVL DX, 12(SP)
	MOVL $1, DI // standard output
	MOVL SP, SI
	MOVL $16, DX
	NACL_SYSCALL(SYS_write)

	// Write actual data.
	MOVL $1, DI // standard output
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	NACL_SYSCALL(SYS_write)

	// Unlock.
	MOVL	$0, runtime·writelock(SB)

	RET

TEXT runtime·nacl_exception_stack(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	NACL_SYSJMP(SYS_exception_stack)

TEXT runtime·nacl_exception_handler(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	NACL_SYSJMP(SYS_exception_handler)

TEXT runtime·nacl_sem_create(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_sem_create)

TEXT runtime·nacl_sem_wait(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_sem_wait)

TEXT runtime·nacl_sem_post(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_sem_post)

TEXT runtime·nacl_mutex_create(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_mutex_create)

TEXT runtime·nacl_mutex_lock(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_mutex_lock)

TEXT runtime·nacl_mutex_trylock(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_mutex_trylock)

TEXT runtime·nacl_mutex_unlock(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_mutex_unlock)

TEXT runtime·nacl_cond_create(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_cond_create)

TEXT runtime·nacl_cond_wait(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	NACL_SYSJMP(SYS_cond_wait)

TEXT runtime·nacl_cond_signal(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_cond_signal)

TEXT runtime·nacl_cond_broadcast(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	NACL_SYSJMP(SYS_cond_broadcast)

TEXT runtime·nacl_cond_timed_wait_abs(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	NACL_SYSJMP(SYS_cond_timed_wait_abs)

TEXT runtime·nacl_thread_create(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	MOVL arg4+12(FP), CX
	NACL_SYSJMP(SYS_thread_create)

TEXT runtime·mstart_nacl(SB),NOSPLIT,$0
	NACL_SYSCALL(SYS_tls_get)
	SUBL	$8, AX
	MOVL	AX, TLS
	JMP runtime·mstart(SB)

TEXT runtime·nacl_nanosleep(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	NACL_SYSJMP(SYS_nanosleep)

TEXT runtime·osyield(SB),NOSPLIT,$0
	NACL_SYSJMP(SYS_sched_yield)

TEXT runtime·mmap(SB),NOSPLIT,$8
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	MOVL arg4+12(FP), CX
	MOVL arg5+16(FP), R8
	MOVL arg6+20(FP), AX
	MOVQ AX, 0(SP)
	MOVL SP, R9
	NACL_SYSCALL(SYS_mmap)
	CMPL AX, $-4095
	JNA 2(PC)
	NEGL AX
	RET

TEXT time·now(SB),NOSPLIT,$16
	MOVQ runtime·timens(SB), AX
	CMPQ AX, $0
	JEQ realtime
	MOVQ $0, DX
	MOVQ $1000000000, CX
	DIVQ CX
	MOVQ AX, sec+0(FP)
	MOVL DX, nsec+8(FP)
	RET
realtime:
	MOVL $0, DI // real time clock
	LEAL 0(SP), AX
	MOVL AX, SI // timespec
	NACL_SYSCALL(SYS_clock_gettime)
	MOVL 0(SP), AX // low 32 sec
	MOVL 4(SP), CX // high 32 sec
	MOVL 8(SP), BX // nsec

	// sec is in AX, nsec in BX
	MOVL	AX, sec+0(FP)
	MOVL	CX, sec+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

TEXT syscall·now(SB),NOSPLIT,$0
	JMP time·now(SB)

TEXT runtime·nacl_clock_gettime(SB),NOSPLIT,$0
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	NACL_SYSJMP(SYS_clock_gettime)

TEXT runtime·nanotime(SB),NOSPLIT,$16
	MOVQ runtime·timens(SB), AX
	CMPQ AX, $0
	JEQ 2(PC)
	RET
	MOVL $0, DI // real time clock
	LEAL 0(SP), AX
	MOVL AX, SI // timespec
	NACL_SYSCALL(SYS_clock_gettime)
	MOVQ 0(SP), AX // sec
	MOVL 8(SP), DX // nsec

	// sec is in AX, nsec in DX
	// return nsec in AX
	IMULQ	$1000000000, AX
	ADDQ	DX, AX
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$80
	// restore TLS register at time of execution,
	// in case it's been smashed.
	// the TLS register is really BP, but for consistency
	// with non-NaCl systems it is referred to here as TLS.
	// NOTE: Cannot use SYS_tls_get here (like we do in mstart_nacl),
	// because the main thread never calls tls_set.
	LEAL ctxt+0(FP), AX
	MOVL (16*4+5*8)(AX), AX
	MOVL	AX, TLS

	// check that g exists
	get_tls(CX)
	MOVL	g(CX), DI
	
	CMPL	DI, $0
	JEQ	nog

	// save g
	MOVL	DI, 20(SP)
	
	// g = m->gsignal
	MOVL	g_m(DI), BX
	MOVL	m_gsignal(BX), BX
	MOVL	BX, g(CX)

//JMP debughandler

	// copy arguments for sighandler
	MOVL	$11, 0(SP) // signal
	MOVL	$0, 4(SP) // siginfo
	LEAL	ctxt+0(FP), AX
	MOVL	AX, 8(SP) // context
	MOVL	DI, 12(SP) // g

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(CX)
	MOVL	20(SP), BX
	MOVL	BX, g(CX)

sigtramp_ret:
	// Enable exceptions again.
	NACL_SYSCALL(SYS_exception_clear_flag)

	// Restore registers as best we can. Impossible to do perfectly.
	// See comment in sys_nacl_386.s for extended rationale.
	LEAL	ctxt+0(FP), SI
	ADDL	$64, SI
	MOVQ	0(SI), AX
	MOVQ	8(SI), CX
	MOVQ	16(SI), DX
	MOVQ	24(SI), BX
	MOVL	32(SI), SP	// MOVL for SP sandboxing
	// 40(SI) is saved BP aka TLS, already restored above
	// 48(SI) is saved SI, never to be seen again
	MOVQ	56(SI), DI
	MOVQ	64(SI), R8
	MOVQ	72(SI), R9
	MOVQ	80(SI), R10
	MOVQ	88(SI), R11
	MOVQ	96(SI), R12
	MOVQ	104(SI), R13
	MOVQ	112(SI), R14
	// 120(SI) is R15, which is owned by Native Client and must not be modified
	MOVQ	128(SI), SI // saved PC
	// 136(SI) is saved EFLAGS, never to be seen again
	JMP	SI

debughandler:
	// print basic information
	LEAL	ctxt+0(FP), DI
	MOVL	$runtime·sigtrampf(SB), AX
	MOVL	AX, 0(SP)
	MOVQ	(16*4+16*8)(DI), BX // rip
	MOVQ	BX, 8(SP)
	MOVQ	(16*4+0*8)(DI), BX // rax
	MOVQ	BX, 16(SP)
	MOVQ	(16*4+1*8)(DI), BX // rcx
	MOVQ	BX, 24(SP)
	MOVQ	(16*4+2*8)(DI), BX // rdx
	MOVQ	BX, 32(SP)
	MOVQ	(16*4+3*8)(DI), BX // rbx
	MOVQ	BX, 40(SP)
	MOVQ	(16*4+7*8)(DI), BX // rdi
	MOVQ	BX, 48(SP)
	MOVQ	(16*4+15*8)(DI), BX // r15
	MOVQ	BX, 56(SP)
	MOVQ	(16*4+4*8)(DI), BX // rsp
	MOVQ	0(BX), BX
	MOVQ	BX, 64(SP)
	CALL	runtime·printf(SB)
	
	LEAL	ctxt+0(FP), DI
	MOVQ	(16*4+16*8)(DI), BX // rip
	MOVL	BX, 0(SP)
	MOVQ	(16*4+4*8)(DI), BX // rsp
	MOVL	BX, 4(SP)
	MOVL	$0, 8(SP)	// lr
	get_tls(CX)
	MOVL	g(CX), BX
	MOVL	BX, 12(SP)	// gp
	CALL	runtime·traceback(SB)

notls:
	MOVL	0, AX
	RET

nog:
	MOVL	0, AX
	RET

// cannot do real signal handling yet, because gsignal has not been allocated.
MOVL $1, DI; NACL_SYSCALL(SYS_exit)

TEXT runtime·nacl_sysinfo(SB),NOSPLIT,$16
/*
	MOVL	di+0(FP), DI
	LEAL	12(DI), BX
	MOVL	8(DI), AX
	ADDL	4(DI), AX
	ADDL	$2, AX
	LEAL	(BX)(AX*4), BX
	MOVL	BX, runtime·nacl_irt_query(SB)
auxloop:
	MOVL	0(BX), DX
	CMPL	DX, $0
	JNE	2(PC)
	RET
	CMPL	DX, $32
	JEQ	auxfound
	ADDL	$8, BX
	JMP	auxloop
auxfound:
	MOVL	4(BX), BX
	MOVL	BX, runtime·nacl_irt_query(SB)

	LEAL	runtime·nacl_irt_basic_v0_1_str(SB), DI
	LEAL	runtime·nacl_irt_basic_v0_1(SB), SI
	MOVL	runtime·nacl_irt_basic_v0_1_size(SB), DX
	MOVL	runtime·nacl_irt_query(SB), BX
	CALL	BX

	LEAL	runtime·nacl_irt_memory_v0_3_str(SB), DI
	LEAL	runtime·nacl_irt_memory_v0_3(SB), SI
	MOVL	runtime·nacl_irt_memory_v0_3_size(SB), DX
	MOVL	runtime·nacl_irt_query(SB), BX
	CALL	BX

	LEAL	runtime·nacl_irt_thread_v0_1_str(SB), DI
	LEAL	runtime·nacl_irt_thread_v0_1(SB), SI
	MOVL	runtime·nacl_irt_thread_v0_1_size(SB), DX
	MOVL	runtime·nacl_irt_query(SB), BX
	CALL	BX

	// TODO: Once we have a NaCl SDK with futex syscall support,
	// try switching to futex syscalls and here load the
	// nacl-irt-futex-0.1 table.
*/
	RET
