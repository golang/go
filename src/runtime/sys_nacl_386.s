// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "syscall_nacl.h"

#define NACL_SYSCALL(code) \
	MOVL $(0x10000 + ((code)<<5)), AX; CALL AX

TEXT runtime·exit(SB),NOSPLIT,$4
	MOVL code+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_exit)
	JMP 0(PC)

TEXT runtime·exit1(SB),NOSPLIT,$4
	MOVL code+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_thread_exit)
	RET

TEXT runtime·open(SB),NOSPLIT,$12
	MOVL name+0(FP), AX
	MOVL AX, 0(SP)
	MOVL mode+4(FP), AX
	MOVL AX, 4(SP)
	MOVL perm+8(FP), AX
	MOVL AX, 8(SP)
	NACL_SYSCALL(SYS_open)
	MOVL AX, ret+12(FP)
	RET

TEXT runtime·closefd(SB),NOSPLIT,$4
	MOVL fd+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_close)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·read(SB),NOSPLIT,$12
	MOVL fd+0(FP), AX
	MOVL AX, 0(SP)
	MOVL p+4(FP), AX
	MOVL AX, 4(SP)
	MOVL n+8(FP), AX
	MOVL AX, 8(SP)
	NACL_SYSCALL(SYS_read)
	MOVL AX, ret+12(FP)
	RET

TEXT syscall·naclWrite(SB), NOSPLIT, $16-16
	MOVL arg1+0(FP), DI
	MOVL arg2+4(FP), SI
	MOVL arg3+8(FP), DX
	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL DX, 8(SP)
	CALL runtime·write(SB)
	MOVL AX, ret+16(FP)
	RET

TEXT runtime·write(SB),NOSPLIT,$12
	MOVL fd+0(FP), AX
	MOVL AX, 0(SP)
	MOVL p+4(FP), AX
	MOVL AX, 4(SP)
	MOVL n+8(FP), AX
	MOVL AX, 8(SP)
	NACL_SYSCALL(SYS_write)
	MOVL AX, ret+12(FP)
	RET

TEXT runtime·nacl_exception_stack(SB),NOSPLIT,$8
	MOVL p+0(FP), AX
	MOVL AX, 0(SP)
	MOVL size+4(FP), AX
	MOVL AX, 4(SP)
	NACL_SYSCALL(SYS_exception_stack)
	MOVL AX, ret+8(FP)
	RET

TEXT runtime·nacl_exception_handler(SB),NOSPLIT,$8
	MOVL fn+0(FP), AX
	MOVL AX, 0(SP)
	MOVL arg+4(FP), AX
	MOVL AX, 4(SP)
	NACL_SYSCALL(SYS_exception_handler)
	MOVL AX, ret+8(FP)
	RET

TEXT runtime·nacl_sem_create(SB),NOSPLIT,$4
	MOVL flag+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_sem_create)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_sem_wait(SB),NOSPLIT,$4
	MOVL sem+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_sem_wait)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_sem_post(SB),NOSPLIT,$4
	MOVL sem+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_sem_post)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_create(SB),NOSPLIT,$4
	MOVL flag+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_mutex_create)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_lock(SB),NOSPLIT,$4
	MOVL mutex+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_mutex_lock)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_trylock(SB),NOSPLIT,$4
	MOVL mutex+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_mutex_trylock)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_mutex_unlock(SB),NOSPLIT,$4
	MOVL mutex+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_mutex_unlock)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_cond_create(SB),NOSPLIT,$4
	MOVL flag+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_cond_create)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_cond_wait(SB),NOSPLIT,$8
	MOVL cond+0(FP), AX
	MOVL AX, 0(SP)
	MOVL n+4(FP), AX
	MOVL AX, 4(SP)
	NACL_SYSCALL(SYS_cond_wait)
	MOVL AX, ret+8(FP)
	RET

TEXT runtime·nacl_cond_signal(SB),NOSPLIT,$4
	MOVL cond+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_cond_signal)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_cond_broadcast(SB),NOSPLIT,$4
	MOVL cond+0(FP), AX
	MOVL AX, 0(SP)
	NACL_SYSCALL(SYS_cond_broadcast)
	MOVL AX, ret+4(FP)
	RET

TEXT runtime·nacl_cond_timed_wait_abs(SB),NOSPLIT,$12
	MOVL cond+0(FP), AX
	MOVL AX, 0(SP)
	MOVL lock+4(FP), AX
	MOVL AX, 4(SP)
	MOVL ts+8(FP), AX
	MOVL AX, 8(SP)
	NACL_SYSCALL(SYS_cond_timed_wait_abs)
	MOVL AX, ret+12(FP)
	RET

TEXT runtime·nacl_thread_create(SB),NOSPLIT,$16
	MOVL fn+0(FP), AX
	MOVL AX, 0(SP)
	MOVL stk+4(FP), AX
	MOVL AX, 4(SP)
	MOVL tls+8(FP), AX
	MOVL AX, 8(SP)
	MOVL xx+12(FP), AX
	MOVL AX, 12(SP)
	NACL_SYSCALL(SYS_thread_create)
	MOVL AX, ret+16(FP)
	RET

TEXT runtime·mstart_nacl(SB),NOSPLIT,$0
	JMP runtime·mstart(SB)

TEXT runtime·nacl_nanosleep(SB),NOSPLIT,$8
	MOVL ts+0(FP), AX
	MOVL AX, 0(SP)
	MOVL extra+4(FP), AX
	MOVL AX, 4(SP)
	NACL_SYSCALL(SYS_nanosleep)
	MOVL AX, ret+8(FP)
	RET

TEXT runtime·osyield(SB),NOSPLIT,$0
	NACL_SYSCALL(SYS_sched_yield)
	RET

TEXT runtime·mmap(SB),NOSPLIT,$32
	MOVL	addr+0(FP), AX
	MOVL	AX, 0(SP)
	MOVL	n+4(FP), AX
	MOVL	AX, 4(SP)
	MOVL	prot+8(FP), AX
	MOVL	AX, 8(SP)
	MOVL	flags+12(FP), AX
	MOVL	AX, 12(SP)
	MOVL	fd+16(FP), AX
	MOVL	AX, 16(SP)
	MOVL	off+20(FP), AX
	MOVL	AX, 24(SP)
	MOVL	$0, 28(SP)
	LEAL	24(SP), AX
	MOVL	AX, 20(SP)
	NACL_SYSCALL(SYS_mmap)
	CMPL	AX, $-4095
	JNA	2(PC)
	NEGL	AX
	MOVL	AX, ret+24(FP)
	RET

TEXT runtime·walltime(SB),NOSPLIT,$20
	MOVL $0, 0(SP) // real time clock
	LEAL 8(SP), AX
	MOVL AX, 4(SP) // timespec
	NACL_SYSCALL(SYS_clock_gettime)
	MOVL 8(SP), AX // low 32 sec
	MOVL 12(SP), CX // high 32 sec
	MOVL 16(SP), BX // nsec

	// sec is in AX, nsec in BX
	MOVL	AX, sec_lo+0(FP)
	MOVL	CX, sec_hi+4(FP)
	MOVL	BX, nsec+8(FP)
	RET

TEXT syscall·now(SB),NOSPLIT,$0
	JMP runtime·walltime(SB)

TEXT runtime·nacl_clock_gettime(SB),NOSPLIT,$8
	MOVL arg1+0(FP), AX
	MOVL AX, 0(SP)
	MOVL arg2+4(FP), AX
	MOVL AX, 4(SP)
	NACL_SYSCALL(SYS_clock_gettime)
	MOVL AX, ret+8(FP)
	RET
	
TEXT runtime·nanotime(SB),NOSPLIT,$20
	MOVL $0, 0(SP) // real time clock
	LEAL 8(SP), AX
	MOVL AX, 4(SP) // timespec
	NACL_SYSCALL(SYS_clock_gettime)
	MOVL 8(SP), AX // low 32 sec
	MOVL 16(SP), BX // nsec

	// sec is in AX, nsec in BX
	// convert to DX:AX nsec
	MOVL	$1000000000, CX
	MULL	CX
	ADDL	BX, AX
	ADCL	$0, DX

	MOVL	AX, ret_lo+0(FP)
	MOVL	DX, ret_hi+4(FP)
	RET

TEXT runtime·setldt(SB),NOSPLIT,$8
	MOVL	addr+4(FP), BX // aka base
	ADDL	$0x8, BX
	MOVL	BX, 0(SP)
	NACL_SYSCALL(SYS_tls_init)
	RET

TEXT runtime·sigtramp(SB),NOSPLIT,$0
	get_tls(CX)

	// check that g exists
	MOVL	g(CX), DI
	CMPL	DI, $0
	JNE	6(PC)
	MOVL	$11, BX
	MOVL	$0, 0(SP)
	MOVL	$runtime·badsignal(SB), AX
	CALL	AX
	JMP 	ret

	// save g
	MOVL	DI, 20(SP)
	
	// g = m->gsignal
	MOVL	g_m(DI), BX
	MOVL	m_gsignal(BX), BX
	MOVL	BX, g(CX)
	
	// copy arguments for sighandler
	MOVL	$11, 0(SP) // signal
	MOVL	$0, 4(SP) // siginfo
	LEAL	ctxt+4(FP), AX
	MOVL	AX, 8(SP) // context
	MOVL	DI, 12(SP) // g

	CALL	runtime·sighandler(SB)

	// restore g
	get_tls(CX)
	MOVL	20(SP), BX
	MOVL	BX, g(CX)

ret:
	// Enable exceptions again.
	NACL_SYSCALL(SYS_exception_clear_flag)

	// NaCl has abdicated its traditional operating system responsibility
	// and declined to implement 'sigreturn'. Instead the only way to return
	// to the execution of our program is to restore the registers ourselves.
	// Unfortunately, that is impossible to do with strict fidelity, because
	// there is no way to do the final update of PC that ends the sequence
	// without either (1) jumping to a register, in which case the register ends
	// holding the PC value instead of its intended value or (2) storing the PC
	// on the stack and using RET, which imposes the requirement that SP is
	// valid and that is okay to smash the word below it. The second would
	// normally be the lesser of the two evils, except that on NaCl, the linker
	// must rewrite RET into "POP reg; AND $~31, reg; JMP reg", so either way
	// we are going to lose a register as a result of the incoming signal.
	// Similarly, there is no way to restore EFLAGS; the usual way is to use
	// POPFL, but NaCl rejects that instruction. We could inspect the bits and
	// execute a sequence of instructions designed to recreate those flag
	// settings, but that's a lot of work.
	//
	// Thankfully, Go's signal handlers never try to return directly to the
	// executing code, so all the registers and EFLAGS are dead and can be
	// smashed. The only registers that matter are the ones that are setting
	// up for the simulated call that the signal handler has created.
	// Today those registers are just PC and SP, but in case additional registers
	// are relevant in the future (for example DX is the Go func context register)
	// we restore as many registers as possible.
	// 
	// We smash BP, because that's what the linker smashes during RET.
	//
	LEAL	ctxt+4(FP), BP
	ADDL	$64, BP
	MOVL	0(BP), AX
	MOVL	4(BP), CX
	MOVL	8(BP), DX
	MOVL	12(BP), BX
	MOVL	16(BP), SP
	// 20(BP) is saved BP, never to be seen again
	MOVL	24(BP), SI
	MOVL	28(BP), DI
	// 36(BP) is saved EFLAGS, never to be seen again
	MOVL	32(BP), BP // saved PC
	JMP	BP

// func getRandomData([]byte)
TEXT runtime·getRandomData(SB),NOSPLIT,$8-12
	MOVL arg_base+0(FP), AX
	MOVL AX, 0(SP)
	MOVL arg_len+4(FP), AX
	MOVL AX, 4(SP)
	NACL_SYSCALL(SYS_get_random_bytes)
	RET
