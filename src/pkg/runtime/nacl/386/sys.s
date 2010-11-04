// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for 386, Linux
//

#include "386/asm.h"

// http://code.google.com/p/nativeclient/source/browse/trunk/src/native_client/src/trusted/service_runtime/include/bits/nacl_syscalls.h
#define SYS_exit 30
#define SYS_mmap 21
#define SYS_munmap 22
#define SYS_thread_create 80
#define SYS_thread_exit 81
#define SYS_tls_init 82
#define SYS_write 13
#define SYS_close 11
#define SYS_mutex_create 70
#define SYS_mutex_lock  71
#define SYS_mutex_unlock 73
#define SYS_gettimeofday 40
#define SYS_dyncode_copy 104


#define SYSCALL(x)	$(0x10000+SYS_/**/x * 32)

TEXT runtime·exit(SB),7,$4
	MOVL	code+0(FP), AX
	MOVL	AX, 0(SP)
	CALL	SYSCALL(exit)
	INT $3	// not reached
	RET

TEXT runtime·exit1(SB),7,$4
	MOVL	code+0(FP), AX
	MOVL	AX, 0(SP)
	CALL	SYSCALL(thread_exit)
	INT $3	// not reached
	RET

TEXT runtime·write(SB),7,$0
	JMP	SYSCALL(write)

TEXT runtime·close(SB),7,$0
	JMP	SYSCALL(close)

TEXT runtime·mutex_create(SB),7,$0
	JMP	SYSCALL(mutex_create)

TEXT runtime·mutex_lock(SB),7,$0
	JMP	SYSCALL(mutex_lock)

TEXT runtime·mutex_unlock(SB),7,$0
	JMP	SYSCALL(mutex_unlock)

TEXT runtime·thread_create(SB),7,$0
	JMP	SYSCALL(thread_create)

TEXT runtime·dyncode_copy(SB),7,$0
	JMP	SYSCALL(dyncode_copy)

// For Native Client: a simple no-op function.
// Inserting a call to this no-op is a simple way
// to trigger an alignment.
TEXT runtime·naclnop(SB),7,$0
	RET

TEXT runtime·mmap(SB),7,$24
	MOVL	a1+0(FP), BX
	MOVL	a2+4(FP), CX	// round up to 64 kB boundary; silences nacl warning
	ADDL	$(64*1024-1), CX
	ANDL	$~(64*1024-1), CX
	MOVL	a3+8(FP), DX
	MOVL	a4+12(FP), SI
	MOVL	a5+16(FP), DI
	MOVL	a6+20(FP), BP
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)
	MOVL	DI, 16(SP)
	MOVL	BP, 20(SP)
	CALL	SYSCALL(mmap)
	CMPL	AX, $0xfffff001
	JLS	6(PC)
	MOVL	$1, 0(SP)
	MOVL	$runtime·mmap_failed(SB), 4(SP)
	MOVL	$12, 8(SP)	// "mmap failed\n"
	CALL	SYSCALL(write)
	INT $3
	RET

TEXT runtime·munmap(SB),7,$0
	JMP	SYSCALL(munmap)

TEXT runtime·gettime(SB),7,$32
	LEAL	8(SP), BX
	MOVL	BX, 0(SP)
	MOVL	$0, 4(SP)
	CALL	SYSCALL(gettimeofday)
	
	MOVL	8(SP), BX	// sec
	MOVL	sec+0(FP), DI
	MOVL	BX, (DI)
	MOVL	$0, 4(DI)	// zero extend 32 -> 64 bits

	MOVL	12(SP), BX	// usec
	MOVL	usec+4(FP), DI
	MOVL	BX, (DI)
	RET

// setldt(int entry, int address, int limit)
TEXT runtime·setldt(SB),7,$32
	// entry is ignored - nacl tells us the
	// segment selector to use and stores it in GS.
	MOVL	address+4(FP), BX
	MOVL	limit+8(FP), CX
	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	CALL	SYSCALL(tls_init)
	CMPL	AX, $0xfffff001
	JLS	6(PC)
	MOVL	$1, 0(SP)
	MOVL	$runtime·tls_init_failed(SB), 4(SP)
	MOVL	$16, 8(SP)	// "tls_init failed\n"
	CALL	SYSCALL(write)
	INT $3
	RET

// There's no good way (yet?) to get stack traces out of a
// broken NaCl process, so if something goes wrong,
// print an error string before dying.

DATA runtime·mmap_failed(SB)/8, $"mmap fai"
DATA mmap_failed+8(SB)/4, $"led\n"
GLOBL runtime·mmap_failed(SB), $12

DATA runtime·tls_init_failed(SB)/8, $"tls_init"
DATA tls_init_failed+8(SB)/8, $" failed\n"
GLOBL runtime·tls_init_failed(SB), $16
