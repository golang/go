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
#define SYS_thread_create 80
#define SYS_thread_exit 81
#define SYS_tls_init 82
#define SYS_write 13
#define SYS_close 11
#define SYS_mutex_create 70
#define SYS_mutex_lock  71
#define SYS_mutex_unlock 73

#define SYSCALL(x)	$(0x10000+SYS_/**/x * 32)

TEXT exit(SB),7,$4
	MOVL	code+0(FP), AX
	MOVL	AX, 0(SP)
	CALL	SYSCALL(exit)
	INT $3	// not reached
	RET

TEXT exit1(SB),7,$4
	MOVL	code+0(FP), AX
	MOVL	AX, 0(SP)
	CALL	SYSCALL(thread_exit)
	INT $3	// not reached
	RET

TEXT write(SB),7,$0
	JMP	SYSCALL(write)

TEXT close(SB),7,$0
	JMP	SYSCALL(close)

TEXT mutex_create(SB),7,$0
	JMP	SYSCALL(mutex_create)

TEXT mutex_lock(SB),7,$0
	JMP	SYSCALL(mutex_lock)

TEXT	mutex_unlock(SB),7,$0
	JMP	SYSCALL(mutex_unlock)

TEXT thread_create(SB),7,$0
	JMP	SYSCALL(thread_create)

TEXT sys·mmap(SB),7,$24
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
	MOVL	$mmap_failed(SB), 4(SP)
	MOVL	$12, 8(SP)	// "mmap failed\n"
	CALL	SYSCALL(write)
	INT $3
	RET

// setldt(int entry, int address, int limit)
TEXT setldt(SB),7,$32
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
	MOVL	$tls_init_failed(SB), 4(SP)
	MOVL	$16, 8(SP)	// "tls_init failed\n"
	CALL	SYSCALL(write)
	INT $3
	RET

// There's no good way (yet?) to get stack traces out of a
// broken NaCl process, so if something goes wrong,
// print an error string before dying.

DATA mmap_failed(SB)/8, $"mmap fai"
DATA mmap_failed+8(SB)/4, $"led\n"
GLOBL mmap_failed(SB), $12

DATA tls_init_failed(SB)/8, $"tls_init"
DATA tls_init_failed+8(SB)/8, $" failed\n"
GLOBL tls_init_failed(SB), $16
