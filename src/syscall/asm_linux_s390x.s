// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System calls for s390x, Linux
//

// func rawVforkSyscall(trap, a1, a2 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT|NOFRAME,$0-40
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok2
	MOVD	$-1, r1+24(FP)
	NEG	R2, R2
	MOVD	R2, err+32(FP)	// errno
	RET
ok2:
	MOVD	R2, r1+24(FP)
	MOVD	$0, err+32(FP)	// errno
	RET

// func rawSyscallNoError(trap, a1, a2, a3 uintptr) (r1, r2 uintptr)
TEXT ·rawSyscallNoError(SB),NOSPLIT,$0-48
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	R2, r1+32(FP)
	MOVD	R3, r2+40(FP)
	RET

#define SYS_SOCKETCALL 102	/* from zsysnum_linux_s390x.go */

// func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, err int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT ·socketcall(SB),NOSPLIT,$0-72
	BL	runtime·entersyscall(SB)
	MOVD	$SYS_SOCKETCALL, R1	// syscall entry
	MOVD	call+0(FP), R2		// socket call number
	MOVD	$a0+8(FP), R3		// pointer to call arguments
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, oksock
	MOVD	$-1, n+56(FP)
	NEG	R2, R2
	MOVD	R2, err+64(FP)
	BL	runtime·exitsyscall(SB)
	RET
oksock:
	MOVD	R2, n+56(FP)
	MOVD	$0, err+64(FP)
	CALL	runtime·exitsyscall(SB)
	RET

// func rawsocketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, err int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT ·rawsocketcall(SB),NOSPLIT,$0-72
	MOVD	$SYS_SOCKETCALL, R1	// syscall entry
	MOVD	call+0(FP), R2		// socket call number
	MOVD	$a0+8(FP), R3		// pointer to call arguments
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, oksock1
	MOVD	$-1, n+56(FP)
	NEG	R2, R2
	MOVD	R2, err+64(FP)
	RET
oksock1:
	MOVD	R2, n+56(FP)
	MOVD	$0, err+64(FP)
	RET
