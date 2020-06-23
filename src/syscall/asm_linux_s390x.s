// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System calls for s390x, Linux
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64)
TEXT ·Syscall(SB),NOSPLIT,$0-56
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok
	MOVD	$-1, r1+32(FP)
	MOVD	$0, r2+40(FP)
	NEG	R2, R2
	MOVD	R2, err+48(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R2, r1+32(FP)
	MOVD	R3, r2+40(FP)
	MOVD	$0, err+48(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

// func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	a4+32(FP), R5
	MOVD	a5+40(FP), R6
	MOVD	a6+48(FP), R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok6
	MOVD	$-1, r1+56(FP)
	MOVD	$0, r2+64(FP)
	NEG	R2, R2
	MOVD	R2, err+72(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVD	R2, r1+56(FP)
	MOVD	R3, r2+64(FP)
	MOVD	$0, err+72(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok1
	MOVD	$-1, r1+32(FP)
	MOVD	$0, r2+40(FP)
	NEG	R2, R2
	MOVD	R2, err+48(FP)	// errno
	RET
ok1:
	MOVD	R2, r1+32(FP)
	MOVD	R3, r2+40(FP)
	MOVD	$0, err+48(FP)	// errno
	RET

// func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	a4+32(FP), R5
	MOVD	a5+40(FP), R6
	MOVD	a6+48(FP), R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok2
	MOVD	$-1, r1+56(FP)
	MOVD	$0, r2+64(FP)
	NEG	R2, R2
	MOVD	R2, err+72(FP)	// errno
	RET
ok2:
	MOVD	R2, r1+56(FP)
	MOVD	R3, r2+64(FP)
	MOVD	$0, err+72(FP)	// errno
	RET

// func rawVforkSyscall(trap, a1 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT|NOFRAME,$0-32
	MOVD	$0, R2
	MOVD	a1+8(FP), R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	$0, R6
	MOVD	$0, R7
	MOVD	trap+0(FP), R1	// syscall entry
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok2
	MOVD	$-1, r1+16(FP)
	NEG	R2, R2
	MOVD	R2, err+24(FP)	// errno
	RET
ok2:
	MOVD	R2, r1+16(FP)
	MOVD	$0, err+24(FP)	// errno
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
