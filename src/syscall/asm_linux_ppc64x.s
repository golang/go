// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

#include "textflag.h"

//
// System calls for ppc64, Linux
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
TEXT ·Syscall(SB),NOSPLIT,$0-56
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R3
	MOVD	a2+16(FP), R4
	MOVD	a3+24(FP), R5
	MOVD	R0, R6
	MOVD	R0, R7
	MOVD	R0, R8
	MOVD	trap+0(FP), R9	// syscall entry
	SYSCALL R9
	BVC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+32(FP)	// r1
	MOVD	R0, r2+40(FP)	// r2
	MOVD	R3, err+48(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R3, r1+32(FP)	// r1
	MOVD	R4, r2+40(FP)	// r2
	MOVD	R0, err+48(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R3
	MOVD	a2+16(FP), R4
	MOVD	a3+24(FP), R5
	MOVD	a4+32(FP), R6
	MOVD	a5+40(FP), R7
	MOVD	a6+48(FP), R8
	MOVD	trap+0(FP), R9	// syscall entry
	SYSCALL R9
	BVC	ok6
	MOVD	$-1, R4
	MOVD	R4, r1+56(FP)	// r1
	MOVD	R0, r2+64(FP)	// r2
	MOVD	R3, err+72(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVD	R3, r1+56(FP)	// r1
	MOVD	R4, r2+64(FP)	// r2
	MOVD	R0, err+72(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVD	a1+8(FP), R3
	MOVD	a2+16(FP), R4
	MOVD	a3+24(FP), R5
	MOVD	R0, R6
	MOVD	R0, R7
	MOVD	R0, R8
	MOVD	trap+0(FP), R9	// syscall entry
	SYSCALL R9
	BVC	ok1
	MOVD	$-1, R4
	MOVD	R4, r1+32(FP)	// r1
	MOVD	R0, r2+40(FP)	// r2
	MOVD	R3, err+48(FP)	// errno
	RET
ok1:
	MOVD	R3, r1+32(FP)	// r1
	MOVD	R4, r2+40(FP)	// r2
	MOVD	R0, err+48(FP)	// errno
	RET

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	MOVD	a1+8(FP), R3
	MOVD	a2+16(FP), R4
	MOVD	a3+24(FP), R5
	MOVD	a4+32(FP), R6
	MOVD	a5+40(FP), R7
	MOVD	a6+48(FP), R8
	MOVD	trap+0(FP), R9	// syscall entry
	SYSCALL R9
	BVC	ok2
	MOVD	$-1, R4
	MOVD	R4, r1+56(FP)	// r1
	MOVD	R0, r2+64(FP)	// r2
	MOVD	R3, err+72(FP)	// errno
	RET
ok2:
	MOVD	R3, r1+56(FP)	// r1
	MOVD	R4, r2+64(FP)	// r2
	MOVD	R0, err+72(FP)	// errno
	RET

// func rawVforkSyscall(trap, a1 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT|NOFRAME,$0-32
	MOVD	a1+8(FP), R3
	MOVD	R0, R4
	MOVD	R0, R5
	MOVD	R0, R6
	MOVD	R0, R7
	MOVD	R0, R8
	MOVD	trap+0(FP), R9	// syscall entry
	SYSCALL R9
	BVC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+16(FP)	// r1
	MOVD	R3, err+24(FP)	// errno
	RET
ok:
	MOVD	R3, r1+16(FP)	// r1
	MOVD	R0, err+24(FP)	// errno
	RET

TEXT ·rawSyscallNoError(SB),NOSPLIT,$0-48
	MOVD	a1+8(FP), R3
	MOVD	a2+16(FP), R4
	MOVD	a3+24(FP), R5
	MOVD	R0, R6
	MOVD	R0, R7
	MOVD	R0, R8
	MOVD	trap+0(FP), R9	// syscall entry
	SYSCALL R9
	MOVD	R3, r1+32(FP)
	MOVD	R4, r2+40(FP)
	RET
