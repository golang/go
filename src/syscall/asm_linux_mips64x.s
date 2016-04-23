// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips64 mips64le

#include "textflag.h"

//
// System calls for mips64, Linux
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);

TEXT	·Syscall(SB),NOSPLIT,$0-56
	JAL	runtime·entersyscall(SB)
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	R0, R7
	MOVV	R0, R8
	MOVV	R0, R9
	MOVV	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok
	MOVV	$-1, R1
	MOVV	R1, r1+32(FP)	// r1
	MOVV	R0, r2+40(FP)	// r2
	MOVV	R2, err+48(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET
ok:
	MOVV	R2, r1+32(FP)	// r1
	MOVV	R3, r2+40(FP)	// r2
	MOVV	R0, err+48(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	JAL	runtime·entersyscall(SB)
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	a4+32(FP), R7
	MOVV	a5+40(FP), R8
	MOVV	a6+48(FP), R9
	MOVV	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok6
	MOVV	$-1, R1
	MOVV	R1, r1+56(FP)	// r1
	MOVV	R0, r2+64(FP)	// r2
	MOVV	R2, err+72(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVV	R2, r1+56(FP)	// r1
	MOVV	R3, r2+64(FP)	// r2
	MOVV	R0, err+72(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	R0, R7
	MOVV	R0, R8
	MOVV	R0, R9
	MOVV	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok1
	MOVV	$-1, R1
	MOVV	R1, r1+32(FP)	// r1
	MOVV	R0, r2+40(FP)	// r2
	MOVV	R2, err+48(FP)	// errno
	RET
ok1:
	MOVV	R2, r1+32(FP)	// r1
	MOVV	R3, r2+40(FP)	// r2
	MOVV	R0, err+48(FP)	// errno
	RET

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	a4+32(FP), R7
	MOVV	a5+40(FP), R8
	MOVV	a6+48(FP), R9
	MOVV	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok2
	MOVV	$-1, R1
	MOVV	R1, r1+56(FP)	// r1
	MOVV	R0, r2+64(FP)	// r2
	MOVV	R2, err+72(FP)	// errno
	RET
ok2:
	MOVV	R2, r1+56(FP)	// r1
	MOVV	R3, r2+64(FP)	// r2
	MOVV	R0, err+72(FP)	// errno
	RET
