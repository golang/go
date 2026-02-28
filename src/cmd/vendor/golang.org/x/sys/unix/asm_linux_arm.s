// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc

#include "textflag.h"

//
// System calls for arm, Linux
//

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT ·Syscall(SB),NOSPLIT,$0-28
	B	syscall·Syscall(SB)

TEXT ·Syscall6(SB),NOSPLIT,$0-40
	B	syscall·Syscall6(SB)

TEXT ·SyscallNoError(SB),NOSPLIT,$0-24
	BL	runtime·entersyscall(SB)
	MOVW	trap+0(FP), R7
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	MOVW	$0, R3
	MOVW	$0, R4
	MOVW	$0, R5
	SWI	$0
	MOVW	R0, r1+16(FP)
	MOVW	$0, R0
	MOVW	R0, r2+20(FP)
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	B	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB),NOSPLIT,$0-40
	B	syscall·RawSyscall6(SB)

TEXT ·RawSyscallNoError(SB),NOSPLIT,$0-24
	MOVW	trap+0(FP), R7	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	SWI	$0
	MOVW	R0, r1+16(FP)
	MOVW	$0, R0
	MOVW	R0, r2+20(FP)
	RET

TEXT ·seek(SB),NOSPLIT,$0-28
	B	syscall·seek(SB)
