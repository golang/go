// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build arm64
// +build gc

#include "textflag.h"

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT ·Syscall(SB),NOSPLIT,$0-56
	B	syscall·Syscall(SB)

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	B	syscall·Syscall6(SB)

TEXT ·SyscallNoError(SB),NOSPLIT,$0-48
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	$0, R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	trap+0(FP), R8	// syscall entry
	SVC
	MOVD	R0, r1+32(FP)	// r1
	MOVD	R1, r2+40(FP)	// r2
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	B	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	B	syscall·RawSyscall6(SB)

TEXT ·RawSyscallNoError(SB),NOSPLIT,$0-48
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	$0, R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	trap+0(FP), R8	// syscall entry
	SVC
	MOVD	R0, r1+32(FP)
	MOVD	R1, r2+40(FP)
	RET
