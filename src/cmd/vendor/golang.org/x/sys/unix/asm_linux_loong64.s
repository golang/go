// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && loong64 && gc

#include "textflag.h"


// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT ·Syscall(SB),NOSPLIT,$0-56
	JMP	syscall·Syscall(SB)

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	JMP	syscall·Syscall6(SB)

TEXT ·SyscallNoError(SB),NOSPLIT,$0-48
	JAL	runtime·entersyscall(SB)
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	R0, R7
	MOVV	R0, R8
	MOVV	R0, R9
	MOVV	trap+0(FP), R11	// syscall entry
	SYSCALL
	MOVV	R4, r1+32(FP)
	MOVV	R0, r2+40(FP)	// r2 is not used. Always set to 0
	JAL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	JMP	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	JMP	syscall·RawSyscall6(SB)

TEXT ·RawSyscallNoError(SB),NOSPLIT,$0-48
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	R0, R7
	MOVV	R0, R8
	MOVV	R0, R9
	MOVV	trap+0(FP), R11	// syscall entry
	SYSCALL
	MOVV	R4, r1+32(FP)
	MOVV	R0, r2+40(FP)	// r2 is not used. Always set to 0
	RET
