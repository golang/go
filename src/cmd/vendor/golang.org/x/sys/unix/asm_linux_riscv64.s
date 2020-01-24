// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build riscv64,!gccgo

#include "textflag.h"

//
// System calls for linux/riscv64.
//
// Where available, just jump to package syscall's implementation of
// these functions.

TEXT ·Syscall(SB),NOSPLIT,$0-56
	JMP	syscall·Syscall(SB)

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	JMP	syscall·Syscall6(SB)

TEXT ·SyscallNoError(SB),NOSPLIT,$0-48
	CALL	runtime·entersyscall(SB)
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	$0, A3
	MOV	$0, A4
	MOV	$0, A5
	MOV	$0, A6
	MOV	trap+0(FP), A7	// syscall entry
	ECALL
	MOV	A0, r1+32(FP)	// r1
	MOV	A1, r2+40(FP)	// r2
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	JMP	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	JMP	syscall·RawSyscall6(SB)

TEXT ·RawSyscallNoError(SB),NOSPLIT,$0-48
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	ZERO, A3
	MOV	ZERO, A4
	MOV	ZERO, A5
	MOV	trap+0(FP), A7	// syscall entry
	ECALL
	MOV	A0, r1+32(FP)
	MOV	A1, r2+40(FP)
	RET
