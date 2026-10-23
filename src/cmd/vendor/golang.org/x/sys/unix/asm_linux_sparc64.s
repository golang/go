// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && sparc64 && gc

#include "textflag.h"

//
// System calls for sparc64, Linux
//

// Just jump to package syscall's implementation for all these functions.
// The runtime may know about them.

TEXT ·Syscall(SB), NOSPLIT, $0-56
	JMP	syscall·Syscall(SB)

TEXT ·Syscall6(SB), NOSPLIT, $0-80
	JMP	syscall·Syscall6(SB)

TEXT ·SyscallNoError(SB), NOSPLIT, $0-48
	CALL	runtime·entersyscall(SB)
	MOVD	trap+0(FP), R1	// syscall entry
	MOVD	a1+8(FP), R8
	MOVD	a2+16(FP), R9
	MOVD	a3+24(FP), R10
	TA	$0x6d
	MOVD	R8, r1+32(FP)
	MOVD	R9, r2+40(FP)
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB), NOSPLIT, $0-56
	JMP	syscall·RawSyscall(SB)

TEXT ·RawSyscall6(SB), NOSPLIT, $0-80
	JMP	syscall·RawSyscall6(SB)

TEXT ·RawSyscallNoError(SB), NOSPLIT, $0-48
	MOVD	trap+0(FP), R1	// syscall entry
	MOVD	a1+8(FP), R8
	MOVD	a2+16(FP), R9
	MOVD	a3+24(FP), R10
	TA	$0x6d
	MOVD	R8, r1+32(FP)
	MOVD	R9, r2+40(FP)
	RET
