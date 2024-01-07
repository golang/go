// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

#include "textflag.h"

//
// System calls for ppc64, Linux
//

// func rawVforkSyscall(trap, a1, a2, a3 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT|NOFRAME,$0-48
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
	MOVD	R3, err+40(FP)	// errno
	RET
ok:
	MOVD	R3, r1+32(FP)	// r1
	MOVD	R0, err+40(FP)	// errno
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
	MOVD	R0, r2+40(FP)
	RET
