// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func rawVforkSyscall(trap, a1, a2 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT,$0-40
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	$0, R2
	MOVD	$0, R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	trap+0(FP), R8	// syscall entry
	SVC
	CMN	$4095, R0
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+24(FP)	// r1
	NEG	R0, R0
	MOVD	R0, err+32(FP)	// errno
	RET
ok:
	MOVD	R0, r1+24(FP)	// r1
	MOVD	ZR, err+32(FP)	// errno
	RET

// func rawSyscallNoError(trap uintptr, a1, a2, a3 uintptr) (r1, r2 uintptr);
TEXT ·rawSyscallNoError(SB),NOSPLIT,$0-48
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
