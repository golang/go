// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (mips || mipsle)

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
//
// The 5th and 6th arg go at sp+16, sp+20.
// Note that frame size of 20 means that 24 bytes gets reserved on stack.
TEXT ·Syscall6(SB),NOSPLIT,$20-40
	MOVW	num+0(FP), R2	// syscall entry
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	a4+16(FP), R7
	MOVW	a5+20(FP), R8
	MOVW	a6+24(FP), R9
	MOVW	R8, 16(R29)
	MOVW	R9, 20(R29)
	MOVW	R0, R3	// reset R3 to 0 as 1-ret SYSCALL keeps it
	SYSCALL
	BEQ	R7, ok
	MOVW	$-1, R1
	MOVW	R1, r1+28(FP)
	MOVW	R0, r2+32(FP)
	MOVW	R2, errno+36(FP)
	RET
ok:
	MOVW	R2, r1+28(FP)
	MOVW	R3, r2+32(FP)
	MOVW	R0, errno+36(FP)
	RET
