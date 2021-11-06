// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (mips || mipsle)

#include "textflag.h"
#include "funcdata.h"

//
// System calls for mips, Linux
//

// func Syscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
TEXT ·Syscall(SB),NOSPLIT,$0-28
	JAL	runtime·entersyscall(SB)
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	R0, R7
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok
	MOVW	$-1, R1
	MOVW	R1, r1+16(FP)	// r1
	MOVW	R0, r2+20(FP)	// r2
	MOVW	R2, err+24(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET
ok:
	MOVW	R2, r1+16(FP)	// r1
	MOVW	R3, r2+20(FP)	// r2
	MOVW	R0, err+24(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET

// func Syscall6(trap trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
// 5th and 6th arg go at sp+16, sp+20.
// Note that frame size of 20 means that 24 bytes gets reserved on stack.
TEXT ·Syscall6(SB),NOSPLIT,$20-40
	NO_LOCAL_POINTERS
	JAL	runtime·entersyscall(SB)
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	a4+16(FP), R7
	MOVW	a5+20(FP), R8
	MOVW	a6+24(FP), R9
	MOVW	R8, 16(R29)
	MOVW	R9, 20(R29)
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok6
	MOVW	$-1, R1
	MOVW	R1, r1+28(FP)	// r1
	MOVW	R0, r2+32(FP)	// r2
	MOVW	R2, err+36(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVW	R2, r1+28(FP)	// r1
	MOVW	R3, r2+32(FP)	// r2
	MOVW	R0, err+36(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET

// func Syscall9(trap trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr);
// Actually Syscall8 but the rest of the code expects it to be named Syscall9.
TEXT ·Syscall9(SB),NOSPLIT,$28-52
	NO_LOCAL_POINTERS
	JAL	runtime·entersyscall(SB)
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	a4+16(FP), R7
	MOVW	a5+20(FP), R8
	MOVW	a6+24(FP), R9
	MOVW	a7+28(FP), R10
	MOVW	a8+32(FP), R11
	MOVW	R8, 16(R29)
	MOVW	R9, 20(R29)
	MOVW	R10, 24(R29)
	MOVW	R11, 28(R29)
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok9
	MOVW	$-1, R1
	MOVW	R1, r1+40(FP)	// r1
	MOVW	R0, r2+44(FP)	// r2
	MOVW	R2, err+48(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET
ok9:
	MOVW	R2, r1+40(FP)	// r1
	MOVW	R3, r2+44(FP)	// r2
	MOVW	R0, err+48(FP)	// errno
	JAL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$24-28
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok1
	MOVW	$-1, R1
	MOVW	R1, r1+16(FP)	// r1
	MOVW	R0, r2+20(FP)	// r2
	MOVW	R2, err+24(FP)	// errno
	RET
ok1:
	MOVW	R2, r1+16(FP)	// r1
	MOVW	R3, r2+20(FP)	// r2
	MOVW	R0, err+24(FP)	// errno
	RET

TEXT ·RawSyscall6(SB),NOSPLIT,$20-40
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	a4+16(FP), R7
	MOVW	a5+20(FP), R8
	MOVW	a6+24(FP), R9
	MOVW	R8, 16(R29)
	MOVW	R9, 20(R29)
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok2
	MOVW	$-1, R1
	MOVW	R1, r1+28(FP)	// r1
	MOVW	R0, r2+32(FP)	// r2
	MOVW	R2, err+36(FP)	// errno
	RET
ok2:
	MOVW	R2, r1+28(FP)	// r1
	MOVW	R3, r2+32(FP)	// r2
	MOVW	R0, err+36(FP)	// errno
	RET

// func rawVforkSyscall(trap, a1 uintptr) (r1, err uintptr)
TEXT ·rawVforkSyscall(SB),NOSPLIT|NOFRAME,$0-16
	MOVW	a1+4(FP), R4
	MOVW	R0, R5
	MOVW	R0, R6
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	BEQ	R7, ok
	MOVW	$-1, R1
	MOVW	R1, r1+8(FP)	// r1
	MOVW	R2, err+12(FP)	// errno
	RET
ok:
	MOVW	R2, r1+8(FP)	// r1
	MOVW	R0, err+12(FP)	// errno
	RET

TEXT ·rawSyscallNoError(SB),NOSPLIT,$20-24
	MOVW	a1+4(FP), R4
	MOVW	a2+8(FP), R5
	MOVW	a3+12(FP), R6
	MOVW	trap+0(FP), R2	// syscall entry
	SYSCALL
	MOVW	R2, r1+16(FP)	// r1
	MOVW	R3, r2+20(FP)	// r2
	RET
