// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System call support for ARM, Darwin
//

// func Syscall(syscall uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall(SB),NOSPLIT,$0-32
	BL		runtime·entersyscall(SB)
	MOVW	4(SP), R12
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	SWI		$0x80
	BCC		ok
	MOVW	$-1, R1
	MOVW	R1, 20(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, 24(SP)	// r2
	MOVW	R0, 28(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET
ok:
	MOVW	R0, 20(SP) // r1
	MOVW	R1, 24(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, 28(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-32
	MOVW	4(SP), R12	// syscall entry
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	SWI		$0x80
	BCC		ok1
	MOVW	$-1, R1
	MOVW	R1, 20(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, 24(SP)	// r2
	MOVW	R0, 28(SP)	// errno
	RET
ok1:
	MOVW	R0, 20(SP) // r1
	MOVW	R1, 24(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, 28(SP)	// errno
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall6(SB),NOSPLIT,$0-44
	BL		runtime·entersyscall(SB)
	MOVW	4(SP), R12	// syscall entry
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	MOVW	20(SP), R3
	MOVW	24(SP), R4
	MOVW	28(SP), R5
	SWI		$0x80
	BCC		ok6
	MOVW	$-1, R1
	MOVW	R1, 32(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, 36(SP)	// r2
	MOVW	R0, 40(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET
ok6:
	MOVW	R0, 32(SP) // r1
	MOVW	R1, 36(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, 40(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET

// func RawSyscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·RawSyscall6(SB),NOSPLIT,$0-44
	MOVW	4(SP), R12	// syscall entry
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	MOVW	20(SP), R3
	MOVW	24(SP), R4
	MOVW	28(SP), R5
	SWI		$0x80
	BCC		ok2
	MOVW	$-1, R1
	MOVW	R1, 32(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, 36(SP)	// r2
	MOVW	R0, 40(SP)	// errno
	RET
ok2:
	MOVW	R0, 32(SP) // r1
	MOVW	R1, 36(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, 40(SP)	// errno
	RET

// Actually Syscall7.
TEXT	·Syscall9(SB),NOSPLIT,$0-56
	BL runtime·entersyscall(SB)
	MOVW	4(SP), R12	// syscall entry
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	MOVW	20(SP), R3
	MOVW	24(SP), R4
	MOVW	28(SP), R5
	MOVW	32(SP), R6
	SWI		$0x80
	BCC		ok9
	MOVW	$-1, R1
	MOVW	R1, 44(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, 48(SP)	// r2
	MOVW	R0, 52(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET
ok9:
	MOVW	R0, 44(SP) // r1
	MOVW	R1, 48(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, 52(SP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

