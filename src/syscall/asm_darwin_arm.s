// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System call support for ARM, Darwin
//

// func Syscall(syscall uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall(SB),NOSPLIT,$0-28
	BL		runtime·entersyscall(SB)
	MOVW	trap+0(FP), R12
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	SWI		$0x80
	BCC		ok
	MOVW	$-1, R1
	MOVW	R1, r1+16(FP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+20(FP)	// r2
	MOVW	R0, err+24(FP)	// err
	BL		runtime·exitsyscall(SB)
	RET
ok:
	MOVW	R0, r1+16(FP) // r1
	MOVW	R1, r2+20(FP)	// r2
	MOVW	$0, R0
	MOVW	R0, err+24(FP)	// err
	BL		runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVW	trap+0(FP), R12	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	SWI		$0x80
	BCC		ok1
	MOVW	$-1, R1
	MOVW	R1, r1+16(FP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+20(FP)	// r2
	MOVW	R0, err+24(FP)	// err
	RET
ok1:
	MOVW	R0, r1+16(FP) // r1
	MOVW	R1, r2+20(FP)	// r2
	MOVW	$0, R0
	MOVW	R0, err+24(FP)	// err
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall6(SB),NOSPLIT,$0-40
	BL		runtime·entersyscall(SB)
	MOVW	trap+0(FP), R12	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	MOVW	a4+16(FP), R3
	MOVW	a5+20(FP), R4
	MOVW	a6+24(FP), R5
	SWI		$0x80
	BCC		ok6
	MOVW	$-1, R1
	MOVW	R1, r1+28(FP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+32(FP)	// r2
	MOVW	R0, err+36(FP)	// err
	BL		runtime·exitsyscall(SB)
	RET
ok6:
	MOVW	R0, r1+28(FP) // r1
	MOVW	R1, r2+32(FP)	// r2
	MOVW	$0, R0
	MOVW	R0, err+36(FP)	// err
	BL		runtime·exitsyscall(SB)
	RET

// func RawSyscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVW	trap+0(FP), R12	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	MOVW	a4+16(FP), R3
	MOVW	a5+20(FP), R4
	MOVW	a6+24(FP), R5
	SWI		$0x80
	BCC		ok2
	MOVW	$-1, R1
	MOVW	R1, r1+28(FP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+32(FP)	// r2
	MOVW	R0, err+36(FP)	// err
	RET
ok2:
	MOVW	R0, r1+28(FP) // r1
	MOVW	R1, r2+32(FP)	// r2
	MOVW	$0, R0
	MOVW	R0, err+36(FP)	// err
	RET

// Actually Syscall7.
TEXT	·Syscall9(SB),NOSPLIT,$0-52
	BL runtime·entersyscall(SB)
	MOVW	num+0(FP), R12	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	MOVW	a4+16(FP), R3
	MOVW	a5+20(FP), R4
	MOVW	a6+24(FP), R5
	MOVW	a7+28(FP), R6
	SWI		$0x80
	BCC		ok9
	MOVW	$-1, R1
	MOVW	R1, r1+40(FP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+44(FP)	// r2
	MOVW	R0, err+48(FP)	// err
	BL		runtime·exitsyscall(SB)
	RET
ok9:
	MOVW	R0, r1+40(FP) // r1
	MOVW	R1, r2+44(FP)	// r2
	MOVW	$0, R0
	MOVW	R0, err+48(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET
