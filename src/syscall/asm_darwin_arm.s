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
	MOVW	syscall+4(SP), R12
	MOVW	a1+8(SP), R0
	MOVW	a2+12(SP), R1
	MOVW	a3+16(SP), R2
	SWI		$0x80
	BCC		ok
	MOVW	$-1, R1
	MOVW	R1, r1+20(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+24(SP)	// r2
	MOVW	R0, errno+28(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET
ok:
	MOVW	R0, r1+20(SP) // r1
	MOVW	R1, r2+24(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, errno+28(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVW	syscall+4(SP), R12	// syscall entry
	MOVW	a1+8(SP), R0
	MOVW	a2+12(SP), R1
	MOVW	a3+16(SP), R2
	SWI		$0x80
	BCC		ok1
	MOVW	$-1, R1
	MOVW	R1, r1+20(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+24(SP)	// r2
	MOVW	R0, errno+28(SP)	// errno
	RET
ok1:
	MOVW	R0, r1+20(SP) // r1
	MOVW	R1, r2+24(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, errno+28(SP)	// errno
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall6(SB),NOSPLIT,$0-40
	BL		runtime·entersyscall(SB)
	MOVW	syscall+4(SP), R12	// syscall entry
	MOVW	a1+8(SP), R0
	MOVW	a2+12(SP), R1
	MOVW	a3+16(SP), R2
	MOVW	a4+20(SP), R3
	MOVW	a5+24(SP), R4
	MOVW	a6+28(SP), R5
	SWI		$0x80
	BCC		ok6
	MOVW	$-1, R1
	MOVW	R1, r1+32(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+36(SP)	// r2
	MOVW	R0, errno+40(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET
ok6:
	MOVW	R0, r1+32(SP) // r1
	MOVW	R1, r2+36(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, errno+40(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET

// func RawSyscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVW	trap+4(SP), R12	// syscall entry
	MOVW	a1+8(SP), R0
	MOVW	a2+12(SP), R1
	MOVW	a3+16(SP), R2
	MOVW	a4+20(SP), R3
	MOVW	a5+24(SP), R4
	MOVW	a6+28(SP), R5
	SWI		$0x80
	BCC		ok2
	MOVW	$-1, R1
	MOVW	R1, r1+32(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+36(SP)	// r2
	MOVW	R0, errno+40(SP)	// errno
	RET
ok2:
	MOVW	R0, r1+32(SP) // r1
	MOVW	R1, r2+36(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, errno+40(SP)	// errno
	RET

// Actually Syscall7.
TEXT	·Syscall9(SB),NOSPLIT,$0-52
	BL runtime·entersyscall(SB)
	MOVW	syscall+4(SP), R12	// syscall entry
	MOVW	a1+8(SP), R0
	MOVW	a2+12(SP), R1
	MOVW	a3+16(SP), R2
	MOVW	a4+20(SP), R3
	MOVW	a5+24(SP), R4
	MOVW	a6+28(SP), R5
	MOVW	a7+32(SP), R6
	SWI		$0x80
	BCC		ok9
	MOVW	$-1, R1
	MOVW	R1, r1+44(SP)	// r1
	MOVW	$0, R2
	MOVW	R2, r2+48(SP)	// r2
	MOVW	R0, errno+52(SP)	// errno
	BL		runtime·exitsyscall(SB)
	RET
ok9:
	MOVW	R0, r1+44(SP) // r1
	MOVW	R1, r2+48(SP)	// r2
	MOVW	$0, R0
	MOVW	R0, errno+52(SP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

