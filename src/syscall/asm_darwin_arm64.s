// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System call support for ARM64, Darwin
//

// func Syscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall(SB),NOSPLIT,$0-56
	BL	runtime·entersyscall(SB)
	MOVD	trap+0(FP), R16
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	SVC	$0x80
	BCC	ok
	MOVD	$-1, R1
	MOVD	R1, r1+32(FP)	// r1
	MOVD	ZR, r2+40(FP)	// r2
	MOVD	R0, err+48(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R0, r1+32(FP) // r1
	MOVD	R1, r2+40(FP)	// r2
	MOVD	ZR, err+48(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVD	trap+0(FP), R16	// syscall entry
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	SVC	$0x80
	BCC	ok
	MOVD	$-1, R1
	MOVD	R1, r1+32(FP)	// r1
	MOVD	ZR, r2+40(FP)	// r2
	MOVD	R0, err+48(FP)	// err
	RET
ok:
	MOVD	R0, r1+32(FP) // r1
	MOVD	R1, r2+40(FP)	// r2
	MOVD	ZR, err+48(FP)	// err
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·Syscall6(SB),NOSPLIT,$0-80
	BL	runtime·entersyscall(SB)
	MOVD	trap+0(FP), R16	// syscall entry
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	SVC	$0x80
	BCC	ok
	MOVD	$-1, R1
	MOVD	R1, r1+56(FP)	// r1
	MOVD	ZR, r2+64(FP)	// r2
	MOVD	R0, err+72(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R0, r1+56(FP) // r1
	MOVD	R1, r2+64(FP)	// r2
	MOVD	ZR, err+72(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET

// func RawSyscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT	·RawSyscall6(SB),NOSPLIT,$0-80
	MOVD	trap+0(FP), R16	// syscall entry
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	SVC	$0x80
	BCC	ok
	MOVD	$-1, R1
	MOVD	R1, r1+56(FP)	// r1
	MOVD	ZR, r2+64(FP)	// r2
	MOVD	R0, err+72(FP)	// err
	RET
ok:
	MOVD	R0, r1+56(FP) // r1
	MOVD	R1, r2+64(FP)	// r2
	MOVD	ZR, R0
	MOVD	R0, err+72(FP)	// err
	RET

// Actually Syscall7
TEXT	·Syscall9(SB),NOSPLIT,$0-104
	BL	runtime·entersyscall(SB)
	MOVD	num+0(FP), R16	// syscall entry
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	MOVD	a7+56(FP), R6
	//MOVD	a8+64(FP), R7
	//MOVD	a9+72(FP), R8
	SVC	$0x80
	BCC	ok
	MOVD	$-1, R1
	MOVD	R1, r1+80(FP)	// r1
	MOVD	ZR, r2+88(FP)	// r2
	MOVD	R0, err+96(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R0, r1+80(FP) // r1
	MOVD	R1, r2+88(FP)	// r2
	MOVD	ZR, err+96(FP)	// err
	BL	runtime·exitsyscall(SB)
	RET

