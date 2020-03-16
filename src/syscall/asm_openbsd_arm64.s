// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
TEXT ·Syscall(SB),NOSPLIT,$0-56
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	$0, R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	trap+0(FP), R8	// syscall number
	SVC
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+32(FP)	// r1
	MOVD	ZR, r2+40(FP)	// r2
	MOVD	R0, err+48(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R0, r1+32(FP)	// r1
	MOVD	R1, r2+40(FP)	// r2
	MOVD	ZR, err+48(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·Syscall6(SB),NOSPLIT,$0-80
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	MOVD	trap+0(FP), R8	// syscall number
	SVC
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+56(FP)	// r1
	MOVD	ZR, r2+64(FP)	// r2
	MOVD	R0, err+72(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R0, r1+56(FP)	// r1
	MOVD	R1, r2+64(FP)	// r2
	MOVD	ZR, err+72(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·Syscall9(SB),NOSPLIT,$0-104
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	MOVD	a7+56(FP), R6
	MOVD	a8+64(FP), R7
	MOVD	a9+72(FP), R8	// on stack
	MOVD	R8, 8(RSP)
	MOVD	num+0(FP), R8	// syscall number
	SVC
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+80(FP)	// r1
	MOVD	ZR, r2+88(FP)	// r2
	MOVD	R0, err+96(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET
ok:
	MOVD	R0, r1+80(FP)	// r1
	MOVD	R1, r2+88(FP)	// r2
	MOVD	ZR, err+96(FP)	// errno
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	$0, R3
	MOVD	$0, R4
	MOVD	$0, R5
	MOVD	trap+0(FP), R8	// syscall number
	SVC
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+32(FP)	// r1
	MOVD	ZR, r2+40(FP)	// r2
	MOVD	R0, err+48(FP)	// errno
	RET
ok:
	MOVD	R0, r1+32(FP)	// r1
	MOVD	R1, r2+40(FP)	// r2
	MOVD	ZR, err+48(FP)	// errno
	RET

TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	MOVD	trap+0(FP), R8	// syscall number
	SVC
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+56(FP)	// r1
	MOVD	ZR, r2+64(FP)	// r2
	MOVD	R0, err+72(FP)	// errno
	RET
ok:
	MOVD	R0, r1+56(FP)	// r1
	MOVD	R1, r2+64(FP)	// r2
	MOVD	ZR, err+72(FP)	// errno
	RET
