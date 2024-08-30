// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

//
// System calls for riscv64, FreeBSD
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64)
TEXT ·Syscall(SB),NOSPLIT,$0-56
	CALL	runtime·entersyscall(SB)
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	trap+0(FP), T0	// syscall entry
	ECALL
	BNE	T0, ZERO, err
	MOV	A0, r1+32(FP)	// r1
	MOV	A1, r2+40(FP)	// r2
	MOV	ZERO, err+48(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
err:
	MOV	$-1, T0
	MOV	T0, r1+32(FP)	// r1
	MOV	ZERO, r2+40(FP)	// r2
	MOV	A0, err+48(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET


// func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	CALL	runtime·entersyscall(SB)
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	a4+32(FP), A3
	MOV	a5+40(FP), A4
	MOV	a6+48(FP), A5
	MOV	trap+0(FP), T0	// syscall entry
	ECALL
	BNE	T0, ZERO, err
	MOV	A0, r1+56(FP)	// r1
	MOV	A1, r2+64(FP)	// r2
	MOV	ZERO, err+72(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
err:
	MOV	$-1, T0
	MOV	T0, r1+56(FP)	// r1
	MOV	ZERO, r2+64(FP)	// r2
	MOV	A0, err+72(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

// func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	trap+0(FP), T0	// syscall entry
	ECALL
	BNE	T0, ZERO, err
	MOV	A0, r1+32(FP)	// r1
	MOV	A1, r2+40(FP)	// r2
	MOV	ZERO, err+48(FP)	// errno
	RET
err:
	MOV	$-1, T0
	MOV	T0, r1+32(FP)	// r1
	MOV	ZERO, r2+40(FP)	// r2
	MOV	A0, err+48(FP)	// errno
	RET

// func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall6(SB),NOSPLIT,$0-80
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	a4+32(FP), A3
	MOV	a5+40(FP), A4
	MOV	a6+48(FP), A5
	MOV	trap+0(FP), T0	// syscall entry
	ECALL
	BNE	T0, ZERO, err
	MOV	A0, r1+56(FP)	// r1
	MOV	A1, r2+64(FP)	// r2
	MOV	ZERO, err+72(FP)	// errno
	RET
err:
	MOV	$-1, T0
	MOV	T0, r1+56(FP)	// r1
	MOV	ZERO, r2+64(FP)	// r2
	MOV	A0, err+72(FP)	// errno
	RET

// Actually Syscall7
// func Syscall9(num uintptr, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr)
TEXT ·Syscall9(SB),NOSPLIT,$0-104
	CALL	runtime·entersyscall(SB)
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	a4+32(FP), A3
	MOV	a5+40(FP), A4
	MOV	a6+48(FP), A5
	MOV	a7+56(FP), A6
	MOV	num+0(FP), T0	// syscall entry
	ECALL
	BNE	T0, ZERO, err
	MOV	A0, r1+80(FP)	// r1
	MOV	A1, r2+88(FP)	// r2
	MOV	ZERO, err+96(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
err:
	MOV	$-1, T0
	MOV	T0, r1+80(FP)	// r1
	MOV	ZERO, r2+88(FP)	// r2
	MOV	A0, err+96(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
