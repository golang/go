// +build netbsd freebsd openbsd

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// System call support for some 386 unixes
//

// func Syscall(trap int32, a1, a2, a3 int32) (r1, r2, err int32);
// func Syscall6(trap int32, a1, a2, a3, a4, a5, a6 int32) (r1, r2, err int32);
// Trap # in AX, args on stack above caller pc.

TEXT	·Syscall(SB),NOSPLIT,$0-28
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		a1+4(FP), SI
	LEAL		trap+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok
	MOVL	$-1, r1+16(FP)	// r1
	MOVL	$-1, r2+20(FP)	// r2
	MOVL	AX, err+24(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVL	AX, r1+16(FP)	// r1
	MOVL	DX, r2+20(FP)	// r2
	MOVL	$0, err+24(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),NOSPLIT,$0-40
	CALL	runtime·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		a1+4(FP), SI
	LEAL		trap+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok6
	MOVL	$-1, r1+28(FP)	// r1
	MOVL	$-1, r2+32(FP)	// r2
	MOVL	AX, err+36(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVL	AX, r1+28(FP)	// r1
	MOVL	DX, r2+32(FP)	// r2
	MOVL	$0, err+36(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall9(SB),NOSPLIT,$0-52
	CALL	runtime·entersyscall(SB)
	MOVL	num+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		a1+4(FP), SI
	LEAL		num+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok9
	MOVL	$-1, r1+40(FP)	// r1
	MOVL	$-1, r2+44(FP)	// r2
	MOVL	AX, err+48(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok9:
	MOVL	AX, r1+40(FP)	// r1
	MOVL	DX, r2+44(FP)	// r2
	MOVL	$0, err+48(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-28
	MOVL	trap+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		a1+4(FP), SI
	LEAL		trap+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok1
	MOVL	$-1, r1+16(FP)	// r1
	MOVL	$-1, r2+20(FP)	// r2
	MOVL	AX, err+24(FP)	// errno
	RET
ok1:
	MOVL	AX, r1+16(FP)	// r1
	MOVL	DX, r2+20(FP)	// r2
	MOVL	$0, err+24(FP)	// errno
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVL	trap+0(FP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		a1+4(FP), SI
	LEAL		trap+0(FP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok2
	MOVL	$-1, r1+28(FP)	// r1
	MOVL	$-1, r2+32(FP)	// r2
	MOVL	AX, err+36(FP)	// errno
	RET
ok2:
	MOVL	AX, r1+28(FP)	// r1
	MOVL	DX, r2+32(FP)	// r2
	MOVL	$0, err+36(FP)	// errno
	RET
