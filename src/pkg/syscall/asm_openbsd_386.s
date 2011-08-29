// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System call support for 386, OpenBSD
//

// func Syscall(trap int32, a1, a2, a3 int32) (r1, r2, err int32);
// func Syscall6(trap int32, a1, a2, a3, a4, a5, a6 int32) (r1, r2, err int32);
// Trap # in AX, args on stack above caller pc.

TEXT	·Syscall(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		8(SP), SI
	LEAL		4(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok
	MOVL	$-1, 20(SP)	// r1
	MOVL	$-1, 24(SP)	// r2
	MOVL	AX, 28(SP)		// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVL	AX, 20(SP)	// r1
	MOVL	DX, 24(SP)	// r2
	MOVL	$0, 28(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		8(SP), SI
	LEAL		4(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok6
	MOVL	$-1, 32(SP)	// r1
	MOVL	$-1, 36(SP)	// r2
	MOVL	AX, 40(SP)		// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVL	AX, 32(SP)	// r1
	MOVL	DX, 36(SP)	// r2
	MOVL	$0, 40(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall9(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		8(SP), SI
	LEAL		4(SP), DI
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
	MOVL	$-1, 44(SP)	// r1
	MOVL	$-1, 48(SP)	// r2
	MOVL	AX, 52(SP)		// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok9:
	MOVL	AX, 44(SP)	// r1
	MOVL	DX, 48(SP)	// r2
	MOVL	$0, 52(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),7,$0
	MOVL	4(SP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		8(SP), SI
	LEAL		4(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok1
	MOVL	$-1, 20(SP)	// r1
	MOVL	$-1, 24(SP)	// r2
	MOVL	AX, 28(SP)		// errno
	RET
ok1:
	MOVL	AX, 20(SP)	// r1
	MOVL	DX, 24(SP)	// r2
	MOVL	$0, 28(SP)	// errno
	RET

TEXT	·RawSyscall6(SB),7,$0
	MOVL	4(SP), AX	// syscall entry
	// slide args down on top of system call number
	LEAL		8(SP), SI
	LEAL		4(SP), DI
	CLD
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	MOVSL
	INT	$0x80
	JAE	ok2
	MOVL	$-1, 32(SP)	// r1
	MOVL	$-1, 36(SP)	// r2
	MOVL	AX, 40(SP)		// errno
	RET
ok2:
	MOVL	AX, 32(SP)	// r1
	MOVL	DX, 36(SP)	// r2
	MOVL	$0, 40(SP)	// errno
	RET
