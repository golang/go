// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for 386, Linux
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
// Trap # in AX, args in BX CX DX SI DI, return in AX

TEXT	syscall·Syscall(SB),7,$0
	CALL	sys·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	ok
	MOVL	$-1, 20(SP)	// r1
	MOVL	$0, 24(SP)	// r2
	NEGL	AX
	MOVL	AX, 28(SP)  // errno
	CALL	sys·exitsyscall(SB)
	RET
ok:
	MOVL	AX, 20(SP)	// r1
	MOVL	DX, 24(SP)	// r2
	MOVL	$0, 28(SP)	// errno
	CALL	sys·exitsyscall(SB)
	RET

// Actually Syscall5 but the rest of the code expects it to be named Syscall6.
TEXT	syscall·Syscall6(SB),7,$0
	CALL	sys·entersyscall(SB)
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	20(SP), SI
	MOVL	24(SP), DI
	// 28(SP) is ignored
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	ok6
	MOVL	$-1, 32(SP)	// r1
	MOVL	$0, 36(SP)	// r2
	NEGL	AX
	MOVL	AX, 40(SP)  // errno
	CALL	sys·exitsyscall(SB)
	RET
ok6:
	MOVL	AX, 32(SP)	// r1
	MOVL	DX, 36(SP)	// r2
	MOVL	$0, 40(SP)	// errno
	CALL	sys·exitsyscall(SB)
	RET

TEXT syscall·RawSyscall(SB),7,$0
	MOVL	4(SP), AX	// syscall entry
	MOVL	8(SP), BX
	MOVL	12(SP), CX
	MOVL	16(SP), DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	ok1
	MOVL	$-1, 20(SP)	// r1
	MOVL	$0, 24(SP)	// r2
	NEGL	AX
	MOVL	AX, 28(SP)  // errno
	CALL	sys·exitsyscall(SB)
	RET
ok1:
	MOVL	AX, 20(SP)	// r1
	MOVL	DX, 24(SP)	// r2
	MOVL	$0, 28(SP)	// errno
	RET

#define SYS_SOCKETCALL 102	/* from zsysnum_linux_386.go */

// func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, errno int)
// Kernel interface gets call sub-number and pointer to a0.
TEXT syscall·socketcall(SB),7,$0
	CALL	sys·entersyscall(SB)
	MOVL	$SYS_SOCKETCALL, AX	// syscall entry
	MOVL	4(SP), BX	// socket call number
	LEAL		8(SP), CX	// pointer to call arguments
	MOVL	$0, DX
	MOVL	$0, SI
	MOVL	$0,  DI
	INT	$0x80
	CMPL	AX, $0xfffff001
	JLS	oksock
	MOVL	$-1, 28(SP)	// n
	NEGL	AX
	MOVL	AX, 32(SP)  // errno
	CALL	sys·exitsyscall(SB)
	RET
oksock:
	MOVL	AX, 28(SP)	// n
	MOVL	$0, 32(SP)	// errno
	CALL	sys·exitsyscall(SB)
	RET
