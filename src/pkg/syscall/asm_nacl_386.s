// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for 386, Native Client
//

#define SYSCALL(x)	$(0x10000+x * 32)

// func Syscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
// Trap # in AX, args in BX CX DX SI DI, return in AX

TEXT	syscall·Syscall(SB),7,$20
	CALL	sys·entersyscall(SB)
	MOVL	trap+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	$0, SI
	MOVL	$0,  DI

	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)
	MOVL	DI, 16(SP)

	// Call $(0x10000+32*AX)
	SHLL	$5, AX
	ADDL	$0x10000, AX
	CALL	AX

	CMPL	AX, $0xfffff001
	JLS	ok
	MOVL	$-1, r1+16(FP)
	MOVL	$0, r2+20(FP)
	NEGL	AX
	MOVL	AX, errno+24(FP)
	CALL	sys·exitsyscall(SB)
	RET
ok:
	MOVL	AX, r1+16(FP)
	MOVL	DX, r2+20(FP)
	MOVL	$0, errno+24(FP)
	CALL	sys·exitsyscall(SB)
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
TEXT	syscall·Syscall6(SB),7,$24
	CALL	sys·entersyscall(SB)
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	a4+16(FP), SI
	MOVL	a5+20(FP), DI
	MOVL	a6+24(FP), AX

	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)
	MOVL	DI, 16(SP)
	MOVL	AX, 20(SP)

	// Call $(0x10000+32*trap)
	MOVL	trap+0(FP), AX	// syscall entry
	SHLL	$5, AX
	ADDL	$0x10000, AX
	CALL	AX

	CMPL	AX, $0xfffff001
	JLS	ok6
	MOVL	$-1, r1+28(FP)
	MOVL	$0, r2+32(FP)
	NEGL	AX
	MOVL	AX, errno+36(FP)
	CALL	sys·exitsyscall(SB)
	RET
ok6:
	MOVL	AX, r1+28(FP)
	MOVL	DX, r2+32(FP)
	MOVL	$0, errno+36(FP)
	CALL	sys·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
TEXT syscall·RawSyscall(SB),7,$20
	MOVL	trap+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	$0, SI
	MOVL	$0,  DI

	MOVL	BX, 0(SP)
	MOVL	CX, 4(SP)
	MOVL	DX, 8(SP)
	MOVL	SI, 12(SP)
	MOVL	DI, 16(SP)

	// Call $(0x10000+32*AX)
	SHLL	$5, AX
	ADDL	$0x10000, AX
	CALL	AX

	CMPL	AX, $0xfffff001
	JLS	ok1
	MOVL	$-1, r1+16(FP)
	MOVL	$0, r2+20(FP)
	NEGL	AX
	MOVL	AX, errno+24(FP)
	RET
ok1:
	MOVL	AX, r1+16(FP)
	MOVL	DX, r2+20(FP)
	MOVL	$0, errno+24(FP)
	RET

