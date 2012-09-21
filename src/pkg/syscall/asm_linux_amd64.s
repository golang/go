// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for AMD64, Linux
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
// Trap # in AX, args in DI SI DX R10 R8 R9, return in AX DX
// Note that this differs from "standard" ABI convention, which
// would pass 4th arg in CX, not R10.

TEXT	·Syscall(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	$0, R10
	MOVQ	$0, R8
	MOVQ	$0, R9
	MOVQ	8(SP), AX	// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	ok
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	NEGQ	AX
	MOVQ	AX, 56(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·Syscall6(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	MOVQ	8(SP), AX	// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	ok6
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	NEGQ	AX
	MOVQ	AX, 80(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),7,$0
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	$0, R10
	MOVQ	$0, R8
	MOVQ	$0, R9
	MOVQ	8(SP), AX	// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	ok1
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	NEGQ	AX
	MOVQ	AX, 56(SP)  // errno
	RET
ok1:
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	RET

TEXT ·RawSyscall6(SB),7,$0
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	MOVQ	8(SP), AX	// syscall entry
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	ok2
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	NEGQ	AX
	MOVQ	AX, 80(SP)  // errno
	RET
ok2:
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	RET

TEXT ·Gettimeofday(SB),7,$0
	MOVQ	8(SP), DI
	MOVQ	$0, SI
	MOVQ	runtime·__vdso_gettimeofday_sym(SB), AX
	CALL	AX

	CMPQ	AX, $0xfffffffffffff001
	JLS	ok7
	NEGQ	AX
	MOVQ	AX, 16(SP)  // errno
	RET
ok7:
	MOVQ	$0, 16(SP)  // errno
	RET

TEXT ·Time(SB),7,$0
	MOVQ	8(SP), DI
	MOVQ	runtime·__vdso_time_sym(SB), AX
	CALL	AX
	MOVQ	AX, 16(SP)  // tt
	MOVQ	$0, 24(SP)  // errno
	RET
