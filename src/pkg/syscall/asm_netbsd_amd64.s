// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System call support for AMD64, NetBSD
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
// func Syscall6(trap int64, a1, a2, a3, a4, a5, a6 int64) (r1, r2, err int64);
// func Syscall9(trap int64, a1, a2, a3, a4, a5, a6, a7, a8, a9 int64) (r1, r2, err int64);
// Trap # in AX, args in DI SI DX, return in AX DX

TEXT	·Syscall(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVQ	8(SP), AX	// syscall entry
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	$0, R10
	MOVQ	$0, R8
	MOVQ	$0, R9
	SYSCALL
	JCC	ok
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	MOVQ	AX, 56(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVQ	8(SP), AX	// syscall entry
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	SYSCALL
	JCC	ok6
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	MOVQ	AX, 80(SP)  	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall9(SB),7,$0
	CALL	runtime·entersyscall(SB)
	MOVQ	8(SP), AX	// syscall entry
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	MOVQ	64(SP), R11
	MOVQ	72(SP), R12
	MOVQ	80(SP), R13
	SUBQ    $32, SP
	MOVQ	R11, 8(SP)	// arg 7
	MOVQ	R12, 16(SP)	// arg 8
	MOVQ	R13, 24(SP)	// arg 9
	SYSCALL
	JCC	ok9
	ADDQ    $32, SP
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	MOVQ	AX, 80(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok9:
	ADDQ    $32, SP
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·RawSyscall(SB),7,$0
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	$0, R10
	MOVQ	$0, R8
	MOVQ	$0, R9
	MOVQ	8(SP), AX	// syscall entry
	SYSCALL
	JCC	ok1
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	MOVQ	AX, 56(SP)	// errno
	RET
ok1:
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	RET

TEXT	·RawSyscall6(SB),7,$0
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	MOVQ	8(SP), AX	// syscall entry
	SYSCALL
	JCC	ok2
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	MOVQ	AX, 80(SP)	// errno
	RET
ok2:
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	RET
