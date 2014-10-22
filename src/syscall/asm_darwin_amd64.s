// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Rewrite all nn(SP) references into name+(nn-8)(FP)
// so that go vet can check that they are correct.

#include "textflag.h"
#include "funcdata.h"

//
// System call support for AMD64, Darwin
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64);
// func Syscall6(trap int64, a1, a2, a3, a4, a5, a6 int64) (r1, r2, err int64);
// Trap # in AX, args in DI SI DX, return in AX DX

TEXT	·Syscall(SB),NOSPLIT,$0-56
	CALL	runtime·entersyscall(SB)
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	$0, R10
	MOVQ	$0, R8
	MOVQ	$0, R9
	MOVQ	8(SP), AX	// syscall entry
	ADDQ	$0x2000000, AX
	SYSCALL
	JCC	ok
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	MOVQ	AX, 56(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
ok:
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),NOSPLIT,$0-80
	CALL	runtime·entersyscall(SB)
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	MOVQ	8(SP), AX	// syscall entry
	ADDQ	$0x2000000, AX
	SYSCALL
	JCC	ok6
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	MOVQ	AX, 80(SP)  // errno
	CALL	runtime·exitsyscall(SB)
	RET
ok6:
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET

TEXT ·RawSyscall(SB),NOSPLIT,$0-56
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	$0, R10
	MOVQ	$0, R8
	MOVQ	$0, R9
	MOVQ	8(SP), AX	// syscall entry
	ADDQ	$0x2000000, AX
	SYSCALL
	JCC	ok1
	MOVQ	$-1, 40(SP)	// r1
	MOVQ	$0, 48(SP)	// r2
	MOVQ	AX, 56(SP)  // errno
	RET
ok1:
	MOVQ	AX, 40(SP)	// r1
	MOVQ	DX, 48(SP)	// r2
	MOVQ	$0, 56(SP)	// errno
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-80
	MOVQ	16(SP), DI
	MOVQ	24(SP), SI
	MOVQ	32(SP), DX
	MOVQ	40(SP), R10
	MOVQ	48(SP), R8
	MOVQ	56(SP), R9
	MOVQ	8(SP), AX	// syscall entry
	ADDQ	$0x2000000, AX
	SYSCALL
	JCC	ok2
	MOVQ	$-1, 64(SP)	// r1
	MOVQ	$0, 72(SP)	// r2
	MOVQ	AX, 80(SP)  // errno
	RET
ok2:
	MOVQ	AX, 64(SP)	// r1
	MOVQ	DX, 72(SP)	// r2
	MOVQ	$0, 80(SP)	// errno
	RET
