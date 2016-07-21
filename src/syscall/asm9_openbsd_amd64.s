// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Rewrite all nn(SP) references into name+(nn-8)(FP)
// so that go vet can check that they are correct.

#include "textflag.h"
#include "funcdata.h"

//
// Syscall9 support for AMD64, OpenBSD
//

// func Syscall9(trap int64, a1, a2, a3, a4, a5, a6, a7, a8, a9 int64) (r1, r2, err int64);
TEXT	·Syscall9(SB),NOSPLIT,$0-104
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
	MOVQ	$-1, 88(SP)	// r1
	MOVQ	$0, 96(SP)	// r2
	MOVQ	AX, 104(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok9:
	ADDQ    $32, SP
	MOVQ	AX, 88(SP)	// r1
	MOVQ	DX, 96(SP)	// r2
	MOVQ	$0, 104(SP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
