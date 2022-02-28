// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// Syscall9 support for AMD64, NetBSD
//

// func Syscall9(trap int64, a1, a2, a3, a4, a5, a6, a7, a8, a9 int64) (r1, r2, err int64);
TEXT	·Syscall9(SB),NOSPLIT,$0-104
	CALL	runtime·entersyscall(SB)
	MOVQ	num+0(FP), AX	// syscall entry
	MOVQ	a1+8(FP), DI
	MOVQ	a2+16(FP), SI
	MOVQ	a3+24(FP), DX
	MOVQ	a4+32(FP), R10
	MOVQ	a5+40(FP), R8
	MOVQ	a6+48(FP), R9
	MOVQ	a7+56(FP), R11
	MOVQ	a8+64(FP), R12
	MOVQ	a9+72(FP), R13
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
