// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd

#include "textflag.h"
#include "funcdata.h"

//
// Syscall9 support for AMD64, DragonFly, FreeBSD and NetBSD
//

// func Syscall9(trap int64, a1, a2, a3, a4, a5, a6, a7, a8, a9 int64) (r1, r2, err int64);
TEXT	·Syscall9(SB),NOSPLIT,$32-104
	NO_LOCAL_POINTERS
	CALL	runtime·entersyscall<ABIInternal>(SB)
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

	// only the first 6 arguments can be passed in registers,
	// the last three should be placed at the top of the stack.
	MOVQ	R11, 8(SP)	// arg 7
	MOVQ	R12, 16(SP)	// arg 8
	MOVQ	R13, 24(SP)	// arg 9

	SYSCALL
	JCC	ok9
	MOVQ	$-1, r1+80(FP)	// r1
	MOVQ	$0, r2+88(FP)	// r2
	MOVQ	AX, err+96(FP)	// errno
	CALL	runtime·exitsyscall<ABIInternal>(SB)
	RET
ok9:
	MOVQ	AX, r1+80(FP)	// r1
	MOVQ	DX, r2+88(FP)	// r2
	MOVQ	$0, err+96(FP)	// errno
	CALL	runtime·exitsyscall<ABIInternal>(SB)
	RET
