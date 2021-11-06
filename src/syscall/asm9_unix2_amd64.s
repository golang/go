// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd

#include "textflag.h"
#include "funcdata.h"

//
// Syscall9 support for AMD64, DragonFly and FreeBSD
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

	// shift around the last three arguments so they're at the
	// top of the stack when the syscall is called.
	// note that we are scribbling over the Go arguments now.
	MOVQ	SP, CX	// hide (SP) writes from vet
	MOVQ	a7+56(FP), R11 // arg 7
	MOVQ	R11, 8(CX)
	MOVQ	a8+64(FP), R11 // arg 8
	MOVQ	R11, 16(CX)
	MOVQ	a9+72(FP), R11 // arg 9
	MOVQ	R11, 24(CX)

	SYSCALL
	JCC	ok9
	MOVQ	$-1, r1+80(FP)	// r1
	MOVQ	$0, r2+88(FP)	// r2
	MOVQ	AX, err+96(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
ok9:
	MOVQ	AX, r1+80(FP)	// r1
	MOVQ	DX, r2+88(FP)	// r2
	MOVQ	$0, err+96(FP)	// errno
	CALL	runtime·exitsyscall(SB)
	RET
