// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd

#include "textflag.h"
#include "funcdata.h"

//
// System call support for AMD64 unixes
//

// func Syscall(trap int64, a1, a2, a3 int64) (r1, r2, err int64)
// func Syscall6(trap int64, a1, a2, a3, a4, a5, a6 int64) (r1, r2, err int64)
// Trap # in AX, args in DI SI DX, return in AX DX

TEXT	·Syscall(SB),NOSPLIT,$0-56
	CALL	runtime·entersyscall<ABIInternal>(SB)
	MOVQ	trap+0(FP), AX	// syscall entry
	MOVQ	a1+8(FP), DI
	MOVQ	a2+16(FP), SI
	MOVQ	a3+24(FP), DX
	SYSCALL
	JCC	ok
	MOVQ	$-1, r1+32(FP)	// r1
	MOVQ	$0, r2+40(FP)	// r2
	MOVQ	AX, err+48(FP)	// errno
	CALL	runtime·exitsyscall<ABIInternal>(SB)
	RET
ok:
	MOVQ	AX, r1+32(FP)	// r1
	MOVQ	DX, r2+40(FP)	// r2
	MOVQ	$0, err+48(FP)	// errno
	CALL	runtime·exitsyscall<ABIInternal>(SB)
	RET

TEXT	·Syscall6(SB),NOSPLIT,$0-80
	CALL	runtime·entersyscall<ABIInternal>(SB)
	MOVQ	trap+0(FP), AX	// syscall entry
	MOVQ	a1+8(FP), DI
	MOVQ	a2+16(FP), SI
	MOVQ	a3+24(FP), DX
	MOVQ	a4+32(FP), R10
	MOVQ	a5+40(FP), R8
	MOVQ	a6+48(FP), R9
	SYSCALL
	JCC	ok6
	MOVQ	$-1, r1+56(FP)	// r1
	MOVQ	$0, r2+64(FP)	// r2
	MOVQ	AX, err+72(FP)  // errno
	CALL	runtime·exitsyscall<ABIInternal>(SB)
	RET
ok6:
	MOVQ	AX, r1+56(FP)	// r1
	MOVQ	DX, r2+64(FP)	// r2
	MOVQ	$0, err+72(FP)	// errno
	CALL	runtime·exitsyscall<ABIInternal>(SB)
	RET

TEXT	·RawSyscall(SB),NOSPLIT,$0-56
	MOVQ	a1+8(FP), DI
	MOVQ	a2+16(FP), SI
	MOVQ	a3+24(FP), DX
	MOVQ	trap+0(FP), AX	// syscall entry
	SYSCALL
	JCC	ok1
	MOVQ	$-1, r1+32(FP)	// r1
	MOVQ	$0, r2+40(FP)	// r2
	MOVQ	AX, err+48(FP)	// errno
	RET
ok1:
	MOVQ	AX, r1+32(FP)	// r1
	MOVQ	DX, r2+40(FP)	// r2
	MOVQ	$0, err+48(FP)	// errno
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-80
	MOVQ	a1+8(FP), DI
	MOVQ	a2+16(FP), SI
	MOVQ	a3+24(FP), DX
	MOVQ	a4+32(FP), R10
	MOVQ	a5+40(FP), R8
	MOVQ	a6+48(FP), R9
	MOVQ	trap+0(FP), AX	// syscall entry
	SYSCALL
	JCC	ok2
	MOVQ	$-1, r1+56(FP)	// r1
	MOVQ	$0, r2+64(FP)	// r2
	MOVQ	AX, err+72(FP)	// errno
	RET
ok2:
	MOVQ	AX, r1+56(FP)	// r1
	MOVQ	DX, r2+64(FP)	// r2
	MOVQ	$0, err+72(FP)	// errno
	RET
