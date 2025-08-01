// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See ../sys_linux_386.s for the reason why we always use int 0x80
// instead of the glibc-specific "CALL 0x10(GS)".
#define INVOKE_SYSCALL	INT	$0x80

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
//
// Syscall # in AX, args in BX CX DX SI DI BP, return in AX
TEXT ·Syscall6(SB),NOSPLIT,$0-40
	MOVL	num+0(FP), AX	// syscall entry
	MOVL	a1+4(FP), BX
	MOVL	a2+8(FP), CX
	MOVL	a3+12(FP), DX
	MOVL	a4+16(FP), SI
	MOVL	a5+20(FP), DI
	MOVL	a6+24(FP), BP
	INVOKE_SYSCALL
	CMPL	AX, $0xfffff001
	JLS	ok
	MOVL	$-1, r1+28(FP)
	MOVL	$0, r2+32(FP)
	NEGL	AX
	MOVL	AX, errno+36(FP)
	RET
ok:
	MOVL	AX, r1+28(FP)
	MOVL	DX, r2+32(FP)
	MOVL	$0, errno+36(FP)
	RET
