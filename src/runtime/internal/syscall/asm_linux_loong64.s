// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	MOVV	num+0(FP), R11  // syscall entry
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	a4+32(FP), R7
	MOVV	a5+40(FP), R8
	MOVV	a6+48(FP), R9
	SYSCALL
	MOVW	$-4096, R12
	BGEU	R12, R4, ok
	MOVV	$-1, R12
	MOVV	R12, r1+56(FP)
	MOVV	R0, r2+64(FP)
	SUBVU	R4, R0, R4
	MOVV	R4, errno+72(FP)
	RET
ok:
	MOVV	R4, r1+56(FP)
	MOVV	R5, r2+64(FP)
	MOVV	R0, errno+72(FP)
	RET
