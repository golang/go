// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	MOVD	num+0(FP), R1	// syscall entry
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	a4+32(FP), R5
	MOVD	a5+40(FP), R6
	MOVD	a6+48(FP), R7
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok
	MOVD	$-1, r1+56(FP)
	MOVD	$0, r2+64(FP)
	NEG	R2, R2
	MOVD	R2, errno+72(FP)
	RET
ok:
	MOVD	R2, r1+56(FP)
	MOVD	R3, r2+64(FP)
	MOVD	$0, errno+72(FP)
	RET
