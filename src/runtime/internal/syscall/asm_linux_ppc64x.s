// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	MOVD	num+0(FP), R9	// syscall entry
	MOVD	a1+8(FP), R3
	MOVD	a2+16(FP), R4
	MOVD	a3+24(FP), R5
	MOVD	a4+32(FP), R6
	MOVD	a5+40(FP), R7
	MOVD	a6+48(FP), R8
	SYSCALL	R9
	BVC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+56(FP)
	MOVD	R0, r2+64(FP)
	MOVD	R3, errno+72(FP)
	RET
ok:
	MOVD	R3, r1+56(FP)
	MOVD	R4, r2+64(FP)
	MOVD	R0, errno+72(FP)
	RET
