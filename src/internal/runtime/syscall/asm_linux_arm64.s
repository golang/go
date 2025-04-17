// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	MOVD	num+0(FP), R8	// syscall entry
	MOVD	a1+8(FP), R0
	MOVD	a2+16(FP), R1
	MOVD	a3+24(FP), R2
	MOVD	a4+32(FP), R3
	MOVD	a5+40(FP), R4
	MOVD	a6+48(FP), R5
	SVC
	CMN	$4095, R0
	BCC	ok
	MOVD	$-1, R4
	MOVD	R4, r1+56(FP)
	MOVD	ZR, r2+64(FP)
	NEG	R0, R0
	MOVD	R0, errno+72(FP)
	RET
ok:
	MOVD	R0, r1+56(FP)
	MOVD	R1, r2+64(FP)
	MOVD	ZR, errno+72(FP)
	RET
