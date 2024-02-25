// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-40
	MOVW	num+0(FP), R7	// syscall entry
	MOVW	a1+4(FP), R0
	MOVW	a2+8(FP), R1
	MOVW	a3+12(FP), R2
	MOVW	a4+16(FP), R3
	MOVW	a5+20(FP), R4
	MOVW	a6+24(FP), R5
	SWI	$0
	MOVW	$0xfffff001, R6
	CMP	R6, R0
	BLS	ok
	MOVW	$-1, R1
	MOVW	R1, r1+28(FP)
	MOVW	$0, R2
	MOVW	R2, r2+32(FP)
	RSB	$0, R0, R0
	MOVW	R0, errno+36(FP)
	RET
ok:
	MOVW	R0, r1+28(FP)
	MOVW	R1, r2+32(FP)
	MOVW	$0, R0
	MOVW	R0, errno+36(FP)
	RET
