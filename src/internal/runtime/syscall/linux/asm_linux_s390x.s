// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6<ABIInternal>(SB),NOSPLIT,$0-80
	MOVD	R2, R1
	MOVD	R3, R2
	MOVD	R4, R3
	MOVD	R5, R4
	MOVD	R6, R5
	MOVD	R7, R6
	MOVD	R8, R7
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok
	MOVD	$0, R3
	NEG	R2, R4
	MOVD    $-1, R2
	RET
ok:
	MOVD	$0, R4
	RET
