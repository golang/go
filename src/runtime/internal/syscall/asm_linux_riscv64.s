// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6(SB),NOSPLIT,$0-80
	MOV	num+0(FP), A7	// syscall entry
	MOV	a1+8(FP), A0
	MOV	a2+16(FP), A1
	MOV	a3+24(FP), A2
	MOV	a4+32(FP), A3
	MOV	a5+40(FP), A4
	MOV	a6+48(FP), A5
	ECALL
	MOV	$-4096, T0
	BLTU	T0, A0, err
	MOV	A0, r1+56(FP)
	MOV	A1, r2+64(FP)
	MOV	ZERO, errno+72(FP)
	RET
err:
	MOV	$-1, T0
	MOV	T0, r1+56(FP)
	MOV	ZERO, r2+64(FP)
	SUB	A0, ZERO, A0
	MOV	A0, errno+72(FP)
	RET
