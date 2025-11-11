// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6<ABIInternal>(SB),NOSPLIT,$0-80
#ifdef GOEXPERIMENT_regabiargs
	MOVD	R2, R1
	MOVD	R3, R2
	MOVD	R4, R3
	MOVD	R5, R4
	MOVD	R6, R5
	MOVD	R7, R6
	MOVD	R8, R7
#else
	MOVD	num+0(FP), R1	// syscall entry
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	a4+32(FP), R5
	MOVD	a5+40(FP), R6
	MOVD	a6+48(FP), R7
#endif
	SYSCALL
	MOVD	$0xfffffffffffff001, R8
	CMPUBLT	R2, R8, ok
#ifdef GOEXPERIMENT_regabiargs
	MOVD	$0, R3
	NEG	R2, R4
	MOVD    $-1, R2
#else
	MOVD	$-1, r1+56(FP)
	MOVD	$0, r2+64(FP)
	NEG	R2, R2
	MOVD	R2, errno+72(FP)
#endif
	RET
ok:
#ifdef GOEXPERIMENT_regabiargs
	MOVD	$0, R4
#else
	MOVD	R2, r1+56(FP)
	MOVD	R3, r2+64(FP)
	MOVD	$0, errno+72(FP)
#endif
	RET
