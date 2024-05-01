// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
//
// We need to convert to the syscall ABI.
//
// arg | ABIInternal | Syscall
// ---------------------------
// num | R4          | R11
// a1  | R5          | R4
// a2  | R6          | R5
// a3  | R7          | R6
// a4  | R8          | R7
// a5  | R9          | R8
// a6  | R10         | R9
//
// r1  | R4          | R4
// r2  | R5          | R5
// err | R6          | part of R4
TEXT ·Syscall6<ABIInternal>(SB),NOSPLIT,$0-80
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R4, R11  // syscall entry
	MOVV	R5, R4
	MOVV	R6, R5
	MOVV	R7, R6
	MOVV	R8, R7
	MOVV	R9, R8
	MOVV	R10, R9
#else
	MOVV	num+0(FP), R11  // syscall entry
	MOVV	a1+8(FP), R4
	MOVV	a2+16(FP), R5
	MOVV	a3+24(FP), R6
	MOVV	a4+32(FP), R7
	MOVV	a5+40(FP), R8
	MOVV	a6+48(FP), R9
#endif
	SYSCALL
#ifdef GOEXPERIMENT_regabiargs
	MOVV	R0, R5      // r2 is not used. Always set to 0.
	MOVW	$-4096, R12
	BGEU	R12, R4, ok
	SUBVU	R4, R0, R6  // errno
	MOVV	$-1, R4     // r1
#else
	MOVW	$-4096, R12
	BGEU	R12, R4, ok
	MOVV	$-1, R12
	MOVV	R12, r1+56(FP)
	MOVV	R0, r2+64(FP)
	SUBVU	R4, R0, R4
	MOVV	R4, errno+72(FP)
#endif
	RET
ok:
#ifdef GOEXPERIMENT_regabiargs
	// r1 already in R4
	MOVV	R0, R6     // errno
#else
	MOVV	R4, r1+56(FP)
	MOVV	R0, r2+64(FP)	// r2 is not used. Always set to 0.
	MOVV	R0, errno+72(FP)
#endif
	RET
