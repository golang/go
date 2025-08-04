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
	MOVV	R4, R11  // syscall entry
	MOVV	R5, R4
	MOVV	R6, R5
	MOVV	R7, R6
	MOVV	R8, R7
	MOVV	R9, R8
	MOVV	R10, R9
	SYSCALL
	MOVV	R0, R5      // r2 is not used. Always set to 0.
	MOVW	$-4096, R12
	BGEU	R12, R4, ok
	SUBVU	R4, R0, R6  // errno
	MOVV	$-1, R4     // r1
	RET
ok:
	// r1 already in R4
	MOVV	R0, R6     // errno
	RET
