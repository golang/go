// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (ppc64 || ppc64le)

#include "textflag.h"

// func Syscall6(num, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr)
TEXT ·Syscall6<ABIInternal>(SB),NOSPLIT,$0-80
	MOVD	R3, R10	// Move syscall number to R10. SYSCALL will move it R0, and restore R0.
	MOVD	R4, R3
	MOVD	R5, R4
	MOVD	R6, R5
	MOVD	R7, R6
	MOVD	R8, R7
	MOVD	R9, R8
	SYSCALL	R10
	MOVD	$-1, R6
	ISEL	CR0SO, R3, R0, R5 // errno = (error) ? R3 : 0
	ISEL	CR0SO, R6, R3, R3 // r1 = (error) ? -1 : 0
	MOVD	$0, R4            // r2 is not used on linux/ppc64
	RET
