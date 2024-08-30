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
// num | A0          | A7
// a1  | A1          | A0
// a2  | A2          | A1
// a3  | A3          | A2
// a4  | A4          | A3
// a5  | A5          | A4
// a6  | A6          | A5
//
// r1  | A0          | A0
// r2  | A1          | A1
// err | A2          | part of A0
TEXT ·Syscall6<ABIInternal>(SB),NOSPLIT,$0-80
	MOV	A0, A7
	MOV	A1, A0
	MOV	A2, A1
	MOV	A3, A2
	MOV	A4, A3
	MOV	A5, A4
	MOV	A6, A5
	ECALL
	MOV	$-4096, T0
	BLTU	T0, A0, err
	// r1 already in A0
	// r2 already in A1
	MOV	ZERO, A2 // errno
	RET
err:
	SUB	A0, ZERO, A2 // errno
	MOV	$-1, A0	     // r1
	MOV	ZERO, A1     // r2
	RET
