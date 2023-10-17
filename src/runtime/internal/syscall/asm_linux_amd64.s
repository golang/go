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
// num | AX          | AX
// a1  | BX          | DI
// a2  | CX          | SI
// a3  | DI          | DX
// a4  | SI          | R10
// a5  | R8          | R8
// a6  | R9          | R9
//
// r1  | AX          | AX
// r2  | BX          | DX
// err | CX          | part of AX
//
// Note that this differs from "standard" ABI convention, which would pass 4th
// arg in CX, not R10.
TEXT ·Syscall6<ABIInternal>(SB),NOSPLIT,$0
	// a6 already in R9.
	// a5 already in R8.
	MOVQ	SI, R10 // a4
	MOVQ	DI, DX  // a3
	MOVQ	CX, SI  // a2
	MOVQ	BX, DI  // a1
	// num already in AX.
	SYSCALL
	CMPQ	AX, $0xfffffffffffff001
	JLS	ok
	NEGQ	AX
	MOVQ	AX, CX  // errno
	MOVQ	$-1, AX // r1
	MOVQ	$0, BX  // r2
	RET
ok:
	// r1 already in AX.
	MOVQ	DX, BX // r2
	MOVQ	$0, CX // errno
	RET
