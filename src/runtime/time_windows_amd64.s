// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "textflag.h"
#include "funcdata.h"
#include "time_windows.h"

TEXT time·now<ABIInternal>(SB),NOSPLIT,$0-0
	NO_LOCAL_POINTERS
	CALL	runtime·nanotime1<ABIInternal>(SB)
	MOVQ	AX, R8

	MOVQ	$_SYSTEM_TIME, DI
	MOVQ	time_lo(DI), AX
	MOVQ	$116444736000000000, DI
	SUBQ	DI, AX
	IMULQ	$100, AX

	// generated code for
	//	func f(x uint64) (uint64, uint64) { return x/1000000000, x%1000000000 }
	// adapted to reduce duplication
	MOVQ	AX, CX
	MOVQ	$1360296554856532783, AX
	MULQ	CX
	ADDQ	CX, DX
	RCRQ	$1, DX
	SHRQ	$29, DX
	MOVQ	DX, AX
	IMULQ	$1000000000, DX
	SUBQ	DX, CX
	MOVL	CX, BX
	MOVQ	R8, CX
	RET
