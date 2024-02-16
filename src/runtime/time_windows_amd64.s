// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

#include "go_asm.h"
#include "textflag.h"
#include "time_windows.h"

TEXT time·now(SB),NOSPLIT,$0-24
	MOVQ	$_INTERRUPT_TIME, DI
	MOVQ	time_lo(DI), AX
	IMULQ	$100, AX
	MOVQ	AX, mono+16(FP)

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
	MOVQ	DX, sec+0(FP)
	IMULQ	$1000000000, DX
	SUBQ	DX, CX
	MOVL	CX, nsec+8(FP)
	RET
