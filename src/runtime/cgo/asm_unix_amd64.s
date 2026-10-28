// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin || (linux && !android)) && amd64

#include "go_asm.h"
#include "textflag.h"
#include "abi_amd64.h"

// callC calls a C function with up to 6 integer/pointer arguments
// in the System V AMD64 ABI and returns its integer/pointer result.
// The function may be variadic: AX is cleared to tell it that no
// arguments are passed in vector registers.
TEXT ·callC<ABIInternal>(SB),NOSPLIT,$0
	// Convert from Go ABIInternal (AX, BX, CX, DI, SI, R8, R9)
	// to System V AMD64 ABI (fn in R11; args in DI, SI, DX, CX, R8, R9).
	MOVQ	AX, R11
	MOVQ	DI, DX
	MOVQ	BX, DI
	MOVQ	CX, AX
	MOVQ	SI, CX
	MOVQ	AX, SI
	// a4 and a5 are already in R8 and R9.
	MOVQ	SP, BX
	ANDQ	$~15, SP
	XORL	AX, AX	// vararg: say "no float args"
	CALL	R11
	MOVQ	BX, SP
	// Restore X15 (the ABIInternal zero register), which is caller-saved
	// in the System V AMD64 ABI. The return value is already in AX.
#ifndef GOAMD64_v3
#ifndef GOAMD64_v4
	XORPS	X15, X15
	CMPB	internal∕cpu·X86+const_offsetX86HasAVX(SB), $1
	JNE	noavx
#endif
#endif
	VXORPS	X15, X15, X15
noavx:
	RET
