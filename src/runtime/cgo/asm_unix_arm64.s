// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin || (linux && !android)) && arm64

#include "textflag.h"
#include "abi_arm64.h"

// callC calls a C function with up to 6 integer/pointer arguments
// in the AAPCS64 C ABI and returns its integer/pointer result.
TEXT ·callC<ABIInternal>(SB),NOSPLIT,$0
	// Convert from Go ABIInternal (R0=fn, R1..R6=a0..a5)
	// to AAPCS64 (fn in R9; args in R0..R5).
	MOVD	R0, R9
	MOVD	R1, R0
	MOVD	R2, R1
	MOVD	R3, R2
	MOVD	R4, R3
	MOVD	R5, R4
	MOVD	R6, R5
	SUB	$16, RSP	// reserve 16 bytes for sp-8 where fp is saved
	BL	(R9)
	ADD	$16, RSP
	RET
