// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && unix

#include "textflag.h"

// Called on g0 by cgoCallTrampoline with a PC buffer in R2, its size
// in R3, and runtime.Callers in R4. Set up the ABIInternal arguments
// for Callers(0, buf) and preserve the host's callee-saved registers.
TEXT ·trampolineCallers(SB),NOSPLIT,$160-0
	STMG	R6, R12, 40(R15)
	MOVD	g, 96(R15)
	FMOVD	F8, 104(R15)
	FMOVD	F9, 112(R15)
	FMOVD	F10, 120(R15)
	FMOVD	F11, 128(R15)
	FMOVD	F12, 136(R15)
	FMOVD	F13, 144(R15)
	FMOVD	F14, 152(R15)
	FMOVD	F15, 160(R15)
	MOVD	R4, R1
	MOVD	R3, R5
	MOVD	R3, R4
	MOVD	R2, R3
	MOVD	$0, R2
	XOR	R0, R0
	BL	R1
	FMOVD	104(R15), F8
	FMOVD	112(R15), F9
	FMOVD	120(R15), F10
	FMOVD	128(R15), F11
	FMOVD	136(R15), F12
	FMOVD	144(R15), F13
	FMOVD	152(R15), F14
	FMOVD	160(R15), F15
	MOVD	96(R15), g
	LMG	40(R15), R6, R12
	XOR	R0, R0
	RET
