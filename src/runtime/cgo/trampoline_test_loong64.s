// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix

#include "textflag.h"
#include "abi_loong64.h"

// Called on g0 by cgoCallTrampoline with a PC buffer in R4, its size
// in R5, and runtime.Callers in R6. Set up the ABIInternal arguments
// for Callers(0, buf) and preserve the host's callee-saved registers.
TEXT ·trampolineCallers(SB),NOSPLIT,$184-0
	SAVE_R22_TO_R31(5*8)
	SAVE_F24_TO_F31(15*8)
	MOVV	R6, R12
	MOVV	R5, R7
	MOVV	R5, R6
	MOVV	R4, R5
	MOVV	R0, R4
	JAL	(R12)
	RESTORE_F24_TO_F31(15*8)
	RESTORE_R22_TO_R31(5*8)
	RET
