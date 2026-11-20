// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (darwin || (linux && !android))

#include "textflag.h"
#include "abi_arm64.h"

// Called on g0 by cgoCallTrampoline with a PC buffer in R0, its size
// in R1, and runtime.Callers in R2. Set up the ABIInternal arguments
// for Callers(0, buf) and preserve the host's callee-saved registers.
TEXT ·trampolineCallers(SB),NOSPLIT,$192-0
	SAVE_R19_TO_R28(48)
	SAVE_F8_TO_F15(128)
	MOVD	R2, R9
	MOVD	R1, R3
	MOVD	R1, R2
	MOVD	R0, R1
	MOVD	ZR, R0
	CALL	(R9)
	RESTORE_F8_TO_F15(128)
	RESTORE_R19_TO_R28(48)
	RET
