// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix

#include "textflag.h"
#include "abi_riscv64.h"

// Called on g0 by cgoCallTrampoline with a PC buffer in X10, its size
// in X11, and runtime.Callers in X12. Set up the ABIInternal arguments
// for Callers(0, buf) and preserve the host's callee-saved registers.
TEXT ·trampolineCallers(SB),NOSPLIT,$232-0
	SAVE_GPR(5*8)
	SAVE_FPR(17*8)
	MOV	X12, X6
	MOV	X11, X13
	MOV	X11, X12
	MOV	X10, X11
	MOV	ZERO, X10
	JALR	RA, X6
	RESTORE_FPR(17*8)
	RESTORE_GPR(5*8)
	RET
