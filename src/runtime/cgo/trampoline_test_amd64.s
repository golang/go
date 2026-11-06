// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && (darwin || linux || netbsd || openbsd || windows)

#include "textflag.h"
#include "abi_amd64.h"

// Called on g0 with the PC buffer, its length, and the ABIInternal entry
// for runtime.Callers in the host's first three argument registers.
// Return Callers(0, buf).
TEXT ·trampolineCallers(SB),NOSPLIT|NOFRAME,$0-0
	PUSH_REGS_HOST_TO_ABI0()
	ADJSP	$32 // spill space for the four ABIInternal argument registers
	MOVQ	(TLS), R14
	XORPS	X15, X15
	XORL	AX, AX
#ifdef GOOS_windows
	MOVQ	CX, BX
	MOVQ	DX, CX
	MOVQ	DX, DI
	CALL	R8
#else
	MOVQ	DI, BX
	MOVQ	SI, CX
	MOVQ	SI, DI
	CALL	DX
#endif
	ADJSP	$-32
	POP_REGS_HOST_TO_ABI0()
	RET
