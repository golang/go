// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT runtime·wasmDiv(SB), NOSPLIT, $0-0
	Get R0
	I64Const $-0x8000000000000000
	I64Eq
	If
		Get R1
		I64Const $-1
		I64Eq
		If
			I64Const $-0x8000000000000000
			Return
		End
	End
	Get R0
	Get R1
	I64DivS
	Return

TEXT runtime·wasmTruncS(SB), NOSPLIT, $0-0
	Get R0
	Get R0
	F64Ne // NaN
	If
		I64Const $0x8000000000000000
		Return
	End

	Get R0
	F64Const $0x7ffffffffffffc00p0 // Maximum truncated representation of 0x7fffffffffffffff
	F64Gt
	If
		I64Const $0x8000000000000000
		Return
	End

	Get R0
	F64Const $-0x7ffffffffffffc00p0 // Minimum truncated representation of -0x8000000000000000
	F64Lt
	If
		I64Const $0x8000000000000000
		Return
	End

	Get R0
	I64TruncF64S
	Return

TEXT runtime·wasmTruncU(SB), NOSPLIT, $0-0
	Get R0
	Get R0
	F64Ne // NaN
	If
		I64Const $0x8000000000000000
		Return
	End

	Get R0
	F64Const $0xfffffffffffff800p0 // Maximum truncated representation of 0xffffffffffffffff
	F64Gt
	If
		I64Const $0x8000000000000000
		Return
	End

	Get R0
	F64Const $0.
	F64Lt
	If
		I64Const $0x8000000000000000
		Return
	End

	Get R0
	I64TruncF64U
	Return

TEXT runtime·exitThread(SB), NOSPLIT, $0-0
	UNDEF

TEXT runtime·osyield(SB), NOSPLIT, $0-0
	UNDEF

TEXT runtime·currentMemory(SB), NOSPLIT, $0
	Get SP
	CurrentMemory
	I32Store ret+0(FP)
	RET

TEXT runtime·growMemory(SB), NOSPLIT, $0
	Get SP
	I32Load pages+0(FP)
	GrowMemory
	I32Store ret+8(FP)
	RET
