// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Compare(SB), NOSPLIT, $0-56
	Get SP
	I64Load a_base+0(FP)
	I64Load a_len+8(FP)
	I64Load b_base+24(FP)
	I64Load b_len+32(FP)
	Call cmpbody<>(SB)
	I64Store ret+48(FP)
	RET

TEXT bytes·Compare(SB), NOSPLIT, $0-56
	FUNCDATA $0, ·Compare·args_stackmap(SB)
	Get SP
	I64Load a_base+0(FP)
	I64Load a_len+8(FP)
	I64Load b_base+24(FP)
	I64Load b_len+32(FP)
	Call cmpbody<>(SB)
	I64Store ret+48(FP)
	RET

TEXT runtime·cmpstring(SB), NOSPLIT, $0-40
	Get SP
	I64Load a_base+0(FP)
	I64Load a_len+8(FP)
	I64Load b_base+16(FP)
	I64Load b_len+24(FP)
	Call cmpbody<>(SB)
	I64Store ret+32(FP)
	RET

// params: a, alen, b, blen
// ret: -1/0/1
TEXT cmpbody<>(SB), NOSPLIT, $0-0
	// len = min(alen, blen)
	Get R1
	Get R3
	Get R1
	Get R3
	I64LtU
	Select
	Set R4

	Get R0
	I32WrapI64
	Get R2
	I32WrapI64
	Get R4
	I32WrapI64
	Call memcmp<>(SB)
	I64ExtendSI32
	Set R5

	Get R5
	I64Eqz
	If
		// check length
		Get R1
		Get R3
		I64Sub
		Set R5
	End

	I64Const $0
	I64Const $-1
	I64Const $1
	Get R5
	I64Const $0
	I64LtS
	Select
	Get R5
	I64Eqz
	Select
	Return

// compiled with emscripten
// params: a, b, len
// ret: <0/0/>0
TEXT memcmp<>(SB), NOSPLIT, $0-0
	Get R2
	If $1
	Loop
	Get R0
	I32Load8S $0
	Tee R3
	Get R1
	I32Load8S $0
	Tee R4
	I32Eq
	If
	Get R0
	I32Const $1
	I32Add
	Set R0
	Get R1
	I32Const $1
	I32Add
	Set R1
	I32Const $0
	Get R2
	I32Const $-1
	I32Add
	Tee R2
	I32Eqz
	BrIf $3
	Drop
	Br $1
	End
	End
	Get R3
	I32Const $255
	I32And
	Get R4
	I32Const $255
	I32And
	I32Sub
	Else
	I32Const $0
	End
	Return
