// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte(SB), NOSPLIT, $0-40
	I64Load b_base+0(FP)
	I32WrapI64
	I32Load8U c+24(FP)
	I64Load b_len+8(FP)
	I32WrapI64
	Call memchr<>(SB)
	I64ExtendI32S
	Set R0

	Get SP
	I64Const $-1
	Get R0
	I64Load b_base+0(FP)
	I64Sub
	Get R0
	I64Eqz $0
	Select
	I64Store ret+32(FP)

	RET

TEXT ·IndexByteString(SB), NOSPLIT, $0-32
	Get SP
	I64Load s_base+0(FP)
	I32WrapI64
	I32Load8U c+16(FP)
	I64Load s_len+8(FP)
	I32WrapI64
	Call memchr<>(SB)
	I64ExtendI32S
	Set R0

	I64Const $-1
	Get R0
	I64Load s_base+0(FP)
	I64Sub
	Get R0
	I64Eqz $0
	Select
	I64Store ret+24(FP)

	RET

// initially compiled with emscripten and then modified over time.
// params:
//   R0: s
//   R1: c
//   R2: len
// ret: index
TEXT memchr<>(SB), NOSPLIT, $0
	Get R1
	Set R4
	Block
		Block
			Get R2
			I32Const $0
			I32Ne
			Tee R3
			Get R0
			I32Const $3
			I32And
			I32Const $0
			I32Ne
			I32And
			If
				Loop
					Get R0
					I32Load8U $0
					Get R1
					I32Eq
					BrIf $2
					Get R2
					I32Const $-1
					I32Add
					Tee R2
					I32Const $0
					I32Ne
					Tee R3
					Get R0
					I32Const $1
					I32Add
					Tee R0
					I32Const $3
					I32And
					I32Const $0
					I32Ne
					I32And
					BrIf $0
				End
			End
			Get R3
			BrIf $0
			I32Const $0
			Set R1
			Br $1
		End
		Get R0
		I32Load8U $0
		Get R4
		Tee R3
		I32Eq
		If
			Get R2
			Set R1
		Else
			Get R4
			I32Const $16843009
			I32Mul
			Set R4
			Block
				Block
					Get R2
					I32Const $3
					I32GtU
					If
						Get R2
						Set R1
						Loop
							Get R0
							I32Load $0
							Get R4
							I32Xor
							Tee R2
							I32Const $-2139062144
							I32And
							I32Const $-2139062144
							I32Xor
							Get R2
							I32Const $-16843009
							I32Add
							I32And
							I32Eqz
							If
								Get R0
								I32Const $4
								I32Add
								Set R0
								Get R1
								I32Const $-4
								I32Add
								Tee R1
								I32Const $3
								I32GtU
								BrIf $1
								Br $3
							End
						End
					Else
						Get R2
						Set R1
						Br $1
					End
					Br $1
				End
				Get R1
				I32Eqz
				If
					I32Const $0
					Set R1
					Br $3
				End
			End
			Loop
				Get R0
				I32Load8U $0
				Get R3
				I32Eq
				BrIf $2
				Get R0
				I32Const $1
				I32Add
				Set R0
				Get R1
				I32Const $-1
				I32Add
				Tee R1
				BrIf $0
				I32Const $0
				Set R1
			End
		End
	End
	Get R0
	I32Const $0
	Get R1
	Select
	Return
