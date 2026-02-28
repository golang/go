// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove(SB), NOSPLIT, $0-24
	MOVD to+0(FP), R0
	MOVD from+8(FP), R1
	MOVD n+16(FP), R2

	Get R0
	Get R1
	I64LtU
	If // forward
exit_forward_64:
		Block
loop_forward_64:
			Loop
				Get R2
				I64Const $8
				I64LtU
				BrIf exit_forward_64

				MOVD 0(R1), 0(R0)

				Get R0
				I64Const $8
				I64Add
				Set R0

				Get R1
				I64Const $8
				I64Add
				Set R1

				Get R2
				I64Const $8
				I64Sub
				Set R2

				Br loop_forward_64
			End
		End

loop_forward_8:
		Loop
			Get R2
			I64Eqz
			If
				RET
			End

			Get R0
			I32WrapI64
			I64Load8U (R1)
			I64Store8 $0

			Get R0
			I64Const $1
			I64Add
			Set R0

			Get R1
			I64Const $1
			I64Add
			Set R1

			Get R2
			I64Const $1
			I64Sub
			Set R2

			Br loop_forward_8
		End

	Else
		// backward
		Get R0
		Get R2
		I64Add
		Set R0

		Get R1
		Get R2
		I64Add
		Set R1

exit_backward_64:
		Block
loop_backward_64:
			Loop
				Get R2
				I64Const $8
				I64LtU
				BrIf exit_backward_64

				Get R0
				I64Const $8
				I64Sub
				Set R0

				Get R1
				I64Const $8
				I64Sub
				Set R1

				Get R2
				I64Const $8
				I64Sub
				Set R2

				MOVD 0(R1), 0(R0)

				Br loop_backward_64
			End
		End

loop_backward_8:
		Loop
			Get R2
			I64Eqz
			If
				RET
			End

			Get R0
			I64Const $1
			I64Sub
			Set R0

			Get R1
			I64Const $1
			I64Sub
			Set R1

			Get R2
			I64Const $1
			I64Sub
			Set R2

			Get R0
			I32WrapI64
			I64Load8U (R1)
			I64Store8 $0

			Br loop_backward_8
		End
	End

	UNDEF
