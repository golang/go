// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB), NOSPLIT, $0-16
	MOVD ptr+0(FP), R0
	MOVD n+8(FP), R1

loop:
	Loop
		Get R1
		I64Eqz
		If
			RET
		End

		Get R0
		I32WrapI64
		I64Const $0
		I64Store8 $0

		Get R0
		I64Const $1
		I64Add
		Set R0

		Get R1
		I64Const $1
		I64Sub
		Set R1

		Br loop
	End
	UNDEF
