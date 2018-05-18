// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// _rt0_wasm_js does NOT follow the Go ABI. It has two WebAssembly parameters:
// R0: argc (i32)
// R1: argv (i32)
TEXT _rt0_wasm_js(SB),NOSPLIT,$0
	MOVD $runtime·wasmStack+m0Stack__size(SB), SP

	Get SP
	Get R0 // argc
	I64ExtendUI32
	I64Store $0

	Get SP
	Get R1 // argv
	I64ExtendUI32
	I64Store $8

	I32Const $runtime·rt0_go(SB)
	I32Const $16
	I32ShrU
	Set PC_F

// Call the function for the current PC_F. Repeat until SP=0 indicates program end.
// The WebAssembly stack may unwind, e.g. when switching goroutines.
// The Go stack on the linear memory is then used to jump to the correct functions
// with this loop, without having to restore the full WebAssembly stack.
loop:
	Loop
		Get SP
		I32Eqz
		If
			Return
		End

		Get PC_F
		CallIndirect $0
		Drop

		Br loop
	End

TEXT _rt0_wasm_js_lib(SB),NOSPLIT,$0
	UNDEF
