// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// The register RUN indicates the current run state of the program.
// Possible values are:
#define RUN_STARTING 0
#define RUN_RUNNING 1
#define RUN_PAUSED 2
#define RUN_EXITED 3

// _rt0_wasm_js does NOT follow the Go ABI. It has two WebAssembly parameters:
// R0: argc (i32)
// R1: argv (i32)
TEXT _rt0_wasm_js(SB),NOSPLIT,$0
	Get RUN
	I32Const $RUN_STARTING
	I32Eq
	If
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

		I32Const $RUN_RUNNING
		Set RUN
	Else
		Get RUN
		I32Const $RUN_PAUSED
		I32Eq
		If
			I32Const $RUN_RUNNING
			Set RUN
		Else
			Unreachable
		End
	End

// Call the function for the current PC_F. Repeat until RUN != 0 indicates pause or exit.
// The WebAssembly stack may unwind, e.g. when switching goroutines.
// The Go stack on the linear memory is then used to jump to the correct functions
// with this loop, without having to restore the full WebAssembly stack.
loop:
	Loop
		Get PC_F
		CallIndirect $0
		Drop

		Get RUN
		I32Const $RUN_RUNNING
		I32Eq
		BrIf loop
	End

	Return

TEXT runtime·pause(SB), NOSPLIT, $0
	I32Const $RUN_PAUSED
	Set RUN
	RETUNWIND

TEXT runtime·exit(SB), NOSPLIT, $0-4
	Call runtime·wasmExit(SB)
	Drop
	I32Const $RUN_EXITED
	Set RUN
	RETUNWIND

TEXT _rt0_wasm_js_lib(SB),NOSPLIT,$0
	UNDEF
