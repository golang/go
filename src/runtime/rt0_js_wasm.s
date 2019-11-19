// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// _rt0_wasm_js is not used itself. It only exists to mark the exported functions as alive.
TEXT _rt0_wasm_js(SB),NOSPLIT,$0
	I32Const $wasm_export_run(SB)
	Drop
	I32Const $wasm_export_resume(SB)
	Drop
	I32Const $wasm_export_getsp(SB)
	Drop

// wasm_export_run gets called from JavaScript. It initializes the Go runtime and executes Go code until it needs
// to wait for an event. It does NOT follow the Go ABI. It has two WebAssembly parameters:
// R0: argc (i32)
// R1: argv (i32)
TEXT wasm_export_run(SB),NOSPLIT,$0
	MOVD $runtime·wasmStack+(m0Stack__size-16)(SB), SP

	Get SP
	Get R0 // argc
	I64ExtendI32U
	I64Store $0

	Get SP
	Get R1 // argv
	I64ExtendI32U
	I64Store $8

	I32Const $0 // entry PC_B
	Call runtime·rt0_go(SB)
	Drop
	Call wasm_pc_f_loop(SB)

	Return

// wasm_export_resume gets called from JavaScript. It resumes the execution of Go code until it needs to wait for
// an event.
TEXT wasm_export_resume(SB),NOSPLIT,$0
	I32Const $0
	Call runtime·handleEvent(SB)
	Drop
	Call wasm_pc_f_loop(SB)

	Return

TEXT wasm_pc_f_loop(SB),NOSPLIT,$0
// Call the function for the current PC_F. Repeat until PAUSE != 0 indicates pause or exit.
// The WebAssembly stack may unwind, e.g. when switching goroutines.
// The Go stack on the linear memory is then used to jump to the correct functions
// with this loop, without having to restore the full WebAssembly stack.
// It is expected to have a pending call before entering the loop, so check PAUSE first.
	Get PAUSE
	I32Eqz
	If
	loop:
		Loop
			// Get PC_B & PC_F from -8(SP)
			Get SP
			I32Const $8
			I32Sub
			I32Load16U $0 // PC_B

			Get SP
			I32Const $8
			I32Sub
			I32Load16U $2 // PC_F

			CallIndirect $0
			Drop

			Get PAUSE
			I32Eqz
			BrIf loop
		End
	End

	I32Const $0
	Set PAUSE

	Return

// wasm_export_getsp gets called from JavaScript to retrieve the SP.
TEXT wasm_export_getsp(SB),NOSPLIT,$0
	Get SP
	Return

TEXT runtime·pause(SB), NOSPLIT, $0-8
	MOVD newsp+0(FP), SP
	I32Const $1
	Set PAUSE
	RETUNWIND

TEXT runtime·exit(SB), NOSPLIT, $0-4
	I32Const $0
	Call runtime·wasmExit(SB)
	Drop
	I32Const $1
	Set PAUSE
	RETUNWIND

TEXT wasm_export_lib(SB),NOSPLIT,$0
	UNDEF
