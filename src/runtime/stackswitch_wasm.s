// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm.stackswitch

#include "go_asm.h"
#include "textflag.h"

// Accessors for the continuation table (see stackswitch_wasm.go). An I32Load
// with an FP operand pushes SP itself, but an I32Store does not: each result
// store takes its address from an explicit Get SP pushed beforehand.

// func wasmContTableSize() int32
TEXT ·wasmContTableSize(SB), NOSPLIT, $0-4
	Get SP
	TableSize $const_wasmContTable
	I32Store ret+0(FP)
	RET

// func wasmContTableGrow(delta int32) int32
TEXT ·wasmContTableGrow(SB), NOSPLIT, $0-12
	// The operand stack is shown after each instruction.
	Get SP				// SP: address of the result
	RefNull				// SP, null: value of the new slots
	I32Load delta+0(FP)		// SP, null, delta
	TableGrow $const_wasmContTable	// SP, previous size or -1
	I32Store ret+8(FP)
	RET
