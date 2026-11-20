// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Loads and stores on the stack put the offset into the instruction.

//go:noinline
func wasmStackCallee(a, b, c int) int {
	return a + b + c
}

func wasmStackArgs(x int) int {
	// wasm:`I64Store 8\(SP\)` `I64Store 16\(SP\)` -`I64ExtendI32U`
	return wasmStackCallee(x, x+1, x+2)
}

func wasmStackResults(x, y int) (int, int) {
	// wasm:`I64Store .*~r0\+\d+\(SP\)` `I64Store .*~r1\+\d+\(SP\)` -`I64ExtendI32U`
	return y, x
}

func wasmStackFloatResult(x float64) float64 {
	// wasm:`F64Store .*~r0\+\d+\(SP\)` -`I64ExtendI32U`
	return x * 2
}

//go:noinline
func wasmStackUse(p *[4]int) {}

func wasmStackLocal(x int) int {
	var a [4]int
	// wasm:`I64Store .*a\+\d+\(SP\)` -`I64ExtendI32U`
	a[2] = x
	wasmStackUse(&a)
	return a[3]
}
