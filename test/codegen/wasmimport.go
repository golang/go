// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm

package codegen

// Imported 32-bit values should not be widened just to store them on the Go stack.

//go:wasmimport test int32
func ImportInt32() int32 // wasm:`I32Store\s` -`I64ExtendI32U`

//go:wasmimport test bool
func ImportBool() bool // wasm:`I32Store8` -`I64ExtendI32U`

// Pointer results must retain their full Go pointer width on each architecture.

//go:wasmimport test pointer
func ImportPointer() *byte // wasm:`I64ExtendI32U` `I64Store\s`

//go:wasmimport test int64
func ImportInt64() int64 // wasm:`I64Store\s`
