// asmcheck -gcflags=-d=ssa/check/on

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Shuffle SSA values to cover extensions lowered before their loads.

func wasmSignExtendArgs(x int8, y int16, z int32) (int64, int64, int64) {
	// wasm:`I64Load8S` `I64Load16S` `I64Load32S` -`I64Extend(8|16|32)S`
	return int64(x), int64(y), int64(z)
}

func wasmSignExtendLoads(x *int8, y *int16, z *int32) (int64, int64, int64) {
	// wasm:`I64Load8S` `I64Load16S` `I64Load32S` -`I64Extend(8|16|32)S`
	return int64(*x), int64(*y), int64(*z)
}

func wasmZeroExtendLoads(x *uint8, y *uint16, z *uint32) (uint64, uint64, uint64) {
	// wasm:`I64Load8U` `I64Load16U` `I64Load32U` -`I64And`
	return uint64(*x), uint64(*y), uint64(*z)
}

// Unsigned arguments still need sign extension after conversion to a signed type.
func wasmSignExtendUnsignedArgs(x uint8, y uint16, z uint32) (int64, int64, int64) {
	// wasm:`I64Load8U` `I64Load16U` `I64Load32U` `I64Extend8S` `I64Extend16S` `I64Extend32S`
	return int64(int8(x)), int64(int16(y)), int64(int32(z))
}

// Narrowing a signed value changes which bit must be extended.
func wasmSignExtendNarrowedArg(x int32) (int64, int64) {
	// wasm:`I64Load32S` `I64Extend8S` `I64Extend16S`
	return int64(int8(x)), int64(int16(x))
}

func wasmSignExtendNarrowedLoad(x *int32) (int64, int64) {
	// wasm:`I64Load32S` `I64Extend8S` `I64Extend16S`
	return int64(int8(*x)), int64(int16(*x))
}
