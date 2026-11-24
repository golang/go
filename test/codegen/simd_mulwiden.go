// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of widening multiplies.

//go:build goexperiment.simdstable && (arm64 || wasm)

package codegen

import "simd/archsimd"

// Multiplying the high halves of two vectors folds into a single instruction.

func simdMulWidenHiInt8x16(x, y archsimd.Int8x16) archsimd.Int16x8 {
	// arm64:`VSMULL2` -`VDUP`
	// wasm:`I16x8ExtmulHighI8x16S` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().MulWiden(y.HiToLo())
}

func simdMulWidenHiUint8x16(x, y archsimd.Uint8x16) archsimd.Uint16x8 {
	// arm64:`VUMULL2` -`VDUP`
	// wasm:`I16x8ExtmulHighI8x16U` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().MulWiden(y.HiToLo())
}

func simdMulWidenHiInt16x8(x, y archsimd.Int16x8) archsimd.Int32x4 {
	// arm64:`VSMULL2` -`VDUP`
	// wasm:`I32x4ExtmulHighI16x8S` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().MulWiden(y.HiToLo())
}

func simdMulWidenHiUint16x8(x, y archsimd.Uint16x8) archsimd.Uint32x4 {
	// arm64:`VUMULL2` -`VDUP`
	// wasm:`I32x4ExtmulHighI16x8U` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().MulWiden(y.HiToLo())
}

func simdMulWidenHiInt32x4(x, y archsimd.Int32x4) archsimd.Int64x2 {
	// arm64:`VSMULL2` -`VDUP`
	// wasm:`I64x2ExtmulHighI32x4S` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().MulWiden(y.HiToLo())
}

func simdMulWidenHiUint32x4(x, y archsimd.Uint32x4) archsimd.Uint64x2 {
	// arm64:`VUMULL2` -`VDUP`
	// wasm:`I64x2ExtmulHighI32x4U` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().MulWiden(y.HiToLo())
}

func simdDotProductPairsInt16x8(x, y archsimd.Int16x8) archsimd.Int32x4 {
	// arm64:`VSMULL\s` `VSMULL2` `VADDP` -`VDUP`
	// wasm:`I32x4DotI16x8S`
	return x.DotProductPairs(y)
}
