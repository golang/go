// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of SIMD conversions on 128-bit vectors,
// which every architecture has.

//go:build goexperiment.simd

package codegen

import "simd/archsimd"

// Widening the high half of a vector folds into a single instruction where the
// architecture has one.

func simdExtendHiInt8x16(x archsimd.Int8x16) archsimd.Int16x8 {
	// arm64:`VSXTL2` -`VDUP`
	// wasm:`I16x8ExtendHighI8x16S` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().ExtendToInt16()
}

func simdExtendHiUint16x8(x archsimd.Uint16x8) archsimd.Uint32x4 {
	// arm64:`VUXTL2` -`VDUP`
	// wasm:`I32x4ExtendHighI16x8U` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().ExtendToUint32()
}

func simdExtendHiInt32x4(x archsimd.Int32x4) archsimd.Int64x2 {
	// arm64:`VSXTL2` -`VDUP`
	// wasm:`I64x2ExtendHighI32x4S` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().ExtendToInt64()
}

func simdConvertHiFloat32x4(x archsimd.Float32x4) archsimd.Float64x2 {
	// arm64:`VFCVTL2` -`VDUP`
	return x.HiToLo().ConvertToFloat64()
}

func simdExtendHiUint8x16(x archsimd.Uint8x16) archsimd.Uint16x8 {
	// arm64:`VUXTL2` -`VDUP`
	// wasm:`I16x8ExtendHighI8x16U` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().ExtendToUint16()
}

func simdExtendHiUint32x4(x archsimd.Uint32x4) archsimd.Uint64x2 {
	// arm64:`VUXTL2` -`VDUP`
	// wasm:`I64x2ExtendHighI32x4U` -`ExtractLane` -`ReplaceLane`
	return x.HiToLo().ExtendToUint64()
}
