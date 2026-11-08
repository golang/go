// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of amd64 SIMD conversions.

//go:build goexperiment.simd && amd64

package codegen

import "simd/archsimd"

// A widening conversion reads only the low part of its input, so a load of
// the input folds into it, whatever the input's width.

func simdExtendToInt32LoadFold(s []int8) archsimd.Int32x8 {
	// amd64:`VPMOVSXBD \([A-Z0-9]+\), Y[0-9]+`
	return archsimd.LoadInt8x32(s).ExtendToInt32()
}

func simdExtendLo8ToInt32LoadFold(s []int8) archsimd.Int32x8 {
	// amd64:`VPMOVSXBD \([A-Z0-9]+\), Y[0-9]+`
	return archsimd.LoadInt8x16(s).ExtendLo8ToInt32()
}

// The 128- and 256-bit 2:1 narrowing conversions are emulated with AVX and
// AVX2 instructions. Check that they stay inlined and short.

func simdTruncToUint8x16(x archsimd.Uint16x8) archsimd.Uint8x16 {
	// amd64:`VPSHUFB` -`CALL`
	return x.TruncToUint8()
}

func simdSaturateToInt8x32(x archsimd.Int16x16) archsimd.Int8x32 {
	// amd64:`VPACKSSWB` `VPERMD` -`CALL`
	return x.SaturateToInt8()
}

func simdSaturateToUint16x8(x archsimd.Uint32x4) archsimd.Uint16x8 {
	// amd64:`VPMINUD` `VPACKUSDW` -`CALL`
	return x.SaturateToUint16()
}

func simdSaturateToInt32x4(x archsimd.Int64x2) archsimd.Int32x4 {
	// amd64:`VPCMPGTQ` `VPBLENDVB` `VSHUFPS` -`CALL`
	return x.SaturateToInt32()
}
