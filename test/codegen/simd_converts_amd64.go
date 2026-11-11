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

// Enlarge is a single VEX-encoded move, which zeroes the upper half of the
// result. It must write a register of the source's width, not of the result's.

func simdEnlargeInt32x4(x archsimd.Int32x4) archsimd.Int32x8 {
	// amd64:`VMOVDQA X[0-9]+, X[0-9]+` -`CALL`
	return x.Enlarge()
}

func simdEnlargeFloat64x4(x archsimd.Float64x4) archsimd.Float64x8 {
	// amd64:`VMOVAPD Y[0-9]+, Y[0-9]+` -`CALL`
	return x.Enlarge()
}

// x.Enlarge().ExtendTo*() replaces the widening ExtendLo* conversions. A
// widening conversion reads only the low part of its input, which is x, so the
// Enlarge disappears and the result is the same single instruction as the
// ExtendLo* conversion, including when it folds a load.

func simdEnlargeExtendToInt32(x archsimd.Int8x16) archsimd.Int32x8 {
	// amd64:`VPMOVSXBD X[0-9]+, Y[0-9]+` -`VMOVDQA` -`CALL`
	return x.Enlarge().ExtendToInt32()
}

func simdEnlargeExtendToUint32(x archsimd.Uint8x16) archsimd.Uint32x8 {
	// amd64:`VPMOVZXBD X[0-9]+, Y[0-9]+` -`VMOVDQA` -`CALL`
	return x.Enlarge().ExtendToUint32()
}

func simdEnlargeExtendToInt64FromInt8(x archsimd.Int8x16) archsimd.Int64x4 {
	// amd64:`VPMOVSXBQ X[0-9]+, Y[0-9]+` -`VMOVDQA` -`CALL`
	return x.Enlarge().ExtendToInt64()
}

func simdEnlargeExtendToUint64FromUint16(x archsimd.Uint16x8) archsimd.Uint64x4 {
	// amd64:`VPMOVZXWQ X[0-9]+, Y[0-9]+` -`VMOVDQA` -`CALL`
	return x.Enlarge().ExtendToUint64()
}

func simdEnlargeEnlargeExtendToInt64(x archsimd.Int8x16) archsimd.Int64x8 {
	// amd64:`VPMOVSXBQ X[0-9]+, Z[0-9]+` -`VMOVDQA` -`CALL`
	return x.Enlarge().Enlarge().ExtendToInt64()
}

func simdEnlargeExtendToInt32LoadFold(s []int8) archsimd.Int32x8 {
	// amd64:`VPMOVSXBD \([A-Z0-9]+\), Y[0-9]+` -`VMOVDQA` -`VMOVDQU`
	return archsimd.LoadInt8x16(s).Enlarge().ExtendToInt32()
}

func simdEnlargeEnlargeExtendToUint64LoadFold(s []uint8) archsimd.Uint64x8 {
	// amd64:`VPMOVZXBQ \([A-Z0-9]+\), Z[0-9]+` -`VMOVDQA` -`VMOVDQU`
	return archsimd.LoadUint8x16(s).Enlarge().Enlarge().ExtendToUint64()
}

// The same applies to the other operations that read only the low part of an
// input, such as widening float conversions, and at 512 bits.

func simdEnlargeConvertToFloat64(x archsimd.Float32x4) archsimd.Float64x4 {
	// amd64:`VCVTPS2PD X[0-9]+, Y[0-9]+` -`VMOVAPS` -`CALL`
	return x.Enlarge().ConvertToFloat64()
}

func simdEnlargeExtendToInt32Wide(x archsimd.Int16x16) archsimd.Int32x16 {
	// amd64:`VPMOVSXWD Y[0-9]+, Z[0-9]+` -`VMOVDQA` -`CALL`
	return x.Enlarge().ExtendToInt32()
}

// Masked and merging forms are built from the unmasked conversion, so they
// read x directly as well.

func simdEnlargeExtendToInt32Masked(x archsimd.Int8x16, m uint8) archsimd.Int32x8 {
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask32x8FromBits(m)
		// amd64:`VPMOVSXBD.Z X[0-9]+, K[0-7], Y[0-9]+` -`VMOVDQA`
		return x.Enlarge().ExtendToInt32().Masked(mask)
	}
	return archsimd.Int32x8{}
}

func simdEnlargeExtendToInt32Merge(x archsimd.Int8x16, y archsimd.Int32x8, m uint8) archsimd.Int32x8 {
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask32x8FromBits(m)
		// amd64:`VPMOVSXBD X[0-9]+, K[0-7], Y[0-9]+` -`VMOVDQA`
		return x.Enlarge().ExtendToInt32().Merge(y, mask)
	}
	return y
}

// If the Enlarge has a use that reads all of it, the Enlarge stays, but the
// conversion still reads x.

func simdEnlargeExtendToInt32Shared(x archsimd.Int8x16, y archsimd.Int8x32) (archsimd.Int32x8, archsimd.Int8x32) {
	e := x.Enlarge()                   // amd64:`VMOVDQA X0, X[0-9]+`
	return e.ExtendToInt32(), e.Add(y) // amd64:`VPMOVSXBD X0, Y[0-9]+`
}
