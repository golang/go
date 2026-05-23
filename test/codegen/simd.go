// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of simd peephole optimizations.

//go:build goexperiment.simd && amd64

package codegen

import (
	"math"
	"simd/archsimd"
)

func vptest1() bool {
	v1 := archsimd.LoadUint64x2([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2([]uint64{0, 0})
	// amd64:`VPTEST (.*)(.*)$`
	// amd64:`SETCS (.*)$`
	return v1.AndNot(v2).IsZero()
}

func vptest2() bool {
	v1 := archsimd.LoadUint64x2([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2([]uint64{0, 0})
	// amd64:`VPTEST (.*)(.*)$`
	// amd64:`SETEQ (.*)$`
	return v1.And(v2).IsZero()
}

type Args2 struct {
	V0 archsimd.Uint8x32
	V1 archsimd.Uint8x32
	x  string
}

//go:noinline
func simdStructNoSpill(a Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	return a.V0.Xor(a.V1)
}

func simdStructWrapperNoSpill(a Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	a.x = "test"
	return simdStructNoSpill(a)
}

//go:noinline
func simdArrayNoSpill(a [1]Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	return a[0].V0.Xor(a[0].V1)
}

func simdArrayWrapperNoSpill(a [1]Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU .*$`
	a[0].x = "test"
	return simdArrayNoSpill(a)
}

func simdFeatureGuardedMaskOpt() archsimd.Int16x16 {
	var x, y archsimd.Int16x16
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask16x16FromBits(5)
		return x.Add(y).Masked(mask) // amd64:`VPADDW.Z .*$`
	}
	mask := archsimd.Mask16x16FromBits(5)
	return x.Add(y).Masked(mask) // amd64:`VPAND .*$`
}

func simdMaskedMerge() archsimd.Int16x16 {
	var x, y archsimd.Int16x16
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask16x16FromBits(5)
		return x.Add(y).Merge(x, mask) // amd64:-`VPBLENDVB .*$`
	}
	mask := archsimd.Mask16x16FromBits(5)
	return x.Add(y).Merge(x, mask) // amd64:`VPBLENDVB .*$`
}

var nan = math.NaN()
var floats64s = []float64{0, 1, 2, nan, 4, nan, 6, 7, 8, 9, 10, 11, nan, 13, 14, 15}
var sinkInt64s = make([]int64, 100)

func simdIsNaN() {
	x := archsimd.LoadFloat64x4(floats64s)
	y := archsimd.LoadFloat64x4(floats64s[4:])
	a := x.IsNaN()
	b := y.IsNaN()
	// amd64:"VCMPPD [$]3," -"VPOR"
	c := a.Or(b)
	c.ToInt64x4().Store(sinkInt64s)
}

func simdIsNaN512() {
	x := archsimd.LoadFloat64x8(floats64s)
	y := archsimd.LoadFloat64x8(floats64s[8:])
	a := x.IsNaN()
	b := y.IsNaN()
	// amd64:"VCMPPD [$]3," -"VPOR"
	c := a.Or(b)
	c.ToInt64x8().Store(sinkInt64s)
}

func sftImmVPSRL() archsimd.Uint32x4 {
	var x archsimd.Uint32x4
	// amd64:`VPSRLD \$1, .*$`
	return x.ShiftAllRight(1)
}

func aLtbLtc8_avx512(a, b, c archsimd.Int8x64) archsimd.Mask8x64 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KANDB`
	return a.Less(b).And(b.Less(c))
}

func aLtbORbLtc8_avx512(a, b, c archsimd.Int8x64) archsimd.Mask8x64 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KORB`
	return a.Less(b).Or(b.Less(c))
}

func aLtbLtc64_avx512(a, b, c archsimd.Int64x8) archsimd.Mask64x8 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KANDQ`
	return a.Less(b).And(b.Less(c))
}

func aLtbORbLtc64_avx512(a, b, c archsimd.Int64x8) archsimd.Mask64x8 {
	// the vector length implies AVX512 implies the mask operations.
	// amd64:`KORQ`
	return a.Less(b).Or(b.Less(c))
}

var globalSlice = []uint32{1, 2, 3, 4, 5, 6, 7, 8}

func simdMemoryOperandMerge() archsimd.Uint32x4 {
	a := archsimd.BroadcastUint32x4(1)
	// amd64:`VPADDD \([A-Z]+\), X\d, X\d`
	a = a.Add(archsimd.LoadUint32x4(globalSlice[0:4]))
	// amd64:`VPADDD 16\([A-Z]+\), X\d, X\d`
	a = a.Add(archsimd.LoadUint32x4(globalSlice[4:8]))
	return a
}
