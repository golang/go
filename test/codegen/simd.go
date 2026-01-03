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
	v1 := archsimd.LoadUint64x2Slice([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2Slice([]uint64{0, 0})
	// amd64:`VPTEST\s(.*)(.*)$`
	// amd64:`SETCS\s(.*)$`
	return v1.AndNot(v2).IsZero()
}

func vptest2() bool {
	v1 := archsimd.LoadUint64x2Slice([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2Slice([]uint64{0, 0})
	// amd64:`VPTEST\s(.*)(.*)$`
	// amd64:`SETEQ\s(.*)$`
	return v1.And(v2).IsZero()
}

type Args2 struct {
	V0 archsimd.Uint8x32
	V1 archsimd.Uint8x32
	x  string
}

//go:noinline
func simdStructNoSpill(a Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU\s.*$`
	return a.V0.Xor(a.V1)
}

func simdStructWrapperNoSpill(a Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU\s.*$`
	a.x = "test"
	return simdStructNoSpill(a)
}

//go:noinline
func simdArrayNoSpill(a [1]Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU\s.*$`
	return a[0].V0.Xor(a[0].V1)
}

func simdArrayWrapperNoSpill(a [1]Args2) archsimd.Uint8x32 {
	// amd64:-`VMOVDQU\s.*$`
	a[0].x = "test"
	return simdArrayNoSpill(a)
}

func simdFeatureGuardedMaskOpt() archsimd.Int16x16 {
	var x, y archsimd.Int16x16
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask16x16FromBits(5)
		return x.Add(y).Masked(mask) // amd64:`VPADDW.Z\s.*$`
	}
	mask := archsimd.Mask16x16FromBits(5)
	return x.Add(y).Masked(mask) // amd64:`VPAND\s.*$`
}

func simdMaskedMerge() archsimd.Int16x16 {
	var x, y archsimd.Int16x16
	if archsimd.X86.AVX512() {
		mask := archsimd.Mask16x16FromBits(5)
		return x.Add(y).Merge(x, mask) // amd64:-`VPBLENDVB\s.*$`
	}
	mask := archsimd.Mask16x16FromBits(5)
	return x.Add(y).Merge(x, mask) // amd64:`VPBLENDVB\s.*$`
}

var nan = math.NaN()
var floats64s = []float64{0, 1, 2, nan, 4, nan, 6, 7, 8, 9, 10, 11, nan, 13, 14, 15}
var sinkInt64s = make([]int64, 100)

func simdIsNaN() {
	x := archsimd.LoadFloat64x4Slice(floats64s)
	y := archsimd.LoadFloat64x4Slice(floats64s[4:])
	a := x.IsNaN()
	b := y.IsNaN()
	// amd64:"VCMPPD [$]3," -"VPOR"
	c := a.Or(b)
	c.ToInt64x4().StoreSlice(sinkInt64s)
}

func simdIsNaN512() {
	x := archsimd.LoadFloat64x8Slice(floats64s)
	y := archsimd.LoadFloat64x8Slice(floats64s[8:])
	a := x.IsNaN()
	b := y.IsNaN()
	// amd64:"VCMPPD [$]3," -"VPOR"
	c := a.Or(b)
	c.ToInt64x8().StoreSlice(sinkInt64s)
}
