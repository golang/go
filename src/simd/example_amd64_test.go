// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"simd"
	"simd/archsimd"
)

// mulAddPairs multiplies corresponding int16 elements of x and y and adds
// their products into acc, using x86's VPMADDWD (DotProductPairs) when
// available and falling back to portable widening multiplies otherwise.
func mulAddPairs(acc simd.Int32s, x, y simd.Int16s) simd.Int32s {
	switch ax := x.ToArch().(type) {
	case archsimd.Int16x8:
		ay := y.ToArch().(archsimd.Int16x8)
		return acc.Add(simd.Int32sFromArch(ax.DotProductPairs(ay)))
	case archsimd.Int16x16:
		ay := y.ToArch().(archsimd.Int16x16)
		return acc.Add(simd.Int32sFromArch(ax.DotProductPairs(ay)))
	case archsimd.Int16x32:
		ay := y.ToArch().(archsimd.Int16x32)
		return acc.Add(simd.Int32sFromArch(ax.DotProductPairs(ay)))
	default:
		lo := x.ExtendToInt32().Mul(y.ExtendToInt32())
		hi := x.HiToLo().ExtendToInt32().Mul(y.HiToLo().ExtendToInt32())
		return acc.Add(lo).Add(hi)
	}
}

// dotProduct returns the int32 dot product of equal-length slices xs and ys.
func dotProduct(xs, ys []int16) int32 {
	var acc simd.Int32s
	step := (simd.Int16s{}).Len()
	i := 0
	for ; i+step <= len(xs); i += step {
		x := simd.LoadInt16s(xs[i:])
		y := simd.LoadInt16s(ys[i:])
		acc = mulAddPairs(acc, x, y)
	}
	if i < len(xs) {
		x, _ := simd.LoadInt16sPart(xs[i:])
		y, _ := simd.LoadInt16sPart(ys[i:])
		acc = mulAddPairs(acc, x, y)
	}
	return acc.ReduceSum()
}

func ExampleInt16s_ToArch() {
	xs := make([]int16, 50)
	ys := make([]int16, 50)
	for i := range xs {
		xs[i] = int16(i + 1)
		ys[i] = 2
	}
	fmt.Println(dotProduct(xs, ys))
	// Output: 2550
}
