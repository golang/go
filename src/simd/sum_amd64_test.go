// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd"
	"simd/archsimd"
)

func sum(x simd.Float32s) float32 {
	switch a := x.ToArch().(type) {
	case archsimd.Float32x8:
		a = a.ConcatAddPairsGrouped(a)
		a = a.ConcatAddPairsGrouped(a)
		return a.GetLo().GetElem(0) + a.GetHi().GetElem(0)
	case archsimd.Float32x16:
		b := a.GetLo().Add(a.GetHi())
		b = b.ConcatAddPairsGrouped(b)
		b = b.ConcatAddPairsGrouped(b)
		return b.GetLo().GetElem(0) + b.GetHi().GetElem(0)
	case archsimd.Float32x4:
		return boringSum(simd.Float32sFromArch(a))
	default:
		return boringSum(x)
	}
	panic("nope")
}
