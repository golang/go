// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && amd64

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
		return a.Lo().Elem(0) + a.Hi().Elem(0)
	case archsimd.Float32x16:
		b := a.Lo().Add(a.Hi())
		b = b.ConcatAddPairsGrouped(b)
		b = b.ConcatAddPairsGrouped(b)
		return b.Lo().Elem(0) + b.Hi().Elem(0)
	case archsimd.Float32x4:
		return boringSum(simd.Float32sFromArch(a))
	default:
		return boringSum(x)
	}
}
