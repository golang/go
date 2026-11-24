// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && arm64

package archsimd

// ReduceSum returns the scalar sum of the elements of x.
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
//	z = (x[0] + x[1]) + (x[2] + x[3])
//
// Emulated
func (x Float32x4) ReduceSum() (z float32) {
	x = x.ConcatAddPairs(x) // [x0+x1, x2+x3, x0+x1, x2+x3]
	x = x.ConcatAddPairs(x) // [(x0+x1)+(x2+x3), ...]
	return x.Elem(0)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated
func (x Int64x2) ReduceMax() (z int64) {
	return max(x.Elem(0), x.Elem(1))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated
func (x Int64x2) ReduceMin() (z int64) {
	return min(x.Elem(0), x.Elem(1))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated
func (x Uint64x2) ReduceMax() (z uint64) {
	return max(x.Elem(0), x.Elem(1))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated
func (x Uint64x2) ReduceMin() (z uint64) {
	return min(x.Elem(0), x.Elem(1))
}

// DotProductPairs multiplies corresponding elements of x and y, and sums
// adjacent pairs, returning a vector of half as many elements, each with twice
// the input element size.
//
//	w[i] = x[i] * y[i]        // Double width
//	z[j] = w[2*j] + w[2*j+1]
//
// While the products can never overflow, the sum wraps on overflow.
//
// This happens in exactly one situation: if all four input elements are MinInt16.
//
// Emulated
func (x Int16x8) DotProductPairs(y Int16x8) (z Int32x4) {
	return x.MulWiden(y).ConcatAddPairs(x.HiToLo().MulWiden(y.HiToLo()))
}

// ConvertToFloat64 converts each element of x to float64.
//
//	z[i] = float64(x[i])
//
// Only the low 2 elements of x are converted.
// The remaining elements of x are ignored.
//
// Emulated
func (x Int32x4) ConvertToFloat64() (z Float64x2) {
	return x.ExtendToInt64().ConvertToFloat64()
}

// ConvertToInt32 converts each element of x to int32.
//
//	z[i] = int32(x[i])
//
// When a conversion is inexact, this truncates the result (rounds toward zero).
//
// The handling of NaN and values whose truncated result is outside the
// representable range of int32 is architecture-dependent. ARM64 and WASM map
// NaN to 0 and saturate out-of-range values (including ±Inf). AMD64 returns
// math.MinInt32 (the integer indefinite value) for NaN and all out-of-range
// values.
//
//	trunc(x)          ARM64/WASM  AMD64
//	----------------  ----------  --------
//	NaN               0           MinInt32
//	< MinInt32, -Inf  MinInt32    MinInt32
//	> MaxInt32, +Inf  MaxInt32    MinInt32
//
// The results are stored in the low 2 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Float64x2) ConvertToInt32() (z Int32x4) {
	return x.ConvertToInt64().SaturateToInt32()
}
