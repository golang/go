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
