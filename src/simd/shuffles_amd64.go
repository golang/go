// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

// FlattenedTranspose tranposes x and y, regarded as a pair of 2x2
// matrices, but then flattens the rows in order, i.e
// x: ABCD ==> a: A1B2
// y: 1234     b: C3D4
func (x Int32x4) FlattenedTranspose(y Int32x4) (a, b Int32x4) {
	return x.InterleaveLo(y), x.InterleaveHi(y)
}
