// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"simd"
	"testing"
)

//go:noinline
func mulAdd(A, B, C simd.Float32s) simd.Float32s {
	return A.MulAdd(B, C)
}

func TestMulAdd(t *testing.T) {
	a := []float32{2, 3, 5, 7}
	b := []float32{11, 13, 17, 19}
	c := []float32{23, 29, 31, 37}

	A, _ := simd.LoadFloat32sPart(a)
	B, _ := simd.LoadFloat32sPart(b)
	C, _ := simd.LoadFloat32sPart(c)

	D := mulAdd(A, B, C)
	d := make([]float32, 4)

	D.StorePart(d)

	if d[0] != a[0]*b[0]+c[0] {
		t.Errorf("MulAdd test failed, expected %f, got %f", a[0]*b[0]+c[0], d[0])
	}
}
