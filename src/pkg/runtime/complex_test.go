// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math/cmplx"
	"testing"
)

var result complex128

func BenchmarkComplex128DivNormal(b *testing.B) {
	d := 15 + 2i
	n := 32 + 3i
	for i := 0; i < b.N; i++ {
		n += n / d
	}
	result = n
}

func BenchmarkComplex128DivNisNaN(b *testing.B) {
	d := cmplx.NaN()
	n := 32 + 3i
	for i := 0; i < b.N; i++ {
		n += n / d
	}
	result = n
}

func BenchmarkComplex128DivDisNaN(b *testing.B) {
	d := 15 + 2i
	n := cmplx.NaN()
	for i := 0; i < b.N; i++ {
		n += n / d
	}
	result = n
}

func BenchmarkComplex128DivNisInf(b *testing.B) {
	d := 15 + 2i
	n := cmplx.Inf()
	for i := 0; i < b.N; i++ {
		n += n / d
	}
	result = n
}

func BenchmarkComplex128DivDisInf(b *testing.B) {
	d := cmplx.Inf()
	n := 32 + 3i
	for i := 0; i < b.N; i++ {
		n += n / d
	}
	result = n
}
