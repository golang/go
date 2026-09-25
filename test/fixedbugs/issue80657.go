// compile

//go:build goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "simd"

type Implementation struct{}

func (Implementation) Foo(x []float64) {
	for i := 0; i < len(x); {
		v, n := simd.LoadFloat64sPart(x[i:])
		_ = v
		i += n
	}
}

func (impl Implementation) Bar(x []float32) {
	for i := 0; i < len(x); {
		v, n := simd.LoadFloat32sPart(x[i:])
		_ = v
		i += n
	}
}

func (Implementation) Baz(x []float64, extra ...int) {
	for i := 0; i < len(x); {
		v, n := simd.LoadFloat64sPart(x[i:])
		_ = v
		i += n
	}
	_ = extra
}
