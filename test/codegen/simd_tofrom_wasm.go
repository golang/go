// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that portable simd <-> archsimd conversions via ToArch and FromArch
// inline and fold away with zero overhead on wasm.

//go:build goexperiment.simd && wasm

package codegen

import (
	"simd"
	"simd/archsimd"
)

func toFromArchWasm(x simd.Int32s) simd.Int32s {
	// wasm:-`Call` -`BrIf`
	switch a := x.ToArch().(type) {
	case archsimd.Int32x4:
		// wasm:`I32x4Add` -`Call` -`BrIf`
		return simd.Int32sFromArch(a.Add(a))
	default:
		return x.Add(x)
	}
}
