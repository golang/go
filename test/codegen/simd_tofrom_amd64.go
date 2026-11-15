// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that portable simd <-> archsimd conversions via ToArch and FromArch
// inline and fold away with zero overhead.

//go:build goexperiment.simd && amd64

package codegen

import (
	"simd"
	"simd/archsimd"
)

func toFromArchAMD64(x simd.Int32s) simd.Int32s {
	// amd64:-`CALL` -`CMP` -`JEQ` -`JNE`
	switch a := x.ToArch().(type) {
	case archsimd.Int32x4:
		// amd64:`VPADDD X[0-9]+, X[0-9]+, X[0-9]+` -`CALL` -`VMOV`
		return simd.Int32sFromArch(a.Add(a))
	case archsimd.Int32x8:
		// amd64:`VPADDD Y[0-9]+, Y[0-9]+, Y[0-9]+` -`CALL` -`VMOV`
		return simd.Int32sFromArch(a.Add(a))
	case archsimd.Int32x16:
		// amd64:`VPADDD Z[0-9]+, Z[0-9]+, Z[0-9]+` -`CALL` -`VMOV`
		return simd.Int32sFromArch(a.Add(a))
	default:
		return x.Add(x)
	}
}
