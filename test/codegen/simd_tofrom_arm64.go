// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that portable simd <-> archsimd conversions via ToArch and FromArch
// inline and fold away with zero overhead on arm64 (including the nclm variant).

//go:build goexperiment.simd && arm64

package codegen

import (
	"simd"
	"simd/archsimd"
)

func toFromArchARM64(x simd.Int32s) simd.Int32s {
	// arm64:-`CALL` -`CMP` -`BNE` -`BEQ` -`TBZ` -`TBNZ`
	switch a := x.ToArch().(type) {
	case archsimd.Int32x4:
		// arm64:`VADD V[0-9]+\.S4, V[0-9]+\.S4, V[0-9]+\.S4` -`CALL` -`CMP` -`TBZ` -`TBNZ`
		return simd.Int32sFromArch(a.Add(a))
	default:
		return x.Add(x)
	}
}
