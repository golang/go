// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check that SIMD code generation assumes the CPU features that
// the GOAMD64 level guarantees, without an explicit feature check.

//go:build goexperiment.simdstable && amd64

package codegen

import "simd/archsimd"

func maskAndNot256(a, b, c, d archsimd.Float32x8) archsimd.Mask32x8 {
	// 256-bit mask logic needs only AVX, but uses the integer instructions
	// when AVX2 is known to be available.
	// amd64/v1,amd64/v2:"VANDNPS" -"VPANDN"
	// amd64/v3,amd64/v4:"VPANDN" -"VANDNPS"
	return a.Less(b).AndNot(c.Equal(d))
}

func maskAnd256K(a, b, c, d archsimd.Int32x8) archsimd.Mask32x8 {
	// With AVX-512, comparisons produce K register masks, and logic on those
	// masks stays in K registers.
	// amd64/v3:-"KANDB"
	// amd64/v4:"KANDB"
	return a.Less(b).And(c.Equal(d))
}
