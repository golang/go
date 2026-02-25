// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package ssa

import (
	"testing"

	"cmd/compile/internal/archsimd"
)

func TestVPTERNLOGDPanic(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test only applies to AVX512")
	}

	resultsMask := archsimd.Mask64x8{}
	a := archsimd.Mask64x8FromBits(0xFF)
	b := archsimd.Float64x8{}.Less(archsimd.Float64x8{})

	for i := 0; i < 1; i++ {
		resultsMask = a.Or(b).Or(resultsMask)
		// This nested logic triggered the panic
		_ = resultsMask.And(resultsMask.And(archsimd.Mask64x8{}))
	}
}
