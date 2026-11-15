// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"internal/cpu"
	"simd/archsimd"
	"testing"
)

func TestHelpersWithoutAVX2(t *testing.T) {
	saved := cpu.X86.HasAVX2
	cpu.X86.HasAVX2 = false
	defer func() { cpu.X86.HasAVX2 = saved }()
	if archsimd.X86.AVX2() {
		t.Skip("AVX2 is guaranteed by GOAMD64")
	}
	t.Run("unary", func(t *testing.T) {
		testUint64x4Unary(t, func(x archsimd.Uint64x4) archsimd.Uint64x4 {
			t.Fatal("called 256-bit operation without AVX2")
			return x
		}, func(x []uint64) []uint64 { return x })
	})
}
