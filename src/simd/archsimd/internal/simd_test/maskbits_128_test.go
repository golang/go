// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestMaskToBits128(t *testing.T) {
	t.Run("Mask8x16", func(t *testing.T) { testMaskBits128(t, archsimd.LoadInt8x16) })
	t.Run("Mask16x8", func(t *testing.T) { testMaskBits128(t, archsimd.LoadInt16x8) })
	t.Run("Mask32x4", func(t *testing.T) { testMaskBits128(t, archsimd.LoadInt32x4) })
	t.Run("Mask64x2", func(t *testing.T) { testMaskBits128(t, archsimd.LoadInt64x2) })
}

func testMaskBits128[E ~int8 | ~int16 | ~int32 | ~int64, V interface {
	Len() int
	ToMask() M
}, M interface {
	ToBits() R
}, R ~uint8 | ~uint16](t *testing.T, load func([]E) V) {
	t.Helper()
	var zero V
	s := make([]E, zero.Len())
	// Exhaust all possible masks, including adjacent lanes and both halves
	// of the vector. This also checks that unused high bits are zero.
	for bitmap := 0; bitmap < 1<<len(s); bitmap++ {
		for i := range s {
			s[i] = 0
			if bitmap&(1<<i) != 0 {
				s[i] = -1
			}
		}
		if got := load(s).ToMask().ToBits(); int(got) != bitmap {
			t.Fatalf("mask %x: ToBits() = %x", bitmap, got)
		}
	}
}
