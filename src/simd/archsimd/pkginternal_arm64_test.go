// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd_test

import (
	"math/rand/v2"
	"simd/archsimd"
	"testing"
)

// clmul8 returns the carryless product of a and b.
func clmul8(a, b uint8) uint16 {
	var z uint16
	for i := range 8 {
		if b>>i&1 != 0 {
			z ^= uint16(a) << i
		}
	}
	return z
}

func TestCarrylessMultiplyWidenLo8(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for range 1000 {
		var xs, ys [16]uint8
		for i := range xs {
			xs[i], ys[i] = uint8(r.Uint32()), uint8(r.Uint32())
		}
		x, y := archsimd.LoadUint8x16Array(&xs), archsimd.LoadUint8x16Array(&ys)

		var lo, hi [8]uint16
		x.ExportTestCarrylessMultiplyWidenLo(y).StoreArray(&lo)
		x.ExportTestCarrylessMultiplyWidenHi(y).StoreArray(&hi)
		for i := range 8 {
			if want := clmul8(xs[i], ys[i]); lo[i] != want {
				t.Fatalf("lo[%d] = clmul(%#x, %#x) = %#x, want %#x", i, xs[i], ys[i], lo[i], want)
			}
			if want := clmul8(xs[8+i], ys[8+i]); hi[i] != want {
				t.Fatalf("hi[%d] = clmul(%#x, %#x) = %#x, want %#x", i, xs[8+i], ys[8+i], hi[i], want)
			}
		}
	}
}
