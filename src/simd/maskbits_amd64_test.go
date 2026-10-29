// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"math/bits"
	"simd"
	"simd/archsimd"
	"testing"
)

// Compare with the existing ToArch workaround from issue 80149. Keep the
// switch in the loop so a helper call does not inflate the baseline cost.
func BenchmarkMaskToBitsToArch(b *testing.B) {
	if simd.Emulated() {
		b.Skip("ToArch requires hardware SIMD")
	}
	var data [64]uint8
	for i := range data {
		data[i] = uint8(i & 1)
	}
	mask := simd.LoadUint8s(data[:]).Equal(simd.BroadcastUint8s(1))
	var word uint64
	b.ReportAllocs()
	for b.Loop() {
		switch m := mask.ToArch().(type) {
		case archsimd.Mask8x16:
			word = uint64(m.ToBits())
		case archsimd.Mask8x32:
			word = uint64(m.ToBits())
		case archsimd.Mask8x64:
			word = m.ToBits()
		}
	}
	maskBitsSink = int(word)
}

func BenchmarkMaskBitsScanToArch(b *testing.B) {
	if simd.Emulated() {
		b.Skip("ToArch requires hardware SIMD")
	}
	for _, density := range maskDensities {
		b.Run(density, func(b *testing.B) {
			data := maskScanData(density)
			tag := simd.BroadcastUint8s(1)
			sum := 0
			b.ReportAllocs()
			for b.Loop() {
				mask := simd.LoadUint8s(data[:]).Equal(tag)
				var word uint64
				switch m := mask.ToArch().(type) {
				case archsimd.Mask8x16:
					word = uint64(m.ToBits())
				case archsimd.Mask8x32:
					word = uint64(m.ToBits())
				case archsimd.Mask8x64:
					word = m.ToBits()
				}
				for word != 0 {
					sum += bits.TrailingZeros64(word)
					word &= word - 1
				}
			}
			maskBitsSink = sum
		})
	}
}
