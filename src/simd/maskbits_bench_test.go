// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"math/bits"
	"simd"
	"testing"
)

var maskBitsSink int

func BenchmarkMaskToBits(b *testing.B) {
	var data [64]uint8
	for i := range data {
		data[i] = uint8(i & 1)
	}
	mask := simd.LoadUint8s(data[:]).Equal(simd.BroadcastUint8s(1))
	var dst [simd.MaxMaskBitsLen]uint
	b.ReportAllocs()
	for b.Loop() {
		mask.ToBits(dst[:])
	}
	maskBitsSink = int(dst[0])
}

var maskDensities = []string{"empty", "one", "sparse", "half", "full"}

func maskScanData(density string) [64]uint8 {
	var data [64]uint8
	lanes := simd.Uint8s{}.Len()
	for i := range lanes {
		switch density {
		case "one":
			if i == lanes-1 {
				data[i] = 1
			}
		case "sparse":
			if i == 1 || i == lanes/2 || i == lanes-1 {
				data[i] = 1
			}
		case "half":
			data[i] = uint8(i & 1)
		case "full":
			data[i] = 1
		}
	}
	return data
}

// Include the load and comparison: this is the path used by tag-table probes.
func BenchmarkMaskBitsScan(b *testing.B) {
	for _, density := range maskDensities {
		b.Run(density, func(b *testing.B) {
			data := maskScanData(density)
			tag := simd.BroadcastUint8s(1)
			var dst [simd.MaxMaskBitsLen]uint
			sum := 0
			b.ReportAllocs()
			for b.Loop() {
				mask := simd.LoadUint8s(data[:]).Equal(tag)
				n := mask.ToBits(dst[:])
				for j, word := range dst[:n] {
					for word != 0 {
						sum += j*bits.UintSize + bits.TrailingZeros(word)
						word &= word - 1
					}
				}
			}
			maskBitsSink = sum
		})
	}
}
