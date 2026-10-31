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
// Keep the probe in a function so it has the same optimization opportunities
// as an ordinary caller. Benchmarks retain the checksum after the timed loop.
func maskBitsScan(data []uint8, tag simd.Uint8s) int {
	mask := simd.LoadUint8s(data).Equal(tag)
	var dst [simd.MaxMaskBitsLen]uint
	n := mask.ToBits(dst[:])
	sum := 0
	// Every mask has at least one word. Handle that word directly so current
	// fixed-width targets do not pay for a loop over a one-word bitmap.
	word := dst[0]
	for word != 0 {
		sum += bits.TrailingZeros(word)
		word &= word - 1
	}
	for j, word := range dst[1:n] {
		for word != 0 {
			sum += (j+1)*bits.UintSize + bits.TrailingZeros(word)
			word &= word - 1
		}
	}
	return sum
}

func BenchmarkMaskBitsScan(b *testing.B) {
	for _, density := range maskDensities {
		b.Run(density, func(b *testing.B) {
			data := maskScanData(density)
			tag := simd.BroadcastUint8s(1)
			sum := 0
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sum += maskBitsScan(data[:], tag)
			}
			maskBitsSink = sum
		})
	}
}

// The general loop is a control for the single-word consumption fast path.
// Use the same function boundary and timing loop for both consumers.
func maskBitsScanLoop(data []uint8, tag simd.Uint8s) int {
	mask := simd.LoadUint8s(data).Equal(tag)
	var dst [simd.MaxMaskBitsLen]uint
	n := mask.ToBits(dst[:])
	sum := 0
	for j, word := range dst[:n] {
		for word != 0 {
			sum += j*bits.UintSize + bits.TrailingZeros(word)
			word &= word - 1
		}
	}
	return sum
}

func BenchmarkMaskBitsScanLoop(b *testing.B) {
	for _, density := range maskDensities {
		b.Run(density, func(b *testing.B) {
			data := maskScanData(density)
			tag := simd.BroadcastUint8s(1)
			sum := 0
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sum += maskBitsScanLoop(data[:], tag)
			}
			maskBitsSink = sum
		})
	}
}
