// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"math/bits"
	"math/rand/v2"
	"simd"
	"slices"
	"testing"
)

func TestMaskBitsScan(t *testing.T) {
	for _, density := range maskDensities {
		t.Run(density, func(t *testing.T) {
			data := maskScanData(density)
			want := 0
			for i, tag := range data[:simd.Uint8s{}.Len()] {
				if tag == 1 {
					want += i
				}
			}
			if got := maskBitsScan(data[:], simd.BroadcastUint8s(1)); got != want {
				t.Fatalf("maskBitsScan = %d, want %d", got, want)
			}
			if got := maskBitsScanLoop(data[:], simd.BroadcastUint8s(1)); got != want {
				t.Fatalf("maskBitsScanLoop = %d, want %d", got, want)
			}
		})
	}
}

func TestMaskToBits(t *testing.T) {
	t.Run("Mask8s", func(t *testing.T) {
		testMaskToBits(t, simd.Int8s{}.Len(), func(s []int8, dst []uint) int {
			return simd.LoadInt8s(s).ToMask().ToBits(dst)
		})
	})
	t.Run("Mask16s", func(t *testing.T) {
		testMaskToBits(t, simd.Int16s{}.Len(), func(s []int16, dst []uint) int {
			return simd.LoadInt16s(s).ToMask().ToBits(dst)
		})
	})
	t.Run("Mask32s", func(t *testing.T) {
		testMaskToBits(t, simd.Int32s{}.Len(), func(s []int32, dst []uint) int {
			return simd.LoadInt32s(s).ToMask().ToBits(dst)
		})
	})
	t.Run("Mask64s", func(t *testing.T) {
		testMaskToBits(t, simd.Int64s{}.Len(), func(s []int64, dst []uint) int {
			return simd.LoadInt64s(s).ToMask().ToBits(dst)
		})
	})
}

func testMaskToBits[E ~int8 | ~int16 | ~int32 | ~int64](t *testing.T, lanes int, extract func([]E, []uint) int) {
	t.Helper()
	n := (lanes + bits.UintSize - 1) / bits.UintSize
	if n > simd.MaxMaskBitsLen {
		t.Fatalf("need %d words, MaxMaskBitsLen = %d", n, simd.MaxMaskBitsLen)
	}
	s := make([]E, lanes)
	rng := rand.New(rand.NewPCG(1, 2))
	// Empty, full, alternating, every individual lane (including the word
	// boundary on wide vectors), and reproducible random masks.
	for trial := 0; trial < lanes+103; trial++ {
		for i := range s {
			selected := false
			switch {
			case trial == 1:
				selected = true
			case trial == 2:
				selected = i%2 == 0
			case trial >= 3 && trial < lanes+3:
				selected = i == trial-3
			case trial >= lanes+3:
				selected = rng.Uint32()&1 != 0
			}
			s[i] = 0
			if selected {
				s[i] = -1
			}
		}
		want := make([]uint, n+2)
		for i, v := range s {
			if v != 0 {
				want[i/bits.UintSize] |= 1 << (i % bits.UintSize)
			}
		}
		want[n], want[n+1] = ^uint(0), ^uint(0)
		dst := make([]uint, n+2)
		for i := range dst {
			dst[i] = ^uint(0)
		}
		if got := extract(s, dst); got != n {
			t.Fatalf("trial %d: wrote %d words, want %d", trial, got, n)
		}
		if !slices.Equal(dst, want) {
			t.Fatalf("trial %d: got %x, want %x", trial, dst, want)
		}
		if got := extract(s, dst[:n]); got != n {
			t.Fatalf("exact buffer: wrote %d words, want %d", got, n)
		}
	}
	for length := 0; length < n; length++ {
		dst := make([]uint, n)
		for i := range dst {
			dst[i] = ^uint(0)
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("length %d: did not panic", length)
				}
			}()
			extract(s, dst[:length])
		}()
		for i, v := range dst {
			if v != ^uint(0) {
				t.Errorf("short buffer modified word %d", i)
			}
		}
	}
}
