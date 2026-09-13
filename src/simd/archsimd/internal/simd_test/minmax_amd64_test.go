// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func TestFloatMinMax32AMD64(t *testing.T) {
	for _, tc := range []struct {
		name  string
		lanes int
		run   func([]float32, []float32, [4][]float32, bool, bool)
	}{
		{"Float32x8", 8, floatMinMax32x8},
		{"Float32x16", 16, floatMinMax32x16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.lanes == 16 && !archsimd.X86.AVX512() {
				t.Skip("requires AVX512")
			}
			testFloatMinMax32(t, tc.lanes, tc.run)
		})
	}
}

func TestFloatMinMax64AMD64(t *testing.T) {
	for _, tc := range []struct {
		name  string
		lanes int
		run   func([]float64, []float64, [4][]float64, bool, bool)
	}{
		{"Float64x4", 4, floatMinMax64x4},
		{"Float64x8", 8, floatMinMax64x8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.lanes == 8 && !archsimd.X86.AVX512() {
				t.Skip("requires AVX512")
			}
			testFloatMinMax64(t, tc.lanes, tc.run)
		})
	}
}

func floatMinMaxPlatformConfig() floatMinMaxConfig {
	return floatMinMaxConfig{allowMasked: archsimd.X86.AVX512()}
}

func floatMinMaxMask32x4() archsimd.Mask32x4 {
	return archsimd.Mask32x4FromBits(0x5)
}

func floatMinMaxMask64x2() archsimd.Mask64x2 {
	return archsimd.Mask64x2FromBits(0x1)
}

//go:noinline
func floatMinMax32x8(a, b []float32, out [4][]float32, reverse, masked bool) {
	x, y := archsimd.LoadFloat32x8(a), archsimd.LoadFloat32x8(b)
	var minXY, minYX, maxXY, maxYX archsimd.Float32x8
	if masked {
		mask := archsimd.Mask32x8FromBits(0x55)
		if reverse {
			minYX, minXY = y.Min(x).Masked(mask), x.Min(y).Masked(mask)
			maxYX, maxXY = y.Max(x).Masked(mask), x.Max(y).Masked(mask)
		} else {
			minXY, minYX = x.Min(y).Masked(mask), y.Min(x).Masked(mask)
			maxXY, maxYX = x.Max(y).Masked(mask), y.Max(x).Masked(mask)
		}
	} else if reverse {
		minYX, minXY = y.Min(x), x.Min(y)
		maxYX, maxXY = y.Max(x), x.Max(y)
	} else {
		minXY, minYX = x.Min(y), y.Min(x)
		maxXY, maxYX = x.Max(y), y.Max(x)
	}
	minXY.Store(out[0])
	minYX.Store(out[1])
	maxXY.Store(out[2])
	maxYX.Store(out[3])
}

//go:noinline
func floatMinMax32x16(a, b []float32, out [4][]float32, reverse, masked bool) {
	x, y := archsimd.LoadFloat32x16(a), archsimd.LoadFloat32x16(b)
	var minXY, minYX, maxXY, maxYX archsimd.Float32x16
	if masked {
		mask := archsimd.Mask32x16FromBits(0x5555)
		if reverse {
			minYX, minXY = y.Min(x).Masked(mask), x.Min(y).Masked(mask)
			maxYX, maxXY = y.Max(x).Masked(mask), x.Max(y).Masked(mask)
		} else {
			minXY, minYX = x.Min(y).Masked(mask), y.Min(x).Masked(mask)
			maxXY, maxYX = x.Max(y).Masked(mask), y.Max(x).Masked(mask)
		}
	} else if reverse {
		minYX, minXY = y.Min(x), x.Min(y)
		maxYX, maxXY = y.Max(x), x.Max(y)
	} else {
		minXY, minYX = x.Min(y), y.Min(x)
		maxXY, maxYX = x.Max(y), y.Max(x)
	}
	minXY.Store(out[0])
	minYX.Store(out[1])
	maxXY.Store(out[2])
	maxYX.Store(out[3])
}

//go:noinline
func floatMinMax64x4(a, b []float64, out [4][]float64, reverse, masked bool) {
	x, y := archsimd.LoadFloat64x4(a), archsimd.LoadFloat64x4(b)
	var minXY, minYX, maxXY, maxYX archsimd.Float64x4
	if masked {
		mask := archsimd.Mask64x4FromBits(0x5)
		if reverse {
			minYX, minXY = y.Min(x).Masked(mask), x.Min(y).Masked(mask)
			maxYX, maxXY = y.Max(x).Masked(mask), x.Max(y).Masked(mask)
		} else {
			minXY, minYX = x.Min(y).Masked(mask), y.Min(x).Masked(mask)
			maxXY, maxYX = x.Max(y).Masked(mask), y.Max(x).Masked(mask)
		}
	} else if reverse {
		minYX, minXY = y.Min(x), x.Min(y)
		maxYX, maxXY = y.Max(x), x.Max(y)
	} else {
		minXY, minYX = x.Min(y), y.Min(x)
		maxXY, maxYX = x.Max(y), y.Max(x)
	}
	minXY.Store(out[0])
	minYX.Store(out[1])
	maxXY.Store(out[2])
	maxYX.Store(out[3])
}

//go:noinline
func floatMinMax64x8(a, b []float64, out [4][]float64, reverse, masked bool) {
	x, y := archsimd.LoadFloat64x8(a), archsimd.LoadFloat64x8(b)
	var minXY, minYX, maxXY, maxYX archsimd.Float64x8
	if masked {
		mask := archsimd.Mask64x8FromBits(0x55)
		if reverse {
			minYX, minXY = y.Min(x).Masked(mask), x.Min(y).Masked(mask)
			maxYX, maxXY = y.Max(x).Masked(mask), x.Max(y).Masked(mask)
		} else {
			minXY, minYX = x.Min(y).Masked(mask), y.Min(x).Masked(mask)
			maxXY, maxYX = x.Max(y).Masked(mask), y.Max(x).Masked(mask)
		}
	} else if reverse {
		minYX, minXY = y.Min(x), x.Min(y)
		maxYX, maxXY = y.Max(x), x.Max(y)
	} else {
		minXY, minYX = x.Min(y), y.Min(x)
		maxXY, maxYX = x.Max(y), y.Max(x)
	}
	minXY.Store(out[0])
	minYX.Store(out[1])
	maxXY.Store(out[2])
	maxYX.Store(out[3])
}
