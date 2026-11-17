// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalaropt

import (
	"math/bits"
	"testing"
)

// These kernels retain calls so benchmark timing includes the same invocation
// overhead in both builds. The input changes on every iteration.

//go:noinline
func maskedArithmetic(x, y uint64) uint64 {
	return ((x &^ 255) + (y &^ 255) + 37) & 255
}

//go:noinline
func rotatedField(x uint64) uint64 {
	x &= 255
	return (x<<8 | x>>56) & 255
}

//go:noinline
func correlated(flag bool) uint64 {
	var x, y uint64
	if flag {
		x, y = 1, 3
	} else {
		x, y = 2, 2
	}
	return x + y
}

//go:noinline
func dynamicArithmetic(x, y uint64) uint64 {
	return ((x &^ 255) + y + 37) & 255
}

//go:noinline
func maskedLoad(p *uint64) uint64 {
	return ((*p &^ 255) + 37) & 255
}

//go:noinline
func maskedIndex(xs []uint64, i int) uint64 {
	return ((xs[i] &^ 255) + 37) & 255
}

//go:noinline
func maskedDivision(x, y uint64) uint64 {
	return (((x / y) &^ 255) + 37) & 255
}

// Only SCCP can infer this loop's multiplicative fixed point. The late
// intrinsic must then participate in the same constant worklist.
//
//go:noinline
func fixedPointDispatch(n int) uint64 {
	x := uint64(1)
	var sum uint64
	for i := 0; i < n; i++ {
		x *= x
		if bits.Len64(x) == 1 {
			sum += uint64(i)
		} else {
			sum ^= uint64(i)
		}
	}
	return sum
}

//go:noinline
func evaluatedBool(p *bool) bool { return *p && false }

//go:noinline
func shortCircuitBool(p *bool) bool { return true || *p }

var sink uint64

func TestScalarParity(t *testing.T) {
	for i := uint64(0); i < 65536; i++ {
		x := i * 0x9e3779b97f4a7c15
		y := ^x
		if got := maskedArithmetic(x, y); got != 37 {
			t.Fatalf("masked arithmetic(%x, %x) = %d", x, y, got)
		}
		if got := rotatedField(x); got != 0 {
			t.Fatalf("rotated field(%x) = %d", x, got)
		}
		if got := correlated(i&1 != 0); got != 4 {
			t.Fatalf("correlated(%d) = %d", i&1, got)
		}
		if got, want := dynamicArithmetic(x, y), uint64(uint8(y+37)); got != want {
			t.Fatalf("dynamic arithmetic(%x, %x) = %d, want %d", x, y, got, want)
		}
	}
	for _, n := range []int{-1, 0, 1, 2, 3, 17, 255} {
		var want uint64
		for i := 0; i < n; i++ {
			want += uint64(i)
		}
		if got := fixedPointDispatch(n); got != want {
			t.Fatalf("fixed-point dispatch(%d) = %d, want %d", n, got, want)
		}
	}
	if !shortCircuitBool(nil) {
		t.Fatal("short-circuit result changed")
	}
}

func TestScalarPanics(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func() uint64
	}{
		{"nil_load", func() uint64 { return maskedLoad(nil) }},
		{"negative_index", func() uint64 { return maskedIndex([]uint64{1}, -1) }},
		{"past_end_index", func() uint64 { return maskedIndex([]uint64{1}, 1) }},
		{"divide_by_zero", func() uint64 { return maskedDivision(1, 0) }},
		{"evaluated_bool_load", func() uint64 {
			if evaluatedBool(nil) {
				return 1
			}
			return 0
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("required panic was eliminated")
				}
			}()
			sink = tc.call()
		})
	}
}

func BenchmarkMaskedArithmetic(b *testing.B) {
	var result uint64
	for i := 0; i < b.N; i++ {
		result += maskedArithmetic(uint64(i), ^uint64(i))
	}
	sink = result
}

func BenchmarkRotatedField(b *testing.B) {
	var result uint64
	for i := 0; i < b.N; i++ {
		result += rotatedField(uint64(i))
	}
	sink = result
}

func BenchmarkCorrelated(b *testing.B) {
	var result uint64
	for i := 0; i < b.N; i++ {
		result += correlated(i&1 != 0)
	}
	sink = result
}

func BenchmarkFixedPointDispatch(b *testing.B) {
	var result uint64
	for i := 0; i < b.N; i++ {
		result += fixedPointDispatch(17)
	}
	sink = result
}
