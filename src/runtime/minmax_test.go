// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	"strings"
	"testing"
	"unsafe"
)

var (
	zero    = math.Copysign(0, +1)
	negZero = math.Copysign(0, -1)
	inf     = math.Inf(+1)
	negInf  = math.Inf(-1)
	nan     = math.NaN()
)

var tests = []struct{ min, max float64 }{
	{1, 2},
	{-2, 1},
	{negZero, zero},
	{zero, inf},
	{negInf, zero},
	{negInf, inf},
	{1, inf},
	{negInf, 1},
}

var all = []float64{1, 2, -1, -2, zero, negZero, inf, negInf, nan}

func eq(x, y float64) bool {
	return x == y && math.Signbit(x) == math.Signbit(y)
}

func TestMinFloat(t *testing.T) {
	for _, tt := range tests {
		if z := min(tt.min, tt.max); !eq(z, tt.min) {
			t.Errorf("min(%v, %v) = %v, want %v", tt.min, tt.max, z, tt.min)
		}
		if z := min(tt.max, tt.min); !eq(z, tt.min) {
			t.Errorf("min(%v, %v) = %v, want %v", tt.max, tt.min, z, tt.min)
		}
	}
	for _, x := range all {
		if z := min(nan, x); !math.IsNaN(z) {
			t.Errorf("min(%v, %v) = %v, want %v", nan, x, z, nan)
		}
		if z := min(x, nan); !math.IsNaN(z) {
			t.Errorf("min(%v, %v) = %v, want %v", nan, x, z, nan)
		}
	}
}

func TestMaxFloat(t *testing.T) {
	for _, tt := range tests {
		if z := max(tt.min, tt.max); !eq(z, tt.max) {
			t.Errorf("max(%v, %v) = %v, want %v", tt.min, tt.max, z, tt.max)
		}
		if z := max(tt.max, tt.min); !eq(z, tt.max) {
			t.Errorf("max(%v, %v) = %v, want %v", tt.max, tt.min, z, tt.max)
		}
	}
	for _, x := range all {
		if z := max(nan, x); !math.IsNaN(z) {
			t.Errorf("min(%v, %v) = %v, want %v", nan, x, z, nan)
		}
		if z := max(x, nan); !math.IsNaN(z) {
			t.Errorf("min(%v, %v) = %v, want %v", nan, x, z, nan)
		}
	}
}

// testMinMax tests that min/max behave correctly on every pair of
// values in vals.
//
// vals should be a sequence of values in strictly ascending order.
func testMinMax[T int | uint8 | string](t *testing.T, vals ...T) {
	for i, x := range vals {
		for _, y := range vals[i+1:] {
			if !(x < y) {
				t.Fatalf("values out of order: !(%v < %v)", x, y)
			}

			if z := min(x, y); z != x {
				t.Errorf("min(%v, %v) = %v, want %v", x, y, z, x)
			}
			if z := min(y, x); z != x {
				t.Errorf("min(%v, %v) = %v, want %v", y, x, z, x)
			}

			if z := max(x, y); z != y {
				t.Errorf("max(%v, %v) = %v, want %v", x, y, z, y)
			}
			if z := max(y, x); z != y {
				t.Errorf("max(%v, %v) = %v, want %v", y, x, z, y)
			}
		}
	}
}

func TestMinMaxInt(t *testing.T)    { testMinMax[int](t, -7, 0, 9) }
func TestMinMaxUint8(t *testing.T)  { testMinMax[uint8](t, 0, 1, 2, 4, 7) }
func TestMinMaxString(t *testing.T) { testMinMax[string](t, "a", "b", "c") }

// TestMinMaxStringTies ensures that min(a, b) returns a when a == b.
func TestMinMaxStringTies(t *testing.T) {
	s := "xxx"
	x := strings.Split(s, "")

	test := func(i, j, k int) {
		if z := min(x[i], x[j], x[k]); unsafe.StringData(z) != unsafe.StringData(x[i]) {
			t.Errorf("min(x[%v], x[%v], x[%v]) = %p, want %p", i, j, k, unsafe.StringData(z), unsafe.StringData(x[i]))
		}
		if z := max(x[i], x[j], x[k]); unsafe.StringData(z) != unsafe.StringData(x[i]) {
			t.Errorf("max(x[%v], x[%v], x[%v]) = %p, want %p", i, j, k, unsafe.StringData(z), unsafe.StringData(x[i]))
		}
	}

	test(0, 1, 2)
	test(0, 2, 1)
	test(1, 0, 2)
	test(1, 2, 0)
	test(2, 0, 1)
	test(2, 1, 0)
}
