// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math_test

import (
	. "math"
	"testing"
)

func TestBesselMinInt(t *testing.T) {
	for _, x := range []float64{SmallestNonzeroFloat64, 0x1p-29, 1, 10} {
		if got := Jn(MinInt, x); !alike(got, 0) {
			t.Errorf("Jn(MinInt, %g) = %g, want 0", x, got)
		}
		if got := Jn(MinInt, -x); !alike(got, 0) {
			t.Errorf("Jn(MinInt, %g) = %g, want 0", -x, got)
		}
		if got := Yn(MinInt, x); !alike(got, Inf(-1)) {
			t.Errorf("Yn(MinInt, %g) = %g, want -Inf", x, got)
		}
	}
}

func TestBesselExtremeOrders(t *testing.T) {
	for _, n := range []int{MaxInt, MaxInt - 1, MinInt + 1, 1 << 30, -(1 << 30)} {
		for _, x := range []float64{0x1p-29, 1, 10} {
			wantJ := 0.0
			wantY := Inf(-1)
			if n < 0 && n&1 != 0 {
				wantJ = Copysign(0, -1)
				wantY = Inf(1)
			}
			if got := Jn(n, x); !alike(got, wantJ) {
				t.Errorf("Jn(%d, %g) = %g, want %g", n, x, got, wantJ)
			}
			wantNegativeX := wantJ
			if n&1 != 0 {
				wantNegativeX = -wantNegativeX
			}
			if got := Jn(n, -x); !alike(got, wantNegativeX) {
				t.Errorf("Jn(%d, %g) = %g, want %g", n, -x, got, wantNegativeX)
			}
			if got := Yn(n, x); !alike(got, wantY) {
				t.Errorf("Yn(%d, %g) = %g, want %g", n, x, got, wantY)
			}
		}
	}
}

func TestBesselExtremeOrdersLargeX(t *testing.T) {
	// For x much larger than n², the existing asymptotic formulas depend
	// on the order only modulo 4. Extreme orders must not return zero
	// unconditionally.
	for _, n := range []int{MinInt, MinInt + 1, MaxInt - 1, MaxInt} {
		reduced := n % 4
		if reduced == 0 {
			reduced = 4
		}
		for _, x := range []float64{0x1p302, 1e100, MaxFloat64} {
			if got, want := Jn(n, x), Jn(reduced, x); !alike(got, want) {
				t.Errorf("Jn(%d, %g) = %g, want %g", n, x, got, want)
			}
			if got, want := Yn(n, x), Yn(reduced, x); !alike(got, want) {
				t.Errorf("Yn(%d, %g) = %g, want %g", n, x, got, want)
			}
		}
	}
}

func TestBesselSubnormal(t *testing.T) {
	// Values from the series J_n(1) = sum_k (-1)^k / (2^(n+2k) k! (n+k)!),
	// evaluated with rational arithmetic. Do not stop the recurrence while
	// the result can still be represented as a subnormal number.
	for _, test := range []struct {
		n    int
		want float64
	}{
		{150, 1.2243010020861067e-308},
		{151, 4.054020986174775e-311},
		{155, 4.565166567573118e-321},
		{156, 1.4821969375237396e-323},
		{157, 0},
	} {
		if got := Jn(test.n, 1); !close(got, test.want) || test.want == 0 && got != 0 {
			t.Errorf("Jn(%d, 1) = %g, want %g", test.n, got, test.want)
		}
	}
}

func TestBesselExtremeOrdersSpecial(t *testing.T) {
	for _, n := range []int{MinInt, MinInt + 1, MaxInt - 1, MaxInt} {
		if got := Jn(n, NaN()); !IsNaN(got) {
			t.Errorf("Jn(%d, NaN) = %g, want NaN", n, got)
		}
		for _, x := range []float64{0, Copysign(0, -1), Inf(1), Inf(-1)} {
			if got := Jn(n, x); !alike(got, 0) {
				t.Errorf("Jn(%d, %g) = %g, want 0", n, x, got)
			}
		}
		want := Inf(-1)
		if n < 0 && n&1 != 0 {
			want = Inf(1)
		}
		for _, x := range []float64{0, Copysign(0, -1)} {
			if got := Yn(n, x); !alike(got, want) {
				t.Errorf("Yn(%d, %g) = %g, want %g", n, x, got, want)
			}
		}
		for _, x := range []float64{-1, Inf(-1), NaN()} {
			if got := Yn(n, x); !IsNaN(got) {
				t.Errorf("Yn(%d, %g) = %g, want NaN", n, x, got)
			}
		}
		if got := Yn(n, Inf(1)); !alike(got, 0) {
			t.Errorf("Yn(%d, +Inf) = %g, want 0", n, got)
		}
	}
}
