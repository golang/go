// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package test_helpers

import "testing"

func TestMaskAllAny[E integer, V interface {
	Len() int
	Equal(V) M
	NotEqual(V) M
}, M interface {
	All() bool
	Any() bool
	None() bool
	TrailingZeros() int
}](t *testing.T, load func([]E) V) {
	t.Helper()
	var zero V
	lanes := zero.Len()
	a := make([]E, lanes)
	b := make([]E, lanes)
	for i := range lanes {
		a[i] = E(i % 2)
		b[i] = E(i % 3)
	}
	b[1] = 0
	va := load(a)
	vb := load(b)
	m := va.Equal(vb)
	if m.All() {
		t.Errorf("m.All(): want false, got true")
	}
	if !m.Any() {
		t.Errorf("m.Any(): want true, got false")
	}
	if m.None() {
		t.Errorf("m.None(): want false, got true")
	}
	mAllOnes := va.Equal(va)
	if !mAllOnes.All() {
		t.Errorf("mAllOnes.All(): want true, got false")
	}
	if !mAllOnes.Any() {
		t.Errorf("mAllOnes.Any(): want true, got false")
	}
	if mAllOnes.None() {
		t.Errorf("mAllOnes.None(): want false, got true")
	}
	if got := mAllOnes.TrailingZeros(); got != 0 {
		t.Errorf("mAllOnes.TrailingZeros(): want 0, got %d", got)
	}
	mAllZeros := va.NotEqual(va)
	if mAllZeros.All() {
		t.Errorf("mAllZeros.All(): want false, got true")
	}
	if mAllZeros.Any() {
		t.Errorf("mAllZeros.Any(): want false, got true")
	}
	if !mAllZeros.None() {
		t.Errorf("mAllZeros.None(): want true, got false")
	}
	if got := mAllZeros.TrailingZeros(); got != lanes {
		t.Errorf("mAllZeros.TrailingZeros(): want %d, got %d", lanes, got)
	}
	vz := load(make([]E, lanes))
	for i := range lanes {
		s1 := make([]E, lanes)
		s1[i] = 1
		m1 := load(s1).NotEqual(vz)
		if m1.All() {
			t.Errorf("single-1 lane %d All: want false, got true", i)
		}
		if !m1.Any() {
			t.Errorf("single-1 lane %d Any: want true, got false", i)
		}
		if m1.None() {
			t.Errorf("single-1 lane %d None: want false, got true", i)
		}
		if got := m1.TrailingZeros(); got != i {
			t.Errorf("single-1 lane %d TrailingZeros: want %d, got %d", i, i, got)
		}
		s0 := make([]E, lanes)
		for j := range lanes {
			if j != i {
				s0[j] = 1
			}
		}
		m0 := load(s0).NotEqual(vz)
		if m0.All() {
			t.Errorf("single-0 lane %d All: want false, got true", i)
		}
		if !m0.Any() {
			t.Errorf("single-0 lane %d Any: want true, got false", i)
		}
		if m0.None() {
			t.Errorf("single-0 lane %d None: want false, got true", i)
		}
		wantTZ := 0
		if i == 0 {
			wantTZ = 1
		}
		if got := m0.TrailingZeros(); got != wantTZ {
			t.Errorf("single-0 lane %d TrailingZeros: want %d, got %d", i, wantTZ, got)
		}
	}
}
