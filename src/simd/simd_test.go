// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"simd"
	"testing"
)

func TestType(t *testing.T) {
	// Testing:
	// - Defined as another struct's field is safe
	// - Pointer is safe.
	// - typedef is safe
	// - type alias is safe
	// - type conversion is safe
	type alias = simd.Int32x4
	type maskT simd.Mask32x4
	type myStruct struct {
		x alias
		y *simd.Int32x4
		z maskT
	}
	vals := [4]int32{1, 2, 3, 4}
	v := myStruct{x: simd.LoadInt32x4(&vals)}
	// masking elements 1 and 2.
	maskv := [4]int32{-1, -1, 0, 0}
	want := []int32{2, 4, 0, 0}
	y := simd.LoadInt32x4(&vals)
	v.y = &y

	if !simd.HasAVX512BW() || !simd.HasAVX512VL() {
		t.Skip("Test requires HasAVX512BW+VL, not available on this hardware")
		return
	}
	v.z = maskT(simd.LoadInt32x4(&maskv).AsMask32x4())
	*v.y = v.y.MaskedAdd(v.x, simd.Mask32x4(v.z))

	got := [4]int32{}
	v.y.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestAdd(t *testing.T) {
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	want := []int32{6, 8, 10, 12}
	x := simd.LoadInt32x4(&xv)
	y := simd.LoadInt32x4(&yv)
	x = x.Add(y)
	got := [4]int32{}
	x.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestVectorConversion(t *testing.T) {
	if !simd.HasAVX512BW() || !simd.HasAVX512VL() {
		t.Skip("Test requires HasAVX512BW+VL, not available on this hardware")
		return
	}
	xv := [4]int32{1, 2, 3, 4}
	x := simd.LoadInt32x4(&xv)
	xPromoted := x.AsInt64x2()
	xPromotedDemoted := xPromoted.AsInt32x4()
	got := [4]int32{}
	xPromotedDemoted.Store(&got)
	for i := range 4 {
		if xv[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, xv[i], got[i])
		}
	}
}

func TestMaskConversion(t *testing.T) {
	if !simd.HasAVX512BW() || !simd.HasAVX512VL() {
		t.Skip("Test requires HasAVX512BW+VL, not available on this hardware")
		return
	}
	v := [4]int32{1, 0, 1, 0}
	x := simd.LoadInt32x4(&v)
	var y simd.Int32x4
	mask := y.Sub(x).AsMask32x4()
	v = [4]int32{5, 6, 7, 8}
	y = simd.LoadInt32x4(&v)
	y = y.MaskedAdd(x, mask)
	got := [4]int32{6, 0, 8, 0}
	y.Store(&v)
	for i := range 4 {
		if v[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, v[i], got[i])
		}
	}
}

func TestMaskedAdd(t *testing.T) {
	if !simd.HasAVX512BW() || !simd.HasAVX512VL() {
		t.Skip("Test requires HasAVX512BW+VL, not available on this hardware")
		return
	}
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	// masking elements 1 and 2.
	maskv := [4]int32{-1, -1, 0, 0}
	want := []int32{6, 8, 0, 0}
	x := simd.LoadInt32x4(&xv)
	y := simd.LoadInt32x4(&yv)
	mask := simd.LoadInt32x4(&maskv).AsMask32x4()
	x = x.MaskedAdd(y, mask)
	got := [4]int32{}
	x.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestCompare(t *testing.T) {
	xv := [4]int32{5, 1, 5, 3}
	yv := [4]int32{3, 3, 3, 3}
	want := []int32{8, 0, 8, 0}
	x := simd.LoadInt32x4(&xv)
	y := simd.LoadInt32x4(&yv)
	if !simd.HasAVX512BW() {
		t.Skip("Test requires HasAVX512BW, not available on this hardware")
		return
	}
	mask := x.Greater(y)
	x = x.MaskedAdd(y, mask)
	got := [4]int32{}
	x.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestSub(t *testing.T) {
	xv := [4]int32{5, 5, 5, 3}
	yv := [4]int32{3, 3, 3, 3}
	want := []int32{2, 2, 2, 0}
	x := simd.LoadInt32x4(&xv)
	y := simd.LoadInt32x4(&yv)
	x = x.Sub(y)
	got := [4]int32{}
	x.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}
