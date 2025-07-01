// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"reflect"
	"simd"
	"testing"
)

var sink any

func TestType(t *testing.T) {
	// Testing:
	// - Defined as another struct's field is ok
	// - Pointer is ok
	// - Type defition is ok
	// - Type alias is ok
	// - Type conversion is ok
	// - Conversion to interface is ok
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
	sink = y

	if !simd.HasAVX512GFNI() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
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

func TestFuncValue(t *testing.T) {
	// Test that simd intrinsic can be used as a function value.
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	want := []int32{6, 8, 10, 12}
	x := simd.LoadInt32x4(&xv)
	y := simd.LoadInt32x4(&yv)
	fn := simd.Int32x4.Add
	sink = fn
	x = fn(x, y)
	got := [4]int32{}
	x.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestReflectMethod(t *testing.T) {
	// Test that simd intrinsic can be accessed via reflection.
	// NOTE: we don't yet support reflect method.Call.
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	want := []int32{6, 8, 10, 12}
	x := simd.LoadInt32x4(&xv)
	y := simd.LoadInt32x4(&yv)
	m, ok := reflect.TypeOf(x).MethodByName("Add")
	if !ok {
		t.Fatal("Add method not found")
	}
	fn := m.Func.Interface().(func(x, y simd.Int32x4) simd.Int32x4)
	x = fn(x, y)
	got := [4]int32{}
	x.Store(&got)
	for i := range 4 {
		if want[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestVectorConversion(t *testing.T) {
	if !simd.HasAVX512GFNI() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
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
	if !simd.HasAVX512GFNI() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
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

func TestAdd(t *testing.T) {
	testInt32x4Binary(t, []int32{1, 2, 3, 4}, []int32{5, 6, 7, 8}, []int32{6, 8, 10, 12}, "Add")
}

func TestSub(t *testing.T) {
	testInt32x4Binary(t, []int32{5, 5, 5, 3}, []int32{3, 3, 3, 3}, []int32{2, 2, 2, 0}, "Sub")
}

func TestMaskedAdd(t *testing.T) {
	if !simd.HasAVX512GFNI() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}
	testInt32x4BinaryMasked(t, []int32{1, 2, 3, 4}, []int32{5, 6, 7, 8}, []int32{-1, -1, 0, 0}, []int32{6, 8, 0, 0}, "MaskedAdd")
}

// checkInt8Slices ensures that b and a are equal, to the end of b.
// also serves to use the slices, to prevent accidental optimization.
func checkInt8Slices(t *testing.T, a, b []int8) {
	for i := range b {
		if a[i] != b[i] {
			t.Errorf("a and b differ at index %d, a=%d, b=%d", i, a[i], b[i])
		}
	}
}

func TestSlicesInt8(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadInt8x32Slice(a)
	b := make([]int8, 32, 32)
	v.StoreSlice(b)
	checkInt8Slices(t, a, b)
}

func TestSlicesInt8SetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadInt8x16Slice(a)

	v = v.SetElem(3, 13)
	a[3] = 13

	b := make([]int8, 16, 16)
	v.StoreSlice(b)
	checkInt8Slices(t, a, b)
}

func TestSlicesInt8GetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadInt8x16Slice(a)
	e := v.GetElem(2)
	if e != a[2] {
		t.Errorf("GetElem(2) = %d != a[2] = %d", e, a[2])
	}

}
func TestSlicesInt8TooShortLoad(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31} // TOO SHORT, should panic
	v := simd.LoadInt8x32Slice(a)
	b := make([]int8, 32, 32)
	v.StoreSlice(b)
	checkInt8Slices(t, a, b)
}

func TestSlicesInt8TooShortStore(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadInt8x32Slice(a)
	b := make([]int8, 31) // TOO SHORT, should panic
	v.StoreSlice(b)
	checkInt8Slices(t, a, b)
}

func TestSlicesFloat64(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8} // too long, should be fine
	v := simd.LoadFloat64x4Slice(a)
	b := make([]float64, 4, 4)
	v.StoreSlice(b)
	for i := range b {
		if a[i] != b[i] {
			t.Errorf("a and b differ at index %d, a=%f, b=%f", i, a[i], b[i])
		}
	}
}
