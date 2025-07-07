// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"reflect"
	"simd"
	"slices"
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
	*v.y = v.y.AddMasked(v.x, simd.Mask32x4(v.z))

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
	y = y.AddMasked(x, mask)
	got := [4]int32{6, 0, 8, 0}
	y.Store(&v)
	for i := range 4 {
		if v[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, v[i], got[i])
		}
	}
}

func TestPermute(t *testing.T) {
	if !simd.HasAVX512() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	indices := []uint64{7, 6, 5, 4, 3, 2, 1, 0}
	want := []int64{8, 7, 6, 5, 4, 3, 2, 1}
	got := make([]int64, 8)
	simd.LoadInt64x8Slice(x).Permute(simd.LoadUint64x8Slice(indices)).StoreSlice(got)
	for i := range 8 {
		if want[i] != got[i] {
			t.Errorf("want and got differ at index %d, want=%d, got=%d", i, want[i], got[i])
		}
	}
}

func TestPermute2(t *testing.T) {
	if !simd.HasAVX512() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	y := []int64{-1, -2, -3, -4, -5, -6, -7, -8}
	indices := []uint64{7 + 8, 6, 5 + 8, 4, 3 + 8, 2, 1 + 8, 0}
	want := []int64{-8, 7, -6, 5, -4, 3, -2, 1}
	got := make([]int64, 8)
	simd.LoadInt64x8Slice(x).Permute2(simd.LoadInt64x8Slice(y), simd.LoadUint64x8Slice(indices)).StoreSlice(got)
	for i := range 8 {
		if want[i] != got[i] {
			t.Errorf("want and got differ at index %d, want=%d, got=%d", i, want[i], got[i])
		}
	}
}

func TestCompress(t *testing.T) {
	if !simd.HasAVX512() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}
	v1234 := simd.LoadInt32x4Slice([]int32{1, 2, 3, 4})
	v0101 := simd.LoadInt32x4Slice([]int32{0, -1, 0, -1})
	v2400 := v1234.Compress(v0101.AsMask32x4())
	got := make([]int32, 4)
	v2400.StoreSlice(got)
	want := []int32{2, 4, 0, 0}
	if !slices.Equal(got, want) {
		t.Errorf("want and got differ, want=%v, got=%v", want, got)
	}
}

func TestPairDotProdAccumulate(t *testing.T) {
	if !simd.HasAVX512GFNI() {
		// TODO: this function is actually VNNI, let's implement and call the right check.
		t.Skip("Test requires HasAVX512GFNI, not available on this hardware")
		return
	}
	x := simd.LoadInt16x8Slice([]int16{2, 2, 2, 2, 2, 2, 2, 2})
	z := simd.LoadInt32x4Slice([]int32{3, 3, 3, 3})
	want := []int32{11, 11, 11, 11}
	got := make([]int32, 4)
	z = x.PairDotProdAccumulate(x, z)
	z.StoreSlice(got)
	for i := range 4 {
		if got[i] != want[i] {
			t.Errorf("a and b differ at index %d, got=%d, want=%d", i, got[i], want[i])
		}
	}
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
	checkSlices(t, a, b)
}

func TestSlicesInt8SetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadInt8x16Slice(a)

	v = v.SetElem(3, 13)
	a[3] = 13

	b := make([]int8, 16, 16)
	v.StoreSlice(b)
	checkSlices(t, a, b)
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

func TestSlicesInt8Set128(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadInt8x16Slice(a) // 1-16
	u := simd.LoadInt8x32Slice(a) // 1-32

	w := u.Set128(1, v) // 1-16:1-16

	b := make([]int8, 32, 32)
	w.StoreSlice(b)

	checkSlices(t, a, b[:16])
	checkSlices(t, a, b[16:])
}

func TestSlicesInt8Get128(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	u := simd.LoadInt8x32Slice(a) // 1-32
	v := u.Get128(0)              // 1-16
	w := u.Get128(1)              // 17-32

	b := make([]int8, 32, 32)
	v.StoreSlice(b[:16])
	w.StoreSlice(b[16:])

	checkSlices(t, a, b)
}

func TestSlicesFloat32Set128(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadFloat32x4Slice(a) // 1-4
	u := simd.LoadFloat32x8Slice(a) // 1-4

	w := u.Set128(1, v) // 1-4:1-4

	b := make([]float32, 8, 8)
	w.StoreSlice(b)

	checkSlices(t, a, b[:4])
	checkSlices(t, a, b[4:])
}

func TestSlicesFloat32Get128(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	u := simd.LoadFloat32x8Slice(a) // 1-8
	v := u.Get128(0)                // 1-4
	w := u.Get128(1)                // 5-8

	b := make([]float32, 8, 8)
	v.StoreSlice(b[:4])
	w.StoreSlice(b[4:])

	checkSlices(t, a, b)
}

func TestSlicesFloat64Set128(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := simd.LoadFloat64x2Slice(a) // 1-2
	u := simd.LoadFloat64x4Slice(a) // 1-2

	w := u.Set128(1, v) // 1-2:1-2

	b := make([]float64, 4, 4)
	w.StoreSlice(b)

	checkSlices(t, a, b[:2])
	checkSlices(t, a, b[2:])
}

func TestSlicesFloat64Get128(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	u := simd.LoadFloat64x4Slice(a) // 1-4
	v := u.Get128(0)                // 1-2
	w := u.Get128(1)                // 3-4

	b := make([]float64, 4, 4)
	v.StoreSlice(b[:2])
	w.StoreSlice(b[2:])

	checkSlices(t, a, b)
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
	checkSlices(t, a, b)
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
	checkSlices(t, a, b)
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

// TODO: try to reduce this test to be smaller.
func TestMergeLocals(t *testing.T) {
	testMergeLocalswrapper(t, simd.Int64x4.Add)
}

//go:noinline
func forceSpill() {}

func testMergeLocalswrapper(t *testing.T, op func(simd.Int64x4, simd.Int64x4) simd.Int64x4) {
	t.Helper()
	s0 := []int64{0, 1, 2, 3}
	s1 := []int64{-1, 0, -1, 0}
	want := []int64{-1, 1, 1, 3}
	v := simd.LoadInt64x4Slice(s0)
	m := simd.LoadInt64x4Slice(s1)
	forceSpill()
	got := make([]int64, 4)
	gotv := op(v, m)
	gotv.StoreSlice(got)
	for i := range len(want) {
		if !(got[i] == want[i]) {
			t.Errorf("Result at %d incorrect: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestBitMaskLoad(t *testing.T) {
	if !simd.HasAVX512() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}
	var bits uint64 = 0b10
	results := [2]int64{}
	want := [2]int64{0, 6}
	m := simd.LoadMask64x2FromBits(&bits)
	simd.LoadInt64x2Slice([]int64{1, 2}).AddMasked(simd.LoadInt64x2Slice([]int64{3, 4}), m).Store(&results)
	for i := range 2 {
		if results[i] != want[i] {
			t.Errorf("Result at %d incorrect: want %v, got %v", i, want[i], results[i])
		}
	}
}

func TestBitMaskStore(t *testing.T) {
	if !simd.HasAVX512() {
		t.Skip("Test requires HasAVX512, not available on this hardware")
		return
	}
	var want uint64 = 0b101
	var got uint64
	x := simd.LoadInt32x4Slice([]int32{1, 2, 3, 4})
	y := simd.LoadInt32x4Slice([]int32{5, 0, 5, 0})
	m := y.Greater(x)
	m.StoreToBits(&got)
	if got != want {
		t.Errorf("Result incorrect: want %b, got %b", want, got)
	}
}
