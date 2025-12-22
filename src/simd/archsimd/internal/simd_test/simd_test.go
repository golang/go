// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"os"
	"reflect"
	"simd/archsimd"
	"slices"
	"testing"
	"unsafe"
)

func TestMain(m *testing.M) {
	if !archsimd.X86.AVX() {
		fmt.Fprintln(os.Stderr, "Skipping tests: AVX is not available")
		os.Exit(0)
	}
	os.Exit(m.Run())
}

var sink any

func TestType(t *testing.T) {
	// Testing:
	// - Defined as another struct's field is ok
	// - Pointer is ok
	// - Type defition is ok
	// - Type alias is ok
	// - Type conversion is ok
	// - Conversion to interface is ok
	type alias = archsimd.Int32x4
	type maskT archsimd.Mask32x4
	type myStruct struct {
		x alias
		y *archsimd.Int32x4
		z maskT
	}
	vals := [4]int32{1, 2, 3, 4}
	v := myStruct{x: archsimd.LoadInt32x4(&vals)}
	// masking elements 1 and 2.
	want := []int32{2, 4, 0, 0}
	y := archsimd.LoadInt32x4(&vals)
	v.y = &y
	sink = y

	if !archsimd.X86.AVX512GFNI() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	v.z = maskT(archsimd.Mask32x4FromBits(0b0011))
	*v.y = v.y.Add(v.x).Masked(archsimd.Mask32x4(v.z))

	got := [4]int32{}
	v.y.Store(&got)
	checkSlices(t, got[:], want)
}

func TestUncomparable(t *testing.T) {
	// Test that simd vectors are not comparable
	var x, y any = archsimd.LoadUint32x4(&[4]uint32{1, 2, 3, 4}), archsimd.LoadUint32x4(&[4]uint32{5, 6, 7, 8})
	shouldPanic := func(fn func()) {
		defer func() {
			if recover() == nil {
				panic("did not panic")
			}
		}()
		fn()
	}
	shouldPanic(func() { _ = x == y })
}

func TestFuncValue(t *testing.T) {
	// Test that simd intrinsic can be used as a function value.
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	want := []int32{6, 8, 10, 12}
	x := archsimd.LoadInt32x4(&xv)
	y := archsimd.LoadInt32x4(&yv)
	fn := archsimd.Int32x4.Add
	sink = fn
	x = fn(x, y)
	got := [4]int32{}
	x.Store(&got)
	checkSlices(t, got[:], want)
}

func TestReflectMethod(t *testing.T) {
	// Test that simd intrinsic can be accessed via reflection.
	// NOTE: we don't yet support reflect method.Call.
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	want := []int32{6, 8, 10, 12}
	x := archsimd.LoadInt32x4(&xv)
	y := archsimd.LoadInt32x4(&yv)
	m, ok := reflect.TypeOf(x).MethodByName("Add")
	if !ok {
		t.Fatal("Add method not found")
	}
	fn := m.Func.Interface().(func(x, y archsimd.Int32x4) archsimd.Int32x4)
	x = fn(x, y)
	got := [4]int32{}
	x.Store(&got)
	checkSlices(t, got[:], want)
}

func TestVectorConversion(t *testing.T) {
	if !archsimd.X86.AVX512GFNI() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	xv := [4]int32{1, 2, 3, 4}
	x := archsimd.LoadInt32x4(&xv)
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
	if !archsimd.X86.AVX512GFNI() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := archsimd.LoadInt32x4Slice([]int32{5, 0, 7, 0})
	mask := archsimd.Int32x4{}.Sub(x).ToMask()
	y := archsimd.LoadInt32x4Slice([]int32{1, 2, 3, 4}).Add(x).Masked(mask)
	want := [4]int32{6, 0, 10, 0}
	got := make([]int32, 4)
	y.StoreSlice(got)
	checkSlices(t, got[:], want[:])
}

func TestPermute(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	indices := []uint64{7, 6, 5, 4, 3, 2, 1, 0}
	want := []int64{8, 7, 6, 5, 4, 3, 2, 1}
	got := make([]int64, 8)
	archsimd.LoadInt64x8Slice(x).Permute(archsimd.LoadUint64x8Slice(indices)).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestPermuteOrZero(t *testing.T) {
	x := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	indices := []int8{7, 6, 5, 4, 3, 2, 1, 0, -1, 8, -1, 9, -1, 10, -1, 11}
	want := []uint8{8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 0, 10, 0, 11, 0, 12}
	got := make([]uint8, len(x))
	archsimd.LoadUint8x16Slice(x).PermuteOrZero(archsimd.LoadInt8x16Slice(indices)).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestConcatPermute(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	y := []int64{-1, -2, -3, -4, -5, -6, -7, -8}
	indices := []uint64{7 + 8, 6, 5 + 8, 4, 3 + 8, 2, 1 + 8, 0}
	want := []int64{-8, 7, -6, 5, -4, 3, -2, 1}
	got := make([]int64, 8)
	archsimd.LoadInt64x8Slice(x).ConcatPermute(archsimd.LoadInt64x8Slice(y), archsimd.LoadUint64x8Slice(indices)).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestCompress(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	v1234 := archsimd.LoadInt32x4Slice([]int32{1, 2, 3, 4})
	v2400 := v1234.Compress(archsimd.Mask32x4FromBits(0b1010))
	got := make([]int32, 4)
	v2400.StoreSlice(got)
	want := []int32{2, 4, 0, 0}
	if !slices.Equal(got, want) {
		t.Errorf("want and got differ, want=%v, got=%v", want, got)
	}
}

func TestExpand(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	v3400 := archsimd.LoadInt32x4Slice([]int32{3, 4, 0, 0})
	v2400 := v3400.Expand(archsimd.Mask32x4FromBits(0b1010))
	got := make([]int32, 4)
	v2400.StoreSlice(got)
	want := []int32{0, 3, 0, 4}
	if !slices.Equal(got, want) {
		t.Errorf("want and got differ, want=%v, got=%v", want, got)
	}
}

var testShiftAllVal uint64 = 3

func TestShiftAll(t *testing.T) {
	got := make([]int32, 4)
	archsimd.LoadInt32x4Slice([]int32{0b11, 0b11, 0b11, 0b11}).ShiftAllLeft(2).StoreSlice(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("expect 0b1100, got %b", v)
		}
	}
	archsimd.LoadInt32x4Slice([]int32{0b11, 0b11, 0b11, 0b11}).ShiftAllLeft(testShiftAllVal).StoreSlice(got)
	for _, v := range got {
		if v != 0b11000 {
			t.Errorf("expect 0b11000, got %b", v)
		}
	}
}

func TestSlicesInt8(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x32Slice(a)
	b := make([]int8, 32, 32)
	v.StoreSlice(b)
	checkSlices(t, a, b)
}

func TestSlicesInt8SetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x16Slice(a)

	v = v.SetElem(3, 13)
	a[3] = 13

	b := make([]int8, 16, 16)
	v.StoreSlice(b)
	checkSlices(t, a, b)
}

func TestSlicesInt8GetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x16Slice(a)
	e := v.GetElem(2)
	if e != a[2] {
		t.Errorf("GetElem(2) = %d != a[2] = %d", e, a[2])
	}

}

func TestSlicesInt8TooShortLoad(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31} // TOO SHORT, should panic
	v := archsimd.LoadInt8x32Slice(a)
	b := make([]int8, 32, 32)
	v.StoreSlice(b)
	checkSlices(t, a, b)
}

func TestSlicesInt8TooShortStore(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x32Slice(a)
	b := make([]int8, 31) // TOO SHORT, should panic
	v.StoreSlice(b)
	checkSlices(t, a, b)
}

func TestSlicesFloat64(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8} // too long, should be fine
	v := archsimd.LoadFloat64x4Slice(a)
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
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	testMergeLocalswrapper(t, archsimd.Int64x4.Add)
}

//go:noinline
func forceSpill() {}

func testMergeLocalswrapper(t *testing.T, op func(archsimd.Int64x4, archsimd.Int64x4) archsimd.Int64x4) {
	t.Helper()
	s0 := []int64{0, 1, 2, 3}
	s1 := []int64{-1, 0, -1, 0}
	want := []int64{-1, 1, 1, 3}
	v := archsimd.LoadInt64x4Slice(s0)
	m := archsimd.LoadInt64x4Slice(s1)
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

func TestBitMaskFromBits(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	results := [2]int64{}
	want := [2]int64{0, 6}
	m := archsimd.Mask64x2FromBits(0b10)
	archsimd.LoadInt64x2Slice([]int64{1, 2}).Add(archsimd.LoadInt64x2Slice([]int64{3, 4})).Masked(m).Store(&results)
	for i := range 2 {
		if results[i] != want[i] {
			t.Errorf("Result at %d incorrect: want %v, got %v", i, want[i], results[i])
		}
	}
}

var maskForTestBitMaskFromBitsLoad = uint8(0b10)

func TestBitMaskFromBitsLoad(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	results := [2]int64{}
	want := [2]int64{0, 6}
	m := archsimd.Mask64x2FromBits(maskForTestBitMaskFromBitsLoad)
	archsimd.LoadInt64x2Slice([]int64{1, 2}).Add(archsimd.LoadInt64x2Slice([]int64{3, 4})).Masked(m).Store(&results)
	for i := range 2 {
		if results[i] != want[i] {
			t.Errorf("Result at %d incorrect: want %v, got %v", i, want[i], results[i])
		}
	}
}

func TestBitMaskToBits(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	if v := archsimd.LoadInt16x8Slice([]int16{1, 0, 1, 0, 0, 0, 0, 0}).ToMask().ToBits(); v != 0b101 {
		t.Errorf("Want 0b101, got %b", v)
	}
}

var maskForTestBitMaskFromBitsStore uint8

func TestBitMaskToBitsStore(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	maskForTestBitMaskFromBitsStore = archsimd.LoadInt16x8Slice([]int16{1, 0, 1, 0, 0, 0, 0, 0}).ToMask().ToBits()
	if maskForTestBitMaskFromBitsStore != 0b101 {
		t.Errorf("Want 0b101, got %b", maskForTestBitMaskFromBitsStore)
	}
}

func TestMergeFloat(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	k := make([]int64, 4, 4)
	s := make([]float64, 4, 4)

	a := archsimd.LoadFloat64x4Slice([]float64{1, 2, 3, 4})
	b := archsimd.LoadFloat64x4Slice([]float64{4, 2, 3, 1})
	g := a.Greater(b)
	g.ToInt64x4().StoreSlice(k)
	c := a.Merge(b, g)

	c.StoreSlice(s)

	checkSlices[int64](t, k, []int64{0, 0, 0, -1})
	checkSlices[float64](t, s, []float64{4, 2, 3, 4})
}

func TestMergeFloat512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	k := make([]int64, 8, 8)
	s := make([]float64, 8, 8)

	a := archsimd.LoadFloat64x8Slice([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	b := archsimd.LoadFloat64x8Slice([]float64{8, 7, 6, 5, 4, 2, 3, 1})
	g := a.Greater(b)
	g.ToInt64x8().StoreSlice(k)
	c := a.Merge(b, g)
	d := a.Masked(g)

	checkSlices[int64](t, k, []int64{0, 0, 0, 0, -1, -1, -1, -1})

	c.StoreSlice(s)
	checkSlices[float64](t, s, []float64{8, 7, 6, 5, 5, 6, 7, 8})

	d.StoreSlice(s)
	checkSlices[float64](t, s, []float64{0, 0, 0, 0, 5, 6, 7, 8})
}

var ro uint8 = 2

func TestRotateAllVariable(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	got := make([]int32, 4)
	archsimd.LoadInt32x4Slice([]int32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(ro).StoreSlice(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("Want 0b1100, got %b", v)
		}
	}
}

func TestBroadcastUint32x4(t *testing.T) {
	s := make([]uint32, 4, 4)
	archsimd.BroadcastUint32x4(123456789).StoreSlice(s)
	checkSlices(t, s, []uint32{123456789, 123456789, 123456789, 123456789})
}

func TestBroadcastFloat32x8(t *testing.T) {
	s := make([]float32, 8, 8)
	archsimd.BroadcastFloat32x8(123456789).StoreSlice(s)
	checkSlices(t, s, []float32{123456789, 123456789, 123456789, 123456789, 123456789, 123456789, 123456789, 123456789})
}

func TestBroadcastFloat64x2(t *testing.T) {
	s := make([]float64, 2, 2)
	archsimd.BroadcastFloat64x2(123456789).StoreSlice(s)
	checkSlices(t, s, []float64{123456789, 123456789})
}

func TestBroadcastUint64x2(t *testing.T) {
	s := make([]uint64, 2, 2)
	archsimd.BroadcastUint64x2(123456789).StoreSlice(s)
	checkSlices(t, s, []uint64{123456789, 123456789})
}

func TestBroadcastUint16x8(t *testing.T) {
	s := make([]uint16, 8, 8)
	archsimd.BroadcastUint16x8(12345).StoreSlice(s)
	checkSlices(t, s, []uint16{12345, 12345, 12345, 12345})
}

func TestBroadcastInt8x32(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	s := make([]int8, 32, 32)
	archsimd.BroadcastInt8x32(-123).StoreSlice(s)
	checkSlices(t, s, []int8{-123, -123, -123, -123, -123, -123, -123, -123,
		-123, -123, -123, -123, -123, -123, -123, -123,
		-123, -123, -123, -123, -123, -123, -123, -123,
		-123, -123, -123, -123, -123, -123, -123, -123,
	})
}

func TestMaskOpt512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	k := make([]int64, 8, 8)
	s := make([]float64, 8, 8)

	a := archsimd.LoadFloat64x8Slice([]float64{2, 0, 2, 0, 2, 0, 2, 0})
	b := archsimd.LoadFloat64x8Slice([]float64{1, 1, 1, 1, 1, 1, 1, 1})
	c := archsimd.LoadFloat64x8Slice([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	d := archsimd.LoadFloat64x8Slice([]float64{2, 4, 6, 8, 10, 12, 14, 16})
	g := a.Greater(b)
	e := c.Add(d).Masked(g)
	e.StoreSlice(s)
	g.ToInt64x8().StoreSlice(k)
	checkSlices[int64](t, k, []int64{-1, 0, -1, 0, -1, 0, -1, 0})
	checkSlices[float64](t, s, []float64{3, 0, 9, 0, 15, 0, 21, 0})
}

// flattenedTranspose tranposes x and y, regarded as a pair of 2x2
// matrices, but then flattens the rows in order, i.e
// x: ABCD ==> a: A1B2
// y: 1234     b: C3D4
func flattenedTranspose(x, y archsimd.Int32x4) (a, b archsimd.Int32x4) {
	return x.InterleaveLo(y), x.InterleaveHi(y)
}

func TestFlattenedTranspose(t *testing.T) {
	r := make([]int32, 4, 4)
	s := make([]int32, 4, 4)

	x := archsimd.LoadInt32x4Slice([]int32{0xA, 0xB, 0xC, 0xD})
	y := archsimd.LoadInt32x4Slice([]int32{1, 2, 3, 4})
	a, b := flattenedTranspose(x, y)

	a.StoreSlice(r)
	b.StoreSlice(s)

	checkSlices[int32](t, r, []int32{0xA, 1, 0xB, 2})
	checkSlices[int32](t, s, []int32{0xC, 3, 0xD, 4})

}

func TestClearAVXUpperBits(t *testing.T) {
	// Test that ClearAVXUpperBits is safe even if there are SIMD values
	// alive (although usually one should not do this).
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}

	r := make([]int64, 4)
	s := make([]int64, 4)

	x := archsimd.LoadInt64x4Slice([]int64{10, 20, 30, 40})
	y := archsimd.LoadInt64x4Slice([]int64{1, 2, 3, 4})

	x.Add(y).StoreSlice(r)
	archsimd.ClearAVXUpperBits()
	x.Sub(y).StoreSlice(s)

	checkSlices[int64](t, r, []int64{11, 22, 33, 44})
	checkSlices[int64](t, s, []int64{9, 18, 27, 36})
}

func TestLeadingZeros(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	src := []uint64{0b1111, 0}
	want := []uint64{60, 64}
	got := make([]uint64, 2)
	archsimd.LoadUint64x2Slice(src).LeadingZeros().StoreSlice(got)
	for i := range 2 {
		if want[i] != got[i] {
			t.Errorf("Result incorrect at %d: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestIsZero(t *testing.T) {
	v1 := archsimd.LoadUint64x2Slice([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2Slice([]uint64{0, 0})
	if v1.IsZero() {
		t.Errorf("Result incorrect, want false, got true")
	}
	if !v2.IsZero() {
		t.Errorf("Result incorrect, want true, got false")
	}
	if !v1.And(v2).IsZero() {
		t.Errorf("Result incorrect, want true, got false")
	}
	if v1.AndNot(v2).IsZero() {
		t.Errorf("Result incorrect, want false, got true")
	}
	if !v2.And(v1).IsZero() {
		t.Errorf("Result incorrect, want true, got false")
	}
	if !v2.AndNot(v1).IsZero() {
		t.Errorf("Result incorrect, want true, got false")
	}
}

func TestSelect4FromPairConst(t *testing.T) {
	x := archsimd.LoadInt32x4Slice([]int32{0, 1, 2, 3})
	y := archsimd.LoadInt32x4Slice([]int32{4, 5, 6, 7})

	llll := x.SelectFromPair(0, 1, 2, 3, y)
	hhhh := x.SelectFromPair(4, 5, 6, 7, y)
	llhh := x.SelectFromPair(0, 1, 6, 7, y)
	hhll := x.SelectFromPair(6, 7, 0, 1, y)

	lllh := x.SelectFromPair(0, 1, 2, 7, y)
	llhl := x.SelectFromPair(0, 1, 7, 2, y)
	lhll := x.SelectFromPair(0, 7, 1, 2, y)
	hlll := x.SelectFromPair(7, 0, 1, 2, y)

	hhhl := x.SelectFromPair(4, 5, 6, 0, y)
	hhlh := x.SelectFromPair(4, 5, 0, 6, y)
	hlhh := x.SelectFromPair(4, 0, 5, 6, y)
	lhhh := x.SelectFromPair(0, 4, 5, 6, y)

	lhlh := x.SelectFromPair(0, 4, 1, 5, y)
	hlhl := x.SelectFromPair(4, 0, 5, 1, y)
	lhhl := x.SelectFromPair(0, 4, 5, 1, y)
	hllh := x.SelectFromPair(4, 0, 1, 5, y)

	r := make([]int32, 4, 4)

	foo := func(v archsimd.Int32x4, a, b, c, d int32) {
		v.StoreSlice(r)
		checkSlices[int32](t, r, []int32{a, b, c, d})
	}

	foo(llll, 0, 1, 2, 3)
	foo(hhhh, 4, 5, 6, 7)
	foo(llhh, 0, 1, 6, 7)
	foo(hhll, 6, 7, 0, 1)

	foo(lllh, 0, 1, 2, 7)
	foo(llhl, 0, 1, 7, 2)
	foo(lhll, 0, 7, 1, 2)
	foo(hlll, 7, 0, 1, 2)

	foo(hhhl, 4, 5, 6, 0)
	foo(hhlh, 4, 5, 0, 6)
	foo(hlhh, 4, 0, 5, 6)
	foo(lhhh, 0, 4, 5, 6)

	foo(lhlh, 0, 4, 1, 5)
	foo(hlhl, 4, 0, 5, 1)
	foo(lhhl, 0, 4, 5, 1)
	foo(hllh, 4, 0, 1, 5)
}

//go:noinline
func selectFromPairInt32x4(x archsimd.Int32x4, a, b, c, d uint8, y archsimd.Int32x4) archsimd.Int32x4 {
	return x.SelectFromPair(a, b, c, d, y)
}

func TestSelect4FromPairVar(t *testing.T) {
	x := archsimd.LoadInt32x4Slice([]int32{0, 1, 2, 3})
	y := archsimd.LoadInt32x4Slice([]int32{4, 5, 6, 7})

	llll := selectFromPairInt32x4(x, 0, 1, 2, 3, y)
	hhhh := selectFromPairInt32x4(x, 4, 5, 6, 7, y)
	llhh := selectFromPairInt32x4(x, 0, 1, 6, 7, y)
	hhll := selectFromPairInt32x4(x, 6, 7, 0, 1, y)

	lllh := selectFromPairInt32x4(x, 0, 1, 2, 7, y)
	llhl := selectFromPairInt32x4(x, 0, 1, 7, 2, y)
	lhll := selectFromPairInt32x4(x, 0, 7, 1, 2, y)
	hlll := selectFromPairInt32x4(x, 7, 0, 1, 2, y)

	hhhl := selectFromPairInt32x4(x, 4, 5, 6, 0, y)
	hhlh := selectFromPairInt32x4(x, 4, 5, 0, 6, y)
	hlhh := selectFromPairInt32x4(x, 4, 0, 5, 6, y)
	lhhh := selectFromPairInt32x4(x, 0, 4, 5, 6, y)

	lhlh := selectFromPairInt32x4(x, 0, 4, 1, 5, y)
	hlhl := selectFromPairInt32x4(x, 4, 0, 5, 1, y)
	lhhl := selectFromPairInt32x4(x, 0, 4, 5, 1, y)
	hllh := selectFromPairInt32x4(x, 4, 0, 1, 5, y)

	r := make([]int32, 4, 4)

	foo := func(v archsimd.Int32x4, a, b, c, d int32) {
		v.StoreSlice(r)
		checkSlices[int32](t, r, []int32{a, b, c, d})
	}

	foo(llll, 0, 1, 2, 3)
	foo(hhhh, 4, 5, 6, 7)
	foo(llhh, 0, 1, 6, 7)
	foo(hhll, 6, 7, 0, 1)

	foo(lllh, 0, 1, 2, 7)
	foo(llhl, 0, 1, 7, 2)
	foo(lhll, 0, 7, 1, 2)
	foo(hlll, 7, 0, 1, 2)

	foo(hhhl, 4, 5, 6, 0)
	foo(hhlh, 4, 5, 0, 6)
	foo(hlhh, 4, 0, 5, 6)
	foo(lhhh, 0, 4, 5, 6)

	foo(lhlh, 0, 4, 1, 5)
	foo(hlhl, 4, 0, 5, 1)
	foo(lhhl, 0, 4, 5, 1)
	foo(hllh, 4, 0, 1, 5)
}

func TestSelect4FromPairConstGrouped(t *testing.T) {
	x := archsimd.LoadFloat32x8Slice([]float32{0, 1, 2, 3, 10, 11, 12, 13})
	y := archsimd.LoadFloat32x8Slice([]float32{4, 5, 6, 7, 14, 15, 16, 17})

	llll := x.SelectFromPairGrouped(0, 1, 2, 3, y)
	hhhh := x.SelectFromPairGrouped(4, 5, 6, 7, y)
	llhh := x.SelectFromPairGrouped(0, 1, 6, 7, y)
	hhll := x.SelectFromPairGrouped(6, 7, 0, 1, y)

	lllh := x.SelectFromPairGrouped(0, 1, 2, 7, y)
	llhl := x.SelectFromPairGrouped(0, 1, 7, 2, y)
	lhll := x.SelectFromPairGrouped(0, 7, 1, 2, y)
	hlll := x.SelectFromPairGrouped(7, 0, 1, 2, y)

	hhhl := x.SelectFromPairGrouped(4, 5, 6, 0, y)
	hhlh := x.SelectFromPairGrouped(4, 5, 0, 6, y)
	hlhh := x.SelectFromPairGrouped(4, 0, 5, 6, y)
	lhhh := x.SelectFromPairGrouped(0, 4, 5, 6, y)

	lhlh := x.SelectFromPairGrouped(0, 4, 1, 5, y)
	hlhl := x.SelectFromPairGrouped(4, 0, 5, 1, y)
	lhhl := x.SelectFromPairGrouped(0, 4, 5, 1, y)
	hllh := x.SelectFromPairGrouped(4, 0, 1, 5, y)

	r := make([]float32, 8, 8)

	foo := func(v archsimd.Float32x8, a, b, c, d float32) {
		v.StoreSlice(r)
		checkSlices[float32](t, r, []float32{a, b, c, d, 10 + a, 10 + b, 10 + c, 10 + d})
	}

	foo(llll, 0, 1, 2, 3)
	foo(hhhh, 4, 5, 6, 7)
	foo(llhh, 0, 1, 6, 7)
	foo(hhll, 6, 7, 0, 1)

	foo(lllh, 0, 1, 2, 7)
	foo(llhl, 0, 1, 7, 2)
	foo(lhll, 0, 7, 1, 2)
	foo(hlll, 7, 0, 1, 2)

	foo(hhhl, 4, 5, 6, 0)
	foo(hhlh, 4, 5, 0, 6)
	foo(hlhh, 4, 0, 5, 6)
	foo(lhhh, 0, 4, 5, 6)

	foo(lhlh, 0, 4, 1, 5)
	foo(hlhl, 4, 0, 5, 1)
	foo(lhhl, 0, 4, 5, 1)
	foo(hllh, 4, 0, 1, 5)
}

func TestSelectFromPairConstGroupedUint32x16(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := archsimd.LoadUint32x16Slice([]uint32{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33})
	y := archsimd.LoadUint32x16Slice([]uint32{4, 5, 6, 7, 14, 15, 16, 17, 24, 25, 26, 27, 34, 35, 36, 37})

	llll := x.SelectFromPairGrouped(0, 1, 2, 3, y)
	hhhh := x.SelectFromPairGrouped(4, 5, 6, 7, y)
	llhh := x.SelectFromPairGrouped(0, 1, 6, 7, y)
	hhll := x.SelectFromPairGrouped(6, 7, 0, 1, y)

	lllh := x.SelectFromPairGrouped(0, 1, 2, 7, y)
	llhl := x.SelectFromPairGrouped(0, 1, 7, 2, y)
	lhll := x.SelectFromPairGrouped(0, 7, 1, 2, y)
	hlll := x.SelectFromPairGrouped(7, 0, 1, 2, y)

	hhhl := x.SelectFromPairGrouped(4, 5, 6, 0, y)
	hhlh := x.SelectFromPairGrouped(4, 5, 0, 6, y)
	hlhh := x.SelectFromPairGrouped(4, 0, 5, 6, y)
	lhhh := x.SelectFromPairGrouped(0, 4, 5, 6, y)

	lhlh := x.SelectFromPairGrouped(0, 4, 1, 5, y)
	hlhl := x.SelectFromPairGrouped(4, 0, 5, 1, y)
	lhhl := x.SelectFromPairGrouped(0, 4, 5, 1, y)
	hllh := x.SelectFromPairGrouped(4, 0, 1, 5, y)

	r := make([]uint32, 16, 16)

	foo := func(v archsimd.Uint32x16, a, b, c, d uint32) {
		v.StoreSlice(r)
		checkSlices[uint32](t, r, []uint32{a, b, c, d,
			10 + a, 10 + b, 10 + c, 10 + d,
			20 + a, 20 + b, 20 + c, 20 + d,
			30 + a, 30 + b, 30 + c, 30 + d,
		})
	}

	foo(llll, 0, 1, 2, 3)
	foo(hhhh, 4, 5, 6, 7)
	foo(llhh, 0, 1, 6, 7)
	foo(hhll, 6, 7, 0, 1)

	foo(lllh, 0, 1, 2, 7)
	foo(llhl, 0, 1, 7, 2)
	foo(lhll, 0, 7, 1, 2)
	foo(hlll, 7, 0, 1, 2)

	foo(hhhl, 4, 5, 6, 0)
	foo(hhlh, 4, 5, 0, 6)
	foo(hlhh, 4, 0, 5, 6)
	foo(lhhh, 0, 4, 5, 6)

	foo(lhlh, 0, 4, 1, 5)
	foo(hlhl, 4, 0, 5, 1)
	foo(lhhl, 0, 4, 5, 1)
	foo(hllh, 4, 0, 1, 5)
}

func TestSelect128FromPair(t *testing.T) {
	x := archsimd.LoadUint64x4Slice([]uint64{0, 1, 2, 3})
	y := archsimd.LoadUint64x4Slice([]uint64{4, 5, 6, 7})

	aa := x.Select128FromPair(0, 0, y)
	ab := x.Select128FromPair(0, 1, y)
	bc := x.Select128FromPair(1, 2, y)
	cd := x.Select128FromPair(2, 3, y)
	da := x.Select128FromPair(3, 0, y)
	dc := x.Select128FromPair(3, 2, y)

	r := make([]uint64, 4, 4)

	foo := func(v archsimd.Uint64x4, a, b uint64) {
		a, b = 2*a, 2*b
		v.StoreSlice(r)
		checkSlices[uint64](t, r, []uint64{a, a + 1, b, b + 1})
	}

	foo(aa, 0, 0)
	foo(ab, 0, 1)
	foo(bc, 1, 2)
	foo(cd, 2, 3)
	foo(da, 3, 0)
	foo(dc, 3, 2)
}

func TestSelect128FromPairError(t *testing.T) {
	x := archsimd.LoadUint64x4Slice([]uint64{0, 1, 2, 3})
	y := archsimd.LoadUint64x4Slice([]uint64{4, 5, 6, 7})

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic %v", r)
		}
	}()
	_ = x.Select128FromPair(0, 4, y)

	t.Errorf("Should have panicked")
}

//go:noinline
func select128FromPair(x archsimd.Uint64x4, lo, hi uint8, y archsimd.Uint64x4) archsimd.Uint64x4 {
	return x.Select128FromPair(lo, hi, y)
}

func TestSelect128FromPairVar(t *testing.T) {
	x := archsimd.LoadUint64x4Slice([]uint64{0, 1, 2, 3})
	y := archsimd.LoadUint64x4Slice([]uint64{4, 5, 6, 7})

	aa := select128FromPair(x, 0, 0, y)
	ab := select128FromPair(x, 0, 1, y)
	bc := select128FromPair(x, 1, 2, y)
	cd := select128FromPair(x, 2, 3, y)
	da := select128FromPair(x, 3, 0, y)
	dc := select128FromPair(x, 3, 2, y)

	r := make([]uint64, 4, 4)

	foo := func(v archsimd.Uint64x4, a, b uint64) {
		a, b = 2*a, 2*b
		v.StoreSlice(r)
		checkSlices[uint64](t, r, []uint64{a, a + 1, b, b + 1})
	}

	foo(aa, 0, 0)
	foo(ab, 0, 1)
	foo(bc, 1, 2)
	foo(cd, 2, 3)
	foo(da, 3, 0)
	foo(dc, 3, 2)
}

func TestSelect2FromPairConst(t *testing.T) {
	x := archsimd.LoadUint64x2Slice([]uint64{0, 1})
	y := archsimd.LoadUint64x2Slice([]uint64{2, 3})

	ll := x.SelectFromPair(0, 1, y)
	hh := x.SelectFromPair(3, 2, y)
	lh := x.SelectFromPair(0, 3, y)
	hl := x.SelectFromPair(2, 1, y)

	r := make([]uint64, 2, 2)

	foo := func(v archsimd.Uint64x2, a, b uint64) {
		v.StoreSlice(r)
		checkSlices[uint64](t, r, []uint64{a, b})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedUint(t *testing.T) {
	x := archsimd.LoadUint64x4Slice([]uint64{0, 1, 10, 11})
	y := archsimd.LoadUint64x4Slice([]uint64{2, 3, 12, 13})

	ll := x.SelectFromPairGrouped(0, 1, y)
	hh := x.SelectFromPairGrouped(3, 2, y)
	lh := x.SelectFromPairGrouped(0, 3, y)
	hl := x.SelectFromPairGrouped(2, 1, y)

	r := make([]uint64, 4, 4)

	foo := func(v archsimd.Uint64x4, a, b uint64) {
		v.StoreSlice(r)
		checkSlices[uint64](t, r, []uint64{a, b, a + 10, b + 10})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedFloat(t *testing.T) {
	x := archsimd.LoadFloat64x4Slice([]float64{0, 1, 10, 11})
	y := archsimd.LoadFloat64x4Slice([]float64{2, 3, 12, 13})

	ll := x.SelectFromPairGrouped(0, 1, y)
	hh := x.SelectFromPairGrouped(3, 2, y)
	lh := x.SelectFromPairGrouped(0, 3, y)
	hl := x.SelectFromPairGrouped(2, 1, y)

	r := make([]float64, 4, 4)

	foo := func(v archsimd.Float64x4, a, b float64) {
		v.StoreSlice(r)
		checkSlices[float64](t, r, []float64{a, b, a + 10, b + 10})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedInt(t *testing.T) {
	x := archsimd.LoadInt64x4Slice([]int64{0, 1, 10, 11})
	y := archsimd.LoadInt64x4Slice([]int64{2, 3, 12, 13})

	ll := x.SelectFromPairGrouped(0, 1, y)
	hh := x.SelectFromPairGrouped(3, 2, y)
	lh := x.SelectFromPairGrouped(0, 3, y)
	hl := x.SelectFromPairGrouped(2, 1, y)

	r := make([]int64, 4, 4)

	foo := func(v archsimd.Int64x4, a, b int64) {
		v.StoreSlice(r)
		checkSlices[int64](t, r, []int64{a, b, a + 10, b + 10})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedInt512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	x := archsimd.LoadInt64x8Slice([]int64{0, 1, 10, 11, 20, 21, 30, 31})
	y := archsimd.LoadInt64x8Slice([]int64{2, 3, 12, 13, 22, 23, 32, 33})

	ll := x.SelectFromPairGrouped(0, 1, y)
	hh := x.SelectFromPairGrouped(3, 2, y)
	lh := x.SelectFromPairGrouped(0, 3, y)
	hl := x.SelectFromPairGrouped(2, 1, y)

	r := make([]int64, 8, 8)

	foo := func(v archsimd.Int64x8, a, b int64) {
		v.StoreSlice(r)
		checkSlices[int64](t, r, []int64{a, b, a + 10, b + 10, a + 20, b + 20, a + 30, b + 30})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestString(t *testing.T) {
	x := archsimd.LoadUint32x4Slice([]uint32{0, 1, 2, 3})
	y := archsimd.LoadInt64x4Slice([]int64{-4, -5, -6, -7})
	z := archsimd.LoadFloat32x4Slice([]float32{0.5, 1.5, -2.5, 3.5e9})
	w := archsimd.LoadFloat64x4Slice([]float64{0.5, 1.5, -2.5, 3.5e9})

	sx := "{0,1,2,3}"
	sy := "{-4,-5,-6,-7}"
	sz := "{0.5,1.5,-2.5,3.5e+09}"
	sw := sz

	if x.String() != sx {
		t.Errorf("x=%s wanted %s", x, sx)
	}
	if y.String() != sy {
		t.Errorf("y=%s wanted %s", y, sy)
	}
	if z.String() != sz {
		t.Errorf("z=%s wanted %s", z, sz)
	}
	if w.String() != sw {
		t.Errorf("w=%s wanted %s", w, sw)
	}
	t.Logf("w=%s", w)
	t.Logf("x=%s", x)
	t.Logf("y=%s", y)
	t.Logf("z=%s", z)
}

// a returns an slice of 16 int32
func a() []int32 {
	return make([]int32, 16, 16)
}

// applyTo3 returns a 16-element slice of the results of
// applying f to the respective elements of vectors x, y, and z.
func applyTo3(x, y, z archsimd.Int32x16, f func(x, y, z int32) int32) []int32 {
	ax, ay, az := a(), a(), a()
	x.StoreSlice(ax)
	y.StoreSlice(ay)
	z.StoreSlice(az)

	r := a()
	for i := range r {
		r[i] = f(ax[i], ay[i], az[i])
	}
	return r
}

// applyTo3 returns a 16-element slice of the results of
// applying f to the respective elements of vectors x, y, z, and w.
func applyTo4(x, y, z, w archsimd.Int32x16, f func(x, y, z, w int32) int32) []int32 {
	ax, ay, az, aw := a(), a(), a(), a()
	x.StoreSlice(ax)
	y.StoreSlice(ay)
	z.StoreSlice(az)
	w.StoreSlice(aw)

	r := make([]int32, len(ax), len(ax))
	for i := range r {
		r[i] = f(ax[i], ay[i], az[i], aw[i])
	}
	return r
}

func TestSelectTernOptInt32x16(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	ax := []int32{0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}
	ay := []int32{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}
	az := []int32{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
	aw := []int32{0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}
	am := []int32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}

	x := archsimd.LoadInt32x16Slice(ax)
	y := archsimd.LoadInt32x16Slice(ay)
	z := archsimd.LoadInt32x16Slice(az)
	w := archsimd.LoadInt32x16Slice(aw)
	m := archsimd.LoadInt32x16Slice(am)

	foo := func(v archsimd.Int32x16, s []int32) {
		r := make([]int32, 16, 16)
		v.StoreSlice(r)
		checkSlices[int32](t, r, s)
	}

	t0 := w.Xor(y).Xor(z)
	ft0 := func(w, y, z int32) int32 {
		return w ^ y ^ z
	}
	foo(t0, applyTo3(w, y, z, ft0))

	t1 := m.And(w.Xor(y).Xor(z.Not()))
	ft1 := func(m, w, y, z int32) int32 {
		return m & (w ^ y ^ ^z)
	}
	foo(t1, applyTo4(m, w, y, z, ft1))

	t2 := x.Xor(y).Xor(z).And(x.Xor(y).Xor(z.Not()))
	ft2 := func(x, y, z int32) int32 {
		return (x ^ y ^ z) & (x ^ y ^ ^z)
	}
	foo(t2, applyTo3(x, y, z, ft2))
}

func TestMaskedMerge(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	x := archsimd.LoadInt64x4Slice([]int64{1, 2, 3, 4})
	y := archsimd.LoadInt64x4Slice([]int64{5, 6, 1, 1})
	z := archsimd.LoadInt64x4Slice([]int64{-1, -2, -3, -4})
	res := make([]int64, 4)
	expected := []int64{6, 8, -3, -4}
	mask := x.Less(y)
	if archsimd.X86.AVX512() {
		x.Add(y).Merge(z, mask).StoreSlice(res)
	} else {
		x.Add(y).Merge(z, mask).StoreSlice(res)
	}
	for i := range 4 {
		if res[i] != expected[i] {
			t.Errorf("got %d wanted %d", res[i], expected[i])
		}
	}
}

func TestPermuteScalars(t *testing.T) {
	x := []int32{11, 12, 13, 14}
	want := []int32{12, 13, 14, 11}
	got := make([]int32, 4)
	archsimd.LoadInt32x4Slice(x).PermuteScalars(1, 2, 3, 0).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsGrouped(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	x := []int32{11, 12, 13, 14, 21, 22, 23, 24}
	want := []int32{12, 13, 14, 11, 22, 23, 24, 21}
	got := make([]int32, 8)
	archsimd.LoadInt32x8Slice(x).PermuteScalarsGrouped(1, 2, 3, 0).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsHi(t *testing.T) {
	x := []int16{-1, -2, -3, -4, 11, 12, 13, 14}
	want := []int16{-1, -2, -3, -4, 12, 13, 14, 11}
	got := make([]int16, len(x))
	archsimd.LoadInt16x8Slice(x).PermuteScalarsHi(1, 2, 3, 0).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsLo(t *testing.T) {
	x := []int16{11, 12, 13, 14, 4, 5, 6, 7}
	want := []int16{12, 13, 14, 11, 4, 5, 6, 7}
	got := make([]int16, len(x))
	archsimd.LoadInt16x8Slice(x).PermuteScalarsLo(1, 2, 3, 0).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsHiGrouped(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	x := []int16{-1, -2, -3, -4, 11, 12, 13, 14, -11, -12, -13, -14, 111, 112, 113, 114}
	want := []int16{-1, -2, -3, -4, 12, 13, 14, 11, -11, -12, -13, -14, 112, 113, 114, 111}
	got := make([]int16, len(x))
	archsimd.LoadInt16x16Slice(x).PermuteScalarsHiGrouped(1, 2, 3, 0).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsLoGrouped(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	x := []int16{11, 12, 13, 14, 4, 5, 6, 7, 111, 112, 113, 114, 14, 15, 16, 17}
	want := []int16{12, 13, 14, 11, 4, 5, 6, 7, 112, 113, 114, 111, 14, 15, 16, 17}
	got := make([]int16, len(x))
	archsimd.LoadInt16x16Slice(x).PermuteScalarsLoGrouped(1, 2, 3, 0).StoreSlice(got)
	checkSlices(t, got, want)
}

func TestClMul(t *testing.T) {
	var x = archsimd.LoadUint64x2Slice([]uint64{1, 5})
	var y = archsimd.LoadUint64x2Slice([]uint64{3, 9})

	foo := func(v archsimd.Uint64x2, s []uint64) {
		r := make([]uint64, 2, 2)
		v.StoreSlice(r)
		checkSlices[uint64](t, r, s)
	}

	foo(x.CarrylessMultiply(0, 0, y), []uint64{3, 0})
	foo(x.CarrylessMultiply(0, 1, y), []uint64{9, 0})
	foo(x.CarrylessMultiply(1, 0, y), []uint64{15, 0})
	foo(x.CarrylessMultiply(1, 1, y), []uint64{45, 0})
	foo(y.CarrylessMultiply(0, 0, y), []uint64{5, 0})

}

func addPairsSlice[T number](a, b []T) []T {
	r := make([]T, len(a))
	for i := range len(a) / 2 {
		r[i] = a[2*i] + a[2*i+1]
		r[i+len(a)/2] = b[2*i] + b[2*i+1]
	}
	return r
}

func subPairsSlice[T number](a, b []T) []T {
	r := make([]T, len(a))
	for i := range len(a) / 2 {
		r[i] = a[2*i] - a[2*i+1]
		r[i+len(a)/2] = b[2*i] - b[2*i+1]
	}
	return r
}

func addPairsGroupedSlice[T number](a, b []T) []T {
	group := int(128 / unsafe.Sizeof(a[0]))
	r := make([]T, 0, len(a))
	for i := range len(a) / group {
		r = append(r, addPairsSlice(a[i*group:(i+1)*group], b[i*group:(i+1)*group])...)
	}
	return r
}

func subPairsGroupedSlice[T number](a, b []T) []T {
	group := int(128 / unsafe.Sizeof(a[0]))
	r := make([]T, 0, len(a))
	for i := range len(a) / group {
		r = append(r, subPairsSlice(a[i*group:(i+1)*group], b[i*group:(i+1)*group])...)
	}
	return r
}

func TestAddSubPairs(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.AddPairs, addPairsSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.SubPairs, subPairsSlice[int16])
	testUint16x8Binary(t, archsimd.Uint16x8.AddPairs, addPairsSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.SubPairs, subPairsSlice[uint16])
	testInt32x4Binary(t, archsimd.Int32x4.AddPairs, addPairsSlice[int32])
	testInt32x4Binary(t, archsimd.Int32x4.SubPairs, subPairsSlice[int32])
	testUint32x4Binary(t, archsimd.Uint32x4.AddPairs, addPairsSlice[uint32])
	testUint32x4Binary(t, archsimd.Uint32x4.SubPairs, subPairsSlice[uint32])
	testFloat32x4Binary(t, archsimd.Float32x4.AddPairs, addPairsSlice[float32])
	testFloat32x4Binary(t, archsimd.Float32x4.SubPairs, subPairsSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.AddPairs, addPairsSlice[float64])
	testFloat64x2Binary(t, archsimd.Float64x2.SubPairs, subPairsSlice[float64])

	// Grouped versions
	if archsimd.X86.AVX2() {
		testInt16x16Binary(t, archsimd.Int16x16.AddPairsGrouped, addPairsGroupedSlice[int16])
		testInt16x16Binary(t, archsimd.Int16x16.SubPairsGrouped, subPairsGroupedSlice[int16])
		testUint16x16Binary(t, archsimd.Uint16x16.AddPairsGrouped, addPairsGroupedSlice[uint16])
		testUint16x16Binary(t, archsimd.Uint16x16.SubPairsGrouped, subPairsGroupedSlice[uint16])
		testInt32x8Binary(t, archsimd.Int32x8.AddPairsGrouped, addPairsGroupedSlice[int32])
		testInt32x8Binary(t, archsimd.Int32x8.SubPairsGrouped, subPairsGroupedSlice[int32])
		testUint32x8Binary(t, archsimd.Uint32x8.AddPairsGrouped, addPairsGroupedSlice[uint32])
		testUint32x8Binary(t, archsimd.Uint32x8.SubPairsGrouped, subPairsGroupedSlice[uint32])
		testFloat32x8Binary(t, archsimd.Float32x8.AddPairsGrouped, addPairsGroupedSlice[float32])
		testFloat32x8Binary(t, archsimd.Float32x8.SubPairsGrouped, subPairsGroupedSlice[float32])
		testFloat64x4Binary(t, archsimd.Float64x4.AddPairsGrouped, addPairsGroupedSlice[float64])
		testFloat64x4Binary(t, archsimd.Float64x4.SubPairsGrouped, subPairsGroupedSlice[float64])
	}
}
