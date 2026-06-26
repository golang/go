// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"os"
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

func TestPermute(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	indices := []uint64{7, 6, 5, 4, 3, 2, 1, 0}
	want := []int64{8, 7, 6, 5, 4, 3, 2, 1}
	got := make([]int64, 8)
	archsimd.LoadInt64x8(x).Permute(archsimd.LoadUint64x8(indices)).Store(got)
	checkSlices(t, got, want)
}

func TestPermuteOrZero(t *testing.T) {
	x := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	indices := []int8{7, 6, 5, 4, 3, 2, 1, 0, -1, 8, -1, 9, -1, 10, -1, 11}
	want := []uint8{8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 0, 10, 0, 11, 0, 12}
	got := make([]uint8, len(x))
	archsimd.LoadUint8x16(x).PermuteOrZero(archsimd.LoadInt8x16(indices)).Store(got)
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
	archsimd.LoadInt64x8(x).ConcatPermute(archsimd.LoadInt64x8(y), archsimd.LoadUint64x8(indices)).Store(got)
	checkSlices(t, got, want)
}

func TestCompress(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	v1234 := archsimd.LoadInt32x4([]int32{1, 2, 3, 4})
	v2400 := v1234.Compress(archsimd.Mask32x4FromBits(0b1010))
	got := make([]int32, 4)
	v2400.Store(got)
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
	v3400 := archsimd.LoadInt32x4([]int32{3, 4, 0, 0})
	v2400 := v3400.Expand(archsimd.Mask32x4FromBits(0b1010))
	got := make([]int32, 4)
	v2400.Store(got)
	want := []int32{0, 3, 0, 4}
	if !slices.Equal(got, want) {
		t.Errorf("want and got differ, want=%v, got=%v", want, got)
	}
}

func TestSlicesInt8(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x32(a)
	b := make([]int8, 32, 32)
	v.Store(b)
	checkSlices(t, a, b)
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
	v := archsimd.LoadInt8x32(a)
	b := make([]int8, 32, 32)
	v.Store(b)
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
	v := archsimd.LoadInt8x32(a)
	b := make([]int8, 31) // TOO SHORT, should panic
	v.Store(b)
	checkSlices(t, a, b)
}

func TestSlicesFloat64(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8} // too long, should be fine
	v := archsimd.LoadFloat64x4(a)
	b := make([]float64, 4, 4)
	v.Store(b)
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
	v := archsimd.LoadInt64x4(s0)
	m := archsimd.LoadInt64x4(s1)
	forceSpill()
	got := make([]int64, 4)
	gotv := op(v, m)
	gotv.Store(got)
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
	archsimd.LoadInt64x2([]int64{1, 2}).Add(archsimd.LoadInt64x2([]int64{3, 4})).Masked(m).StoreArray(&results)
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
	archsimd.LoadInt64x2([]int64{1, 2}).Add(archsimd.LoadInt64x2([]int64{3, 4})).Masked(m).StoreArray(&results)
	for i := range 2 {
		if results[i] != want[i] {
			t.Errorf("Result at %d incorrect: want %v, got %v", i, want[i], results[i])
		}
	}
}

func TestBitMaskToBits(t *testing.T) {
	int8s := []int8{
		0, 1, 1, 0, 0, 1, 0, 1,
		1, 0, 1, 1, 0, 0, 1, 0,
		1, 0, 0, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 1, 0, 1,
		1, 0, 0, 1, 0, 1, 1, 0,
		0, 1, 0, 1, 1, 0, 0, 1,
		1, 0, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 1, 0, 0, 1,
	}
	int16s := make([]int16, 32)
	for i := range int16s {
		int16s[i] = int16(int8s[i])
	}
	int32s := make([]int32, 16)
	for i := range int32s {
		int32s[i] = int32(int8s[i])
	}
	int64s := make([]int64, 8)
	for i := range int64s {
		int64s[i] = int64(int8s[i])
	}
	want64 := uint64(0)
	for i := range int8s {
		want64 |= uint64(int8s[i]) << i
	}
	want32 := uint32(want64)
	want16 := uint16(want64)
	want8 := uint8(want64)
	want4 := want8 & 0b1111
	want2 := want4 & 0b11

	if v := archsimd.LoadInt8x16(int8s[:16]).ToMask().ToBits(); v != want16 {
		t.Errorf("want %b, got %b", want16, v)
	}
	if v := archsimd.LoadInt32x4(int32s[:4]).ToMask().ToBits(); v != want4 {
		t.Errorf("want %b, got %b", want4, v)
	}
	if v := archsimd.LoadInt32x8(int32s[:8]).ToMask().ToBits(); v != want8 {
		t.Errorf("want %b, got %b", want8, v)
	}
	if v := archsimd.LoadInt64x2(int64s[:2]).ToMask().ToBits(); v != want2 {
		t.Errorf("want %b, got %b", want2, v)
	}
	if v := archsimd.LoadInt64x4(int64s[:4]).ToMask().ToBits(); v != want4 {
		t.Errorf("want %b, got %b", want4, v)
	}

	if archsimd.X86.AVX2() {
		if v := archsimd.LoadInt8x32(int8s[:32]).ToMask().ToBits(); v != want32 {
			t.Errorf("want %b, got %b", want32, v)
		}
	}

	if archsimd.X86.AVX512() {
		if v := archsimd.LoadInt8x64(int8s).ToMask().ToBits(); v != want64 {
			t.Errorf("want %b, got %b", want64, v)
		}
		if v := archsimd.LoadInt16x8(int16s[:8]).ToMask().ToBits(); v != want8 {
			t.Errorf("want %b, got %b", want8, v)
		}
		if v := archsimd.LoadInt16x16(int16s[:16]).ToMask().ToBits(); v != want16 {
			t.Errorf("want %b, got %b", want16, v)
		}
		if v := archsimd.LoadInt16x32(int16s).ToMask().ToBits(); v != want32 {
			t.Errorf("want %b, got %b", want32, v)
		}
		if v := archsimd.LoadInt32x16(int32s).ToMask().ToBits(); v != want16 {
			t.Errorf("want %b, got %b", want16, v)
		}
		if v := archsimd.LoadInt64x8(int64s).ToMask().ToBits(); v != want8 {
			t.Errorf("want %b, got %b", want8, v)
		}
	}
}

var maskForTestBitMaskFromBitsStore uint8

func TestBitMaskToBitsStore(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	maskForTestBitMaskFromBitsStore = archsimd.LoadInt16x8([]int16{1, 0, 1, 0, 0, 0, 0, 0}).ToMask().ToBits()
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

	a := archsimd.LoadFloat64x4([]float64{1, 2, 3, 4})
	b := archsimd.LoadFloat64x4([]float64{4, 2, 3, 1})
	g := a.Greater(b)
	g.ToInt64x4().Store(k)
	c := a.Merge(b, g)

	c.Store(s)

	checkSlices[int64](t, k, []int64{0, 0, 0, -1})
	checkSlices[float64](t, s, []float64{4, 2, 3, 4})
}

func TestIfElseFloat(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	k := make([]int64, 4, 4)
	s := make([]float64, 4, 4)

	a := archsimd.LoadFloat64x4([]float64{1, 2, 3, 4})
	b := archsimd.LoadFloat64x4([]float64{4, 2, 3, 1})
	g := a.Greater(b)
	g.ToInt64x4().Store(k)
	c := a.IfElse(g, b)

	c.Store(s)

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

	a := archsimd.LoadFloat64x8([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	b := archsimd.LoadFloat64x8([]float64{8, 7, 6, 5, 4, 2, 3, 1})
	g := a.Greater(b)
	g.ToInt64x8().Store(k)
	c := a.Merge(b, g)
	d := a.Masked(g)

	checkSlices[int64](t, k, []int64{0, 0, 0, 0, -1, -1, -1, -1})

	c.Store(s)
	checkSlices[float64](t, s, []float64{8, 7, 6, 5, 5, 6, 7, 8})

	d.Store(s)
	checkSlices[float64](t, s, []float64{0, 0, 0, 0, 5, 6, 7, 8})
}

func TestIfElseFloat512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	k := make([]int64, 8, 8)
	s := make([]float64, 8, 8)

	a := archsimd.LoadFloat64x8([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	b := archsimd.LoadFloat64x8([]float64{8, 7, 6, 5, 4, 2, 3, 1})
	g := a.Greater(b)
	g.ToInt64x8().Store(k)
	c := a.IfElse(g, b)
	d := a.Masked(g)

	checkSlices[int64](t, k, []int64{0, 0, 0, 0, -1, -1, -1, -1})

	c.Store(s)
	checkSlices[float64](t, s, []float64{8, 7, 6, 5, 5, 6, 7, 8})

	d.Store(s)
	checkSlices[float64](t, s, []float64{0, 0, 0, 0, 5, 6, 7, 8})
}

var ro uint64 = 2
var roBig uint64 = 1024 + 2

func TestRotateAllVariable(t *testing.T) {
	got := make([]int32, 4)
	archsimd.LoadInt32x4([]int32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(ro).Store(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("Want 0b1100, got %b", v)
		}
	}
	archsimd.LoadInt32x4([]int32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(roBig).Store(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("Want 0b1100, got %b", v)
		}
	}
}

func TestRotateAllConst(t *testing.T) {
	got := make([]int32, 4)
	archsimd.LoadInt32x4([]int32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(2).Store(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("Want 0b1100, got %b", v)
		}
	}
}

func TestBroadcastFloat32x8(t *testing.T) {
	s := make([]float32, 8, 8)
	archsimd.BroadcastFloat32x8(123456789).Store(s)
	checkSlices(t, s, []float32{123456789, 123456789, 123456789, 123456789, 123456789, 123456789, 123456789, 123456789})
}

func TestBroadcastInt8x32(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	s := make([]int8, 32, 32)
	archsimd.BroadcastInt8x32(-123).Store(s)
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

	a := archsimd.LoadFloat64x8([]float64{2, 0, 2, 0, 2, 0, 2, 0})
	b := archsimd.LoadFloat64x8([]float64{1, 1, 1, 1, 1, 1, 1, 1})
	c := archsimd.LoadFloat64x8([]float64{1, 2, 3, 4, 5, 6, 7, 8})
	d := archsimd.LoadFloat64x8([]float64{2, 4, 6, 8, 10, 12, 14, 16})
	g := a.Greater(b)
	e := c.Add(d).Masked(g)
	e.Store(s)
	g.ToInt64x8().Store(k)
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

	x := archsimd.LoadInt32x4([]int32{0xA, 0xB, 0xC, 0xD})
	y := archsimd.LoadInt32x4([]int32{1, 2, 3, 4})
	a, b := flattenedTranspose(x, y)

	a.Store(r)
	b.Store(s)

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

	x := archsimd.LoadInt64x4([]int64{10, 20, 30, 40})
	y := archsimd.LoadInt64x4([]int64{1, 2, 3, 4})

	x.Add(y).Store(r)
	archsimd.ClearAVXUpperBits()
	x.Sub(y).Store(s)

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
	archsimd.LoadUint64x2(src).LeadingZeros().Store(got)
	for i := range 2 {
		if want[i] != got[i] {
			t.Errorf("Result incorrect at %d: want %d, got %d", i, want[i], got[i])
		}
	}
}

func TestIsZero(t *testing.T) {
	v1 := archsimd.LoadUint64x2([]uint64{0, 1})
	v2 := archsimd.LoadUint64x2([]uint64{0, 0})
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
	x := archsimd.LoadInt32x4([]int32{0, 1, 2, 3})
	y := archsimd.LoadInt32x4([]int32{4, 5, 6, 7})

	llll := x.ConcatPermuteScalars(0, 1, 2, 3, y)
	hhhh := x.ConcatPermuteScalars(4, 5, 6, 7, y)
	llhh := x.ConcatPermuteScalars(0, 1, 6, 7, y)
	hhll := x.ConcatPermuteScalars(6, 7, 0, 1, y)

	lllh := x.ConcatPermuteScalars(0, 1, 2, 7, y)
	llhl := x.ConcatPermuteScalars(0, 1, 7, 2, y)
	lhll := x.ConcatPermuteScalars(0, 7, 1, 2, y)
	hlll := x.ConcatPermuteScalars(7, 0, 1, 2, y)

	hhhl := x.ConcatPermuteScalars(4, 5, 6, 0, y)
	hhlh := x.ConcatPermuteScalars(4, 5, 0, 6, y)
	hlhh := x.ConcatPermuteScalars(4, 0, 5, 6, y)
	lhhh := x.ConcatPermuteScalars(0, 4, 5, 6, y)

	lhlh := x.ConcatPermuteScalars(0, 4, 1, 5, y)
	hlhl := x.ConcatPermuteScalars(4, 0, 5, 1, y)
	lhhl := x.ConcatPermuteScalars(0, 4, 5, 1, y)
	hllh := x.ConcatPermuteScalars(4, 0, 1, 5, y)

	r := make([]int32, 4, 4)

	foo := func(v archsimd.Int32x4, a, b, c, d int32) {
		v.Store(r)
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
	return x.ConcatPermuteScalars(a, b, c, d, y)
}

func TestSelect4FromPairVar(t *testing.T) {
	x := archsimd.LoadInt32x4([]int32{0, 1, 2, 3})
	y := archsimd.LoadInt32x4([]int32{4, 5, 6, 7})

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
		v.Store(r)
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
	x := archsimd.LoadFloat32x8([]float32{0, 1, 2, 3, 10, 11, 12, 13})
	y := archsimd.LoadFloat32x8([]float32{4, 5, 6, 7, 14, 15, 16, 17})

	llll := x.ConcatPermuteScalarsGrouped(0, 1, 2, 3, y)
	hhhh := x.ConcatPermuteScalarsGrouped(4, 5, 6, 7, y)
	llhh := x.ConcatPermuteScalarsGrouped(0, 1, 6, 7, y)
	hhll := x.ConcatPermuteScalarsGrouped(6, 7, 0, 1, y)

	lllh := x.ConcatPermuteScalarsGrouped(0, 1, 2, 7, y)
	llhl := x.ConcatPermuteScalarsGrouped(0, 1, 7, 2, y)
	lhll := x.ConcatPermuteScalarsGrouped(0, 7, 1, 2, y)
	hlll := x.ConcatPermuteScalarsGrouped(7, 0, 1, 2, y)

	hhhl := x.ConcatPermuteScalarsGrouped(4, 5, 6, 0, y)
	hhlh := x.ConcatPermuteScalarsGrouped(4, 5, 0, 6, y)
	hlhh := x.ConcatPermuteScalarsGrouped(4, 0, 5, 6, y)
	lhhh := x.ConcatPermuteScalarsGrouped(0, 4, 5, 6, y)

	lhlh := x.ConcatPermuteScalarsGrouped(0, 4, 1, 5, y)
	hlhl := x.ConcatPermuteScalarsGrouped(4, 0, 5, 1, y)
	lhhl := x.ConcatPermuteScalarsGrouped(0, 4, 5, 1, y)
	hllh := x.ConcatPermuteScalarsGrouped(4, 0, 1, 5, y)

	r := make([]float32, 8, 8)

	foo := func(v archsimd.Float32x8, a, b, c, d float32) {
		v.Store(r)
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

func TestConcatPermuteScalarsConstGroupedUint32x16(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := archsimd.LoadUint32x16([]uint32{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33})
	y := archsimd.LoadUint32x16([]uint32{4, 5, 6, 7, 14, 15, 16, 17, 24, 25, 26, 27, 34, 35, 36, 37})

	llll := x.ConcatPermuteScalarsGrouped(0, 1, 2, 3, y)
	hhhh := x.ConcatPermuteScalarsGrouped(4, 5, 6, 7, y)
	llhh := x.ConcatPermuteScalarsGrouped(0, 1, 6, 7, y)
	hhll := x.ConcatPermuteScalarsGrouped(6, 7, 0, 1, y)

	lllh := x.ConcatPermuteScalarsGrouped(0, 1, 2, 7, y)
	llhl := x.ConcatPermuteScalarsGrouped(0, 1, 7, 2, y)
	lhll := x.ConcatPermuteScalarsGrouped(0, 7, 1, 2, y)
	hlll := x.ConcatPermuteScalarsGrouped(7, 0, 1, 2, y)

	hhhl := x.ConcatPermuteScalarsGrouped(4, 5, 6, 0, y)
	hhlh := x.ConcatPermuteScalarsGrouped(4, 5, 0, 6, y)
	hlhh := x.ConcatPermuteScalarsGrouped(4, 0, 5, 6, y)
	lhhh := x.ConcatPermuteScalarsGrouped(0, 4, 5, 6, y)

	lhlh := x.ConcatPermuteScalarsGrouped(0, 4, 1, 5, y)
	hlhl := x.ConcatPermuteScalarsGrouped(4, 0, 5, 1, y)
	lhhl := x.ConcatPermuteScalarsGrouped(0, 4, 5, 1, y)
	hllh := x.ConcatPermuteScalarsGrouped(4, 0, 1, 5, y)

	r := make([]uint32, 16, 16)

	foo := func(v archsimd.Uint32x16, a, b, c, d uint32) {
		v.Store(r)
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

func TestConcatPermute128Scalars(t *testing.T) {
	x := archsimd.LoadUint64x4([]uint64{0, 1, 2, 3})
	y := archsimd.LoadUint64x4([]uint64{4, 5, 6, 7})

	aa := x.ConcatPermute128Scalars(0, 0, y)
	ab := x.ConcatPermute128Scalars(0, 1, y)
	bc := x.ConcatPermute128Scalars(1, 2, y)
	cd := x.ConcatPermute128Scalars(2, 3, y)
	da := x.ConcatPermute128Scalars(3, 0, y)
	dc := x.ConcatPermute128Scalars(3, 2, y)

	r := make([]uint64, 4, 4)

	foo := func(v archsimd.Uint64x4, a, b uint64) {
		a, b = 2*a, 2*b
		v.Store(r)
		checkSlices[uint64](t, r, []uint64{a, a + 1, b, b + 1})
	}

	foo(aa, 0, 0)
	foo(ab, 0, 1)
	foo(bc, 1, 2)
	foo(cd, 2, 3)
	foo(da, 3, 0)
	foo(dc, 3, 2)
}

func TestConcatPermute128ScalarsError(t *testing.T) {
	x := archsimd.LoadUint64x4([]uint64{0, 1, 2, 3})
	y := archsimd.LoadUint64x4([]uint64{4, 5, 6, 7})

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic %v", r)
		}
	}()
	_ = x.ConcatPermute128Scalars(0, 4, y)

	t.Errorf("Should have panicked")
}

//go:noinline
func select128FromPair(x archsimd.Uint64x4, lo, hi uint8, y archsimd.Uint64x4) archsimd.Uint64x4 {
	return x.ConcatPermute128Scalars(lo, hi, y)
}

func TestConcatPermute128ScalarsVar(t *testing.T) {
	x := archsimd.LoadUint64x4([]uint64{0, 1, 2, 3})
	y := archsimd.LoadUint64x4([]uint64{4, 5, 6, 7})

	aa := select128FromPair(x, 0, 0, y)
	ab := select128FromPair(x, 0, 1, y)
	bc := select128FromPair(x, 1, 2, y)
	cd := select128FromPair(x, 2, 3, y)
	da := select128FromPair(x, 3, 0, y)
	dc := select128FromPair(x, 3, 2, y)

	r := make([]uint64, 4, 4)

	foo := func(v archsimd.Uint64x4, a, b uint64) {
		a, b = 2*a, 2*b
		v.Store(r)
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
	x := archsimd.LoadUint64x2([]uint64{0, 1})
	y := archsimd.LoadUint64x2([]uint64{2, 3})

	ll := x.ConcatPermuteScalars(0, 1, y)
	hh := x.ConcatPermuteScalars(3, 2, y)
	lh := x.ConcatPermuteScalars(0, 3, y)
	hl := x.ConcatPermuteScalars(2, 1, y)

	r := make([]uint64, 2, 2)

	foo := func(v archsimd.Uint64x2, a, b uint64) {
		v.Store(r)
		checkSlices[uint64](t, r, []uint64{a, b})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedUint(t *testing.T) {
	x := archsimd.LoadUint64x4([]uint64{0, 1, 10, 11})
	y := archsimd.LoadUint64x4([]uint64{2, 3, 12, 13})

	ll := x.ConcatPermuteScalarsGrouped(0, 1, y)
	hh := x.ConcatPermuteScalarsGrouped(3, 2, y)
	lh := x.ConcatPermuteScalarsGrouped(0, 3, y)
	hl := x.ConcatPermuteScalarsGrouped(2, 1, y)

	r := make([]uint64, 4, 4)

	foo := func(v archsimd.Uint64x4, a, b uint64) {
		v.Store(r)
		checkSlices[uint64](t, r, []uint64{a, b, a + 10, b + 10})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedFloat(t *testing.T) {
	x := archsimd.LoadFloat64x4([]float64{0, 1, 10, 11})
	y := archsimd.LoadFloat64x4([]float64{2, 3, 12, 13})

	ll := x.ConcatPermuteScalarsGrouped(0, 1, y)
	hh := x.ConcatPermuteScalarsGrouped(3, 2, y)
	lh := x.ConcatPermuteScalarsGrouped(0, 3, y)
	hl := x.ConcatPermuteScalarsGrouped(2, 1, y)

	r := make([]float64, 4, 4)

	foo := func(v archsimd.Float64x4, a, b float64) {
		v.Store(r)
		checkSlices[float64](t, r, []float64{a, b, a + 10, b + 10})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestSelect2FromPairConstGroupedInt(t *testing.T) {
	x := archsimd.LoadInt64x4([]int64{0, 1, 10, 11})
	y := archsimd.LoadInt64x4([]int64{2, 3, 12, 13})

	ll := x.ConcatPermuteScalarsGrouped(0, 1, y)
	hh := x.ConcatPermuteScalarsGrouped(3, 2, y)
	lh := x.ConcatPermuteScalarsGrouped(0, 3, y)
	hl := x.ConcatPermuteScalarsGrouped(2, 1, y)

	r := make([]int64, 4, 4)

	foo := func(v archsimd.Int64x4, a, b int64) {
		v.Store(r)
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

	x := archsimd.LoadInt64x8([]int64{0, 1, 10, 11, 20, 21, 30, 31})
	y := archsimd.LoadInt64x8([]int64{2, 3, 12, 13, 22, 23, 32, 33})

	ll := x.ConcatPermuteScalarsGrouped(0, 1, y)
	hh := x.ConcatPermuteScalarsGrouped(3, 2, y)
	lh := x.ConcatPermuteScalarsGrouped(0, 3, y)
	hl := x.ConcatPermuteScalarsGrouped(2, 1, y)

	r := make([]int64, 8, 8)

	foo := func(v archsimd.Int64x8, a, b int64) {
		v.Store(r)
		checkSlices[int64](t, r, []int64{a, b, a + 10, b + 10, a + 20, b + 20, a + 30, b + 30})
	}

	foo(ll, 0, 1)
	foo(hh, 3, 2)
	foo(lh, 0, 3)
	foo(hl, 2, 1)
}

func TestStringAMD64(t *testing.T) {
	x := archsimd.LoadUint32x4([]uint32{0, 1, 2, 3})
	y := archsimd.LoadInt64x4([]int64{-4, -5, -6, -7})
	z := archsimd.LoadFloat32x4([]float32{0.5, 1.5, -2.5, 3.5e9})
	w := archsimd.LoadFloat64x4([]float64{0.5, 1.5, -2.5, 3.5e9})

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

func TestMaskString(t *testing.T) {
	x := archsimd.LoadUint32x4([]uint32{0, 1, 2, 3})
	var y archsimd.Uint32x4

	m := x.Equal(y)

	w := "{1,0,0,0}"

	if g := m.String(); g != w {
		t.Errorf("got=%s wanted %s", g, w)
	}
}

// a returns an slice of 16 int32
func a() []int32 {
	return make([]int32, 16, 16)
}

// applyTo3 returns a 16-element slice of the results of
// applying f to the respective elements of vectors x, y, and z.
func applyTo3(x, y, z archsimd.Int32x16, f func(x, y, z int32) int32) []int32 {
	ax, ay, az := a(), a(), a()
	x.Store(ax)
	y.Store(ay)
	z.Store(az)

	r := a()
	for i := range r {
		r[i] = f(ax[i], ay[i], az[i])
	}
	return r
}

// applyTo4 returns a 16-element slice of the results of
// applying f to the respective elements of vectors x, y, z, and w.
func applyTo4(x, y, z, w archsimd.Int32x16, f func(x, y, z, w int32) int32) []int32 {
	ax, ay, az, aw := a(), a(), a(), a()
	x.Store(ax)
	y.Store(ay)
	z.Store(az)
	w.Store(aw)

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

	x := archsimd.LoadInt32x16(ax)
	y := archsimd.LoadInt32x16(ay)
	z := archsimd.LoadInt32x16(az)
	w := archsimd.LoadInt32x16(aw)
	m := archsimd.LoadInt32x16(am)

	foo := func(v archsimd.Int32x16, s []int32) {
		r := make([]int32, 16, 16)
		v.Store(r)
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
	x := archsimd.LoadInt64x4([]int64{1, 2, 3, 4})
	y := archsimd.LoadInt64x4([]int64{5, 6, 1, 1})
	z := archsimd.LoadInt64x4([]int64{-1, -2, -3, -4})
	res := make([]int64, 4)
	expected := []int64{6, 8, -3, -4}
	mask := x.Less(y)
	if archsimd.X86.AVX512() {
		x.Add(y).Merge(z, mask).Store(res)
	} else {
		x.Add(y).Merge(z, mask).Store(res)
	}
	for i := range 4 {
		if res[i] != expected[i] {
			t.Errorf("got %d wanted %d", res[i], expected[i])
		}
	}
}

func TestMaskedIfElse(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	x := archsimd.LoadInt64x4([]int64{1, 2, 3, 4})
	y := archsimd.LoadInt64x4([]int64{5, 6, 1, 1})
	z := archsimd.LoadInt64x4([]int64{-1, -2, -3, -4})
	res := make([]int64, 4)
	expected := []int64{6, 8, -3, -4}
	mask := x.Less(y)
	if archsimd.X86.AVX512() {
		x.Add(y).IfElse(mask, z).Store(res)
	} else {
		x.Add(y).IfElse(mask, z).Store(res)
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
	archsimd.LoadInt32x4(x).PermuteScalars(1, 2, 3, 0).Store(got)
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
	archsimd.LoadInt32x8(x).PermuteScalarsGrouped(1, 2, 3, 0).Store(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsHi(t *testing.T) {
	x := []int16{-1, -2, -3, -4, 11, 12, 13, 14}
	want := []int16{-1, -2, -3, -4, 12, 13, 14, 11}
	got := make([]int16, len(x))
	archsimd.LoadInt16x8(x).PermuteScalarsHi(1, 2, 3, 0).Store(got)
	checkSlices(t, got, want)
}

func TestPermuteScalarsLo(t *testing.T) {
	x := []int16{11, 12, 13, 14, 4, 5, 6, 7}
	want := []int16{12, 13, 14, 11, 4, 5, 6, 7}
	got := make([]int16, len(x))
	archsimd.LoadInt16x8(x).PermuteScalarsLo(1, 2, 3, 0).Store(got)
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
	archsimd.LoadInt16x16(x).PermuteScalarsHiGrouped(1, 2, 3, 0).Store(got)
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
	archsimd.LoadInt16x16(x).PermuteScalarsLoGrouped(1, 2, 3, 0).Store(got)
	checkSlices(t, got, want)
}

func TestClMul(t *testing.T) {
	var x = archsimd.LoadUint64x2([]uint64{1, 5})
	var y = archsimd.LoadUint64x2([]uint64{3, 9})

	foo := func(v archsimd.Uint64x2, s []uint64) {
		r := make([]uint64, 2, 2)
		v.Store(r)
		checkSlices[uint64](t, r, s)
	}

	foo(x.CarrylessMultiplyEven(y), []uint64{3, 0})
	foo(x.CarrylessMultiplyEvenOdd(y), []uint64{9, 0})
	foo(x.CarrylessMultiplyOddEven(y), []uint64{15, 0})
	foo(x.CarrylessMultiplyOdd(y), []uint64{45, 0})
	foo(y.CarrylessMultiplyEven(y), []uint64{5, 0})

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
	testInt16x8Binary(t, archsimd.Int16x8.ConcatAddPairs, addPairsSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.ConcatSubPairs, subPairsSlice[int16])
	testUint16x8Binary(t, archsimd.Uint16x8.ConcatAddPairs, addPairsSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.ConcatSubPairs, subPairsSlice[uint16])
	testInt32x4Binary(t, archsimd.Int32x4.ConcatAddPairs, addPairsSlice[int32])
	testInt32x4Binary(t, archsimd.Int32x4.ConcatSubPairs, subPairsSlice[int32])
	testUint32x4Binary(t, archsimd.Uint32x4.ConcatAddPairs, addPairsSlice[uint32])
	testUint32x4Binary(t, archsimd.Uint32x4.ConcatSubPairs, subPairsSlice[uint32])
	testFloat32x4Binary(t, archsimd.Float32x4.ConcatAddPairs, addPairsSlice[float32])
	testFloat32x4Binary(t, archsimd.Float32x4.ConcatSubPairs, subPairsSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.ConcatAddPairs, addPairsSlice[float64])
	testFloat64x2Binary(t, archsimd.Float64x2.ConcatSubPairs, subPairsSlice[float64])

	// Grouped versions
	if archsimd.X86.AVX2() {
		testInt16x16Binary(t, archsimd.Int16x16.ConcatAddPairsGrouped, addPairsGroupedSlice[int16])
		testInt16x16Binary(t, archsimd.Int16x16.ConcatSubPairsGrouped, subPairsGroupedSlice[int16])
		testUint16x16Binary(t, archsimd.Uint16x16.ConcatAddPairsGrouped, addPairsGroupedSlice[uint16])
		testUint16x16Binary(t, archsimd.Uint16x16.ConcatSubPairsGrouped, subPairsGroupedSlice[uint16])
		testInt32x8Binary(t, archsimd.Int32x8.ConcatAddPairsGrouped, addPairsGroupedSlice[int32])
		testInt32x8Binary(t, archsimd.Int32x8.ConcatSubPairsGrouped, subPairsGroupedSlice[int32])
		testUint32x8Binary(t, archsimd.Uint32x8.ConcatAddPairsGrouped, addPairsGroupedSlice[uint32])
		testUint32x8Binary(t, archsimd.Uint32x8.ConcatSubPairsGrouped, subPairsGroupedSlice[uint32])
		testFloat32x8Binary(t, archsimd.Float32x8.ConcatAddPairsGrouped, addPairsGroupedSlice[float32])
		testFloat32x8Binary(t, archsimd.Float32x8.ConcatSubPairsGrouped, subPairsGroupedSlice[float32])
		testFloat64x4Binary(t, archsimd.Float64x4.ConcatAddPairsGrouped, addPairsGroupedSlice[float64])
		testFloat64x4Binary(t, archsimd.Float64x4.ConcatSubPairsGrouped, subPairsGroupedSlice[float64])
	}
}

func convConcatSlice[T, U number](a, b []T, conv func(T) U) []U {
	r := make([]U, len(a)+len(b))
	for i, v := range a {
		r[i] = conv(v)
	}
	for i, v := range b {
		r[len(a)+i] = conv(v)
	}
	return r
}

func convConcatGroupedSlice[T, U number](a, b []T, conv func(T) U) []U {
	group := int(128 / unsafe.Sizeof(a[0]))
	r := make([]U, 0, len(a)+len(b))
	for i := 0; i < len(a)/group; i++ {
		r = append(r, convConcatSlice(a[i*group:(i+1)*group], b[i*group:(i+1)*group], conv)...)
	}
	return r
}

func TestSaturateConcat(t *testing.T) {
	// Int32x4.SaturateToInt16Concat
	forSlicePair(t, int32s, 4, func(x, y []int32) bool {
		a, b := archsimd.LoadInt32x4(x), archsimd.LoadInt32x4(y)
		var out [8]int16
		a.SaturateToInt16Concat(b).StoreArray(&out)
		want := convConcatSlice(x, y, satToInt16)
		return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
	})
	// Int32x4.SaturateToUint16Concat
	forSlicePair(t, int32s, 4, func(x, y []int32) bool {
		a, b := archsimd.LoadInt32x4(x), archsimd.LoadInt32x4(y)
		var out [8]uint16
		a.SaturateToUint16Concat(b).StoreArray(&out)
		want := convConcatSlice(x, y, satToUint16)
		return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
	})

	if archsimd.X86.AVX2() {
		// Int32x8.SaturateToInt16ConcatGrouped
		forSlicePair(t, int32s, 8, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x8(x), archsimd.LoadInt32x8(y)
			var out [16]int16
			a.SaturateToInt16ConcatGrouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToInt16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
		// Int32x8.SaturateToUint16ConcatGrouped
		forSlicePair(t, int32s, 8, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x8(x), archsimd.LoadInt32x8(y)
			var out [16]uint16
			a.SaturateToUint16ConcatGrouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToUint16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
	}

	if archsimd.X86.AVX512() {
		// Int32x16.SaturateToInt16ConcatGrouped
		forSlicePair(t, int32s, 16, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x16(x), archsimd.LoadInt32x16(y)
			var out [32]int16
			a.SaturateToInt16ConcatGrouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToInt16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
		// Int32x16.SaturateToUint16ConcatGrouped
		forSlicePair(t, int32s, 16, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x16(x), archsimd.LoadInt32x16(y)
			var out [32]uint16
			a.SaturateToUint16ConcatGrouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToUint16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
	}
}

func testMaskOr8x64(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int8, 64)
	want := []int8{-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0,
		-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
		-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0,
		-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1}
	var a archsimd.Int8x64
	b := archsimd.LoadInt8x64(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt8x64()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr16x32(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int16, 32)
	want := []int16{-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0,
		-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1}
	var a archsimd.Int16x32
	b := archsimd.LoadInt16x32(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt16x32()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr32x16(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int32, 16)
	want := []int32{-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1}
	var a archsimd.Int32x16
	b := archsimd.LoadInt32x16(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt32x16()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr64x8(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int64, 8)
	want := []int64{-1, 0, 0, 0, 0, 0, -1, -1}
	var a archsimd.Int64x8
	b := archsimd.LoadInt64x8(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt64x8()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr8x32(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int8, 32)
	want := []int8{-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0,
		-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1}
	var a archsimd.Int8x32
	b := archsimd.LoadInt8x32(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt8x32()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr16x16(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int16, 16)
	want := []int16{-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1}
	var a archsimd.Int16x16
	b := archsimd.LoadInt16x16(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt16x16()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr32x8(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int32, 8)
	want := []int32{-1, 0, 0, 0, 0, 0, -1, -1}
	var a archsimd.Int32x8
	b := archsimd.LoadInt32x8(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt32x8()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr64x4(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int64, 4)
	want := []int64{-1, 0, 0, -1}
	var a archsimd.Int64x4
	b := archsimd.LoadInt64x4(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt64x4()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr8x16(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int8, 16)
	want := []int8{-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1}
	var a archsimd.Int8x16
	b := archsimd.LoadInt8x16(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt8x16()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr16x8(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int16, 8)
	want := []int16{-1, 0, 0, 0, 0, 0, -1, -1}
	var a archsimd.Int16x8
	b := archsimd.LoadInt16x8(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt16x8()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr32x4(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int32, 4)
	want := []int32{-1, 0, 0, -1}
	var a archsimd.Int32x4
	b := archsimd.LoadInt32x4(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt32x4()
	c.Store(s)
	checkSlices(t, s, want)
}

func testMaskOr64x2(t *testing.T) {
	if !archsimd.X86.AVX512() {
		return // compiler needs to see check+return to ensure mask register use
	}
	s := make([]int64, 2)
	want := []int64{-1, 0}
	var a archsimd.Int64x2
	b := archsimd.LoadInt64x2(want)
	m1 := a.Less(a)
	m2 := b.Less(a)
	m3 := m1.Or(m2)
	c := m3.ToInt64x2()
	c.Store(s)
	checkSlices(t, s, want)
}

func TestMaskOr(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	testMaskOr8x64(t)
	testMaskOr16x32(t)
	testMaskOr32x16(t)
	testMaskOr64x8(t)
	testMaskOr8x32(t)
	testMaskOr16x16(t)
	testMaskOr32x8(t)
	testMaskOr64x4(t)
	testMaskOr8x16(t)
	testMaskOr16x8(t)
	testMaskOr32x4(t)
	testMaskOr64x2(t)
}
