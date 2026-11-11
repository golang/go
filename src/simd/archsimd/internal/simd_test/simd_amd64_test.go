// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"math"
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

func TestLookupMod(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	indices := []uint64{7, 6, 5, 4, 3, 2, 1, 0}
	want := []int64{8, 7, 6, 5, 4, 3, 2, 1}
	got := make([]int64, 8)
	archsimd.LoadInt64x8(x).LookupMod(archsimd.LoadUint64x8(indices)).Store(got)
	checkSlices(t, got, want)
}

func TestLookupModOrZero(t *testing.T) {
	x := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	indices := []int8{7, 6, 5, 4, 3, 2, 1, 0, -1, 8, -1, 9, -1, 10, -1, 11}
	want := []uint8{8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 0, 10, 0, 11, 0, 12}
	got := make([]uint8, len(x))
	archsimd.LoadUint8x16(x).LookupModOrZero(archsimd.LoadInt8x16(indices)).Store(got)
	checkSlices(t, got, want)
}

func TestConcatLookupMod(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	x := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	y := []int64{-1, -2, -3, -4, -5, -6, -7, -8}
	indices := []uint64{7 + 8, 6, 5 + 8, 4, 3 + 8, 2, 1 + 8, 0}
	want := []int64{-8, 7, -6, 5, -4, 3, -2, 1}
	got := make([]int64, 8)
	archsimd.LoadInt64x8(x).ConcatLookupMod(archsimd.LoadInt64x8(y), archsimd.LoadUint64x8(indices)).Store(got)
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
	s := make([]float64, 4, 4)

	a := archsimd.LoadFloat64x4([]float64{1, 2, 3, 4})
	b := archsimd.LoadFloat64x4([]float64{4, 2, 3, 1})
	g := a.Greater(b)
	c := a.Merge(b, g)
	d := a.Masked(g)

	c.Store(s)
	checkSlices[float64](t, s, []float64{4, 2, 3, 4})

	d.Store(s)
	checkSlices[float64](t, s, []float64{0, 0, 0, 4})

	if archsimd.X86.AVX2() {
		k := make([]int64, 4, 4)
		g.ToInt64x4().Store(k)
		checkSlices[int64](t, k, []int64{0, 0, 0, -1})
	}
}

func TestIfElseFloat(t *testing.T) {
	s64 := make([]float64, 4, 4)

	a64 := archsimd.LoadFloat64x4([]float64{1, 2, 3, 4})
	b64 := archsimd.LoadFloat64x4([]float64{4, 2, 3, 1})
	g64 := a64.Greater(b64)
	c64 := a64.IfElse(g64, b64)
	d64 := a64.Masked(g64)

	c64.Store(s64)
	checkSlices[float64](t, s64, []float64{4, 2, 3, 4})

	d64.Store(s64)
	checkSlices[float64](t, s64, []float64{0, 0, 0, 4})

	s32 := make([]float32, 8, 8)

	a32 := archsimd.LoadFloat32x8([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	b32 := archsimd.LoadFloat32x8([]float32{8, 7, 6, 5, 4, 2, 3, 1})
	g32 := a32.Greater(b32)
	c32 := a32.IfElse(g32, b32)
	d32 := a32.Masked(g32)

	c32.Store(s32)
	checkSlices[float32](t, s32, []float32{8, 7, 6, 5, 5, 6, 7, 8})

	d32.Store(s32)
	checkSlices[float32](t, s32, []float32{0, 0, 0, 0, 5, 6, 7, 8})

	if archsimd.X86.AVX2() {
		k := make([]int64, 4, 4)
		g64.ToInt64x4().Store(k)
		checkSlices[int64](t, k, []int64{0, 0, 0, -1})
	}
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
	got := make([]uint32, 4)
	archsimd.LoadUint32x4([]uint32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(ro).Store(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("Want 0b1100, got %b", v)
		}
	}
	archsimd.LoadUint32x4([]uint32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(roBig).Store(got)
	for _, v := range got {
		if v != 0b1100 {
			t.Errorf("Want 0b1100, got %b", v)
		}
	}
}

func TestRotateAllConst(t *testing.T) {
	got := make([]uint32, 4)
	archsimd.LoadUint32x4([]uint32{0b11, 0b11, 0b11, 0b11}).RotateAllLeft(2).Store(got)
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

// TestMaskXnor512 tests mask equality, x.Xor(y).Not(), which the compiler
// folds into KXNOR.
func TestMaskXnor512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}

	zero8 := archsimd.Int8x64{}
	x8 := make([]int8, 64)
	y8 := make([]int8, 64)
	want8 := make([]int8, 64)
	for i := range x8 {
		x8[i], y8[i] = int8(i&1), int8(i>>1&1)
		if x8[i] == y8[i] {
			want8[i] = -1
		}
	}
	m8 := archsimd.LoadInt8x64(x8).Greater(zero8)
	n8 := archsimd.LoadInt8x64(y8).Greater(zero8)
	got8 := make([]int8, 64)
	m8.Xor(n8).Not().ToInt8x64().Store(got8)
	checkSlices(t, got8, want8)

	zero64 := archsimd.Int64x8{}
	m64 := archsimd.LoadInt64x8([]int64{0, 1, 0, 1, 0, 1, 0, 1}).Greater(zero64)
	n64 := archsimd.LoadInt64x8([]int64{0, 0, 1, 1, 0, 0, 1, 1}).Greater(zero64)
	got64 := make([]int64, 8)
	m64.Xor(n64).Not().ToInt64x8().Store(got64)
	checkSlices(t, got64, []int64{-1, 0, 0, -1, -1, 0, 0, -1})
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
	if !archsimd.X86.AVXPCLMULQDQ() {
		t.Skip("Test requires X86.AVXPCLMULQDQ, not available on this hardware")
	}
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

func TestClMul256(t *testing.T) {
	if !archsimd.X86.VPCLMULQDQ() {
		t.Skip("Test requires X86.VPCLMULQDQ, not available on this hardware")
	}
	x := archsimd.LoadUint64x4([]uint64{1, 5, 1 << 63, 5})
	y := archsimd.LoadUint64x4([]uint64{3, 9, 2, 9})

	check := func(v archsimd.Uint64x4, want []uint64) {
		t.Helper()
		var got [4]uint64
		v.Store(got[:])
		checkSlices(t, got[:], want)
	}
	check(x.CarrylessMultiplyEven(y), []uint64{3, 0, 0, 1})
	check(x.CarrylessMultiplyEvenOdd(y), []uint64{9, 0, 1 << 63, 4})
	check(x.CarrylessMultiplyOddEven(y), []uint64{15, 0, 10, 0})
	check(x.CarrylessMultiplyOdd(y), []uint64{45, 0, 45, 0})
}

func concatSubPairsSlice[T number](a, b []T) []T {
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
		r = append(r, concatAddPairsSlice(a[i*group:(i+1)*group], b[i*group:(i+1)*group])...)
	}
	return r
}

func subPairsGroupedSlice[T number](a, b []T) []T {
	group := int(128 / unsafe.Sizeof(a[0]))
	r := make([]T, 0, len(a))
	for i := range len(a) / group {
		r = append(r, concatSubPairsSlice(a[i*group:(i+1)*group], b[i*group:(i+1)*group])...)
	}
	return r
}

func TestAddSubPairs(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.ConcatAddPairs, concatAddPairsSlice[int16])
	testInt16x8Binary(t, archsimd.Int16x8.ConcatSubPairs, concatSubPairsSlice[int16])
	testUint16x8Binary(t, archsimd.Uint16x8.ConcatAddPairs, concatAddPairsSlice[uint16])
	testUint16x8Binary(t, archsimd.Uint16x8.ConcatSubPairs, concatSubPairsSlice[uint16])
	testInt32x4Binary(t, archsimd.Int32x4.ConcatAddPairs, concatAddPairsSlice[int32])
	testInt32x4Binary(t, archsimd.Int32x4.ConcatSubPairs, concatSubPairsSlice[int32])
	testUint32x4Binary(t, archsimd.Uint32x4.ConcatAddPairs, concatAddPairsSlice[uint32])
	testUint32x4Binary(t, archsimd.Uint32x4.ConcatSubPairs, concatSubPairsSlice[uint32])
	testFloat32x4Binary(t, archsimd.Float32x4.ConcatAddPairs, concatAddPairsSlice[float32])
	testFloat32x4Binary(t, archsimd.Float32x4.ConcatSubPairs, concatSubPairsSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.ConcatAddPairs, concatAddPairsSlice[float64])
	testFloat64x2Binary(t, archsimd.Float64x2.ConcatSubPairs, concatSubPairsSlice[float64])

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
	// Int32x4.ConcatSaturateToInt16
	forSlicePair(t, int32s, 4, func(x, y []int32) bool {
		a, b := archsimd.LoadInt32x4(x), archsimd.LoadInt32x4(y)
		var out [8]int16
		a.ConcatSaturateToInt16(b).StoreArray(&out)
		want := convConcatSlice(x, y, satToInt16)
		return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
	})
	// Int32x4.ConcatSaturateToUint16
	forSlicePair(t, int32s, 4, func(x, y []int32) bool {
		a, b := archsimd.LoadInt32x4(x), archsimd.LoadInt32x4(y)
		var out [8]uint16
		a.ConcatSaturateToUint16(b).StoreArray(&out)
		want := convConcatSlice(x, y, satToUint16)
		return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
	})

	if archsimd.X86.AVX2() {
		// Int32x8.ConcatSaturateToInt16Grouped
		forSlicePair(t, int32s, 8, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x8(x), archsimd.LoadInt32x8(y)
			var out [16]int16
			a.ConcatSaturateToInt16Grouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToInt16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
		// Int32x8.ConcatSaturateToUint16Grouped
		forSlicePair(t, int32s, 8, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x8(x), archsimd.LoadInt32x8(y)
			var out [16]uint16
			a.ConcatSaturateToUint16Grouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToUint16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
	}

	if archsimd.X86.AVX512() {
		// Int32x16.ConcatSaturateToInt16Grouped
		forSlicePair(t, int32s, 16, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x16(x), archsimd.LoadInt32x16(y)
			var out [32]int16
			a.ConcatSaturateToInt16Grouped(b).StoreArray(&out)
			want := convConcatGroupedSlice(x, y, satToInt16)
			return checkSlicesLogInput(t, out[:], want, 0, func() { t.Logf("x=%v, y=%v", x, y) })
		})
		// Int32x16.ConcatSaturateToUint16Grouped
		forSlicePair(t, int32s, 16, func(x, y []int32) bool {
			a, b := archsimd.LoadInt32x16(x), archsimd.LoadInt32x16(y)
			var out [32]uint16
			a.ConcatSaturateToUint16Grouped(b).StoreArray(&out)
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

func TestReduceSumAmd64(t *testing.T) {
	// 256-bit float available with plain AVX
	testV2S(t, float32s, archsimd.LoadFloat32x8, archsimd.Float32x8.ReduceSum, reduceSum)
	testV2S(t, float64s, archsimd.LoadFloat64x4, archsimd.Float64x4.ReduceSum, reduceSum)
}

func TestReduceSumAmd64AVX2(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("Test requires X86.AVX2, not available on this hardware")
		return
	}
	testV2S(t, float32s, archsimd.LoadFloat32x8, archsimd.Float32x8.ReduceSum, reduceSum)
	testV2S(t, float64s, archsimd.LoadFloat64x4, archsimd.Float64x4.ReduceSum, reduceSum)

	testV2S(t, int8s, archsimd.LoadInt8x32, archsimd.Int8x32.ReduceSum, reduceSum)
	testV2S(t, uint8s, archsimd.LoadUint8x32, archsimd.Uint8x32.ReduceSum, reduceSum)

	testV2S(t, int16s, archsimd.LoadInt16x16, archsimd.Int16x16.ReduceSum, reduceSum)
	testV2S(t, uint16s, archsimd.LoadUint16x16, archsimd.Uint16x16.ReduceSum, reduceSum)

	testV2S(t, int32s, archsimd.LoadInt32x8, archsimd.Int32x8.ReduceSum, reduceSum)
	testV2S(t, uint32s, archsimd.LoadUint32x8, archsimd.Uint32x8.ReduceSum, reduceSum)
}

func TestReduceSumAmd64AVX512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	testV2S(t, float32s, archsimd.LoadFloat32x16, archsimd.Float32x16.ReduceSum, reduceSum)
	testV2S(t, float64s, archsimd.LoadFloat64x8, archsimd.Float64x8.ReduceSum, reduceSum)

	testV2S(t, int8s, archsimd.LoadInt8x64, archsimd.Int8x64.ReduceSum, reduceSum)
	testV2S(t, uint8s, archsimd.LoadUint8x64, archsimd.Uint8x64.ReduceSum, reduceSum)

	testV2S(t, int16s, archsimd.LoadInt16x32, archsimd.Int16x32.ReduceSum, reduceSum)
	testV2S(t, uint16s, archsimd.LoadUint16x32, archsimd.Uint16x32.ReduceSum, reduceSum)

	testV2S(t, int32s, archsimd.LoadInt32x16, archsimd.Int32x16.ReduceSum, reduceSum)
	testV2S(t, uint32s, archsimd.LoadUint32x16, archsimd.Uint32x16.ReduceSum, reduceSum)
}

func TestMaskedZeroingResultSize(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	// Regression test: VCVTPS2PD produces 64-bit elements from 32-bit
	// inputs. A 32-bit mask applied after reinterpreting the result as
	// Int32x8 must not be folded into VCVTPS2PDMasked256, because that
	// would cause it to be treated as a 64-bit mask (using only bits 0-3
	// and zeroing the whole result for 0xf0).
	x := archsimd.LoadFloat32x8([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	got64 := make([]float64, 4)
	x.ConvertToFloat64().AsInt32x8().Masked(archsimd.Mask32x8FromBits(0xf0)).AsFloat64x4().Store(got64)
	checkSlices(t, got64, []float64{0, 0, 3, 4})

	// Conversely, VCVTPD2PSY produces four 32-bit elements from four
	// 64-bit inputs, in the low half of a Float32x8. A 64-bit mask applied
	// after reinterpreting the result as Int64x4 must not be folded into
	// a masked VCVTPD2PSY, because that would treat the 64-bit mask 0b10
	// as the 32-bit mask 0b0010 (producing {0, 2, 0, 0, ...} instead of
	// {0, 0, 3, 4, ...}).
	y := archsimd.LoadFloat64x4([]float64{1, 2, 3, 4})
	got32 := make([]float32, 8)
	y.ConvertToFloat32().AsInt64x4().Masked(archsimd.Mask64x4FromBits(0b10)).AsFloat32x8().Store(got32)
	checkSlices(t, got32, []float32{0, 0, 3, 4, 0, 0, 0, 0})

	// And when a 32-bit mask is applied directly to the Float32x8 result,
	// folding it into a masked VCVTPD2PSY must mask all 4 float32 outputs.
	y.ConvertToFloat32().Masked(archsimd.Mask32x8FromBits(0b1010)).Store(got32)
	checkSlices(t, got32, []float32{0, 2, 0, 4, 0, 0, 0, 0})
}

func TestMaskedMergingResultSize(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	// Regression test: VPMADDWD produces 32-bit elements from 16-bit
	// inputs. Merging the result must convert the mask using the 32-bit
	// result element size, not the 16-bit input element size.
	x := archsimd.LoadInt16x8([]int16{1, 2, 3, 4, 5, 6, 7, 8})
	y := archsimd.LoadInt16x8([]int16{1, 1, 1, 1, 1, 1, 1, 1})
	dst := archsimd.LoadInt32x4([]int32{10, 20, 30, 40})
	got := make([]int32, 4)
	x.DotProductPairs(y).Merge(dst, archsimd.Mask32x4FromBits(0b0010)).Store(got)
	checkSlices(t, got, []int32{10, 7, 30, 40})
}

func TestMaskedMergePartialResult(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	// Regression test: VCVTPD2PSX converts two float64s into the low two
	// float32s of its 128-bit result and zeroes the upper two, even with
	// merging masking. Merging its result into dst must preserve the upper
	// two elements of dst where the mask is false.
	x := archsimd.LoadFloat64x2([]float64{1, 2})
	dst := archsimd.LoadFloat32x4([]float32{10, 20, 30, 40})
	got := make([]float32, 4)
	x.ConvertToFloat32().Merge(dst, archsimd.Mask32x4FromBits(0b0001)).Store(got)
	checkSlices(t, got, []float32{1, 20, 30, 40})
}

func TestMaskedNarrowingConversion(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	// A 256-bit narrowing conversion writes only the low part of its result
	// (VPMOVDB Y, X). Masking with zeroing can fold into the conversion,
	// since the upper part is zero either way, but merging can't, since the
	// conversion would zero the upper part of dst. (The 256-bit 2:1
	// conversions are emulated, so this uses a 4:1 conversion.)
	xs := make([]int32, 8)
	dsts := make([]int8, 32)
	for i := range xs {
		xs[i] = int32(i + 1)
	}
	for i := range dsts {
		dsts[i] = 100
	}
	x := archsimd.LoadInt32x8(xs)
	dst := archsimd.LoadInt8x32(dsts)
	got := make([]int8, 32)

	x.TruncToInt8().Masked(archsimd.Mask8x32FromBits(0xaaaaaaaa)).Store(got)
	want := make([]int8, 32)
	for i := 1; i < 8; i += 2 {
		want[i] = int8(i + 1)
	}
	checkSlices(t, got, want)

	x.TruncToInt8().IfElse(archsimd.Mask8x32FromBits(0x0000000f), dst).Store(got)
	for i := range want {
		if i < 4 {
			want[i] = int8(i + 1)
		} else {
			want[i] = 100
		}
	}
	checkSlices(t, got, want)

	// Likewise at 512 bits (VPMOVSDW Z, Y).
	ys := make([]int32, 16)
	dsts16 := make([]int16, 32)
	for i := range ys {
		ys[i] = int32(i+1) << 16 // Saturates to MaxInt16.
	}
	for i := range dsts16 {
		dsts16[i] = 100
	}
	y := archsimd.LoadInt32x16(ys)
	dst16 := archsimd.LoadInt16x32(dsts16)
	got16 := make([]int16, 32)
	y.SaturateToInt16().IfElse(archsimd.Mask16x32FromBits(0x0000000f), dst16).Store(got16)
	want16 := make([]int16, 32)
	for i := range want16 {
		if i < 4 {
			want16[i] = math.MaxInt16
		} else {
			want16[i] = 100
		}
	}
	checkSlices(t, got16, want16)
}

func TestMaskedWideningConversion(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
	}
	// A 256-bit widening conversion reads only the low half of its input
	// (VPMOVSXBW X, Y), and its result fills the whole register, so both
	// zeroing and merging masks can fold into it.
	xs := make([]int8, 32)
	dsts := make([]int16, 16)
	for i := range xs {
		xs[i] = int8(-i)
	}
	for i := range dsts {
		dsts[i] = 100
	}
	x := archsimd.LoadInt8x32(xs)
	dst := archsimd.LoadInt16x16(dsts)
	got := make([]int16, 16)

	x.ExtendToInt16().Masked(archsimd.Mask16x16FromBits(0xff00)).Store(got)
	want := make([]int16, 16)
	for i := 8; i < 16; i++ {
		want[i] = int16(-i)
	}
	checkSlices(t, got, want)

	x.ExtendToInt16().IfElse(archsimd.Mask16x16FromBits(0x00f0), dst).Store(got)
	for i := range want {
		if 4 <= i && i < 8 {
			want[i] = int16(-i)
		} else {
			want[i] = 100
		}
	}
	checkSlices(t, got, want)
}

func TestEnlarge(t *testing.T) {
	// The upper half of each result must be zero.
	{
		var got [32]int8
		archsimd.LoadInt8x16([]int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	}
	{
		var got [16]uint16
		archsimd.LoadUint16x8([]uint16{1, 2, 3, 4, 5, 6, 7, 0xffff}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []uint16{1, 2, 3, 4, 5, 6, 7, 0xffff, 0, 0, 0, 0, 0, 0, 0, 0})
	}
	{
		var got [8]int32
		archsimd.LoadInt32x4([]int32{-1, -2, -3, -4}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []int32{-1, -2, -3, -4, 0, 0, 0, 0})
	}
	{
		var got [4]uint64
		archsimd.LoadUint64x2([]uint64{1, 1 << 63}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []uint64{1, 1 << 63, 0, 0})
	}
	{
		var got [8]float32
		archsimd.LoadFloat32x4([]float32{1.5, -2.5, 3.5, -4.5}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []float32{1.5, -2.5, 3.5, -4.5, 0, 0, 0, 0})
	}
	{
		var got [4]float64
		archsimd.LoadFloat64x2([]float64{1.5, -2.5}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []float64{1.5, -2.5, 0, 0})
	}

	// Taking the low half of a wider vector may leave the upper bits of
	// the source register set. Enlarge must clear them.
	{
		var got [8]float32
		y := archsimd.LoadFloat32x8([]float32{1, 2, 3, 4, 5, 6, 7, 8})
		y.GetLo().Enlarge().Store(got[:])
		checkSlices(t, got[:], []float32{1, 2, 3, 4, 0, 0, 0, 0})
		y.GetHi().Enlarge().Store(got[:])
		checkSlices(t, got[:], []float32{5, 6, 7, 8, 0, 0, 0, 0})
	}
	if archsimd.X86.AVX2() {
		var got [8]int32
		y := archsimd.LoadInt32x8([]int32{1, 2, 3, 4, 5, 6, 7, 8})
		y.GetLo().Enlarge().Store(got[:])
		checkSlices(t, got[:], []int32{1, 2, 3, 4, 0, 0, 0, 0})
	}

	t.Run("512", testEnlarge512)
}

func testEnlarge512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("Test requires X86.AVX512, not available on this hardware")
		return
	}
	{
		var got [16]int32
		archsimd.LoadInt32x8([]int32{1, 2, 3, 4, 5, 6, 7, 8}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []int32{1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0})
	}
	{
		var got [8]float64
		archsimd.LoadFloat64x4([]float64{1, 2, 3, 4}).Enlarge().Store(got[:])
		checkSlices(t, got[:], []float64{1, 2, 3, 4, 0, 0, 0, 0})
	}
	{
		// The low half of a 512-bit vector, extended back to 512 bits.
		var got [64]uint8
		var src [64]uint8
		for i := range src {
			src[i] = uint8(i + 1)
		}
		archsimd.LoadUint8x64(src[:]).GetLo().Enlarge().Store(got[:])
		want := make([]uint8, 64)
		copy(want, src[:32])
		checkSlices(t, got[:], want)
	}
}
