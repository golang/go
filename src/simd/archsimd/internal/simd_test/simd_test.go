// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package simd_test

import (
	"reflect"
	"simd/archsimd"
	"testing"
)

var sink any

func TestType(t *testing.T) {
	// Testing:
	// - Defined as another struct's field is ok
	// - Pointer is ok
	// - Type definition is ok
	// - Type alias is ok
	// - Type conversion is ok
	// - Conversion to interface is ok
	type alias = archsimd.Int32x4
	type vecT archsimd.Int32x4
	type myStruct struct {
		x alias
		y *archsimd.Int32x4
		z vecT
	}
	vals := [4]int32{1, 2, 3, 4}
	v := myStruct{x: archsimd.LoadInt32x4Array(&vals)}
	want := []int32{12, 24, 36, 48}
	y := archsimd.LoadInt32x4Array(&vals)
	v.y = &y
	sink = y

	v.z = vecT(archsimd.LoadInt32x4Array(&[4]int32{10, 20, 30, 40}))
	*v.y = v.y.Add(v.x).Add(archsimd.Int32x4(v.z))

	got := [4]int32{}
	v.y.StoreArray(&got)
	checkSlices(t, got[:], want)
}

func TestUncomparable(t *testing.T) {
	// Test that simd vectors are not comparable
	var x, y any = archsimd.LoadUint32x4Array(&[4]uint32{1, 2, 3, 4}), archsimd.LoadUint32x4Array(&[4]uint32{5, 6, 7, 8})
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
	x := archsimd.LoadInt32x4Array(&xv)
	y := archsimd.LoadInt32x4Array(&yv)
	fn := archsimd.Int32x4.Add
	sink = fn
	x = fn(x, y)
	got := [4]int32{}
	x.StoreArray(&got)
	checkSlices(t, got[:], want)
}

func TestReflectMethod(t *testing.T) {
	// Test that simd intrinsic can be accessed via reflection.
	// NOTE: we don't yet support reflect method.Call.
	xv := [4]int32{1, 2, 3, 4}
	yv := [4]int32{5, 6, 7, 8}
	want := []int32{6, 8, 10, 12}
	x := archsimd.LoadInt32x4Array(&xv)
	y := archsimd.LoadInt32x4Array(&yv)
	m, ok := reflect.TypeOf(x).MethodByName("Add")
	if !ok {
		t.Fatal("Add method not found")
	}
	fn := m.Func.Interface().(func(x, y archsimd.Int32x4) archsimd.Int32x4)
	x = fn(x, y)
	got := [4]int32{}
	x.StoreArray(&got)
	checkSlices(t, got[:], want)
}

func TestVectorConversion(t *testing.T) {
	xv := [4]int32{1, 2, 3, 4}
	x := archsimd.LoadInt32x4Array(&xv)
	xPromoted := x.ToBits().ReshapeToUint64s().BitsToInt64()
	xPromotedDemoted := xPromoted.ToBits().ReshapeToUint32s().BitsToInt32()
	got := [4]int32{}
	xPromotedDemoted.StoreArray(&got)
	for i := range 4 {
		if xv[i] != got[i] {
			t.Errorf("Result at %d incorrect: want %d, got %d", i, xv[i], got[i])
		}
	}
}

func TestMaskConversion(t *testing.T) {
	x := archsimd.LoadInt32x4([]int32{5, 0, 7, 0})
	mask := archsimd.Int32x4{}.Sub(x).ToMask()
	y := archsimd.LoadInt32x4([]int32{1, 2, 3, 4}).Add(x).Masked(mask)
	want := [4]int32{6, 0, 10, 0}
	got := make([]int32, 4)
	y.Store(got)
	checkSlices(t, got[:], want[:])
}

func TestBroadcastUint32x4(t *testing.T) {
	s := make([]uint32, 4, 4)
	archsimd.BroadcastUint32x4(123456789).Store(s)
	checkSlices(t, s, []uint32{123456789, 123456789, 123456789, 123456789})
}

func TestBroadcastFloat32x4(t *testing.T) {
	s := make([]float32, 4, 4)
	archsimd.BroadcastFloat32x4(3.14).Store(s)
	checkSlices(t, s, []float32{3.14, 3.14, 3.14, 3.14})
}

func TestBroadcastFloat64x2(t *testing.T) {
	s := make([]float64, 2, 2)
	archsimd.BroadcastFloat64x2(3.14).Store(s)
	checkSlices(t, s, []float64{3.14, 3.14})
}

func TestBroadcastUint64x2(t *testing.T) {
	s := make([]uint64, 2, 2)
	archsimd.BroadcastUint64x2(123456789012345).Store(s)
	checkSlices(t, s, []uint64{123456789012345, 123456789012345})
}

func TestBroadcastUint16x8(t *testing.T) {
	s := make([]uint16, 8, 8)
	archsimd.BroadcastUint16x8(12345).Store(s)
	checkSlices(t, s, []uint16{12345, 12345, 12345, 12345, 12345, 12345, 12345, 12345})
}

func TestBroadcastInt8x16(t *testing.T) {
	s := make([]int8, 16, 16)
	archsimd.BroadcastInt8x16(-123).Store(s)
	checkSlices(t, s, []int8{-123, -123, -123, -123, -123, -123, -123, -123,
		-123, -123, -123, -123, -123, -123, -123, -123})
}

func TestBroadcastUint8x16(t *testing.T) {
	s := make([]uint8, 16, 16)
	archsimd.BroadcastUint8x16(200).Store(s)
	checkSlices(t, s, []uint8{200, 200, 200, 200, 200, 200, 200, 200,
		200, 200, 200, 200, 200, 200, 200, 200})
}

func TestBroadcastInt16x8(t *testing.T) {
	s := make([]int16, 8, 8)
	archsimd.BroadcastInt16x8(-12345).Store(s)
	checkSlices(t, s, []int16{-12345, -12345, -12345, -12345, -12345, -12345, -12345, -12345})
}

func TestBroadcastInt32x4(t *testing.T) {
	s := make([]int32, 4, 4)
	archsimd.BroadcastInt32x4(-123456789).Store(s)
	checkSlices(t, s, []int32{-123456789, -123456789, -123456789, -123456789})
}

func TestBroadcastInt64x2(t *testing.T) {
	s := make([]int64, 2, 2)
	archsimd.BroadcastInt64x2(-123456789).Store(s)
	checkSlices(t, s, []int64{-123456789, -123456789})
}

func TestString(t *testing.T) {
	x := archsimd.LoadUint32x4([]uint32{0, 1, 2, 3})
	y := archsimd.LoadInt64x2([]int64{-44, -5})
	z := archsimd.LoadFloat32x4([]float32{0.5, 1.5, -2.5, 3.5e9})
	w := archsimd.LoadFloat64x2([]float64{-2.5, 3.5e9})

	sx := "{0,1,2,3}"
	sy := "{-44,-5}"
	sz := "{0.5,1.5,-2.5,3.5e+09}"
	sw := "{-2.5,3.5e+09}"

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

func stringy[T interface{ String() string }](v T) string {
	return v.String()
}

func double[T interface{ Add(T) T }](v T) T {
	return v.Add(v)
}

// Test that vector type instantiation works correctly, see issue #77444.
func TestTypeParam(t *testing.T) {
	x := archsimd.LoadInt64x2([]int64{1, 1})
	y := archsimd.LoadInt64x2([]int64{1, 1})
	if got := stringy(x); got != y.String() {
		t.Fatalf("string(x) = %q, want %q", got, y.String())
	}
	var want, got [2]int64
	y.Add(y).StoreArray(&want)
	double(x).StoreArray(&got)
	if got != want {
		t.Fatalf("double(x) = %v, want %v", got, want)
	}
}

func TestManyFloats(t *testing.T) {
	// This test doesn't do anything SIMD, just test that we can
	// handle correctly a large number of floating point values,
	// as floating point uses same registers as SIMD, but the SSE
	// instructions can only work on low-numbered ones.
	testManyFloats(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
}

var float64Sink float64

//go:noinline
func testManyFloats(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16,
	a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32 float64) {
	float64Sink += a1
	float64Sink *= a2
	float64Sink -= a3
	float64Sink /= a4
	float64Sink += a5
	float64Sink *= a6
	float64Sink -= a7
	float64Sink /= a8
	float64Sink += a9
	float64Sink *= a10
	float64Sink -= a11
	float64Sink /= a12
	float64Sink += a13
	float64Sink *= a14
	float64Sink -= a15
	float64Sink /= a16
	float64Sink += a17
	float64Sink *= a18
	float64Sink -= a19
	float64Sink /= a20
	float64Sink += a21
	float64Sink *= a22
	float64Sink -= a23
	float64Sink /= a24
	float64Sink += a25
	float64Sink *= a26
	float64Sink -= a27
	float64Sink /= a28
	float64Sink += a29
	float64Sink *= a30
	float64Sink -= a31
	float64Sink /= a32
}

func TestSlicesInt8SetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x16(a)

	v = v.SetElem(3, 13)
	a[3] = 13

	b := make([]int8, 16, 16)
	v.Store(b)
	checkSlices(t, a, b)
}

func TestSlicesInt8GetElem(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	v := archsimd.LoadInt8x16(a)
	e := v.GetElem(2)
	if e != a[2] {
		t.Errorf("GetElem(2) = %d != a[2] = %d", e, a[2])
	}

}

var seventeen = uint8(17)

func TestSlicesInt8GetElem16(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v := archsimd.LoadInt8x16(a)
	e := v.GetElem(seventeen - 1)
	t.Errorf("Should have panicked, e=%v", e)
}

func TestSlicesInt8GetElem16const(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v := archsimd.LoadInt8x16(a)
	e := v.GetElem(16)
	t.Errorf("Should have panicked, e=%v", e)
}

func TestSlicesInt8GetElem15(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v := archsimd.LoadInt8x16(a)
	e := v.GetElem(seventeen - 2)
	if e != a[15] {
		t.Errorf("GetElem(15) = %d != a[15] = %d", e, a[15])
	}
}

func TestSlicesInt8GetElem15const(t *testing.T) {
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v := archsimd.LoadInt8x16(a)
	e := v.GetElem(15)
	if e != a[15] {
		t.Errorf("GetElem(15) = %d != a[15] = %d", e, a[15])
	}
}

func TestSlicesInt8SetElem17(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v := archsimd.LoadInt8x16(a)
	e := v.SetElem(seventeen, 18).GetElem(2)
	t.Errorf("Should have panicked, e=%v", e)
}

func TestSlicesInt8SetElem17const(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw EXPECTED panic %v", r)
		} else {
			t.Errorf("Did not see expected panic")
		}
	}()
	a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v := archsimd.LoadInt8x16(a)
	e := v.SetElem(17, 18).GetElem(2)
	t.Errorf("Should have panicked, e=%v", e)
}
