// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package simd_test

import (
	"runtime"
	"simd/archsimd"
	"testing"
)

func TestAdd(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Add, addSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Add, addSlice[float64])

	testInt16x8Binary(t, archsimd.Int16x8.Add, addSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Add, addSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Add, addSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Add, addSlice[int8])

	testUint32x4Binary(t, archsimd.Uint32x4.Add, addSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Add, addSlice[uint64])
	testUint16x8Binary(t, archsimd.Uint16x8.Add, addSlice[uint16])
	testUint8x16Binary(t, archsimd.Uint8x16.Add, addSlice[uint8])
}

func TestSub(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Sub, subSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Sub, subSlice[float64])

	testInt32x4Binary(t, archsimd.Int32x4.Sub, subSlice[int32])
	testInt16x8Binary(t, archsimd.Int16x8.Sub, subSlice[int16])
	testInt64x2Binary(t, archsimd.Int64x2.Sub, subSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Sub, subSlice[int8])

	testUint32x4Binary(t, archsimd.Uint32x4.Sub, subSlice[uint32])
	testUint16x8Binary(t, archsimd.Uint16x8.Sub, subSlice[uint16])
	testUint64x2Binary(t, archsimd.Uint64x2.Sub, subSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Sub, subSlice[uint8])
}

func TestMax(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		// skip in AMD64 as NaN behavior is different
		testFloat32x4Binary(t, archsimd.Float32x4.Max, maxSlice[float32])
		testFloat64x2Binary(t, archsimd.Float64x2.Max, maxSlice[float64])
	}
	testInt8x16Binary(t, archsimd.Int8x16.Max, maxSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.Max, maxSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Max, maxSlice[int32])

	testUint8x16Binary(t, archsimd.Uint8x16.Max, maxSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.Max, maxSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Max, maxSlice[uint32])
}

func TestMin(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		// skip in AMD64 as NaN behavior is different
		testFloat32x4Binary(t, archsimd.Float32x4.Min, minSlice[float32])
		testFloat64x2Binary(t, archsimd.Float64x2.Min, minSlice[float64])
	}
	testInt8x16Binary(t, archsimd.Int8x16.Min, minSlice[int8])
	testInt16x8Binary(t, archsimd.Int16x8.Min, minSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Min, minSlice[int32])

	testUint8x16Binary(t, archsimd.Uint8x16.Min, minSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.Min, minSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Min, minSlice[uint32])
}

func TestAnd(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.And, andSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.And, andSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.And, andSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.And, andSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.And, andSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.And, andSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.And, andSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.And, andSlice[uint8])
}

func TestAndNot(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.AndNot, andNotSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.AndNot, andNotSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.AndNot, andNotSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.AndNot, andNotSlice[int8])

	testUint8x16Binary(t, archsimd.Uint8x16.AndNot, andNotSlice[uint8])
	testUint16x8Binary(t, archsimd.Uint16x8.AndNot, andNotSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.AndNot, andNotSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.AndNot, andNotSlice[uint64])
}

func TestXor(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.Xor, xorSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Xor, xorSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Xor, xorSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Xor, xorSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.Xor, xorSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Xor, xorSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Xor, xorSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Xor, xorSlice[uint8])
}

func TestOr(t *testing.T) {
	testInt16x8Binary(t, archsimd.Int16x8.Or, orSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Or, orSlice[int32])
	testInt64x2Binary(t, archsimd.Int64x2.Or, orSlice[int64])
	testInt8x16Binary(t, archsimd.Int8x16.Or, orSlice[int8])

	testUint16x8Binary(t, archsimd.Uint16x8.Or, orSlice[uint16])
	testUint32x4Binary(t, archsimd.Uint32x4.Or, orSlice[uint32])
	testUint64x2Binary(t, archsimd.Uint64x2.Or, orSlice[uint64])
	testUint8x16Binary(t, archsimd.Uint8x16.Or, orSlice[uint8])
}

func TestMul(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Mul, mulSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Mul, mulSlice[float64])

	testInt8x16Binary(t, archsimd.Int8x16.Mul, mulSlice[int8])
	testUint8x16Binary(t, archsimd.Uint8x16.Mul, mulSlice[uint8])
	testInt16x8Binary(t, archsimd.Int16x8.Mul, mulSlice[int16])
	testInt32x4Binary(t, archsimd.Int32x4.Mul, mulSlice[int32])
}

func TestDiv(t *testing.T) {
	testFloat32x4Binary(t, archsimd.Float32x4.Div, divSlice[float32])
	testFloat64x2Binary(t, archsimd.Float64x2.Div, divSlice[float64])
}

func TestGetElem(t *testing.T) {
	// Int8x16
	{
		a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		v := archsimd.LoadInt8x16(a)
		if e := v.GetElem(2); e != a[2] {
			t.Errorf("Int8x16.GetElem(2) = %d, want %d", e, a[2])
		}
	}
	// Int16x8
	{
		a := []int16{10, 20, 30, 40, 50, 60, 70, 80}
		v := archsimd.LoadInt16x8(a)
		if e := v.GetElem(3); e != a[3] {
			t.Errorf("Int16x8.GetElem(3) = %d, want %d", e, a[3])
		}
	}
	// Int32x4
	{
		a := []int32{100, 200, 300, 400}
		v := archsimd.LoadInt32x4(a)
		if e := v.GetElem(1); e != a[1] {
			t.Errorf("Int32x4.GetElem(1) = %d, want %d", e, a[1])
		}
	}
	// Int64x2
	{
		a := []int64{1000, 2000}
		v := archsimd.LoadInt64x2(a)
		if e := v.GetElem(0); e != a[0] {
			t.Errorf("Int64x2.GetElem(0) = %d, want %d", e, a[0])
		}
	}
	// Uint8x16
	{
		a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		v := archsimd.LoadUint8x16(a)
		if e := v.GetElem(5); e != a[5] {
			t.Errorf("Uint8x16.GetElem(5) = %d, want %d", e, a[5])
		}
	}
	// Uint16x8
	{
		a := []uint16{100, 200, 300, 400, 500, 600, 700, 800}
		v := archsimd.LoadUint16x8(a)
		if e := v.GetElem(7); e != a[7] {
			t.Errorf("Uint16x8.GetElem(7) = %d, want %d", e, a[7])
		}
	}
	// Uint32x4
	{
		a := []uint32{1000, 2000, 3000, 4000}
		v := archsimd.LoadUint32x4(a)
		if e := v.GetElem(2); e != a[2] {
			t.Errorf("Uint32x4.GetElem(2) = %d, want %d", e, a[2])
		}
	}
	// Uint64x2
	{
		a := []uint64{10000, 20000}
		v := archsimd.LoadUint64x2(a)
		if e := v.GetElem(1); e != a[1] {
			t.Errorf("Uint64x2.GetElem(1) = %d, want %d", e, a[1])
		}
	}
	// Float32x4
	{
		a := []float32{1.0, 2.0, 3.0, 4.0}
		v := archsimd.LoadFloat32x4(a)
		if e := v.GetElem(3); e != a[3] {
			t.Errorf("Float32x4.GetElem(3) = %f, want %f", e, a[3])
		}
	}
	// Float64x2
	{
		a := []float64{10.5, 20.5}
		v := archsimd.LoadFloat64x2(a)
		if e := v.GetElem(0); e != a[0] {
			t.Errorf("Float64x2.GetElem(0) = %f, want %f", e, a[0])
		}
	}
}

func TestSetElem(t *testing.T) {
	// Int8x16
	{
		a := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		v := archsimd.LoadInt8x16(a)
		v = v.SetElem(3, int8(99))
		a[3] = 99
		b := make([]int8, 16)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Int16x8
	{
		a := []int16{10, 20, 30, 40, 50, 60, 70, 80}
		v := archsimd.LoadInt16x8(a)
		v = v.SetElem(5, int16(123))
		a[5] = 123
		b := make([]int16, 8)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Int32x4
	{
		a := []int32{100, 200, 300, 400}
		v := archsimd.LoadInt32x4(a)
		v = v.SetElem(2, int32(999))
		a[2] = 999
		b := make([]int32, 4)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Int64x2
	{
		a := []int64{1000, 2000}
		v := archsimd.LoadInt64x2(a)
		v = v.SetElem(1, int64(5555))
		a[1] = 5555
		b := make([]int64, 2)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Uint8x16
	{
		a := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		v := archsimd.LoadUint8x16(a)
		v = v.SetElem(7, uint8(200))
		a[7] = 200
		b := make([]uint8, 16)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Uint16x8
	{
		a := []uint16{100, 200, 300, 400, 500, 600, 700, 800}
		v := archsimd.LoadUint16x8(a)
		v = v.SetElem(0, uint16(1111))
		a[0] = 1111
		b := make([]uint16, 8)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Uint32x4
	{
		a := []uint32{1000, 2000, 3000, 4000}
		v := archsimd.LoadUint32x4(a)
		v = v.SetElem(3, uint32(9999))
		a[3] = 9999
		b := make([]uint32, 4)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Uint64x2
	{
		a := []uint64{10000, 20000}
		v := archsimd.LoadUint64x2(a)
		v = v.SetElem(0, uint64(55555))
		a[0] = 55555
		b := make([]uint64, 2)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Float32x4
	{
		a := []float32{1.0, 2.0, 3.0, 4.0}
		v := archsimd.LoadFloat32x4(a)
		v = v.SetElem(1, float32(42.5))
		a[1] = 42.5
		b := make([]float32, 4)
		v.Store(b)
		checkSlices(t, b, a)
	}
	// Float64x2
	{
		a := []float64{10.5, 20.5}
		v := archsimd.LoadFloat64x2(a)
		v = v.SetElem(0, float64(99.9))
		a[0] = 99.9
		b := make([]float64, 2)
		v.Store(b)
		checkSlices(t, b, a)
	}
}
