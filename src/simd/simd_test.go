// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"simd"
	"testing"
)

type signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type float interface {
	~float32 | ~float64
}

type number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr | ~float32 | ~float64
}

func TestInt8s(t *testing.T) {
	// 64 elements = 512 bits
	in1 := []int8{
		1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
		17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
		33, -34, 35, -36, 37, -38, 39, -40, 41, -42, 43, -44, 45, -46, 47, -48,
		49, -50, 51, -52, 53, -54, 55, -56, 57, -58, 59, -60, 61, -62, 63, -64,
	}
	in2 := make([]int8, 64)
	for i := range in2 {
		in2[i] = 2
	}

	x := simd.LoadInt8s(in1)
	y := simd.LoadInt8s(in2)

	if x.Len() <= 0 {
		t.Errorf("Int8s.Len() returned <= 0")
	}

	sum := x.Add(y)
	diff := x.Sub(y)
	neg := x.Neg()
	abs := x.Abs()

	buf := make([]int8, x.Len())
	sum.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] + in2[i]
		if buf[i] != expected {
			t.Errorf("Add at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	diff.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] - in2[i]
		if buf[i] != expected {
			t.Errorf("Sub at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	neg.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := -in1[i]
		if buf[i] != expected {
			t.Errorf("Neg at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	abs.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i]
		if expected < 0 {
			expected = -expected
		}
		if buf[i] != expected {
			t.Errorf("Abs at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestInt16s(t *testing.T) {
	// 32 elements = 512 bits
	in1 := make([]int16, 32)
	in2 := make([]int16, 32)
	for i := range in1 {
		in1[i] = int16((i + 1) * 100)
		if i%2 != 0 {
			in1[i] = -in1[i]
		}
		in2[i] = 10
	}

	x := simd.LoadInt16s(in1)
	y := simd.LoadInt16s(in2)

	sum := x.Add(y)
	buf := make([]int16, x.Len())
	sum.Store(buf)

	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] + in2[i]
		if buf[i] != expected {
			t.Errorf("Int16s Add at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllLeft
	rotLeft := x.RotateAllLeft(3)
	rotLeft.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := uint16(in1[i])
		expected := int16((val << 3) | (val >> 13))
		if buf[i] != expected {
			t.Errorf("Int16s RotateAllLeft at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllRight with large distance
	rotRight := x.RotateAllRight(19)
	rotRight.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := uint16(in1[i])
		expected := int16((val >> 3) | (val << 13))
		if buf[i] != expected {
			t.Errorf("Int16s RotateAllRight(19) at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestInt32s(t *testing.T) {
	// 16 elements = 512 bits
	in1 := make([]int32, 16)
	in2 := make([]int32, 16)
	for i := range in1 {
		in1[i] = int32((i + 1) * 1000)
		if i%2 != 0 {
			in1[i] = -in1[i]
		}
		in2[i] = 100
	}

	x := simd.LoadInt32s(in1)
	y := simd.LoadInt32s(in2)

	sum := x.Add(y)
	buf := make([]int32, x.Len())
	sum.Store(buf)

	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] + in2[i]
		if buf[i] != expected {
			t.Errorf("Int32s Add at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllLeft
	rotLeft := x.RotateAllLeft(5)
	rotLeft.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := uint32(in1[i])
		expected := int32((val << 5) | (val >> 27))
		if buf[i] != expected {
			t.Errorf("Int32s RotateAllLeft at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllRight with large distance
	rotRight := x.RotateAllRight(37)
	rotRight.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := uint32(in1[i])
		expected := int32((val >> 5) | (val << 27))
		if buf[i] != expected {
			t.Errorf("Int32s RotateAllRight(37) at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestInt64s(t *testing.T) {
	// 8 elements = 512 bits
	in1 := make([]int64, 8)
	in2 := make([]int64, 8)
	for i := range in1 {
		in1[i] = int64((i + 1) * 10000)
		if i%2 != 0 {
			in1[i] = -in1[i]
		}
		in2[i] = 1000
	}

	x := simd.LoadInt64s(in1)
	y := simd.LoadInt64s(in2)

	sum := x.Add(y)
	buf := make([]int64, x.Len())
	sum.Store(buf)

	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] + in2[i]
		if buf[i] != expected {
			t.Errorf("Int64s Add at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllLeft
	rotLeft := x.RotateAllLeft(7)
	rotLeft.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := uint64(in1[i])
		expected := int64((val << 7) | (val >> 57))
		if buf[i] != expected {
			t.Errorf("Int64s RotateAllLeft at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllRight with large distance
	rotRight := x.RotateAllRight(71)
	rotRight.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := uint64(in1[i])
		expected := int64((val >> 7) | (val << 57))
		if buf[i] != expected {
			t.Errorf("Int64s RotateAllRight(71) at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestUint8s(t *testing.T) {
	// 64 elements = 512 bits
	in1 := make([]uint8, 64)
	in2 := make([]uint8, 64)
	for i := range in1 {
		in1[i] = uint8(i + 1)
		in2[i] = 10
	}

	x := simd.LoadUint8s(in1)
	y := simd.LoadUint8s(in2)

	avg := x.Average(y)
	buf := make([]uint8, x.Len())
	avg.Store(buf)

	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := uint8((int(in1[i]) + int(in2[i]) + 1) >> 1)
		if buf[i] != expected {
			t.Errorf("Uint8s Average at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestFloat32s(t *testing.T) {
	// 16 elements = 512 bits
	in1 := make([]float32, 16)
	in2 := make([]float32, 16)
	for i := range in1 {
		val := float32(i) + 1.5
		if i%2 != 0 {
			val = -val
		}
		in1[i] = val
		in2[i] = 0.5
	}

	x := simd.LoadFloat32s(in1)
	y := simd.LoadFloat32s(in2)

	sum := x.Add(y)
	buf := make([]float32, x.Len())
	sum.Store(buf)

	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] + in2[i]
		if buf[i] != expected {
			t.Errorf("Float32s Add at %d: got %f, want %f", i, buf[i], expected)
		}
	}
}

func TestFloat64s(t *testing.T) {
	// 8 elements = 512 bits
	in1 := make([]float64, 8)
	in2 := make([]float64, 8)
	for i := range in1 {
		val := float64(i)*10.0 + 10.25
		if i%2 != 0 {
			val = -val
		}
		in1[i] = val
		in2[i] = 1.0
	}

	x := simd.LoadFloat64s(in1)
	y := simd.LoadFloat64s(in2)

	mul := x.Mul(y)
	buf := make([]float64, x.Len())
	mul.Store(buf)

	for i := 0; i < x.Len() && i < len(in1); i++ {
		expected := in1[i] * in2[i]
		if buf[i] != expected {
			t.Errorf("Float64s Mul at %d: got %f, want %f", i, buf[i], expected)
		}
	}
}

func TestUint16s(t *testing.T) {
	in1 := make([]uint16, 32)
	for i := range in1 {
		in1[i] = uint16((i + 1) * 100)
	}

	x := simd.LoadUint16s(in1)
	buf := make([]uint16, x.Len())

	// Test RotateAllLeft
	rotLeft := x.RotateAllLeft(3)
	rotLeft.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := in1[i]
		expected := (val << 3) | (val >> 13)
		if buf[i] != expected {
			t.Errorf("Uint16s RotateAllLeft at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllRight with large distance
	rotRight := x.RotateAllRight(19)
	rotRight.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := in1[i]
		expected := (val >> 3) | (val << 13)
		if buf[i] != expected {
			t.Errorf("Uint16s RotateAllRight(19) at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestUint32s(t *testing.T) {
	in1 := make([]uint32, 16)
	for i := range in1 {
		in1[i] = uint32((i + 1) * 1000)
	}

	x := simd.LoadUint32s(in1)
	buf := make([]uint32, x.Len())

	// Test RotateAllLeft
	rotLeft := x.RotateAllLeft(5)
	rotLeft.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := in1[i]
		expected := (val << 5) | (val >> 27)
		if buf[i] != expected {
			t.Errorf("Uint32s RotateAllLeft at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllRight with large distance
	rotRight := x.RotateAllRight(37)
	rotRight.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := in1[i]
		expected := (val >> 5) | (val << 27)
		if buf[i] != expected {
			t.Errorf("Uint32s RotateAllRight(37) at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

func TestUint64s(t *testing.T) {
	in1 := make([]uint64, 8)
	for i := range in1 {
		in1[i] = uint64((i + 1) * 10000)
	}

	x := simd.LoadUint64s(in1)
	buf := make([]uint64, x.Len())

	// Test RotateAllLeft
	rotLeft := x.RotateAllLeft(7)
	rotLeft.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := in1[i]
		expected := (val << 7) | (val >> 57)
		if buf[i] != expected {
			t.Errorf("Uint64s RotateAllLeft at %d: got %d, want %d", i, buf[i], expected)
		}
	}

	// Test RotateAllRight with large distance
	rotRight := x.RotateAllRight(71)
	rotRight.Store(buf)
	for i := 0; i < x.Len() && i < len(in1); i++ {
		val := in1[i]
		expected := (val >> 7) | (val << 57)
		if buf[i] != expected {
			t.Errorf("Uint64s RotateAllRight(71) at %d: got %d, want %d", i, buf[i], expected)
		}
	}
}

type HasStoreLen[E number] interface {
	Store(s []E)
	Len() int
}

func testBroadcast[E number, V HasStoreLen[E]](t *testing.T, x E, f func(e E) V) {
	v := f(x)
	s := make([]E, v.Len())
	v.Store(s)
	for _, e := range s {
		if e != x {
			t.Errorf("Expected %v, saw %v", x, e)
		}
	}
}

func TestBroadcast(t *testing.T) {
	testBroadcast(t, int8(-2), simd.BroadcastInt8s)
	testBroadcast(t, int16(-2), simd.BroadcastInt16s)
	testBroadcast(t, int32(-2), simd.BroadcastInt32s)
	testBroadcast(t, int64(-2), simd.BroadcastInt64s)

	testBroadcast(t, uint8(99), simd.BroadcastUint8s)
	testBroadcast(t, uint16(9999), simd.BroadcastUint16s)
	testBroadcast(t, uint32(99991111), simd.BroadcastUint32s)
	testBroadcast(t, uint64(112233445599887766), simd.BroadcastUint64s)

	testBroadcast(t, float32(99991111), simd.BroadcastFloat32s)
	testBroadcast(t, float64(112233445599887766), simd.BroadcastFloat64s)
}
