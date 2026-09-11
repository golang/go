// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"runtime"
	"simd"
	"simd/internal/test_helpers"
	"slices"
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
	values := test_helpers.Int8s()
	load := simd.LoadInt8s
	test_helpers.TestV2Ve(t, values, load, simd.Int8s.Neg, test_helpers.Neg)
	test_helpers.TestV2Ve(t, values, load, simd.Int8s.Abs, test_helpers.Abs)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.AndNot, test_helpers.AndNot)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Max, test_helpers.Max)
	test_helpers.TestVV2Ve(t, values, load, simd.Int8s.Min, test_helpers.Min)
	test_helpers.TestVV2Me(t, values, load, simd.Int8s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Int8s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int8s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Int8s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int8s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Int8s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: AddSaturated, ConvertToUint8, IfElse, Len,
	// Masked, Not, ReduceSum (tested in TestReduceSum), Store, StorePart,
	// String, SubSaturated, ToBits, ToMask.
}

func TestInt16s(t *testing.T) {
	values := test_helpers.Int16s()
	load := simd.LoadInt16s
	shifts := test_helpers.Shift16s()
	test_helpers.TestV2Ve(t, values, load, simd.Int16s.Neg, test_helpers.Neg)
	test_helpers.TestV2Ve(t, values, load, simd.Int16s.Abs, test_helpers.Abs)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.AndNot, test_helpers.AndNot)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Max, test_helpers.Max)
	test_helpers.TestVV2Ve(t, values, load, simd.Int16s.Min, test_helpers.Min)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int16s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int16s.ShiftAllRight, test_helpers.ShiftRight)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int16s.RotateAllLeft, test_helpers.RotateLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int16s.RotateAllRight, test_helpers.RotateRight)
	test_helpers.TestVV2Me(t, values, load, simd.Int16s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Int16s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int16s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Int16s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int16s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Int16s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: AddSaturated, ConvertToUint16, IfElse, Len,
	// Masked, Not, ReduceSum (tested in TestReduceSum), Store, StorePart,
	// String, SubSaturated, ToBits, ToMask.
}

func TestInt32s(t *testing.T) {
	values := test_helpers.Int32s()
	load := simd.LoadInt32s
	shifts := test_helpers.Shift32s()
	test_helpers.TestV2Ve(t, values, load, simd.Int32s.Neg, test_helpers.Neg)
	test_helpers.TestV2Ve(t, values, load, simd.Int32s.Abs, test_helpers.Abs)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.AndNot, test_helpers.AndNot)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Max, test_helpers.Max)
	test_helpers.TestVV2Ve(t, values, load, simd.Int32s.Min, test_helpers.Min)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int32s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int32s.ShiftAllRight, test_helpers.ShiftRight)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int32s.RotateAllLeft, test_helpers.RotateLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int32s.RotateAllRight, test_helpers.RotateRight)
	test_helpers.TestVV2Me(t, values, load, simd.Int32s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Int32s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int32s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Int32s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int32s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Int32s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: ConvertToFloat32, ConvertToUint32, IfElse, Len,
	// Masked, Not, ReduceSum (tested in TestReduceSum), Store, StorePart,
	// String, ToBits, ToMask.
}

func TestInt64s(t *testing.T) {
	values := test_helpers.Int64s()
	load := simd.LoadInt64s
	shifts := test_helpers.Shift64s()
	test_helpers.TestV2Ve(t, values, load, simd.Int64s.Neg, test_helpers.Neg)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.AndNot, test_helpers.AndNot)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int64s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int64s.RotateAllLeft, test_helpers.RotateLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int64s.RotateAllRight, test_helpers.RotateRight)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: ConvertToUint64, IfElse, Len, Masked, Not, Store,
	// StorePart, String, ToBits, ToMask.
}

func TestUint8s(t *testing.T) {
	values := test_helpers.Uint8s()
	load := simd.LoadUint8s
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.AndNot, test_helpers.AndNot)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Max, test_helpers.Max)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint8s.Min, test_helpers.Min)
	test_helpers.TestVV2Me(t, values, load, simd.Uint8s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Uint8s.NotEqual, test_helpers.NotEqual)

	// TODO: untested methods: AddSaturated, Average, BitsToInt8, ConvertToInt8,
	// IfElse, Len, Masked, Not, ReduceSum (tested in TestReduceSum),
	// ReshapeToUint16s, ReshapeToUint32s, ReshapeToUint64s, Store, StorePart,
	// String, SubSaturated.
}

func TestFloat32s(t *testing.T) {
	values := test_helpers.Float32s()
	load := simd.LoadFloat32s
	test_helpers.TestV2Ve(t, values, load, simd.Float32s.Neg, test_helpers.Neg)
	test_helpers.TestV2Ve(t, values, load, simd.Float32s.Abs, test_helpers.Abs)
	test_helpers.TestVV2Ve(t, values, load, simd.Float32s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Float32s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Float32s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Float32s.Div, test_helpers.Div)
	if runtime.GOARCH != "amd64" {
		test_helpers.TestVV2Ve(t, values, load, simd.Float32s.Max, test_helpers.Max)
		test_helpers.TestVV2Ve(t, values, load, simd.Float32s.Min, test_helpers.Min)
	} else {
		t.Logf("Skipping FP min/max on %s because of NaN anomalies", runtime.GOARCH)
	}
	test_helpers.TestVV2Me(t, values, load, simd.Float32s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Float32s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Float32s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Float32s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Float32s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Float32s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: ConvertToInt32, IfElse, Len, Masked, MulAdd,
	// ReduceSum (tested in TestReduceSum), Sqrt, Store, StorePart, String,
	// ToBits.
}

func TestFloat64s(t *testing.T) {
	values := test_helpers.Float64s()
	load := simd.LoadFloat64s
	test_helpers.TestV2Ve(t, values, load, simd.Float64s.Neg, test_helpers.Neg)
	test_helpers.TestV2Ve(t, values, load, simd.Float64s.Abs, test_helpers.Abs)
	test_helpers.TestVV2Ve(t, values, load, simd.Float64s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Float64s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Float64s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Float64s.Div, test_helpers.Div)
	if runtime.GOARCH != "amd64" {
		test_helpers.TestVV2Ve(t, values, load, simd.Float64s.Max, test_helpers.Max)
		test_helpers.TestVV2Ve(t, values, load, simd.Float64s.Min, test_helpers.Min)
	} else {
		t.Logf("Skipping FP min/max on %s because of NaN anomalies", runtime.GOARCH)
	}
	test_helpers.TestVV2Me(t, values, load, simd.Float64s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Float64s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Float64s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Float64s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Float64s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Float64s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: IfElse, Len, Masked, MulAdd,
	// ReduceSum (tested in TestReduceSum), Sqrt, Store, StorePart, String,
	// ToBits.
}

func TestUint16s(t *testing.T) {
	values := test_helpers.Uint16s()
	load := simd.LoadUint16s
	shifts := test_helpers.Shift16s()
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.AndNot, test_helpers.AndNot)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Max, test_helpers.Max)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint16s.Min, test_helpers.Min)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint16s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint16s.ShiftAllRight, test_helpers.ShiftRight)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint16s.RotateAllLeft, test_helpers.RotateLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint16s.RotateAllRight, test_helpers.RotateRight)
	test_helpers.TestVV2Me(t, values, load, simd.Uint16s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Uint16s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint16s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Uint16s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint16s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Uint16s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: AddSaturated, Average, BitsToInt16,
	// ConvertToInt16, IfElse, Len, Masked, Not,
	// ReduceSum (tested in TestReduceSum), ReshapeToUint32s, ReshapeToUint64s,
	// ReshapeToUint8s, Store, StorePart, String, SubSaturated.
}

func TestUint32s(t *testing.T) {
	values := test_helpers.Uint32s()
	load := simd.LoadUint32s
	shifts := test_helpers.Shift32s()
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Mul, test_helpers.Mul)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.AndNot, test_helpers.AndNot)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Max, test_helpers.Max)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint32s.Min, test_helpers.Min)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint32s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint32s.ShiftAllRight, test_helpers.ShiftRight)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint32s.RotateAllLeft, test_helpers.RotateLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint32s.RotateAllRight, test_helpers.RotateRight)
	test_helpers.TestVV2Me(t, values, load, simd.Uint32s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Uint32s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint32s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Uint32s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint32s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Uint32s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: BitsToFloat32, BitsToInt32, ConvertToInt32,
	// IfElse, Len, Masked, Not, ReduceSum (tested in TestReduceSum),
	// ReshapeToUint16s, ReshapeToUint64s, ReshapeToUint8s, Store, StorePart,
	// String.
}

func TestUint64s(t *testing.T) {
	values := test_helpers.Uint64s()
	load := simd.LoadUint64s
	shifts := test_helpers.Shift64s()
	test_helpers.TestVV2Ve(t, values, load, simd.Uint64s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint64s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint64s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint64s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint64s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Uint64s.AndNot, test_helpers.AndNot)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint64s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint64s.ShiftAllRight, test_helpers.ShiftRight)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint64s.RotateAllLeft, test_helpers.RotateLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Uint64s.RotateAllRight, test_helpers.RotateRight)
	test_helpers.TestVV2Me(t, values, load, simd.Uint64s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Uint64s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint64s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Uint64s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint64s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Uint64s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: BitsToFloat64, BitsToInt64, CarrylessMultiplyEven,
	// CarrylessMultiplyOdd, ConvertToInt64, IfElse, Len, Masked, Not,
	// ReshapeToUint16s, ReshapeToUint32s, ReshapeToUint8s, Store, StorePart,
	// String.
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

func TestMaskToInt(t *testing.T) {
	topBits := simd.BroadcastUint8s(0x80)
	got := make([]int8, topBits.Len())
	topBits.Equal(topBits).ToInt8s().Store(got)
	want := slices.Repeat([]int8{-1}, topBits.Len())
	if !slices.Equal(want, got) {
		t.Errorf("Wanted %v, got %v", want, got)
	}
}

//go:noinline
func four() uint64 {
	return 4
}

func TestShiftAllLeft(t *testing.T) {
	// Int16s on 512-bit vector has 32 elements.
	in := []int16{
		1, 2, 4, 8, 16, 32, 64, 128,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	v := simd.LoadInt16s(in)
	want := []int16{
		1 << 4, 2 << 4, 4 << 4, 8 << 4, 16 << 4, 32 << 4, 64 << 4, 128 << 4,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	}[:v.Len()]

	{
		// Shift all elements left by 4 bits.
		res := v.ShiftAllLeft(4)
		got := make([]int16, res.Len())
		res.Store(got)
		if !slices.Equal(want, got) {
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
	{
		// Shift all elements left by 4 bits.
		res := v.ShiftAllLeft(four())
		got := make([]int16, res.Len())
		res.Store(got)
		if !slices.Equal(want, got) {
			t.Errorf("Wanted %v, got %v", want, got)
		}
	}
}

func TestReduceSum(t *testing.T) {
	test_helpers.TestV2S(t, test_helpers.Float32s(), simd.LoadFloat32s, simd.Float32s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Float64s(), simd.LoadFloat64s, simd.Float64s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Int32s(), simd.LoadInt32s, simd.Int32s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Uint32s(), simd.LoadUint32s, simd.Uint32s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Int16s(), simd.LoadInt16s, simd.Int16s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Uint16s(), simd.LoadUint16s, simd.Uint16s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Int8s(), simd.LoadInt8s, simd.Int8s.ReduceSum, test_helpers.ReduceSum)
	test_helpers.TestV2S(t, test_helpers.Uint8s(), simd.LoadUint8s, simd.Uint8s.ReduceSum, test_helpers.ReduceSum)
}
