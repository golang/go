// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package simd_test

import (
	"math"
	"math/rand"
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

	test_helpers.TestV2Ve(t, values, load, simd.Int8s.OnesCount, test_helpers.OnesCount)

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
	// Include the extremes: Abs(MinInt64) is 1<<63.
	test_helpers.TestV2Ve(t, append(slices.Clone(values), -1<<63, 1<<63-1), load, simd.Int64s.Abs, test_helpers.Abs)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Add, test_helpers.Add)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Sub, test_helpers.Sub)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.And, test_helpers.And)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Or, test_helpers.Or)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.Xor, test_helpers.Xor)
	test_helpers.TestVV2Ve(t, values, load, simd.Int64s.AndNot, test_helpers.AndNot)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int64s.ShiftAllLeft, test_helpers.ShiftLeft)
	test_helpers.TestVR2Ve(t, values, shifts, load, simd.Int64s.ShiftAllRight, test_helpers.ShiftRight)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.Equal, test_helpers.Equal)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.NotEqual, test_helpers.NotEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Int64s.GreaterEqual, test_helpers.GreaterEqual)

	// TODO: untested methods: ConvertToUint64, IfElse, Len, Masked, Not, Store,
	// StorePart, String, ToBits, ToMask.
}

func TestMask8s(t *testing.T) {
	load := simd.LoadInt8s
	test_helpers.TestMM2Me(t, load, simd.Mask8s.And, test_helpers.And)
	test_helpers.TestMM2Me(t, load, simd.Mask8s.Or, test_helpers.Or)
	test_helpers.TestMM2Me(t, load, simd.Mask8s.Xor, test_helpers.Xor)
	test_helpers.TestMM2Me(t, load, simd.Mask8s.AndNot, test_helpers.AndNot)
	test_helpers.TestM2Me(t, load, simd.Mask8s.Not, test_helpers.Not)
}

func TestMask16s(t *testing.T) {
	load := simd.LoadInt16s
	test_helpers.TestMM2Me(t, load, simd.Mask16s.And, test_helpers.And)
	test_helpers.TestMM2Me(t, load, simd.Mask16s.Or, test_helpers.Or)
	test_helpers.TestMM2Me(t, load, simd.Mask16s.Xor, test_helpers.Xor)
	test_helpers.TestMM2Me(t, load, simd.Mask16s.AndNot, test_helpers.AndNot)
	test_helpers.TestM2Me(t, load, simd.Mask16s.Not, test_helpers.Not)
}

func TestMask32s(t *testing.T) {
	load := simd.LoadInt32s
	test_helpers.TestMM2Me(t, load, simd.Mask32s.And, test_helpers.And)
	test_helpers.TestMM2Me(t, load, simd.Mask32s.Or, test_helpers.Or)
	test_helpers.TestMM2Me(t, load, simd.Mask32s.Xor, test_helpers.Xor)
	test_helpers.TestMM2Me(t, load, simd.Mask32s.AndNot, test_helpers.AndNot)
	test_helpers.TestM2Me(t, load, simd.Mask32s.Not, test_helpers.Not)
}

func TestMask64s(t *testing.T) {
	load := simd.LoadInt64s
	test_helpers.TestMM2Me(t, load, simd.Mask64s.And, test_helpers.And)
	test_helpers.TestMM2Me(t, load, simd.Mask64s.Or, test_helpers.Or)
	test_helpers.TestMM2Me(t, load, simd.Mask64s.Xor, test_helpers.Xor)
	test_helpers.TestMM2Me(t, load, simd.Mask64s.AndNot, test_helpers.AndNot)
	test_helpers.TestM2Me(t, load, simd.Mask64s.Not, test_helpers.Not)
}

func TestUint8s(t *testing.T) {
	values := test_helpers.Uint8s()
	load := simd.LoadUint8s

	test_helpers.TestV2Ve(t, values, load, simd.Uint8s.OnesCount, test_helpers.OnesCount)

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
	test_helpers.TestVV2Me(t, values, load, simd.Uint8s.Less, test_helpers.Less)
	test_helpers.TestVV2Me(t, values, load, simd.Uint8s.LessEqual, test_helpers.LessEqual)
	test_helpers.TestVV2Me(t, values, load, simd.Uint8s.Greater, test_helpers.Greater)
	test_helpers.TestVV2Me(t, values, load, simd.Uint8s.GreaterEqual, test_helpers.GreaterEqual)

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
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat32s(), load, simd.Float32s.Round, test_helpers.Round)
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat32s(), load, simd.Float32s.Ceil, test_helpers.Ceil)
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat32s(), load, simd.Float32s.Floor, test_helpers.Floor)
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat32s(), load, simd.Float32s.Trunc, test_helpers.Trunc)
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
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat64s(), load, simd.Float64s.Round, test_helpers.Round)
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat64s(), load, simd.Float64s.Ceil, test_helpers.Ceil)
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat64s(), load, simd.Float64s.Floor, test_helpers.Floor)
	test_helpers.TestV2Ve(t, test_helpers.RoundingFloat64s(), load, simd.Float64s.Trunc, test_helpers.Trunc)
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

	// TODO: untested methods: BitsToFloat64, BitsToInt64, CarrylessMulEven,
	// CarrylessMulOdd, ConvertToInt64, IfElse, Len, Masked, Not,
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

func testHiToLo[E number, V interface {
	HasStoreLen[E]
	HiToLo() V
}](t *testing.T, load func([]E) V) {
	t.Helper()
	var zero V
	n := zero.Len()
	x := make([]E, n)
	for i := range x {
		x[i] = E(i + 1)
	}
	got := make([]E, n)
	load(x).HiToLo().Store(got)
	want := make([]E, n)
	copy(want, x[n/2:])
	if !slices.Equal(want, got) {
		t.Errorf("%T.HiToLo(%v): want %v, got %v", zero, x, want, got)
	}
}

func TestHiToLo(t *testing.T) {
	testHiToLo(t, simd.LoadInt8s)
	testHiToLo(t, simd.LoadInt16s)
	testHiToLo(t, simd.LoadInt32s)
	testHiToLo(t, simd.LoadInt64s)
	testHiToLo(t, simd.LoadUint8s)
	testHiToLo(t, simd.LoadUint16s)
	testHiToLo(t, simd.LoadUint32s)
	testHiToLo(t, simd.LoadUint64s)
	testHiToLo(t, simd.LoadFloat32s)
	testHiToLo(t, simd.LoadFloat64s)
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

func TestReduceMax(t *testing.T) {
	test_helpers.TestV2S(t, test_helpers.Int8s(), simd.LoadInt8s, simd.Int8s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Int16s(), simd.LoadInt16s, simd.Int16s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Int32s(), simd.LoadInt32s, simd.Int32s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Int64s(), simd.LoadInt64s, simd.Int64s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Uint8s(), simd.LoadUint8s, simd.Uint8s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Uint16s(), simd.LoadUint16s, simd.Uint16s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Uint32s(), simd.LoadUint32s, simd.Uint32s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Uint64s(), simd.LoadUint64s, simd.Uint64s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Float32s(), simd.LoadFloat32s, simd.Float32s.ReduceMax, test_helpers.ReduceMax)
	test_helpers.TestV2S(t, test_helpers.Float64s(), simd.LoadFloat64s, simd.Float64s.ReduceMax, test_helpers.ReduceMax)
}

func TestReduceMin(t *testing.T) {
	test_helpers.TestV2S(t, test_helpers.Int8s(), simd.LoadInt8s, simd.Int8s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Int16s(), simd.LoadInt16s, simd.Int16s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Int32s(), simd.LoadInt32s, simd.Int32s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Int64s(), simd.LoadInt64s, simd.Int64s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Uint8s(), simd.LoadUint8s, simd.Uint8s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Uint16s(), simd.LoadUint16s, simd.Uint16s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Uint32s(), simd.LoadUint32s, simd.Uint32s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Uint64s(), simd.LoadUint64s, simd.Uint64s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Float32s(), simd.LoadFloat32s, simd.Float32s.ReduceMin, test_helpers.ReduceMin)
	test_helpers.TestV2S(t, test_helpers.Float64s(), simd.LoadFloat64s, simd.Float64s.ReduceMin, test_helpers.ReduceMin)
}

func TestMaskRange(t *testing.T) {
	test_helpers.TestMaskAllAny(t, simd.LoadUint8s)
}

func TestMask16Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, simd.LoadUint16s)
}

func TestMask32Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, simd.LoadUint32s)
}

func TestMask64Range(t *testing.T) {
	test_helpers.TestMaskAllAny(t, simd.LoadUint64s)
}

func testMaskTrailingZeros[E integer, V interface {
	Len() int
	Equal(V) M
	NotEqual(V) M
}, M interface {
	TrailingZeros() int
}](t *testing.T, load func([]E) V) {
	t.Helper()
	var zero V
	lanes := zero.Len()
	vz := load(make([]E, lanes))

	if got := vz.NotEqual(vz).TrailingZeros(); got != lanes {
		t.Errorf("all-zeros TrailingZeros(): want %d, got %d", lanes, got)
	}
	if got := vz.Equal(vz).TrailingZeros(); got != 0 {
		t.Errorf("all-ones TrailingZeros(): want 0, got %d", got)
	}

	for i := range lanes {
		// Single lane i set
		s := make([]E, lanes)
		s[i] = 1
		m := load(s).NotEqual(vz)
		if got := m.TrailingZeros(); got != i {
			t.Errorf("single lane %d TrailingZeros(): want %d, got %d", i, i, got)
		}

		// All lanes >= i set
		for j := i; j < lanes; j++ {
			s[j] = 1
		}
		m = load(s).NotEqual(vz)
		if got := m.TrailingZeros(); got != i {
			t.Errorf("lanes >= %d TrailingZeros(): want %d, got %d", i, i, got)
		}
	}
}

func TestMaskTrailingZeros(t *testing.T) {
	testMaskTrailingZeros(t, simd.LoadUint8s)
	testMaskTrailingZeros(t, simd.LoadUint16s)
	testMaskTrailingZeros(t, simd.LoadUint32s)
	testMaskTrailingZeros(t, simd.LoadUint64s)
}

func TestLookupOrZeroInt8s(t *testing.T) {
	var z simd.Int8s
	x := make([]int8, z.Len())
	indices := make([]uint8, z.Len())

	for i := range z.Len() {
		a, b := rand.Int31n(256), rand.Int31n(256)
		x[i] = int8(a - 128)
		indices[i] = uint8(b)
	}

	v := simd.LoadInt8s(x)
	u := simd.LoadUint8s(indices)

	got := make([]int8, v.Len())
	want := make([]int8, v.Len())
	for i := range want {
		j := indices[i]
		if int(j) < len(want) {
			want[i] = x[j]
		}
	}
	v.LookupOrZero(u).StorePart(got)
	if !slices.Equal(want, got) {
		t.Errorf("Wanted %v, got %v", want, got)
	}
}

func TestStoreShort(t *testing.T) {
	// Store panics if the slice is shorter than the vector, and StorePart
	// stores as much as fits.
	x := simd.BroadcastInt8s(1)
	s := make([]int8, x.Len()-1)
	if n := x.StorePart(s); n != len(s) || slices.ContainsFunc(s, func(e int8) bool { return e != 1 }) {
		t.Errorf("StorePart into %d elements returned %d and stored %v", len(s), n, s)
	}
	defer func() {
		if recover() == nil {
			t.Errorf("Store into %d elements of a %d-element vector didn't panic", len(s), x.Len())
		}
	}()
	x.Store(s)
}

func TestStoreShortFloat64s(t *testing.T) {
	x := simd.BroadcastFloat64s(1)
	s := make([]float64, x.Len()-1)
	if n := x.StorePart(s); n != len(s) || slices.ContainsFunc(s, func(e float64) bool { return e != 1 }) {
		t.Errorf("StorePart into %d elements returned %d and stored %v", len(s), n, s)
	}
	defer func() {
		if recover() == nil {
			t.Errorf("Store into %d elements of a %d-element vector didn't panic", len(s), x.Len())
		}
	}()
	x.Store(s)
}

func TestLookupOrZeroUint8s(t *testing.T) {
	var z simd.Uint8s
	x := make([]uint8, z.Len())
	indices := make([]uint8, z.Len())

	for i := range z.Len() {
		a, b := rand.Int31n(256), rand.Int31n(256)
		x[i] = uint8(a)
		indices[i] = uint8(b)
	}

	v := simd.LoadUint8s(x)
	u := simd.LoadUint8s(indices)

	got := make([]uint8, v.Len())
	want := make([]uint8, v.Len())
	for i := range want {
		j := indices[i]
		if int(j) < len(want) {
			want[i] = x[j]
		}
	}
	v.LookupOrZero(u).StorePart(got)
	if !slices.Equal(want, got) {
		t.Errorf("Wanted %v, got %v", want, got)
	}
}

func TestDotProductPairs(t *testing.T) {
	vals := test_helpers.Int16s()
	n := (simd.Int16s{}).Len()
	nOut := (simd.Int32s{}).Len()
	got := make([]int32, nOut)
	want := make([]int32, nOut)

	for i := 0; i+n <= len(vals); i += n {
		x := vals[i : i+n]
		for j := 0; j+n <= len(vals); j += n {
			y := vals[j : j+n]
			simd.LoadInt16s(x).DotProductPairs(simd.LoadInt16s(y)).Store(got)
			for k := range nOut {
				want[k] = int32(x[2*k])*int32(y[2*k]) + int32(x[2*k+1])*int32(y[2*k+1])
			}
			if !slices.Equal(got, want) {
				t.Fatalf("DotProductPairs(%v, %v) = %v, want %v", x, y, got, want)
			}
		}
	}

	// Test the overflow case where all four inputs to a sum are MinInt16,
	// wrapping to MinInt32, alongside the largest non-overflowing sum.
	m := int16(math.MinInt16)
	x := make([]int16, n)
	y := make([]int16, n)
	for i := range n {
		x[i] = m
		y[i] = m
	}
	y[3] = m + 1
	x[4], x[5] = 1, 2
	y[4], y[5] = 3, 4
	simd.LoadInt16s(x).DotProductPairs(simd.LoadInt16s(y)).Store(got)
	for k := range nOut {
		want[k] = int32(x[2*k])*int32(y[2*k]) + int32(x[2*k+1])*int32(y[2*k+1])
	}
	if !slices.Equal(got, want) {
		t.Fatalf("DotProductPairs overflow(%v, %v) = %v, want %v", x, y, got, want)
	}
}

func testTwoInputShuffles[E ~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64, V interface {
	Len() int
	Store([]E)
	InterleaveLo(V) V
	InterleaveHi(V) V
	ConcatEven(V) V
	ConcatOdd(V) V
}](t *testing.T, name string, load func([]E) V) {
	t.Helper()
	var zero V
	n := zero.Len()
	half := n / 2
	xs := make([]E, n)
	ys := make([]E, n)
	for i := range n {
		xs[i] = E(i + 1)
		ys[i] = E(n + i + 1)
	}
	xv := load(xs)
	yv := load(ys)

	check := func(op string, gotV V, want []E) {
		t.Helper()
		got := make([]E, n)
		gotV.Store(got)
		if !slices.Equal(got, want) {
			t.Errorf("%s.%s: got %v, want %v", name, op, got, want)
		}
	}

	wantLo := make([]E, n)
	wantHi := make([]E, n)
	wantEven := make([]E, n)
	wantOdd := make([]E, n)
	for i := range half {
		wantLo[2*i] = xs[i]
		wantLo[2*i+1] = ys[i]
		wantHi[2*i] = xs[half+i]
		wantHi[2*i+1] = ys[half+i]
		wantEven[i] = xs[2*i]
		wantEven[half+i] = ys[2*i]
		wantOdd[i] = xs[2*i+1]
		wantOdd[half+i] = ys[2*i+1]
	}

	check("InterleaveLo", xv.InterleaveLo(yv), wantLo)
	check("InterleaveHi", xv.InterleaveHi(yv), wantHi)
	check("ConcatEven", xv.ConcatEven(yv), wantEven)
	check("ConcatOdd", xv.ConcatOdd(yv), wantOdd)
}

func TestTwoInputShuffles(t *testing.T) {
	testTwoInputShuffles(t, "Int8s", simd.LoadInt8s)
	testTwoInputShuffles(t, "Int16s", simd.LoadInt16s)
	testTwoInputShuffles(t, "Int32s", simd.LoadInt32s)
	testTwoInputShuffles(t, "Int64s", simd.LoadInt64s)
	testTwoInputShuffles(t, "Uint8s", simd.LoadUint8s)
	testTwoInputShuffles(t, "Uint16s", simd.LoadUint16s)
	testTwoInputShuffles(t, "Uint32s", simd.LoadUint32s)
	testTwoInputShuffles(t, "Uint64s", simd.LoadUint64s)
}
