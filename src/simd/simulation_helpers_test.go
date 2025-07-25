// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"math"
)

func less[T number](x, y T) bool {
	return x < y
}
func lessEqual[T number](x, y T) bool {
	return x <= y
}
func greater[T number](x, y T) bool {
	return x > y
}
func greaterEqual[T number](x, y T) bool {
	return x >= y
}
func equal[T number](x, y T) bool {
	return x == y
}
func notEqual[T number](x, y T) bool {
	return x != y
}

func abs[T number](x T) T {
	// TODO this will need a non-standard FP-equality test.
	if x == 0 { // true if x is -0.
		return x // this is not a negative zero
	}
	if x < 0 {
		return -x
	}
	return x
}

func ceil[T float](x T) T {
	return T(math.Ceil(float64(x)))
}
func floor[T float](x T) T {
	return T(math.Floor(float64(x)))
}
func not[T integer](x T) T {
	return ^x
}
func round[T float](x T) T {
	return T(math.RoundToEven(float64(x)))
}
func sqrt[T float](x T) T {
	return T(math.Sqrt(float64(x)))
}
func trunc[T float](x T) T {
	return T(math.Trunc(float64(x)))
}

func add[T number](x, y T) T {
	return x + y
}

func sub[T number](x, y T) T {
	return x - y
}

func max_[T number](x, y T) T { // "max" lands in infinite recursion
	return max(x, y)
}

func min_[T number](x, y T) T { // "min" lands in infinite recursion
	return min(x, y)
}

// Also mulLow for integers
func mul[T number](x, y T) T {
	return x * y
}

func div[T number](x, y T) T {
	return x / y
}

func and[T integer](x, y T) T {
	return x & y
}

func andNotI[T integer](x, y T) T {
	return x & ^y // order corrected to match expectations
}

func orI[T integer](x, y T) T {
	return x | y
}

func xorI[T integer](x, y T) T {
	return x ^ y
}

func ima[T integer](x, y, z T) T {
	return x*y + z
}

func fma[T float](x, y, z T) T {
	return T(math.FMA(float64(x), float64(y), float64(z)))
}

func toInt32[T number](x T) int32 {
	return int32(x)
}

func toUint32[T number](x T) uint32 {
	switch y := (any(x)).(type) {
	case float32:
		if y < 0 || y > float32(math.MaxUint32) || y != y {
			return math.MaxUint32
		}
	case float64:
		if y < 0 || y > float64(math.MaxUint32) || y != y {
			return math.MaxUint32
		}
	}
	return uint32(x)
}

func ceilResidueForPrecision[T float](i int) func(T) T {
	f := 1.0
	for i > 0 {
		f *= 2
		i--
	}
	return func(x T) T {
		y := float64(x)
		if math.IsInf(float64(x*T(f)), 0) {
			return 0
		}
		// TODO sort out the rounding issues when T === float32
		return T(y - math.Ceil(y*f)/f)
	}
}

// Slice versions of all these elementwise operations

func addSlice[T number](x, y []T) []T {
	return map2[T](add)(x, y)
}

func subSlice[T number](x, y []T) []T {
	return map2[T](sub)(x, y)
}

func maxSlice[T number](x, y []T) []T {
	return map2[T](max_)(x, y)
}

func minSlice[T number](x, y []T) []T {
	return map2[T](min_)(x, y)
}

// mulLow for integers
func mulSlice[T number](x, y []T) []T {
	return map2[T](mul)(x, y)
}

func divSlice[T number](x, y []T) []T {
	return map2[T](div)(x, y)
}

func andSlice[T integer](x, y []T) []T {
	return map2[T](and)(x, y)
}

func andNotSlice[T integer](x, y []T) []T {
	return map2[T](andNotI)(x, y)
}

func orSlice[T integer](x, y []T) []T {
	return map2[T](orI)(x, y)
}

func xorSlice[T integer](x, y []T) []T {
	return map2[T](xorI)(x, y)
}

func lessSlice[T number](x, y []T) []int64 {
	return mapCompare[T](less)(x, y)
}

func lessEqualSlice[T number](x, y []T) []int64 {
	return mapCompare[T](lessEqual)(x, y)
}

func greaterSlice[T number](x, y []T) []int64 {
	return mapCompare[T](greater)(x, y)
}

func greaterEqualSlice[T number](x, y []T) []int64 {
	return mapCompare[T](greaterEqual)(x, y)
}

func equalSlice[T number](x, y []T) []int64 {
	return mapCompare[T](equal)(x, y)
}

func notEqualSlice[T number](x, y []T) []int64 {
	return mapCompare[T](notEqual)(x, y)
}

func ceilSlice[T float](x []T) []T {
	return map1[T](ceil)(x)
}

func floorSlice[T float](x []T) []T {
	return map1[T](floor)(x)
}

func notSlice[T integer](x []T) []T {
	return map1[T](not)(x)
}

func roundSlice[T float](x []T) []T {
	return map1[T](round)(x)
}

func sqrtSlice[T float](x []T) []T {
	return map1[T](sqrt)(x)
}

func truncSlice[T float](x []T) []T {
	return map1[T](trunc)(x)
}

func imaSlice[T integer](x, y, z []T) []T {
	return map3[T](ima)(x, y, z)
}

func fmaSlice[T float](x, y, z []T) []T {
	return map3[T](fma)(x, y, z)
}

func toInt32Slice[T number](x []T) []int32 {
	return map1[T](toInt32)(x)
}

func toUint32Slice[T number](x []T) []uint32 {
	return map1[T](toUint32)(x)
}
