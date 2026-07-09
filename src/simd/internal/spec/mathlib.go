// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
	"math"
	"math/bits"
)

func isSigned[T Ints | Uints]() bool {
	return T(0)-1 < 0
}

func maxVal[T Ints | Uints]() T {
	if isSigned[T]() {
		var zero T
		switch any(zero).(type) {
		case int8:
			return any(int8(math.MaxInt8)).(T)
		case int16:
			return any(int16(math.MaxInt16)).(T)
		case int32:
			return any(int32(math.MaxInt32)).(T)
		case int64:
			return any(int64(math.MaxInt64)).(T)
		}
		panic("unhandled type")
	}
	return ^T(0)
}

func minVal[T Ints | Uints]() T {
	if isSigned[T]() {
		return ^maxVal[T]()
	}
	return 0
}

// saturate converts x to type T, with saturation.
func saturate[T Ints | Uints, U Ints | Uints](x T) U {
	if isSigned[T]() {
		return saturateS[U](int64(x))
	}
	return saturateU[U](uint64(x))
}

// saturateS converts signed x to type T, with saturation.
func saturateS[T Ints | Uints](x int64) T {
	if int64(T(x)) == x && (x >= 0 || isSigned[T]()) {
		// It's in range.
		return T(x)
	}

	// Out of range
	if x > 0 {
		return maxVal[T]()
	}
	return minVal[T]()
}

// saturateU converts unsigned x to type T, with saturation.
func saturateU[T Ints | Uints](x uint64) T {
	if x < uint64(maxVal[T]()) {
		return T(x)
	}
	return maxVal[T]()
}

func addSaturated[T Ints | Uints](x, y T) T {
	if isSigned[T]() {
		return saturateS[T](addSaturatedSSS64(int64(x), int64(y)))
	}
	sum, carry := bits.Add64(uint64(x), uint64(y), 0)
	if carry > 0 {
		return maxVal[T]()
	}
	return saturateU[T](sum)
}

func addSaturatedSSS64(x, y int64) int64 {
	sum := x + y

	// Overflow can only happen if x and y have the same sign, and the sum has a
	// different sign.
	//
	// (x ^ sum) & (y ^ sum) checks if the sign bit of sum matches neither x nor y.
	if (x^sum)&(y^sum) < 0 {
		if x > 0 {
			return math.MaxInt64
		}
		return math.MinInt64
	}

	return sum
}

func mulSaturatedUSS[X Uints, Y Ints](x X, y Y) Y {
	// Expand to 64 bits and perform saturated multiplication
	z := mulSaturatedUSS64(uint64(x), int64(y))
	return saturateS[Y](z)
}

func mulSaturatedUSS64(x uint64, y int64) int64 {
	if x == 0 || y == 0 {
		return 0
	}

	// Get the absolute value of i as a uint64
	var absI uint64
	if y == math.MinInt64 {
		absI = math.MaxInt64 + 1
	} else if y < 0 {
		absI = uint64(-y)
	} else {
		absI = uint64(y)
	}

	// 128-bit multiplication
	hi, lo := bits.Mul64(x, absI)

	if y > 0 {
		// Positive result. Check for overflow.
		if hi > 0 || lo >= math.MaxInt64 {
			return math.MaxInt64
		}
		return int64(lo)
	} else {
		// Negative result. Check for underflow.
		if hi > 0 || lo >= uint64(math.MaxInt64)+1 {
			return math.MinInt64
		}
		return -int64(lo)
	}
}
