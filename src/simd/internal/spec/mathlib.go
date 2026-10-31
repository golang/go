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

// subSaturated returns (x - y), saturated to the range of T.
func subSaturated[T Ints | Uints](x, y T) T {
	if isSigned[T]() {
		return saturateS[T](subSaturatedSSS64(int64(x), int64(y)))
	}
	diff, borrow := bits.Sub64(uint64(x), uint64(y), 0)
	if borrow > 0 {
		return minVal[T]()
	}
	return saturateU[T](diff)
}

func subSaturatedSSS64(x, y int64) int64 {
	sub := x - y

	// Overflow can only happen if x and y have different signs, and the result has a
	// different sign than x.
	//
	// (x ^ y) & (x ^ sub) checks if the sign bit of x matches neither y nor sub.
	if (x^y)&(x^sub) < 0 {
		if x >= 0 {
			return math.MaxInt64
		}
		return math.MinInt64
	}

	return sub
}

// scaleSaturated returns x * 2^s, saturated to the range of T.
func scaleSaturated[T Ints | Uints, S Ints | int](x T, s S) T {
	n := S(elemBits[T]())
	if s < 0 {
		if s <= -n {
			if isSigned[T]() && x < 0 {
				return ^T(0)
			}
			return 0
		}
		return x >> uint(-s)
	}
	if s >= n {
		if x == 0 {
			return 0
		}
		if isSigned[T]() && x < 0 {
			return minVal[T]()
		}
		return maxVal[T]()
	}
	res := x << uint(s)
	if res>>uint(s) != x {
		if isSigned[T]() && x < 0 {
			return minVal[T]()
		}
		return maxVal[T]()
	}
	return res
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

// clmul computes the carryless multiplication of a and b, returning the result
// split into high and low halves.
func clmul[E Uints](a, b E) (hi, lo E) {
	n := elemBits[E]()
	for i := range uint(n) {
		if (a>>i)&1 == 1 {
			// In Go, b >> n is 0 for an n-bit unsigned integer, so when i == 0,
			// b >> (n - 0) evaluates to 0 while b << 0 leaves b unshifted.
			lo ^= b << i
			hi ^= b >> (uint(n) - i)
		}
	}
	return
}

// fma32 returns x*y + z, computed with only one rounding.
func fma32(x, y, z float32) float32 {
	// Because float32 has a 24-bit significand and exponent in [-149, 127], the
	// product of two finite float32s has at most a 48-bit significand and
	// exponent in [-298, 255], which is always exact in float64.
	p := float64(x) * float64(y)
	zf := float64(z)
	s := p + zf
	// p+zf performs one 53-bit rounding, and a naive conversion to float32
	// performs a second. Specifically, if the true value of p + zf is
	// slightly off the midpoint between two float32s, p + zf in float64 can
	// round onto that exact float32 midpoint (whose low 28 float64
	// significand bits are 0, so bit 0 is even), and the subsequent float32
	// round-to-even conversion may round in the wrong direction. We
	// implement round-to-odd per Boldo & Melquiond to counteract this.
	if !math.IsInf(s, 0) && !math.IsNaN(s) && math.Float64bits(s)&1 == 0 {
		// s is even, so we need to nudge s to perform round-to-odd. Compute
		// the exact round-off error for the p+zf addition using Fast2Sum.
		hi, lo := p, zf
		if math.Abs(hi) < math.Abs(lo) {
			hi, lo = lo, hi
		}
		if err := (hi - s) + lo; err != 0 {
			// Nudge s by one float64 ULP toward odd. Choose the odd
			// neighbor in the direction of err to keep s on the same side
			// of the midpoint as the true value of p + zf.
			if (err > 0) == (s > 0) {
				s = math.Float64frombits(math.Float64bits(s) | 1)
			} else {
				s = math.Float64frombits(math.Float64bits(s) - 1)
			}
		}
	}
	return float32(s)
}
