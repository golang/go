// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"math"
	"math/bits"
	"unsafe"
)

func rotl[T unsigned](x T, dist uint64) T {
	size := uint64(unsafe.Sizeof(x)) * 8
	dist = dist & (size - 1)
	if dist == 0 {
		return x
	}
	return (x << dist) | (x >> (size - dist))
}

func rotr[T unsigned](x T, dist uint64) T {
	size := uint64(unsafe.Sizeof(x)) * 8
	dist = dist & (size - 1)
	if dist == 0 {
		return x
	}
	return (x >> dist) | (x << (size - dist))
}

// rotlOfSlice returns a slice simulation of a left rotate
// of a specified distance.
func rotlOfSlice[T unsigned](dist uint64) func(x []T) []T {
	return map1[T](func(x T) T { return rotl(x, dist) })
}

// rotrOfSlice returns a slice simulation of a right rotate
// of a specified distance.
func rotrOfSlice[T unsigned](dist uint64) func(x []T) []T {
	return map1[T](func(x T) T { return rotr(x, dist) })
}

func curry2[T, U, V any](f func(T, U) V, y U) func(x T) V {
	return func(x T) V { return f(x, y) }
}

func curry1[T, U, V any](f func(T, U) V, x T) func(y U) V {
	return func(y U) V { return f(x, y) }
}

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

func isNaN[T float](x T) bool {
	return x != x
}

func abs[T number](x T) T {
	// TODO this will need a non-standard FP-equality test.
	if x == 0 { // true if x is -0.
		return 0 // this is not a negative zero
	}
	if x < 0 {
		return -x
	}
	return x
}

func neg[T number](x T) T {
	return -x
}

func onesCount[T integer](x T) T {
	size := uint64(unsafe.Sizeof(x)) * 8
	return T(bits.OnesCount64(uint64(x) & ((1 << size) - 1)))
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

func orNotI[T integer](x, y T) T {
	return x | ^y
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

func toUint8[T number](x T) uint8 {
	return uint8(x)
}

func toUint16[T number](x T) uint16 {
	return uint16(x)
}

func toUint64[T number](x T) uint64 {
	return uint64(x)
}

func toUint32[T number](x T) uint32 {
	return uint32(x)
}

func toInt8[T number](x T) int8 {
	return int8(x)
}

func toInt16[T number](x T) int16 {
	return int16(x)
}

func toInt32[T number](x T) int32 {
	return int32(x)
}

func toInt64[T number](x T) int64 {
	return int64(x)
}

func toFloat32[T number](x T) float32 {
	return float32(x)
}

func toFloat64[T number](x T) float64 {
	return float64(x)
}

// X86 specific behavior for conversion from float to int32.
// If the value cannot be represented as int32, it returns -0x80000000.
func floatToInt32_x86[T float](x T) int32 {
	switch y := (any(x)).(type) {
	case float32:
		if y != y || y < math.MinInt32 ||
			y >= math.MaxInt32 { // float32(MaxInt32) == 0x80000000, actually overflows
			return -0x80000000
		}
	case float64:
		if y != y || y < math.MinInt32 ||
			y > math.MaxInt32 { // float64(MaxInt32) is exact, no overflow
			return -0x80000000
		}
	}
	return int32(x)
}

// X86 specific behavior for conversion from float to int64.
// If the value cannot be represented as int64, it returns -0x80000000_00000000.
func floatToInt64_x86[T float](x T) int64 {
	switch y := (any(x)).(type) {
	case float32:
		if y != y || y < math.MinInt64 ||
			y >= math.MaxInt64 { // float32(MaxInt64) == 0x80000000_00000000, actually overflows
			return -0x80000000_00000000
		}
	case float64:
		if y != y || y < math.MinInt64 ||
			y >= math.MaxInt64 { // float64(MaxInt64) == 0x80000000_00000000, also overflows
			return -0x80000000_00000000
		}
	}
	return int64(x)
}

// X86 specific behavior for conversion from float to uint32.
// If the value cannot be represented as uint32, it returns 1<<32 - 1.
func floatToUint32_x86[T float](x T) uint32 {
	switch y := (any(x)).(type) {
	case float32:
		if y < 0 || y > math.MaxUint32 || y != y {
			return 1<<32 - 1
		}
	case float64:
		if y < 0 || y > math.MaxUint32 || y != y {
			return 1<<32 - 1
		}
	}
	return uint32(x)
}

// X86 specific behavior for conversion from float to uint64.
// If the value cannot be represented as uint64, it returns 1<<64 - 1.
func floatToUint64_x86[T float](x T) uint64 {
	switch y := (any(x)).(type) {
	case float32:
		if y < 0 || y > math.MaxUint64 || y != y {
			return 1<<64 - 1
		}
	case float64:
		if y < 0 || y > math.MaxUint64 || y != y {
			return 1<<64 - 1
		}
	}
	return uint64(x)
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

func orNotSlice[T integer](x, y []T) []T {
	return map2[T](orNotI)(x, y)
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

func isNaNSlice[T float](x []T) []int64 {
	return map1[T](func(x T) int64 {
		if isNaN(x) {
			return -1
		}
		return 0
	})(x)
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

// lanewiseSlice is the common helper for interleave, deinterleave, and transpose
// simulations. It handles lane computation, allocation, and iteration.
// laneBits is the lane size in bits (128 for NEON/x86 128-bit, 0 for whole-input/SVE).
// hi selects the half-lane offset (offHalf = 0 or half, for interleave hi/lo).
// odd selects the single-element offset (offOne = 0 or 1, for deinterleave/transpose odd/even).
// body receives (out, x, y, base, i, half, offHalf, offOne) for each pair within each lane
// and performs the operation-specific element assignment.
func lanewiseSlice[T number](laneBits int, hi bool, odd bool, body func(out, x, y []T, base, i, half, offHalf, offOne int)) func(x, y []T) []T {
	return func(x, y []T) []T {
		lane := laneBits / (8 * int(unsafe.Sizeof(x[0])))
		if lane == 0 || lane > len(x) {
			lane = len(x)
		}
		half := lane / 2
		offHalf := 0
		if hi {
			offHalf = half
		}
		offOne := 0
		if odd {
			offOne = 1
		}
		out := make([]T, len(x))
		for base := 0; base < len(x); base += lane {
			for i := 0; i < half; i++ {
				body(out, x, y, base, i, half, offHalf, offOne)
			}
		}
		return out
	}
}

func interleaveSlice[T number](laneBits int, hi bool) func(x, y []T) []T {
	return lanewiseSlice(laneBits, hi, false, func(out, x, y []T, base, i, half, offHalf, _ int) {
		out[base+2*i] = x[base+offHalf+i]
		out[base+2*i+1] = y[base+offHalf+i]
	})
}

func deinterleaveSlice[T number](laneBits int, odd bool) func(x, y []T) []T {
	return lanewiseSlice(laneBits, false, odd, func(out, x, y []T, base, i, half, _, offOne int) {
		out[base+i] = x[base+2*i+offOne]
		out[base+half+i] = y[base+2*i+offOne]
	})
}

func transposeSlice[T number](laneBits int, odd bool) func(x, y []T) []T {
	return lanewiseSlice(laneBits, false, odd, func(out, x, y []T, base, i, half, _, offOne int) {
		out[base+2*i] = x[base+2*i+offOne]
		out[base+2*i+1] = y[base+2*i+offOne]
	})
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

// reduceSlice reduces x using fn as the combining operation.
func reduceSlice[T number](x []T, fn func(a, b T) T) T {
	acc := x[0]
	for _, v := range x[1:] {
		acc = fn(acc, v)
	}
	return acc
}

func satToInt8[T integer](x T) int8 {
	var m int8 = -128
	var M int8 = 127
	if T(M) < T(m) { // expecting T being a larger type
		panic("bad input type")
	}
	if x < T(m) {
		return m
	}
	if x > T(M) {
		return M
	}
	return int8(x)
}

func satToUint8[T integer](x T) uint8 {
	var M uint8 = 255
	if T(M) < 0 { // expecting T being a larger type
		panic("bad input type")
	}
	if x < 0 {
		return 0
	}
	if x > T(M) {
		return M
	}
	return uint8(x)
}

func satToInt16[T integer](x T) int16 {
	var m int16 = -32768
	var M int16 = 32767
	if T(M) < T(m) { // expecting T being a larger type
		panic("bad input type")
	}
	if x < T(m) {
		return m
	}
	if x > T(M) {
		return M
	}
	return int16(x)
}

func satToUint16[T integer](x T) uint16 {
	var M uint16 = 65535
	if T(M) < 0 { // expecting T being a larger type
		panic("bad input type")
	}
	if x < 0 {
		return 0
	}
	if x > T(M) {
		return M
	}
	return uint16(x)
}

func satToInt32[T integer](x T) int32 {
	var m int32 = -1 << 31
	var M int32 = 1<<31 - 1
	if T(M) < T(m) { // expecting T being a larger type
		panic("bad input type")
	}
	if x < T(m) {
		return m
	}
	if x > T(M) {
		return M
	}
	return int32(x)
}

func satToUint32[T integer](x T) uint32 {
	var M uint32 = 1<<32 - 1
	if T(M) < 0 { // expecting T being a larger type
		panic("bad input type")
	}
	if x < 0 {
		return 0
	}
	if x > T(M) {
		return M
	}
	return uint32(x)
}

// shiftAmount extracts the signed shift amount from the least significant byte of s.
// ARM64 SSHL/USHL use only bits [7:0] of the shift amount element, sign-extended.
func shiftAmount[T integer](s T) int8 {
	return int8(uint8(s))
}

// shiftBy shifts x by signed amount: positive = left, negative = right.
func shiftBy[T integer](x T, amt int8) T {
	a := int(amt)
	if a > 0 {
		return x << uint(a)
	}
	if a < 0 {
		return x >> uint(-a)
	}
	return x
}

// shiftSaturatingSigned shifts x by signed amount with signed saturation on overflow.
func shiftSaturatingSigned[T signed](x T, amt int8) T {
	a := int(amt)
	if a > 0 {
		r := x << uint(a)
		if r>>uint(a) != x { // overflow
			bits := uint(unsafe.Sizeof(x)) * 8
			if x >= 0 {
				return ^T(0) ^ (T(1) << (bits - 1)) // MaxSigned
			}
			return T(1) << (bits - 1) // MinSigned
		}
		return r
	}
	if a < 0 {
		return x >> uint(-a)
	}
	return x
}

// shiftSaturatingUnsigned shifts x by signed amount with unsigned saturation on overflow.
func shiftSaturatingUnsigned[T unsigned](x T, amt int8) T {
	a := int(amt)
	if a > 0 {
		r := x << uint(a)
		if r>>uint(a) != x { // overflow
			return ^T(0) // MaxUnsigned
		}
		return r
	}
	if a < 0 {
		return x >> uint(-a)
	}
	return x
}

// Slice versions for shift operations

// shiftSlice applies shiftBy element-wise using same-type slices.
func shiftSlice[T integer](x, y []T) []T {
	return map2(func(a, b T) T { return shiftBy(a, shiftAmount(b)) })(x, y)
}

// shiftMixedSlice applies shiftBy element-wise using mixed-type slices (unsigned data, signed amounts).
func shiftMixedSlice[D integer, S integer](x []D, y []S) []D {
	r := make([]D, len(x))
	for i := range r {
		r[i] = shiftBy(x[i], shiftAmount(y[i]))
	}
	return r
}

// shiftSaturatingSignedSlice applies saturating shift element-wise (same-type).
func shiftSaturatingSignedSlice[T signed](x, y []T) []T {
	return map2(func(a, b T) T { return shiftSaturatingSigned(a, shiftAmount(b)) })(x, y)
}

// shiftSaturatingUnsignedSlice applies saturating shift element-wise (mixed-type).
func shiftSaturatingUnsignedSlice[D unsigned, S integer](x []D, y []S) []D {
	r := make([]D, len(x))
	for i := range r {
		r[i] = shiftSaturatingUnsigned(x[i], shiftAmount(y[i]))
	}
	return r
}

// Slice versions for const shift operations (same constant amount for all elements)

// shiftLeftByConstSlice shifts all elements left by constant amount.
func shiftLeftByConstSlice[T integer](x []T, amt uint64) []T {
	return map1(func(a T) T { return a << amt })(x)
}

// shiftRightByConstSlice shifts all elements right by constant amount.
// Signed types use arithmetic shift, unsigned types use logical shift.
func shiftRightByConstSlice[T integer](x []T, amt uint64) []T {
	return map1(func(a T) T { return a >> amt })(x)
}

// shiftLeftSaturatingByConstSlice shifts all elements left by constant amount with signed saturation.
func shiftLeftSaturatingByConstSlice[T signed](x []T, amt uint64) []T {
	return map1(func(a T) T { return shiftSaturatingSigned(a, int8(amt)) })(x)
}

// shiftLeftSaturatingUByConstSlice shifts all elements left by constant amount with unsigned saturation.
func shiftLeftSaturatingUByConstSlice[T unsigned](x []T, amt uint64) []T {
	return map1(func(a T) T { return shiftSaturatingUnsigned(a, int8(amt)) })(x)
}

// shiftAllLeftSlice shifts all elements left by the same amount.
func shiftAllLeftSlice[T integer](x []T, amt uint64) []T {
	return map1(func(a T) T { return a << amt })(x)
}

// shiftAllRightSlice shifts all elements right by the same amount.
// Signed types use arithmetic shift, unsigned types use logical shift.
func shiftAllRightSlice[T integer](x []T, amt uint64) []T {
	return map1(func(a T) T { return a >> amt })(x)
}

// ARM64-specific float-to-int conversion saturation helpers.
// ARM64 uses IEEE 754 saturation: out-of-range values clamp to min/max of the target type.
// NaN converts to 0. Negative values convert to 0 for unsigned types.

func floatToInt32_arm64[T float](x T) int32 {
	if x != x { // NaN
		return 0
	}
	if x >= math.MaxInt32 {
		return math.MaxInt32
	}
	if x < math.MinInt32 {
		return math.MinInt32
	}
	return int32(x)
}

func floatToInt64_arm64[T float](x T) int64 {
	if x != x { // NaN
		return 0
	}
	if x >= math.MaxInt64 {
		return math.MaxInt64
	}
	if x < math.MinInt64 {
		return math.MinInt64
	}
	return int64(x)
}

func floatToUint32_arm64[T float](x T) uint32 {
	if x != x || x < 0 { // NaN or negative
		return 0
	}
	if x >= math.MaxUint32 {
		return math.MaxUint32
	}
	return uint32(x)
}

func floatToUint64_arm64[T float](x T) uint64 {
	if x != x || x < 0 { // NaN or negative
		return 0
	}
	if x >= math.MaxUint64 {
		return math.MaxUint64
	}
	return uint64(x)
}
