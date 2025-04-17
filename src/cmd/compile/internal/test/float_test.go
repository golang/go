// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"math"
	"testing"
)

//go:noinline
func compare1(a, b float64) bool {
	return a < b
}

//go:noinline
func compare2(a, b float32) bool {
	return a < b
}

func TestFloatCompare(t *testing.T) {
	if !compare1(3, 5) {
		t.Errorf("compare1 returned false")
	}
	if !compare2(3, 5) {
		t.Errorf("compare2 returned false")
	}
}

func TestFloatCompareFolded(t *testing.T) {
	// float64 comparisons
	d1, d3, d5, d9 := float64(1), float64(3), float64(5), float64(9)
	if d3 == d5 {
		t.Errorf("d3 == d5 returned true")
	}
	if d3 != d3 {
		t.Errorf("d3 != d3 returned true")
	}
	if d3 > d5 {
		t.Errorf("d3 > d5 returned true")
	}
	if d3 >= d9 {
		t.Errorf("d3 >= d9 returned true")
	}
	if d5 < d1 {
		t.Errorf("d5 < d1 returned true")
	}
	if d9 <= d1 {
		t.Errorf("d9 <= d1 returned true")
	}
	if math.NaN() == math.NaN() {
		t.Errorf("math.NaN() == math.NaN() returned true")
	}
	if math.NaN() >= math.NaN() {
		t.Errorf("math.NaN() >= math.NaN() returned true")
	}
	if math.NaN() <= math.NaN() {
		t.Errorf("math.NaN() <= math.NaN() returned true")
	}
	if math.Copysign(math.NaN(), -1) < math.NaN() {
		t.Errorf("math.Copysign(math.NaN(), -1) < math.NaN() returned true")
	}
	if math.Inf(1) != math.Inf(1) {
		t.Errorf("math.Inf(1) != math.Inf(1) returned true")
	}
	if math.Inf(-1) != math.Inf(-1) {
		t.Errorf("math.Inf(-1) != math.Inf(-1) returned true")
	}
	if math.Copysign(0, -1) != 0 {
		t.Errorf("math.Copysign(0, -1) != 0 returned true")
	}
	if math.Copysign(0, -1) < 0 {
		t.Errorf("math.Copysign(0, -1) < 0 returned true")
	}
	if 0 > math.Copysign(0, -1) {
		t.Errorf("0 > math.Copysign(0, -1) returned true")
	}

	// float32 comparisons
	s1, s3, s5, s9 := float32(1), float32(3), float32(5), float32(9)
	if s3 == s5 {
		t.Errorf("s3 == s5 returned true")
	}
	if s3 != s3 {
		t.Errorf("s3 != s3 returned true")
	}
	if s3 > s5 {
		t.Errorf("s3 > s5 returned true")
	}
	if s3 >= s9 {
		t.Errorf("s3 >= s9 returned true")
	}
	if s5 < s1 {
		t.Errorf("s5 < s1 returned true")
	}
	if s9 <= s1 {
		t.Errorf("s9 <= s1 returned true")
	}
	sPosNaN, sNegNaN := float32(math.NaN()), float32(math.Copysign(math.NaN(), -1))
	if sPosNaN == sPosNaN {
		t.Errorf("sPosNaN == sPosNaN returned true")
	}
	if sPosNaN >= sPosNaN {
		t.Errorf("sPosNaN >= sPosNaN returned true")
	}
	if sPosNaN <= sPosNaN {
		t.Errorf("sPosNaN <= sPosNaN returned true")
	}
	if sNegNaN < sPosNaN {
		t.Errorf("sNegNaN < sPosNaN returned true")
	}
	sPosInf, sNegInf := float32(math.Inf(1)), float32(math.Inf(-1))
	if sPosInf != sPosInf {
		t.Errorf("sPosInf != sPosInf returned true")
	}
	if sNegInf != sNegInf {
		t.Errorf("sNegInf != sNegInf returned true")
	}
	sNegZero := float32(math.Copysign(0, -1))
	if sNegZero != 0 {
		t.Errorf("sNegZero != 0 returned true")
	}
	if sNegZero < 0 {
		t.Errorf("sNegZero < 0 returned true")
	}
	if 0 > sNegZero {
		t.Errorf("0 > sNegZero returned true")
	}
}

//go:noinline
func cvt1(a float64) uint64 {
	return uint64(a)
}

//go:noinline
func cvt2(a float64) uint32 {
	return uint32(a)
}

//go:noinline
func cvt3(a float32) uint64 {
	return uint64(a)
}

//go:noinline
func cvt4(a float32) uint32 {
	return uint32(a)
}

//go:noinline
func cvt5(a float64) int64 {
	return int64(a)
}

//go:noinline
func cvt6(a float64) int32 {
	return int32(a)
}

//go:noinline
func cvt7(a float32) int64 {
	return int64(a)
}

//go:noinline
func cvt8(a float32) int32 {
	return int32(a)
}

// make sure to cover int, uint cases (issue #16738)
//
//go:noinline
func cvt9(a float64) int {
	return int(a)
}

//go:noinline
func cvt10(a float64) uint {
	return uint(a)
}

//go:noinline
func cvt11(a float32) int {
	return int(a)
}

//go:noinline
func cvt12(a float32) uint {
	return uint(a)
}

//go:noinline
func f2i64p(v float64) *int64 {
	return ip64(int64(v / 0.1))
}

//go:noinline
func ip64(v int64) *int64 {
	return &v
}

func TestFloatConvert(t *testing.T) {
	if got := cvt1(3.5); got != 3 {
		t.Errorf("cvt1 got %d, wanted 3", got)
	}
	if got := cvt2(3.5); got != 3 {
		t.Errorf("cvt2 got %d, wanted 3", got)
	}
	if got := cvt3(3.5); got != 3 {
		t.Errorf("cvt3 got %d, wanted 3", got)
	}
	if got := cvt4(3.5); got != 3 {
		t.Errorf("cvt4 got %d, wanted 3", got)
	}
	if got := cvt5(3.5); got != 3 {
		t.Errorf("cvt5 got %d, wanted 3", got)
	}
	if got := cvt6(3.5); got != 3 {
		t.Errorf("cvt6 got %d, wanted 3", got)
	}
	if got := cvt7(3.5); got != 3 {
		t.Errorf("cvt7 got %d, wanted 3", got)
	}
	if got := cvt8(3.5); got != 3 {
		t.Errorf("cvt8 got %d, wanted 3", got)
	}
	if got := cvt9(3.5); got != 3 {
		t.Errorf("cvt9 got %d, wanted 3", got)
	}
	if got := cvt10(3.5); got != 3 {
		t.Errorf("cvt10 got %d, wanted 3", got)
	}
	if got := cvt11(3.5); got != 3 {
		t.Errorf("cvt11 got %d, wanted 3", got)
	}
	if got := cvt12(3.5); got != 3 {
		t.Errorf("cvt12 got %d, wanted 3", got)
	}
	if got := *f2i64p(10); got != 100 {
		t.Errorf("f2i64p got %d, wanted 100", got)
	}
}

func TestFloatConvertFolded(t *testing.T) {
	// Assign constants to variables so that they are (hopefully) constant folded
	// by the SSA backend rather than the frontend.
	u64, u32, u16, u8 := uint64(1<<63), uint32(1<<31), uint16(1<<15), uint8(1<<7)
	i64, i32, i16, i8 := int64(-1<<63), int32(-1<<31), int16(-1<<15), int8(-1<<7)
	du64, du32, du16, du8 := float64(1<<63), float64(1<<31), float64(1<<15), float64(1<<7)
	di64, di32, di16, di8 := float64(-1<<63), float64(-1<<31), float64(-1<<15), float64(-1<<7)
	su64, su32, su16, su8 := float32(1<<63), float32(1<<31), float32(1<<15), float32(1<<7)
	si64, si32, si16, si8 := float32(-1<<63), float32(-1<<31), float32(-1<<15), float32(-1<<7)

	// integer to float
	if float64(u64) != du64 {
		t.Errorf("float64(u64) != du64")
	}
	if float64(u32) != du32 {
		t.Errorf("float64(u32) != du32")
	}
	if float64(u16) != du16 {
		t.Errorf("float64(u16) != du16")
	}
	if float64(u8) != du8 {
		t.Errorf("float64(u8) != du8")
	}
	if float64(i64) != di64 {
		t.Errorf("float64(i64) != di64")
	}
	if float64(i32) != di32 {
		t.Errorf("float64(i32) != di32")
	}
	if float64(i16) != di16 {
		t.Errorf("float64(i16) != di16")
	}
	if float64(i8) != di8 {
		t.Errorf("float64(i8) != di8")
	}
	if float32(u64) != su64 {
		t.Errorf("float32(u64) != su64")
	}
	if float32(u32) != su32 {
		t.Errorf("float32(u32) != su32")
	}
	if float32(u16) != su16 {
		t.Errorf("float32(u16) != su16")
	}
	if float32(u8) != su8 {
		t.Errorf("float32(u8) != su8")
	}
	if float32(i64) != si64 {
		t.Errorf("float32(i64) != si64")
	}
	if float32(i32) != si32 {
		t.Errorf("float32(i32) != si32")
	}
	if float32(i16) != si16 {
		t.Errorf("float32(i16) != si16")
	}
	if float32(i8) != si8 {
		t.Errorf("float32(i8) != si8")
	}

	// float to integer
	if uint64(du64) != u64 {
		t.Errorf("uint64(du64) != u64")
	}
	if uint32(du32) != u32 {
		t.Errorf("uint32(du32) != u32")
	}
	if uint16(du16) != u16 {
		t.Errorf("uint16(du16) != u16")
	}
	if uint8(du8) != u8 {
		t.Errorf("uint8(du8) != u8")
	}
	if int64(di64) != i64 {
		t.Errorf("int64(di64) != i64")
	}
	if int32(di32) != i32 {
		t.Errorf("int32(di32) != i32")
	}
	if int16(di16) != i16 {
		t.Errorf("int16(di16) != i16")
	}
	if int8(di8) != i8 {
		t.Errorf("int8(di8) != i8")
	}
	if uint64(su64) != u64 {
		t.Errorf("uint64(su64) != u64")
	}
	if uint32(su32) != u32 {
		t.Errorf("uint32(su32) != u32")
	}
	if uint16(su16) != u16 {
		t.Errorf("uint16(su16) != u16")
	}
	if uint8(su8) != u8 {
		t.Errorf("uint8(su8) != u8")
	}
	if int64(si64) != i64 {
		t.Errorf("int64(si64) != i64")
	}
	if int32(si32) != i32 {
		t.Errorf("int32(si32) != i32")
	}
	if int16(si16) != i16 {
		t.Errorf("int16(si16) != i16")
	}
	if int8(si8) != i8 {
		t.Errorf("int8(si8) != i8")
	}
}

func TestFloat32StoreToLoadConstantFold(t *testing.T) {
	// Test that math.Float32{,from}bits constant fold correctly.
	// In particular we need to be careful that signaling NaN (sNaN) values
	// are not converted to quiet NaN (qNaN) values during compilation.
	// See issue #27193 for more information.

	// signaling NaNs
	{
		const nan = uint32(0x7f800001) // sNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}
	{
		const nan = uint32(0x7fbfffff) // sNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}
	{
		const nan = uint32(0xff800001) // sNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}
	{
		const nan = uint32(0xffbfffff) // sNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}

	// quiet NaNs
	{
		const nan = uint32(0x7fc00000) // qNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}
	{
		const nan = uint32(0x7fffffff) // qNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}
	{
		const nan = uint32(0x8fc00000) // qNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}
	{
		const nan = uint32(0x8fffffff) // qNaN
		if x := math.Float32bits(math.Float32frombits(nan)); x != nan {
			t.Errorf("got %#x, want %#x", x, nan)
		}
	}

	// infinities
	{
		const inf = uint32(0x7f800000) // +∞
		if x := math.Float32bits(math.Float32frombits(inf)); x != inf {
			t.Errorf("got %#x, want %#x", x, inf)
		}
	}
	{
		const negInf = uint32(0xff800000) // -∞
		if x := math.Float32bits(math.Float32frombits(negInf)); x != negInf {
			t.Errorf("got %#x, want %#x", x, negInf)
		}
	}

	// numbers
	{
		const zero = uint32(0) // +0.0
		if x := math.Float32bits(math.Float32frombits(zero)); x != zero {
			t.Errorf("got %#x, want %#x", x, zero)
		}
	}
	{
		const negZero = uint32(1 << 31) // -0.0
		if x := math.Float32bits(math.Float32frombits(negZero)); x != negZero {
			t.Errorf("got %#x, want %#x", x, negZero)
		}
	}
	{
		const one = uint32(0x3f800000) // 1.0
		if x := math.Float32bits(math.Float32frombits(one)); x != one {
			t.Errorf("got %#x, want %#x", x, one)
		}
	}
	{
		const negOne = uint32(0xbf800000) // -1.0
		if x := math.Float32bits(math.Float32frombits(negOne)); x != negOne {
			t.Errorf("got %#x, want %#x", x, negOne)
		}
	}
	{
		const frac = uint32(0x3fc00000) // +1.5
		if x := math.Float32bits(math.Float32frombits(frac)); x != frac {
			t.Errorf("got %#x, want %#x", x, frac)
		}
	}
	{
		const negFrac = uint32(0xbfc00000) // -1.5
		if x := math.Float32bits(math.Float32frombits(negFrac)); x != negFrac {
			t.Errorf("got %#x, want %#x", x, negFrac)
		}
	}
}

// Signaling NaN values as constants.
const (
	snan32bits uint32 = 0x7f800001
	snan64bits uint64 = 0x7ff0000000000001
)

// Signaling NaNs as variables.
var snan32bitsVar uint32 = snan32bits
var snan64bitsVar uint64 = snan64bits

func TestFloatSignalingNaN(t *testing.T) {
	// Make sure we generate a signaling NaN from a constant properly.
	// See issue 36400.
	f32 := math.Float32frombits(snan32bits)
	g32 := math.Float32frombits(snan32bitsVar)
	x32 := math.Float32bits(f32)
	y32 := math.Float32bits(g32)
	if x32 != y32 {
		t.Errorf("got %x, want %x (diff=%x)", x32, y32, x32^y32)
	}

	f64 := math.Float64frombits(snan64bits)
	g64 := math.Float64frombits(snan64bitsVar)
	x64 := math.Float64bits(f64)
	y64 := math.Float64bits(g64)
	if x64 != y64 {
		t.Errorf("got %x, want %x (diff=%x)", x64, y64, x64^y64)
	}
}

func TestFloatSignalingNaNConversion(t *testing.T) {
	// Test to make sure when we convert a signaling NaN, we get a NaN.
	// (Ideally we want a quiet NaN, but some platforms don't agree.)
	// See issue 36399.
	s32 := math.Float32frombits(snan32bitsVar)
	if s32 == s32 {
		t.Errorf("converting a NaN did not result in a NaN")
	}
	s64 := math.Float64frombits(snan64bitsVar)
	if s64 == s64 {
		t.Errorf("converting a NaN did not result in a NaN")
	}
}

func TestFloatSignalingNaNConversionConst(t *testing.T) {
	// Test to make sure when we convert a signaling NaN, it converts to a NaN.
	// (Ideally we want a quiet NaN, but some platforms don't agree.)
	// See issue 36399 and 36400.
	s32 := math.Float32frombits(snan32bits)
	if s32 == s32 {
		t.Errorf("converting a NaN did not result in a NaN")
	}
	s64 := math.Float64frombits(snan64bits)
	if s64 == s64 {
		t.Errorf("converting a NaN did not result in a NaN")
	}
}

var sinkFloat float64

func BenchmarkMul2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var m float64 = 1
		for j := 0; j < 500; j++ {
			m *= 2
		}
		sinkFloat = m
	}
}
func BenchmarkMulNeg2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var m float64 = 1
		for j := 0; j < 500; j++ {
			m *= -2
		}
		sinkFloat = m
	}
}
