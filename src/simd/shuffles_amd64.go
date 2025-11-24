// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

// These constants represent the source pattern for the four parameters
// (a, b, c, d) passed to SelectFromPair and SelectFromPairGrouped.
// L means the element comes from the 'x' vector (Low), and
// H means it comes from the 'y' vector (High).
// The order of the letters corresponds to elements a, b, c, d.
// The underlying integer value is a bitmask where:
// Bit 0: Source of element 'a' (0 for x, 1 for y)
// Bit 1: Source of element 'b' (0 for x, 1 for y)
// Bit 2: Source of element 'c' (0 for x, 1 for y)
// Bit 3: Source of element 'd' (0 for x, 1 for y)
// Note that the least-significant bit is on the LEFT in this encoding.
const (
	_LLLL = iota // a:x, b:x, c:x, d:x
	_HLLL        // a:y, b:x, c:x, d:x
	_LHLL        // a:x, b:y, c:x, d:x
	_HHLL        // a:y, b:y, c:x, d:x
	_LLHL        // a:x, b:x, c:y, d:x
	_HLHL        // a:y, b:x, c:y, d:x
	_LHHL        // a:x, b:y, c:y, d:x
	_HHHL        // a:y, b:y, c:y, d:x
	_LLLH        // a:x, b:x, c:x, d:y
	_HLLH        // a:y, b:x, c:x, d:y
	_LHLH        // a:x, b:y, c:x, d:y
	_HHLH        // a:y, b:y, c:x, d:y
	_LLHH        // a:x, b:x, c:y, d:y
	_HLHH        // a:y, b:x, c:y, d:y
	_LHHH        // a:x, b:y, c:y, d:y
	_HHHH        // a:y, b:y, c:y, d:y
)

// These constants represent the source pattern for the four parameters
// (a, b, c, d) passed to SelectFromPair and SelectFromPairGrouped for
// two-element vectors.
const (
	_LL = iota
	_HL
	_LH
	_HH
)

// SelectFromPair returns the selection of four elements from the two
// vectors x and y, where selector values in the range 0-3 specify
// elements from x and values in the range 4-7 specify the 0-3 elements
// of y.  When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it
// requires two.  a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
// {1,2,4,8}.SelectFromPair(2,3,5,7,{9,25,49,81}) returns {4,8,25,81}
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Int32x4) SelectFromPair(a, b, c, d uint8, y Int32x4) Int32x4 {
	// pattern gets the concatenation of "x or y?" bits
	// (0 == x, 1 == y)
	// This will determine operand choice/order and whether a second
	// instruction is needed.
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	// a-d are masked down to their offsets within x or y
	// this is not necessary for x, but this is easier on the
	// eyes and reduces the risk of an error now or later.
	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstant(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstant(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstant(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstant(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstant(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstant(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstant(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstant(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPair returns the selection of four elements from the two
// vectors x and y, where selector values in the range 0-3 specify
// elements from x and values in the range 4-7 specify the 0-3 elements
// of y.  When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
// {1,2,4,8}.SelectFromPair(2,3,5,7,{9,25,49,81}) returns {4,8,25,81}
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Uint32x4) SelectFromPair(a, b, c, d uint8, y Uint32x4) Uint32x4 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstant(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstant(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstant(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstant(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstant(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstant(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstant(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstant(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPair returns the selection of four elements from the two
// vectors x and y, where selector values in the range 0-3 specify
// elements from x and values in the range 4-7 specify the 0-3 elements
// of y.  When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
// {1,2,4,8}.SelectFromPair(2,3,5,7,{9,25,49,81}) returns {4,8,25,81}
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Float32x4) SelectFromPair(a, b, c, d uint8, y Float32x4) Float32x4 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstant(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstant(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstant(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstant(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstant(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstant(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstant(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstant(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstant(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstant(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstant(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstant(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstant(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstant(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstant(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstant(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
// {1,2,4,8,16,32,64,128}.SelectFromPair(2,3,5,7,{9,25,49,81,121,169,225,289})
//
//	returns {4,8,25,81,64,128,169,289}
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Int32x8) SelectFromPairGrouped(a, b, c, d uint8, y Int32x8) Int32x8 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
// {1,2,4,8,16,32,64,128}.SelectFromPair(2,3,5,7,{9,25,49,81,121,169,225,289})
//
//	returns {4,8,25,81,64,128,169,289}
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Uint32x8) SelectFromPairGrouped(a, b, c, d uint8, y Uint32x8) Uint32x8 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
// {1,2,4,8,16,32,64,128}.SelectFromPair(2,3,5,7,{9,25,49,81,121,169,225,289})
//
//	returns {4,8,25,81,64,128,169,289}
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Float32x8) SelectFromPairGrouped(a, b, c, d uint8, y Float32x8) Float32x8 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX512
func (x Int32x16) SelectFromPairGrouped(a, b, c, d uint8, y Int32x16) Int32x16 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX512
func (x Uint32x16) SelectFromPairGrouped(a, b, c, d uint8, y Uint32x16) Uint32x16 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// When the selectors are constants and can be the selection
// can be implemented in a single instruction, it will be, otherwise
// it requires two.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPS, CPU Feature: AVX512
func (x Float32x16) SelectFromPairGrouped(a, b, c, d uint8, y Float32x16) Float32x16 {
	pattern := a>>2 + (b&4)>>1 + (c & 4) + (d&4)<<1

	a, b, c, d = a&3, b&3, c&3, d&3

	switch pattern {
	case _LLLL:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)
	case _HHHH:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _LLHH:
		return x.concatSelectedConstantGrouped(cscimm4(a, b, c, d), y)
	case _HHLL:
		return y.concatSelectedConstantGrouped(cscimm4(a, b, c, d), x)

	case _HLLL:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)
	case _LHLL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), x)

	case _HLHH:
		z := y.concatSelectedConstantGrouped(cscimm4(a, a, b, b), x)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)
	case _LHHH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, a, b, b), y)
		return z.concatSelectedConstantGrouped(cscimm4(0, 2, c, d), y)

	case _LLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _LLHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return x.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(c, c, d, d), y)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)
	case _HHHL:
		z := y.concatSelectedConstantGrouped(cscimm4(c, c, d, d), x)
		return y.concatSelectedConstantGrouped(cscimm4(a, b, 0, 2), z)

	case _LHLH:
		z := x.concatSelectedConstantGrouped(cscimm4(a, c, b, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_10_00 /* =cscimm4(0, 2, 1, 3) */, z)
	case _HLHL:
		z := x.concatSelectedConstantGrouped(cscimm4(b, d, a, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_00_10 /* =cscimm4(2, 0, 3, 1) */, z)
	case _HLLH:
		z := x.concatSelectedConstantGrouped(cscimm4(b, c, a, d), y)
		return z.concatSelectedConstantGrouped(0b11_01_00_10 /* =cscimm4(2, 0, 1, 3) */, z)
	case _LHHL:
		z := x.concatSelectedConstantGrouped(cscimm4(a, d, b, c), y)
		return z.concatSelectedConstantGrouped(0b01_11_10_00 /* =cscimm4(0, 2, 3, 1) */, z)
	}
	panic("missing case, switch should be exhaustive")
}

// cscimm4 converts the 4 vector element indices into a single
// uint8 for use as an immediate.
func cscimm4(a, b, c, d uint8) uint8 {
	return uint8(a + b<<2 + c<<4 + d<<6)
}

// cscimm2 converts the 2 vector element indices into a single
// uint8 for use as an immediate.
func cscimm2(a, b uint8) uint8 {
	return uint8(a + b<<1)
}

// cscimm2g2 converts the 2 vector element indices into a single
// uint8 for use as an immediate, but duplicated for VSHUFPD
// to emulate grouped behavior of VSHUFPS
func cscimm2g2(a, b uint8) uint8 {
	g := cscimm2(a, b)
	return g + g<<2
}

// cscimm2g4 converts the 2 vector element indices into a single
// uint8 for use as an immediate, but with four copies for VSHUFPD
// to emulate grouped behavior of VSHUFPS
func cscimm2g4(a, b uint8) uint8 {
	g := cscimm2g2(a, b)
	return g + g<<4
}

// SelectFromPair returns the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Uint64x2) SelectFromPair(a, b uint8, y Uint64x2) Uint64x2 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstant(cscimm2(a, b), x)
	case _HH:
		return y.concatSelectedConstant(cscimm2(a, b), y)
	case _LH:
		return x.concatSelectedConstant(cscimm2(a, b), y)
	case _HL:
		return y.concatSelectedConstant(cscimm2(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Uint64x4) SelectFromPairGrouped(a, b uint8, y Uint64x4) Uint64x4 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstantGrouped(cscimm2g2(a, b), x)
	case _HH:
		return y.concatSelectedConstantGrouped(cscimm2g2(a, b), y)
	case _LH:
		return x.concatSelectedConstantGrouped(cscimm2g2(a, b), y)
	case _HL:
		return y.concatSelectedConstantGrouped(cscimm2g2(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX512
func (x Uint64x8) SelectFromPairGrouped(a, b uint8, y Uint64x8) Uint64x8 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstantGrouped(cscimm2g4(a, b), x)
	case _HH:
		return y.concatSelectedConstantGrouped(cscimm2g4(a, b), y)
	case _LH:
		return x.concatSelectedConstantGrouped(cscimm2g4(a, b), y)
	case _HL:
		return y.concatSelectedConstantGrouped(cscimm2g4(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPair returns the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Float64x2) SelectFromPair(a, b uint8, y Float64x2) Float64x2 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstant(cscimm2(a, b), x)
	case _HH:
		return y.concatSelectedConstant(cscimm2(a, b), y)
	case _LH:
		return x.concatSelectedConstant(cscimm2(a, b), y)
	case _HL:
		return y.concatSelectedConstant(cscimm2(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Float64x4) SelectFromPairGrouped(a, b uint8, y Float64x4) Float64x4 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstantGrouped(cscimm2g2(a, b), x)
	case _HH:
		return y.concatSelectedConstantGrouped(cscimm2g2(a, b), y)
	case _LH:
		return x.concatSelectedConstantGrouped(cscimm2g2(a, b), y)
	case _HL:
		return y.concatSelectedConstantGrouped(cscimm2g2(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX512
func (x Float64x8) SelectFromPairGrouped(a, b uint8, y Float64x8) Float64x8 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstantGrouped(cscimm2g4(a, b), x)
	case _HH:
		return y.concatSelectedConstantGrouped(cscimm2g4(a, b), y)
	case _LH:
		return x.concatSelectedConstantGrouped(cscimm2g4(a, b), y)
	case _HL:
		return y.concatSelectedConstantGrouped(cscimm2g4(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPair returns the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Int64x2) SelectFromPair(a, b uint8, y Int64x2) Int64x2 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstant(cscimm2(a, b), x)
	case _HH:
		return y.concatSelectedConstant(cscimm2(a, b), y)
	case _LH:
		return x.concatSelectedConstant(cscimm2(a, b), y)
	case _HL:
		return y.concatSelectedConstant(cscimm2(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Int64x4) SelectFromPairGrouped(a, b uint8, y Int64x4) Int64x4 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstantGrouped(cscimm2g2(a, b), x)
	case _HH:
		return y.concatSelectedConstantGrouped(cscimm2g2(a, b), y)
	case _LH:
		return x.concatSelectedConstantGrouped(cscimm2g2(a, b), y)
	case _HL:
		return y.concatSelectedConstantGrouped(cscimm2g2(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

// SelectFromPairGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.  When the selectors are constants the selection can be
// implemented in a single instruction.
//
// If the selectors are not constant this will translate to a function
// call.
//
// Asm: VSHUFPD, CPU Feature: AVX512
func (x Int64x8) SelectFromPairGrouped(a, b uint8, y Int64x8) Int64x8 {
	pattern := (a&2)>>1 + (b & 2)

	a, b = a&1, b&1

	switch pattern {
	case _LL:
		return x.concatSelectedConstantGrouped(cscimm2g4(a, b), x)
	case _HH:
		return y.concatSelectedConstantGrouped(cscimm2g4(a, b), y)
	case _LH:
		return x.concatSelectedConstantGrouped(cscimm2g4(a, b), y)
	case _HL:
		return y.concatSelectedConstantGrouped(cscimm2g4(a, b), x)
	}
	panic("missing case, switch should be exhaustive")
}

/* PermuteScalars */

// PermuteScalars performs a permutation of vector x's elements using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX
func (x Int32x4) PermuteScalars(a, b, c, d uint8) Int32x4 {
	return x.permuteScalars(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalars performs a permutation of vector x's elements using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX
func (x Uint32x4) PermuteScalars(a, b, c, d uint8) Uint32x4 {
	return x.permuteScalars(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsGrouped */

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX2
func (x Int32x8) PermuteScalarsGrouped(a, b, c, d uint8) Int32x8 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		 {  x[a], x[b], x[c], x[d],         x[a+4], x[b+4], x[c+4], x[d+4],
//			x[a+8], x[b+8], x[c+8], x[d+8], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX512
func (x Int32x16) PermuteScalarsGrouped(a, b, c, d uint8) Int32x16 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFD, CPU Feature: AVX2
func (x Uint32x8) PermuteScalarsGrouped(a, b, c, d uint8) Uint32x8 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		 {  x[a], x[b], x[c], x[d],         x[a+4], x[b+4], x[c+4], x[d+4],
//			x[a+8], x[b+8], x[c+8], x[d+8], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFD, CPU Feature: AVX512
func (x Uint32x16) PermuteScalarsGrouped(a, b, c, d uint8) Uint32x16 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsHi */

// PermuteScalarsHi performs a permutation of vector x using the supplied indices:
//
// result = {x[0], x[1], x[2], x[3], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX512
func (x Int16x8) PermuteScalarsHi(a, b, c, d uint8) Int16x8 {
	return x.permuteScalarsHi(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHi performs a permutation of vector x using the supplied indices:
//
// result = {x[0], x[1], x[2], x[3], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX512
func (x Uint16x8) PermuteScalarsHi(a, b, c, d uint8) Uint16x8 {
	return x.permuteScalarsHi(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsHiGrouped */

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		  {x[0], x[1], x[2], x[3],   x[a+4], x[b+4], x[c+4], x[d+4],
//			x[8], x[9], x[10], x[11], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX2
func (x Int16x16) PermuteScalarsHiGrouped(a, b, c, d uint8) Int16x16 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		  {x[0], x[1], x[2], x[3],     x[a+4], x[b+4], x[c+4], x[d+4],
//			x[8], x[9], x[10], x[11],   x[a+12], x[b+12], x[c+12], x[d+12],
//			x[16], x[17], x[18], x[19], x[a+20], x[b+20], x[c+20], x[d+20],
//			x[24], x[25], x[26], x[27], x[a+28], x[b+28], x[c+28], x[d+28]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX512
func (x Int16x32) PermuteScalarsHiGrouped(a, b, c, d uint8) Int16x32 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	  {x[0], x[1], x[2], x[3],   x[a+4], x[b+4], x[c+4], x[d+4],
//		x[8], x[9], x[10], x[11], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Each group is of size 128-bit.
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX2
func (x Uint16x16) PermuteScalarsHiGrouped(a, b, c, d uint8) Uint16x16 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		 {  x[0], x[1], x[2], x[3],     x[a+4], x[b+4], x[c+4], x[d+4],
//			x[8], x[9], x[10], x[11],   x[a+12], x[b+12], x[c+12], x[d+12],
//			x[16], x[17], x[18], x[19], x[a+20], x[b+20], x[c+20], x[d+20],
//			x[24], x[25], x[26], x[27], x[a+28], x[b+28], x[c+28], x[d+28]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX512
func (x Uint16x32) PermuteScalarsHiGrouped(a, b, c, d uint8) Uint16x32 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsLo */

// PermuteScalarsLo performs a permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[4], x[5], x[6], x[7]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX512
func (x Int16x8) PermuteScalarsLo(a, b, c, d uint8) Int16x8 {
	return x.permuteScalarsLo(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLo performs a permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[4], x[5], x[6], x[7]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX512
func (x Uint16x8) PermuteScalarsLo(a, b, c, d uint8) Uint16x8 {
	return x.permuteScalarsLo(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsLoGrouped */

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	 {x[a], x[b], x[c], x[d],         x[4], x[5], x[6], x[7],
//		 x[a+8], x[b+8], x[c+8], x[d+8], x[12], x[13], x[14], x[15]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX2
func (x Int16x16) PermuteScalarsLoGrouped(a, b, c, d uint8) Int16x16 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	 {x[a], x[b], x[c], x[d],    x[4], x[5], x[6], x[7],
//		x[a+8], x[b+8], x[c+8], x[d+8],     x[12], x[13], x[14], x[15],
//		x[a+16], x[b+16], x[c+16], x[d+16], x[20], x[21], x[22], x[23],
//		x[a+24], x[b+24], x[c+24], x[d+24], x[28], x[29], x[30], x[31]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX512
func (x Int16x32) PermuteScalarsLoGrouped(a, b, c, d uint8) Int16x32 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result = {x[a], x[b], x[c], x[d],         x[4], x[5], x[6], x[7],
//		x[a+8], x[b+8], x[c+8], x[d+8], x[12], x[13], x[14], x[15]}
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX2
func (x Uint16x16) PermuteScalarsLoGrouped(a, b, c, d uint8) Uint16x16 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	 {x[a], x[b], x[c], x[d],    x[4], x[5], x[6], x[7],
//		x[a+8], x[b+8], x[c+8], x[d+8],     x[12], x[13], x[14], x[15],
//		x[a+16], x[b+16], x[c+16], x[d+16], x[20], x[21], x[22], x[23],
//		x[a+24], x[b+24], x[c+24], x[d+24], x[28], x[29], x[30], x[31]}
//
// Each group is of size 128-bit.
//
// Parameters a,b,c,d should have values between 0 and 3.
// If a through d are constants, then an instruction will be inlined, otherwise
// a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX512
func (x Uint16x32) PermuteScalarsLoGrouped(a, b, c, d uint8) Uint16x32 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}
