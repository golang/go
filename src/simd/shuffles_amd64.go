// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

// FlattenedTranspose tranposes x and y, regarded as a pair of 2x2
// matrices, but then flattens the rows in order, i.e
// x: ABCD ==> a: A1B2
// y: 1234     b: C3D4
func (x Int32x4) FlattenedTranspose(y Int32x4) (a, b Int32x4) {
	return x.InterleaveLo(y), x.InterleaveHi(y)
}

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
