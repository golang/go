// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && amd64

package archsimd

func (x Int32x4) concatPermuteScalars(a, b, c, d uint8, y Int32x4) Int32x4 {
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

func (x Uint32x4) concatPermuteScalars(a, b, c, d uint8, y Uint32x4) Uint32x4 {
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

func (x Float32x4) concatPermuteScalars(a, b, c, d uint8, y Float32x4) Float32x4 {
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

func (x Int32x8) concatPermuteScalarsGrouped(a, b, c, d uint8, y Int32x8) Int32x8 {
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

func (x Uint32x8) concatPermuteScalarsGrouped(a, b, c, d uint8, y Uint32x8) Uint32x8 {
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

func (x Float32x8) concatPermuteScalarsGrouped(a, b, c, d uint8, y Float32x8) Float32x8 {
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

func (x Int32x16) concatPermuteScalarsGrouped(a, b, c, d uint8, y Int32x16) Int32x16 {
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

func (x Uint32x16) concatPermuteScalarsGrouped(a, b, c, d uint8, y Uint32x16) Uint32x16 {
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

func (x Float32x16) concatPermuteScalarsGrouped(a, b, c, d uint8, y Float32x16) Float32x16 {
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

func (x Uint64x2) concatPermuteScalars(a, b uint8, y Uint64x2) Uint64x2 {
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

func (x Uint64x4) concatPermuteScalarsGrouped(a, b uint8, y Uint64x4) Uint64x4 {
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

func (x Uint64x8) concatPermuteScalarsGrouped(a, b uint8, y Uint64x8) Uint64x8 {
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

func (x Float64x2) concatPermuteScalars(a, b uint8, y Float64x2) Float64x2 {
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

func (x Float64x4) concatPermuteScalarsGrouped(a, b uint8, y Float64x4) Float64x4 {
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

func (x Float64x8) concatPermuteScalarsGrouped(a, b uint8, y Float64x8) Float64x8 {
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

func (x Int64x2) concatPermuteScalars(a, b uint8, y Int64x2) Int64x2 {
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

func (x Int64x4) concatPermuteScalarsGrouped(a, b uint8, y Int64x4) Int64x4 {
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

func (x Int64x8) concatPermuteScalarsGrouped(a, b uint8, y Int64x8) Int64x8 {
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
