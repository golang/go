// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package archsimd

var nn = [2]int64{-1 << 63, -1 << 63}
var f0s = [16]int8{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}
var ff00s = [8]int16{-1, 0, -1, 0, -1, 0, -1, 0}
var ffff0000s = [4]int32{-1, 0, -1, 0}

// For unsigned comparison, the trick for converting it into
// signed comparisonm is to notice that the unsigned range is
// the same as the signed range plus 1 << bitwidth-1.
// And adding or subtracting the sign bit is the same as XORing
// it.  Thus, XOR both sign bits and then used the signed
// comparison operations.

// Less return a mask vector of x[i] < y[i]
func (x Uint64x2) Less(y Uint64x2) Mask64x2 {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.Less(iy)
}

// LessEqual return a mask vector of x[i] <= y[i]
func (x Uint64x2) LessEqual(y Uint64x2) Mask64x2 {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.LessEqual(iy)
}

// Greater return a mask vector of x[i] > y[i]
func (x Uint64x2) Greater(y Uint64x2) Mask64x2 {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.Greater(iy)
}

// GreaterEqual return a mask vector of x[i] >= y[i]
func (x Uint64x2) GreaterEqual(y Uint64x2) Mask64x2 {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.GreaterEqual(iy)
}

// Max returns the elementswise maximum of elements in x and y
func (x Int64x2) Max(y Int64x2) Int64x2 {
	mask := x.Greater(y).ToInt64x2()
	return x.And(mask).Or(y.AndNot(mask))
}

// Min returns the elementswise minimum of elements in x and y
func (x Int64x2) Min(y Int64x2) Int64x2 {
	mask := x.Less(y).ToInt64x2()
	return x.And(mask).Or(y.AndNot(mask))
}

// Max returns the elementswise maximum of elements in x and y
func (x Uint64x2) Max(y Uint64x2) Uint64x2 {
	mask := x.Greater(y).ToInt64x2().ToBits()
	return x.And(mask).Or(y.AndNot(mask))
}

// Min returns the elementswise minimum of elements in x and y
func (x Uint64x2) Min(y Uint64x2) Uint64x2 {
	mask := x.Less(y).ToInt64x2().ToBits()
	return x.And(mask).Or(y.AndNot(mask))
}

// Mul returns the elementswise product of elements in x and y
func (x Int8x16) Mul(y Int8x16) Int8x16 {
	// To obtain an 8-bit multiply, split the vectors into even and odd
	// elements, shift odds into even position, widen elements in both
	// vectors, multiply, discard high parts, realign the odd results
	// and combine.
	mask := LoadInt8x16Array(&f0s)
	mask16 := mask.ToBits().ReshapeToUint16s()
	xe := x.And(mask).ToBits().ReshapeToUint16s()
	xo := x.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ToBits().ReshapeToUint16s()
	yo := y.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s().BitsToInt8()
}

// Mul returns the elementswise product of elements in x and y
func (x Uint8x16) Mul(y Uint8x16) Uint8x16 {
	mask := LoadInt8x16Array(&f0s).ToBits()
	mask16 := mask.ReshapeToUint16s()
	xe := x.And(mask).ReshapeToUint16s()
	xo := x.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ReshapeToUint16s()
	yo := y.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s()
}

// OnesCount returns the number of set bits in each vector element
func (x Int16x8) OnesCount() Int16x8 {
	mask := LoadInt8x16Array(&f0s)
	c := x.ToBits().ReshapeToUint8s().BitsToInt8().OnesCount()                      // per-byte counts
	ce := c.And(mask).ToBits().ReshapeToUint16s().BitsToInt16()                     // even-element per-byte counts, as 16-bit elements
	co := c.AndNot(mask).ToBits().ReshapeToUint16s().BitsToInt16().ShiftAllRight(8) // odd-element per-byte counts, as 16-bit elements, aligned
	return ce.Add(co)                                                               // return their elementwise sum
}

// OnesCount returns the number of set bits in each vector element
func (x Int32x4) OnesCount() Int32x4 {
	mask := LoadInt8x16Array(&f0s)
	c := x.ToBits().ReshapeToUint8s().BitsToInt8().OnesCount()                      // per-byte counts
	ce := c.And(mask).ToBits().ReshapeToUint16s().BitsToInt16()                     // even-element per-byte counts, as 16-bit elements
	co := c.AndNot(mask).ToBits().ReshapeToUint16s().BitsToInt16().ShiftAllRight(8) // odd-element per-byte counts, as 16-bit elements, aligned
	mask16 := LoadInt16x8Array(&ff00s)
	y := ce.Add(co) // per int16 counts, etc.
	ye := y.And(mask16).ToBits().ReshapeToUint32s().BitsToInt32()
	yo := y.AndNot(mask16).ToBits().ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	return ye.Add(yo)
}

// OnesCount returns the number of set bits in each vector element
func (x Int64x2) OnesCount() Int64x2 {
	mask := LoadInt8x16Array(&f0s)
	c := x.ToBits().ReshapeToUint8s().BitsToInt8().OnesCount()
	ce := c.And(mask).ToBits().ReshapeToUint16s().BitsToInt16()
	co := c.AndNot(mask).ToBits().ReshapeToUint16s().BitsToInt16().ShiftAllRight(8)
	mask16 := LoadInt16x8Array(&ff00s)
	y := ce.Add(co)
	ye := y.And(mask16).ToBits().ReshapeToUint32s().BitsToInt32()
	yo := y.AndNot(mask16).ToBits().ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	mask32 := LoadInt32x4Array(&ffff0000s)
	z := ye.Add(yo)
	ze := z.And(mask32).ToBits().ReshapeToUint64s().BitsToInt64()
	zo := z.AndNot(mask32).ToBits().ReshapeToUint64s().BitsToInt64().ShiftAllRight(32)
	return ze.Add(zo)
}

// OnesCount returns the number of set bits in each vector element
func (x Uint8x16) OnesCount() Uint8x16 {
	return x.BitsToInt8().OnesCount().ToBits()
}

// OnesCount returns the number of set bits in each vector element
func (x Uint16x8) OnesCount() Uint16x8 {
	return x.BitsToInt16().OnesCount().ToBits()
}

// OnesCount returns the number of set bits in each vector element
func (x Uint32x4) OnesCount() Uint32x4 {
	return x.BitsToInt32().OnesCount().ToBits()
}

// OnesCount returns the number of set bits in each vector element
func (x Uint64x2) OnesCount() Uint64x2 {
	return x.BitsToInt64().OnesCount().ToBits()
}

// CarrylessMultiplyEven computes the carryless
// multiplications of selected even halves of the elements of x and y.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
//
// Emulated
func (x Uint64x2) CarrylessMultiplyEven(y Uint64x2) Uint64x2 {
	return x.carrylessMultiply(y)
}

// CarrylessMultiplyOdd computes the carryless
// multiplications of selected odd halves of the elements of x and y.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
//
// Emulated
func (x Uint64x2) CarrylessMultiplyOdd(y Uint64x2) Uint64x2 {
	x = x.SetElem(0, x.GetElem(1))
	y = y.SetElem(0, x.GetElem(1))
	return x.carrylessMultiply(y)
}
