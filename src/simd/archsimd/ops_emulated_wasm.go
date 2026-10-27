// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package archsimd

var nn = [2]int64{-1 << 63, -1 << 63}
var evenInt8s = [16]int8{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}
var evenInt16s = [8]int16{-1, 0, -1, 0, -1, 0, -1, 0}
var evenInt32s = [4]int32{-1, 0, -1, 0}

// For unsigned comparison, the trick for converting it into
// signed comparisonm is to notice that the unsigned range is
// the same as the signed range plus 1 << bitwidth-1.
// And adding or subtracting the sign bit is the same as XORing
// it.  Thus, XOR both sign bits and then used the signed
// comparison operations.

// Less return a mask vector of x[i] < y[i]
func (x Uint64x2) Less(y Uint64x2) (z Mask64x2) {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.Less(iy)
}

// LessEqual return a mask vector of x[i] <= y[i]
func (x Uint64x2) LessEqual(y Uint64x2) (z Mask64x2) {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.LessEqual(iy)
}

// Greater return a mask vector of x[i] > y[i]
func (x Uint64x2) Greater(y Uint64x2) (z Mask64x2) {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.Greater(iy)
}

// GreaterEqual return a mask vector of x[i] >= y[i]
func (x Uint64x2) GreaterEqual(y Uint64x2) (z Mask64x2) {
	signs := LoadInt64x2Array(&nn)
	ix := x.BitsToInt64().Xor(signs)
	iy := y.BitsToInt64().Xor(signs)
	return ix.GreaterEqual(iy)
}

// Max returns the elementswise maximum of elements in x and y
func (x Int64x2) Max(y Int64x2) (z Int64x2) {
	mask := x.Greater(y).ToInt64x2()
	return x.And(mask).Or(y.AndNot(mask))
}

// Min returns the elementswise minimum of elements in x and y
func (x Int64x2) Min(y Int64x2) (z Int64x2) {
	mask := x.Less(y).ToInt64x2()
	return x.And(mask).Or(y.AndNot(mask))
}

// Max returns the elementswise maximum of elements in x and y
func (x Uint64x2) Max(y Uint64x2) (z Uint64x2) {
	mask := x.Greater(y).ToInt64x2().ToBits()
	return x.And(mask).Or(y.AndNot(mask))
}

// Min returns the elementswise minimum of elements in x and y
func (x Uint64x2) Min(y Uint64x2) (z Uint64x2) {
	mask := x.Less(y).ToInt64x2().ToBits()
	return x.And(mask).Or(y.AndNot(mask))
}

// Mul returns the elementswise product of elements in x and y
func (x Int8x16) Mul(y Int8x16) (z Int8x16) {
	// To obtain an 8-bit multiply, split the vectors into even and odd
	// elements, shift odds into even position, widen elements in both
	// vectors, multiply, discard high parts, realign the odd results
	// and combine.
	mask := LoadInt8x16Array(&evenInt8s)
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
func (x Uint8x16) Mul(y Uint8x16) (z Uint8x16) {
	mask := LoadInt8x16Array(&evenInt8s).ToBits()
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
func (x Int16x8) OnesCount() (z Int16x8) {
	mask8 := LoadInt8x16Array(&evenInt8s)
	c8 := x.ToBits().ReshapeToUint8s().BitsToInt8().OnesCount()                         // per-byte counts
	c16e := c8.And(mask8).ToBits().ReshapeToUint16s().BitsToInt16()                     // even-element per-byte counts, as 16-bit elements
	c16o := c8.AndNot(mask8).ToBits().ReshapeToUint16s().BitsToInt16().ShiftAllRight(8) // odd-element per-byte counts, as 16-bit elements, aligned
	return c16e.Add(c16o)                                                               // return their elementwise sum
}

// OnesCount returns the number of set bits in each vector element
func (x Int32x4) OnesCount() (z Int32x4) {
	mask8 := LoadInt8x16Array(&evenInt8s)
	c8 := x.ToBits().ReshapeToUint8s().BitsToInt8().OnesCount()                         // per-byte counts
	c16e := c8.And(mask8).ToBits().ReshapeToUint16s().BitsToInt16()                     // even-element per-byte counts, as 16-bit elements
	c16o := c8.AndNot(mask8).ToBits().ReshapeToUint16s().BitsToInt16().ShiftAllRight(8) // odd-element per-byte counts, as 16-bit elements, aligned
	mask16 := LoadInt16x8Array(&evenInt16s)
	c16 := c16e.Add(c16o) // per int16 counts, etc.
	c32e := c16.And(mask16).ToBits().ReshapeToUint32s().BitsToInt32()
	c32o := c16.AndNot(mask16).ToBits().ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	return c32e.Add(c32o)
}

// OnesCount returns the number of set bits in each vector element
func (x Int64x2) OnesCount() (z Int64x2) {
	mask8 := LoadInt8x16Array(&evenInt8s)
	c8 := x.ToBits().ReshapeToUint8s().BitsToInt8().OnesCount()
	c8e := c8.And(mask8).ToBits().ReshapeToUint16s().BitsToInt16()
	c8o := c8.AndNot(mask8).ToBits().ReshapeToUint16s().BitsToInt16().ShiftAllRight(8)
	mask16 := LoadInt16x8Array(&evenInt16s)
	c16 := c8e.Add(c8o)
	c32e := c16.And(mask16).ToBits().ReshapeToUint32s().BitsToInt32()
	c32o := c16.AndNot(mask16).ToBits().ReshapeToUint32s().BitsToInt32().ShiftAllRight(16)
	mask32 := LoadInt32x4Array(&evenInt32s)
	c32 := c32e.Add(c32o)
	c64e := c32.And(mask32).ToBits().ReshapeToUint64s().BitsToInt64()
	c64o := c32.AndNot(mask32).ToBits().ReshapeToUint64s().BitsToInt64().ShiftAllRight(32)
	return c64e.Add(c64o)
}

// OnesCount returns the number of set bits in each vector element
func (x Uint8x16) OnesCount() (z Uint8x16) {
	return x.BitsToInt8().OnesCount().ToBits()
}

// OnesCount returns the number of set bits in each vector element
func (x Uint16x8) OnesCount() (z Uint16x8) {
	return x.BitsToInt16().OnesCount().ToBits()
}

// OnesCount returns the number of set bits in each vector element
func (x Uint32x4) OnesCount() (z Uint32x4) {
	return x.BitsToInt32().OnesCount().ToBits()
}

// OnesCount returns the number of set bits in each vector element
func (x Uint64x2) OnesCount() (z Uint64x2) {
	return x.BitsToInt64().OnesCount().ToBits()
}

// MulAdd returns elementwise x * y + z.
func (x Float32x4) MulAdd(y Float32x4, z Float32x4) (w Float32x4) {
	return x.Mul(y).Add(z)
}

// MulAdd returns elementwise x * y + z.
func (x Float64x2) MulAdd(y Float64x2, z Float64x2) (w Float64x2) {
	return x.Mul(y).Add(z)
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
func (x Uint64x2) CarrylessMultiplyEven(y Uint64x2) (z Uint64x2) {
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
func (x Uint64x2) CarrylessMultiplyOdd(y Uint64x2) (z Uint64x2) {
	x = x.SetElem(0, x.GetElem(1))
	y = y.SetElem(0, y.GetElem(1))
	return x.carrylessMultiply(y)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Float32x4) ReduceSum() (z float32) {
	// (x0+x1) + (x2 + x3) is a shorter evaluation tree,
	// and associates the same as horizontal addition
	return (x.GetElem(0) + x.GetElem(1)) + (x.GetElem(2) + x.GetElem(3))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Float64x2) ReduceSum() (z float64) {
	return x.GetElem(0) + x.GetElem(1)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Int16x8) ReduceSum() (z int16) {
	return ((x.GetElem(0) + x.GetElem(1)) + (x.GetElem(2) + x.GetElem(3))) +
		((x.GetElem(4) + x.GetElem(5)) + (x.GetElem(6) + x.GetElem(7)))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Uint16x8) ReduceSum() (z uint16) {
	return ((x.GetElem(0) + x.GetElem(1)) + (x.GetElem(2) + x.GetElem(3))) +
		((x.GetElem(4) + x.GetElem(5)) + (x.GetElem(6) + x.GetElem(7)))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Int32x4) ReduceSum() (z int32) {
	return (x.GetElem(0) + x.GetElem(1)) + (x.GetElem(2) + x.GetElem(3))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Uint32x4) ReduceSum() (z uint32) {
	return (x.GetElem(0) + x.GetElem(1)) + (x.GetElem(2) + x.GetElem(3))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Int8x16) ReduceSum() (z int8) {
	return int8(x.ExtendLo8ToInt16().Add(x.ExtendHi8ToInt16()).ReduceSum())
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated
func (x Uint8x16) ReduceSum() (z uint8) {
	return uint8(x.ExtendLo8ToUint16().Add(x.ExtendHi8ToUint16()).ReduceSum())
}
