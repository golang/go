// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package archsimd

// Deprecated operations that are wrappers around their replacements.

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Int8x16) GetElem(i uint8) (z int8) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Uint8x16) GetElem(i uint8) (z uint8) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Int16x8) GetElem(i uint8) (z int16) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Uint16x8) GetElem(i uint8) (z uint16) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Int32x4) GetElem(i uint8) (z int32) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Uint32x4) GetElem(i uint8) (z uint32) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Float32x4) GetElem(i uint8) (z float32) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Int64x2) GetElem(i uint8) (z int64) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Uint64x2) GetElem(i uint8) (z uint64) {
	return x.Elem(i)
}

// GetElem returns the i'th element of x.
//
// Deprecated: use x.Elem(i) instead.
//
//go:fix inline
func (x Float64x2) GetElem(i uint8) (z float64) {
	return x.Elem(i)
}

// CarrylessMultiplyEven computes the carryless multiplication of
// the even-indexed elements of x and y.
//
// Deprecated: use x.CarrylessMulEven(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyEven(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulEven(y)
}

// CarrylessMultiplyOdd computes the carryless multiplication of
// the odd-indexed elements of x and y.
//
// Deprecated: use x.CarrylessMulOdd(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyOdd(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulOdd(y)
}

// CarrylessMultiplyEvenOdd computes the carryless multiplication of
// the even-indexed elements of x and the odd-indexed elements of y.
//
// Deprecated: use x.CarrylessMulEvenOdd(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyEvenOdd(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulEvenOdd(y)
}

// CarrylessMultiplyOddEven computes the carryless multiplication of
// the odd-indexed elements of x and the even-indexed elements of y.
//
// Deprecated: use x.CarrylessMulOddEven(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyOddEven(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulOddEven(y)
}

// MulWidenLo multiplies corresponding elements in the lower half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = int16(x[i]) * int16(y[i])
//
// Deprecated: use x.MulWiden(y) instead.
//
//go:fix inline
func (x Int8x16) MulWidenLo(y Int8x16) (z Int16x8) {
	return x.MulWiden(y)
}

// MulWidenLo multiplies corresponding elements in the lower half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = int32(x[i]) * int32(y[i])
//
// Deprecated: use x.MulWiden(y) instead.
//
//go:fix inline
func (x Int16x8) MulWidenLo(y Int16x8) (z Int32x4) {
	return x.MulWiden(y)
}

// MulWidenLo multiplies corresponding elements in the lower half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = int64(x[i]) * int64(y[i])
//
// Deprecated: use x.MulWiden(y) instead.
//
//go:fix inline
func (x Int32x4) MulWidenLo(y Int32x4) (z Int64x2) {
	return x.MulWiden(y)
}

// MulWidenLo multiplies corresponding elements in the lower half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = uint16(x[i]) * uint16(y[i])
//
// Deprecated: use x.MulWiden(y) instead.
//
//go:fix inline
func (x Uint8x16) MulWidenLo(y Uint8x16) (z Uint16x8) {
	return x.MulWiden(y)
}

// MulWidenLo multiplies corresponding elements in the lower half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = uint32(x[i]) * uint32(y[i])
//
// Deprecated: use x.MulWiden(y) instead.
//
//go:fix inline
func (x Uint16x8) MulWidenLo(y Uint16x8) (z Uint32x4) {
	return x.MulWiden(y)
}

// MulWidenLo multiplies corresponding elements in the lower half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = uint64(x[i]) * uint64(y[i])
//
// Deprecated: use x.MulWiden(y) instead.
//
//go:fix inline
func (x Uint32x4) MulWidenLo(y Uint32x4) (z Uint64x2) {
	return x.MulWiden(y)
}

// MulWidenHi multiplies corresponding elements in the upper half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = int16(x[8+i]) * int16(y[8+i])
//
// Deprecated: use x.HiToLo().MulWiden(y.HiToLo()) instead, which compiles to
// the same single instruction.
//
//go:fix inline
func (x Int8x16) MulWidenHi(y Int8x16) (z Int16x8) {
	return x.HiToLo().MulWiden(y.HiToLo())
}

// MulWidenHi multiplies corresponding elements in the upper half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = uint16(x[8+i]) * uint16(y[8+i])
//
// Deprecated: use x.HiToLo().MulWiden(y.HiToLo()) instead, which compiles to
// the same single instruction.
//
//go:fix inline
func (x Uint8x16) MulWidenHi(y Uint8x16) (z Uint16x8) {
	return x.HiToLo().MulWiden(y.HiToLo())
}

// MulWidenHi multiplies corresponding elements in the upper half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = int32(x[4+i]) * int32(y[4+i])
//
// Deprecated: use x.HiToLo().MulWiden(y.HiToLo()) instead, which compiles to
// the same single instruction.
//
//go:fix inline
func (x Int16x8) MulWidenHi(y Int16x8) (z Int32x4) {
	return x.HiToLo().MulWiden(y.HiToLo())
}

// MulWidenHi multiplies corresponding elements in the upper half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = uint32(x[4+i]) * uint32(y[4+i])
//
// Deprecated: use x.HiToLo().MulWiden(y.HiToLo()) instead, which compiles to
// the same single instruction.
//
//go:fix inline
func (x Uint16x8) MulWidenHi(y Uint16x8) (z Uint32x4) {
	return x.HiToLo().MulWiden(y.HiToLo())
}

// MulWidenHi multiplies corresponding elements in the upper half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = int64(x[2+i]) * int64(y[2+i])
//
// Deprecated: use x.HiToLo().MulWiden(y.HiToLo()) instead, which compiles to
// the same single instruction.
//
//go:fix inline
func (x Int32x4) MulWidenHi(y Int32x4) (z Int64x2) {
	return x.HiToLo().MulWiden(y.HiToLo())
}

// MulWidenHi multiplies corresponding elements in the upper half of x and y,
// returning a vector of half as many elements, each with twice the input
// element size.
//
//	z[i] = uint64(x[2+i]) * uint64(y[2+i])
//
// Deprecated: use x.HiToLo().MulWiden(y.HiToLo()) instead, which compiles to
// the same single instruction.
//
//go:fix inline
func (x Uint32x4) MulWidenHi(y Uint32x4) (z Uint64x2) {
	return x.HiToLo().MulWiden(y.HiToLo())
}
