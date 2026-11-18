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
