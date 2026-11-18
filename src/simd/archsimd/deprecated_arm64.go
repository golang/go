// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package archsimd

// Deprecated operations that are wrappers around their replacements.

// ExtendLo8ToInt16 sign-extends the low 8 elements of x to int16.
//
//	z[i] = int16(x[i])
//
// Asm: VSXTL
//
// Deprecated: use x.ExtendToInt16() instead.
//
//go:fix inline
func (x Int8x16) ExtendLo8ToInt16() (z Int16x8) {
	return x.ExtendToInt16()
}

// ExtendLo4ToInt32 sign-extends the low 4 elements of x to int32.
//
//	z[i] = int32(x[i])
//
// Asm: VSXTL
//
// Deprecated: use x.ExtendToInt32() instead.
//
//go:fix inline
func (x Int16x8) ExtendLo4ToInt32() (z Int32x4) {
	return x.ExtendToInt32()
}

// ExtendLo2ToInt64 sign-extends the low 2 elements of x to int64.
//
//	z[i] = int64(x[i])
//
// Asm: VSXTL
//
// Deprecated: use x.ExtendToInt64() instead.
//
//go:fix inline
func (x Int32x4) ExtendLo2ToInt64() (z Int64x2) {
	return x.ExtendToInt64()
}

// ExtendLo8ToUint16 zero-extends the low 8 elements of x to uint16.
//
//	z[i] = uint16(x[i])
//
// Asm: VUXTL
//
// Deprecated: use x.ExtendToUint16() instead.
//
//go:fix inline
func (x Uint8x16) ExtendLo8ToUint16() (z Uint16x8) {
	return x.ExtendToUint16()
}

// ExtendLo4ToUint32 zero-extends the low 4 elements of x to uint32.
//
//	z[i] = uint32(x[i])
//
// Asm: VUXTL
//
// Deprecated: use x.ExtendToUint32() instead.
//
//go:fix inline
func (x Uint16x8) ExtendLo4ToUint32() (z Uint32x4) {
	return x.ExtendToUint32()
}

// ExtendLo2ToUint64 zero-extends the low 2 elements of x to uint64.
//
//	z[i] = uint64(x[i])
//
// Asm: VUXTL
//
// Deprecated: use x.ExtendToUint64() instead.
//
//go:fix inline
func (x Uint32x4) ExtendLo2ToUint64() (z Uint64x2) {
	return x.ExtendToUint64()
}

// ConvertLo2ToFloat64 converts the low-indexed 2 elements of x to float64.
//
//	z[i] = float64(x[i])
//
// Asm: VFCVTL
//
// Deprecated: use x.ConvertToFloat64() instead.
//
//go:fix inline
func (x Float32x4) ConvertLo2ToFloat64() (z Float64x2) {
	return x.ConvertToFloat64()
}

// ConcatShiftBytesRight concatenates x and y and shifts it right by shift bytes.
// The result vector will be the lower half of the concatenated vector.
//
// Asm: VEXT
//
// Deprecated: use y.ConcatWindow(x, uint8(shift)) instead. ConcatWindow takes
// x and y in the opposite order.
//
//go:fix inline
func (x Uint8x16) ConcatShiftBytesRight(y Uint8x16, shift uint64) Uint8x16 {
	return y.ConcatWindow(x, uint8(shift))
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
func (x Float64x2) GetElem(i uint8) (z float64) {
	return x.Elem(i)
}

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
func (x Int16x8) GetElem(i uint8) (z int16) {
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
func (x Int64x2) GetElem(i uint8) (z int64) {
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
func (x Uint16x8) GetElem(i uint8) (z uint16) {
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
func (x Uint64x2) GetElem(i uint8) (z uint64) {
	return x.Elem(i)
}

// CarrylessMultiplyEven computes the carryless multiplication of
// the even-indexed elements of x and y.
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
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
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
//
// Deprecated: use x.CarrylessMulOdd(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyOdd(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulOdd(y)
}

// CarrylessMultiplyOddEven computes the carryless multiplication of
// the odd-indexed elements of x and the even-indexed elements of y.
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
//
// Deprecated: use x.CarrylessMulOddEven(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyOddEven(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulOddEven(y)
}

// CarrylessMultiplyEvenOdd computes the carryless multiplication of
// the even-indexed elements of x and the odd-indexed elements of y.
//
// Asm: PMULL if CPU feature PMULL, Emulated otherwise
//
// Deprecated: use x.CarrylessMulEvenOdd(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyEvenOdd(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulEvenOdd(y)
}
