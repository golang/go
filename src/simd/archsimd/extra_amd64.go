// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// ClearAVXUpperBits clears the high bits of Y0-Y15 and Z0-Z15 registers.
// It is intended for transitioning from AVX to SSE, eliminating the
// performance penalties caused by false dependencies.
//
// Note: in the future the compiler may automatically generate the
// instruction, making this function unnecessary.
//
// Asm: VZEROUPPER, CPU Feature: AVX
func ClearAVXUpperBits()

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int8x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int8x32) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int16x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int16x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int32x4) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int32x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int64x2) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int64x4) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint8x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint8x32) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint16x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint16x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint32x4) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint32x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint64x2) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y.
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint64x4) IsZero() bool

// IsNaN returns a mask whose elements indicate whether the corresponding
// elements of x are NaN.
//
// Asm: VCMPPS, CPU Feature: AVX
func (x Float32x4) IsNaN() Mask32x4

// IsNaN returns a mask whose elements indicate whether the corresponding
// elements of x are NaN.
//
// Asm: VCMPPS, CPU Feature: AVX
func (x Float32x8) IsNaN() Mask32x8

// IsNaN returns a mask whose elements indicate whether the corresponding
// elements of x are NaN.
//
// Asm: VCMPPS, CPU Feature: AVX512
func (x Float32x16) IsNaN() Mask32x16

// IsNaN returns a mask whose elements indicate whether the corresponding
// elements of x are NaN.
//
// Asm: VCMPPD, CPU Feature: AVX
func (x Float64x2) IsNaN() Mask64x2

// IsNaN returns a mask whose elements indicate whether the corresponding
// elements of x are NaN.
//
// Asm: VCMPPD, CPU Feature: AVX
func (x Float64x4) IsNaN() Mask64x4

// IsNaN returns a mask whose elements indicate whether the corresponding
// elements of x are NaN.
//
// Asm: VCMPPD, CPU Feature: AVX512
func (x Float64x8) IsNaN() Mask64x8

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float32x4) Abs() Float32x4 {
	mask := BroadcastUint32x4(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float32x8) Abs() Float32x8 {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint32x8(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float32x16) Abs() Float32x16 {
	mask := BroadcastUint32x16(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float64x2) Abs() Float64x2 {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint64x2(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float64x4) Abs() Float64x4 {
	mask := BroadcastUint64x4(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float64x8) Abs() Float64x8 {
	mask := BroadcastUint64x8(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float32x4) Neg() Float32x4 {
	mask := BroadcastUint32x4(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float32x8) Neg() Float32x8 {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint32x8(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float32x16) Neg() Float32x16 {
	mask := BroadcastUint32x16(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float64x2) Neg() Float64x2 {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint64x2(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float64x4) Neg() Float64x4 {
	mask := BroadcastUint64x4(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float64x8) Neg() Float64x8 {
	mask := BroadcastUint64x8(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

var f0x16 = [16]int8{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}
var f0x32 = [32]int8{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
	-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}
var f0x64 = [64]int8{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
	-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
	-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
	-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}

// Mul multiplies corresponding elements of two vectors, modulo 2ⁿ.
//
// Emulated, CPU Feature: AVX
func (x Int8x16) Mul(y Int8x16) Int8x16 {
	mask := LoadInt8x16Array(&f0x16)
	mask16 := mask.ToBits().ReshapeToUint16s()
	xe := x.And(mask).ToBits().ReshapeToUint16s()
	xo := x.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ToBits().ReshapeToUint16s()
	yo := y.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s().BitsToInt8()
}

// Mul multiplies corresponding elements of two vectors, modulo 2ⁿ.
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) Mul(y Uint8x16) Uint8x16 {
	mask := LoadInt8x16Array(&f0x16).ToBits()
	mask16 := mask.ReshapeToUint16s()
	xe := x.And(mask).ReshapeToUint16s()
	xo := x.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ReshapeToUint16s()
	yo := y.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s()
}

// Mul multiplies corresponding elements of two vectors, modulo 2ⁿ.
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) Mul(y Int8x32) Int8x32 {
	mask := LoadInt8x32Array(&f0x32)
	mask16 := mask.ToBits().ReshapeToUint16s()
	xe := x.And(mask).ToBits().ReshapeToUint16s()
	xo := x.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ToBits().ReshapeToUint16s()
	yo := y.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s().BitsToInt8()
}

// Mul multiplies corresponding elements of two vectors, modulo 2ⁿ.
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) Mul(y Int8x64) Int8x64 {
	mask := LoadInt8x64Array(&f0x64)
	mask16 := mask.ToBits().ReshapeToUint16s()
	xe := x.And(mask).ToBits().ReshapeToUint16s()
	xo := x.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ToBits().ReshapeToUint16s()
	yo := y.AndNot(mask).ToBits().ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s().BitsToInt8()
}

// Mul multiplies corresponding elements of two vectors, modulo 2ⁿ.
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) Mul(y Uint8x32) Uint8x32 {
	mask := LoadInt8x32Array(&f0x32).ToBits()
	mask16 := mask.ReshapeToUint16s()
	xe := x.And(mask).ReshapeToUint16s()
	xo := x.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ReshapeToUint16s()
	yo := y.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s()
}

// Mul multiplies corresponding elements of two vectors, modulo 2ⁿ.
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) Mul(y Uint8x64) Uint8x64 {
	mask := LoadInt8x64Array(&f0x64).ToBits()
	mask16 := mask.ReshapeToUint16s()
	xe := x.And(mask).ReshapeToUint16s()
	xo := x.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	ye := y.And(mask).ReshapeToUint16s()
	yo := y.AndNot(mask).ReshapeToUint16s().ShiftAllRight(8)
	pe := xe.Mul(ye).And(mask16)
	po := xo.Mul(yo).And(mask16).ShiftAllLeft(8)
	return pe.Or(po).ReshapeToUint8s()
}
