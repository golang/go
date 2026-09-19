// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

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

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float32x4) ReduceSum() float32 {
	x = x.ConcatAddPairs(x) // [x0+x1, x2+x3, x0+x1, x2+x3]
	x = x.ConcatAddPairs(x) // [(x0+x1)+(x2+x3), ...]
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float64x2) ReduceSum() float64 {
	return x.ConcatAddPairs(x).GetElem(0) // [x0+x1, x0+x1]
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float32x8) ReduceSum() float32 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float64x4) ReduceSum() float64 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) ReduceSum() float32 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Float64x8) ReduceSum() float64 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Int16x8) ReduceSum() int16 {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Uint16x8) ReduceSum() uint16 {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Int32x4) ReduceSum() int32 {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Uint32x4) ReduceSum() uint32 {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Int16x16) ReduceSum() int16 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Uint16x16) ReduceSum() uint16 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Int32x8) ReduceSum() int32 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Uint32x8) ReduceSum() uint32 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Int16x32) ReduceSum() int16 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Uint16x32) ReduceSum() uint16 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Int32x16) ReduceSum() int32 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Uint32x16) ReduceSum() uint32 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Int8x16) ReduceSum() int8 {
	s := x.ToBits().SumOf8AbsDiff(Uint8x16{})
	return int8(s.GetElem(0) + s.GetElem(1))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) ReduceSum() uint8 {
	s := x.SumOf8AbsDiff(Uint8x16{})
	return uint8(s.GetElem(0) + s.GetElem(1))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) ReduceSum() int8 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) ReduceSum() uint8 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) ReduceSum() int8 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) ReduceSum() uint8 {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}
