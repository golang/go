// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float32x4) Abs() (z Float32x4) {
	mask := BroadcastUint32x4(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float32x8) Abs() (z Float32x8) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint32x8(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float32x16) Abs() (z Float32x16) {
	mask := BroadcastUint32x16(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float64x2) Abs() (z Float64x2) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint64x2(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float64x4) Abs() (z Float64x4) {
	mask := BroadcastUint64x4(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Abs returns the absolute values of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float64x8) Abs() (z Float64x8) {
	mask := BroadcastUint64x8(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float32x4) Neg() (z Float32x4) {
	mask := BroadcastUint32x4(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float32x8) Neg() (z Float32x8) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint32x8(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float32x16) Neg() (z Float32x16) {
	mask := BroadcastUint32x16(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX
func (x Float64x2) Neg() (z Float64x2) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint64x2(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX2
func (x Float64x4) Neg() (z Float64x4) {
	mask := BroadcastUint64x4(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

// Neg returns the negation of the elements of x
//
// Emulated, CPU Feature AVX512
func (x Float64x8) Neg() (z Float64x8) {
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
func (x Int8x16) Mul(y Int8x16) (z Int8x16) {
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
func (x Uint8x16) Mul(y Uint8x16) (z Uint8x16) {
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
func (x Int8x32) Mul(y Int8x32) (z Int8x32) {
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
func (x Int8x64) Mul(y Int8x64) (z Int8x64) {
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
func (x Uint8x32) Mul(y Uint8x32) (z Uint8x32) {
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
func (x Uint8x64) Mul(y Uint8x64) (z Uint8x64) {
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

var popcnt4x16 = [16]int8{0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4}
var popcnt4x32 = [32]int8{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
}

// OnesCount counts the number of set bits in each element.
//
// Emulated, CPU Feature: AVX
func (x Int8x16) OnesCount() (z Int8x16) {
	if X86.AVX512BITALG() {
		return x.onesCount()
	}
	lut := LoadInt8x16Array(&popcnt4x16)
	mask0f := BroadcastInt8x16(0x0f)
	lo := x.And(mask0f)
	hi := x.ToBits().ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().BitsToInt8().And(mask0f)
	return lut.PermuteOrZero(lo).Add(lut.PermuteOrZero(hi))
}

// OnesCount counts the number of set bits in each element.
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) OnesCount() (z Uint8x16) {
	if X86.AVX512BITALG() {
		return x.BitsToInt8().onesCount().ToBits()
	}
	lut := LoadInt8x16Array(&popcnt4x16).ToBits()
	mask0f := BroadcastInt8x16(0x0f).ToBits()
	lo := x.And(mask0f)
	hi := x.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0f)
	return lut.PermuteOrZero(lo.BitsToInt8()).Add(lut.PermuteOrZero(hi.BitsToInt8()))
}

// OnesCount counts the number of set bits in each element.
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) OnesCount() (z Int8x32) {
	if X86.AVX512BITALG() {
		return x.onesCount()
	}
	lut := LoadInt8x32Array(&popcnt4x32)
	mask0f := BroadcastInt8x32(0x0f)
	lo := x.And(mask0f)
	hi := x.ToBits().ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().BitsToInt8().And(mask0f)
	return lut.PermuteOrZeroGrouped(lo).Add(lut.PermuteOrZeroGrouped(hi))
}

// OnesCount counts the number of set bits in each element.
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) OnesCount() (z Uint8x32) {
	if X86.AVX512BITALG() {
		return x.BitsToInt8().onesCount().ToBits()
	}
	lut := LoadInt8x32Array(&popcnt4x32).ToBits()
	mask0f := BroadcastInt8x32(0x0f).ToBits()
	lo := x.And(mask0f)
	hi := x.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0f)
	return lut.PermuteOrZeroGrouped(lo.BitsToInt8()).Add(lut.PermuteOrZeroGrouped(hi.BitsToInt8()))
}

// OnesCount counts the number of set bits in each element.
//
// Asm: VPOPCNTB, CPU Feature: AVX512BITALG
func (x Int8x64) OnesCount() (z Int8x64) {
	return x.onesCount()
}

// OnesCount counts the number of set bits in each element.
//
// Asm: VPOPCNTB, CPU Feature: AVX512BITALG
func (x Uint8x64) OnesCount() (z Uint8x64) {
	return x.onesCount()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float32x4) ReduceSum() (z float32) {
	x = x.ConcatAddPairs(x) // [x0+x1, x2+x3, x0+x1, x2+x3]
	x = x.ConcatAddPairs(x) // [(x0+x1)+(x2+x3), ...]
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float64x2) ReduceSum() (z float64) {
	return x.ConcatAddPairs(x).GetElem(0) // [x0+x1, x0+x1]
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float32x8) ReduceSum() (z float32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Float64x4) ReduceSum() (z float64) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) ReduceSum() (z float32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Float64x8) ReduceSum() (z float64) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Int16x8) ReduceSum() (z int16) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Uint16x8) ReduceSum() (z uint16) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Int32x4) ReduceSum() (z int32) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Uint32x4) ReduceSum() (z uint32) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Int16x16) ReduceSum() (z int16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Uint16x16) ReduceSum() (z uint16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Int32x8) ReduceSum() (z int32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Uint32x8) ReduceSum() (z uint32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Int16x32) ReduceSum() (z int16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Uint16x32) ReduceSum() (z uint16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Int32x16) ReduceSum() (z int32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Uint32x16) ReduceSum() (z uint32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Int8x16) ReduceSum() (z int8) {
	s := x.ToBits().SumOf8AbsDiff(Uint8x16{})
	return int8(s.GetElem(0) + s.GetElem(1))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) ReduceSum() (z uint8) {
	s := x.SumOf8AbsDiff(Uint8x16{})
	return uint8(s.GetElem(0) + s.GetElem(1))
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) ReduceSum() (z int8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) ReduceSum() (z uint8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) ReduceSum() (z int8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the sum of all elements in x.
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) ReduceSum() (z uint8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}
