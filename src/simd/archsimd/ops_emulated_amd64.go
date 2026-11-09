// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX
func (x Float32x4) Abs() (z Float32x4) {
	mask := BroadcastUint32x4(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX2
func (x Float32x8) Abs() (z Float32x8) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint32x8(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) Abs() (z Float32x16) {
	mask := BroadcastUint32x16(0x80000000)
	return x.ToBits().AndNot(mask).BitsToFloat32()
}

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX
func (x Float64x2) Abs() (z Float64x2) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint64x2(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX2
func (x Float64x4) Abs() (z Float64x4) {
	mask := BroadcastUint64x4(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX512
func (x Float64x8) Abs() (z Float64x8) {
	mask := BroadcastUint64x8(0x8000000000000000)
	return x.ToBits().AndNot(mask).BitsToFloat64()
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
//
// Emulated, CPU Feature: AVX
func (x Float32x4) Neg() (z Float32x4) {
	mask := BroadcastUint32x4(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
//
// Emulated, CPU Feature: AVX2
func (x Float32x8) Neg() (z Float32x8) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint32x8(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) Neg() (z Float32x16) {
	mask := BroadcastUint32x16(0x80000000)
	return x.ToBits().Xor(mask).BitsToFloat32()
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
//
// Emulated, CPU Feature: AVX
func (x Float64x2) Neg() (z Float64x2) {
	// mask will have a 1 in the sign bit UNLESS x is NaN
	mask := BroadcastUint64x2(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
//
// Emulated, CPU Feature: AVX2
func (x Float64x4) Neg() (z Float64x4) {
	mask := BroadcastUint64x4(0x8000000000000000)
	return x.ToBits().Xor(mask).BitsToFloat64()
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
//
// Emulated, CPU Feature: AVX512
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

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX
func (x Int64x2) Abs() (z Uint64x2) {
	// VPABSQ needs AVX-512. Negate the negative elements instead:
	// with m all ones where x < 0 and zero elsewhere, (x ^ m) - m is -x
	// where x < 0 and x elsewhere. MinInt64 stays MinInt64, which as an
	// unsigned value is its magnitude, 1<<63.
	var zero Int64x2
	m := zero.Greater(x).ToInt64x2()
	return x.Xor(m).Sub(m).ToBits()
}

// Abs returns the elementwise absolute value of x.
//
// Emulated, CPU Feature: AVX2
func (x Int64x4) Abs() (z Uint64x4) {
	// See Int64x2.Abs.
	var zero Int64x4
	m := zero.Greater(x).ToInt64x4()
	return x.Xor(m).Sub(m).ToBits()
}

// ShiftAllRight arithmetically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0 or -1.
//
//	z[i] = x[i] >> shift
//
// Emulated, CPU Feature: AVX
func (x Int64x2) ShiftAllRight(shift uint64) (z Int64x2) {
	// VPSRAQ needs AVX-512. Shift logically, and fill the vacated upper
	// bits with the sign: m is all ones where x < 0, and m << (64-shift)
	// has ones in exactly the top shift bits. Shifting by 64 or more gives
	// 0 for both shifts, so clamping shift to 63 handles large shifts, and
	// shift == 0 gives m << 64 == 0.
	shift = min(shift, 63)
	var zero Int64x2
	m := zero.Greater(x).ToInt64x2().ToBits()
	return x.ToBits().ShiftAllRight(shift).Or(m.ShiftAllLeft(64 - shift)).BitsToInt64()
}

// ShiftAllRight arithmetically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0 or -1.
//
//	z[i] = x[i] >> shift
//
// Emulated, CPU Feature: AVX2
func (x Int64x4) ShiftAllRight(shift uint64) (z Int64x4) {
	// See Int64x2.ShiftAllRight.
	shift = min(shift, 63)
	var zero Int64x4
	m := zero.Greater(x).ToInt64x4().ToBits()
	return x.ToBits().ShiftAllRight(shift).Or(m.ShiftAllLeft(64 - shift)).BitsToInt64()
}

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
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

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
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
	return lut.LookupModOrZero(lo).Add(lut.LookupModOrZero(hi))
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
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
	return lut.LookupModOrZero(lo.BitsToInt8()).Add(lut.LookupModOrZero(hi.BitsToInt8()))
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
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
	return lut.LookupModOrZeroGrouped(lo).Add(lut.LookupModOrZeroGrouped(hi))
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
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
	return lut.LookupModOrZeroGrouped(lo.BitsToInt8()).Add(lut.LookupModOrZeroGrouped(hi.BitsToInt8()))
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
//
// Asm: VPOPCNTB, CPU Feature: AVX512BITALG
func (x Int8x64) OnesCount() (z Int8x64) {
	return x.onesCount()
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
//
// Asm: VPOPCNTB, CPU Feature: AVX512BITALG
func (x Uint8x64) OnesCount() (z Uint8x64) {
	return x.onesCount()
}

// ReduceSum returns the scalar sum of the elements of x.
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
//	z = (x[0] + x[1]) + (x[2] + x[3])
//
// Emulated, CPU Feature: AVX
func (x Float32x4) ReduceSum() (z float32) {
	x = x.ConcatAddPairs(x) // [x0+x1, x2+x3, x0+x1, x2+x3]
	x = x.ConcatAddPairs(x) // [(x0+x1)+(x2+x3), ...]
	return x.GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1]
//
// Emulated, CPU Feature: AVX
func (x Float64x2) ReduceSum() (z float64) {
	return x.ConcatAddPairs(x).GetElem(0) // [x0+x1, x0+x1]
}

// ReduceSum returns the scalar sum of the elements of x.
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
//	z = ((x[0] + x[1]) + (x[2] + x[3])) + ...
//
// Emulated, CPU Feature: AVX
func (x Float32x8) ReduceSum() (z float32) {
	// Sum adjacent pairs within each 128-bit half before combining the halves,
	// to match the pairwise summation order of the other platforms.
	x = x.ConcatAddPairsGrouped(x) // [x0+x1, x2+x3, ... | x4+x5, x6+x7, ...]
	x = x.ConcatAddPairsGrouped(x) // [(x0+x1)+(x2+x3), ... | (x4+x5)+(x6+x7), ...]
	return x.GetLo().Add(x.GetHi()).GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
//	z = (x[0] + x[1]) + (x[2] + x[3])
//
// Emulated, CPU Feature: AVX
func (x Float64x4) ReduceSum() (z float64) {
	x = x.ConcatAddPairsGrouped(x) // [x0+x1, x0+x1 | x2+x3, x2+x3]
	return x.GetLo().Add(x.GetHi()).GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
//	z = ((x[0] + x[1]) + (x[2] + x[3])) + ...
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) ReduceSum() (z float32) {
	// There is no 512-bit horizontal add, so start with the 256-bit halves.
	lo, hi := x.GetLo(), x.GetHi()
	y := lo.ConcatAddPairsGrouped(hi) // [x0+x1, x2+x3, x8+x9, x10+x11 | x4+x5, x6+x7, x12+x13, x14+x15]
	y = y.ConcatAddPairsGrouped(y)    // [x0..3, x8..11, ... | x4..7, x12..15, ...]
	w := y.GetLo().Add(y.GetHi())     // [x0..7, x8..15, ...]
	return w.ConcatAddPairs(w).GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
//	z = ((x[0] + x[1]) + (x[2] + x[3])) + ...
//
// Emulated, CPU Feature: AVX512
func (x Float64x8) ReduceSum() (z float64) {
	lo, hi := x.GetLo(), x.GetHi()
	y := lo.ConcatAddPairsGrouped(hi) // [x0+x1, x4+x5 | x2+x3, x6+x7]
	w := y.GetLo().Add(y.GetHi())     // [x0..3, x4..7]
	return w.ConcatAddPairs(w).GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Int16x8) ReduceSum() (z int16) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Uint16x8) ReduceSum() (z uint16) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Int32x4) ReduceSum() (z int32) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Uint32x4) ReduceSum() (z uint32) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.GetElem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Int16x16) ReduceSum() (z int16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Uint16x16) ReduceSum() (z uint16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Int32x8) ReduceSum() (z int32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Uint32x8) ReduceSum() (z uint32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Int16x32) ReduceSum() (z int16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Uint16x32) ReduceSum() (z uint16) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Int32x16) ReduceSum() (z int32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Uint32x16) ReduceSum() (z uint32) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Int8x16) ReduceSum() (z int8) {
	s := x.ToBits().SumOf8AbsDiff(Uint8x16{})
	return int8(s.GetElem(0) + s.GetElem(1))
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) ReduceSum() (z uint8) {
	s := x.SumOf8AbsDiff(Uint8x16{})
	return uint8(s.GetElem(0) + s.GetElem(1))
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) ReduceSum() (z int8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) ReduceSum() (z uint8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) ReduceSum() (z int8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) ReduceSum() (z uint8) {
	return x.GetLo().Add(x.GetHi()).ReduceSum()
}

// LookupOrZero permutes x, returning 0 for out-of-range indices.
//
//	if indices[i] < 16 {
//	    z[i] = x[indices[i]]
//	} else {
//	    z[i] = 0
//	}
//
// Emulated, CPU Feature: AVX
func (x Int8x16) LookupOrZero(indices Uint8x16) (z Int8x16) {
	c70 := BroadcastUint8x16(0x70)
	// Adding 0x70 with unsigned saturation maps 0..15 to 0x70..0x7f
	// (non-negative, preserving the low 4 bits) and >= 16 to >= 0x80
	// (negative, which LookupModOrZero zeroes).
	return x.LookupModOrZero(indices.AddSaturated(c70).BitsToInt8())
}

// LookupOrZero permutes x, returning 0 for out-of-range indices.
//
//	if indices[i] < 32 {
//	    z[i] = x[indices[i]]
//	} else {
//	    z[i] = 0
//	}
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) LookupOrZero(indices Uint8x32) (z Int8x32) {
	if X86.AVX512VBMI() {
		return x.LookupMod(indices).Masked(indices.Less(BroadcastUint8x32(32)))
	}
	ll := x.ConcatPermute128Scalars(0, 0, x) // Double the low 128 bits of x. (The x argument is unused.)
	hh := x.ConcatPermute128Scalars(1, 1, x) // Double the high 128 bits of x.
	c70 := BroadcastUint8x32(0x70)
	c16 := BroadcastUint8x32(16)
	// Adding 0x70 with unsigned saturation maps 0..15 to 0x70..0x7f
	// (non-negative, preserving the low 4 bits) and >= 16 to >= 0x80
	// (negative, which LookupModOrZeroGrouped zeroes).
	yl := indices.AddSaturated(c70)
	yh := indices.Sub(c16).AddSaturated(c70)        // Map 16..31 to 0x70..0x7f, anything else to < 0.
	ll = ll.LookupModOrZeroGrouped(yl.BitsToInt8()) // Elements with 0 <= indexes < 16, zero elsewhere.
	hh = hh.LookupModOrZeroGrouped(yh.BitsToInt8()) // Elements with 16 <= indexes < 32, zero elsewhere.
	return ll.Or(hh)
}

// LookupOrZero permutes x, returning 0 for out-of-range indices.
//
//	if indices[i] < 64 {
//	    z[i] = x[indices[i]]
//	} else {
//	    z[i] = 0
//	}
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) LookupOrZero(indices Uint8x64) (z Int8x64) {
	if X86.AVX512VBMI() {
		return x.LookupMod(indices).Masked(indices.Less(BroadcastUint8x64(64)))
	}
	xl, xh := x.GetLo(), x.GetHi()
	ll := xl.ConcatPermute128Scalars(0, 0, xl) // 2 copies of elements 0-15
	lh := xl.ConcatPermute128Scalars(1, 1, xl) // 2 copies of elements 16-31
	hl := xh.ConcatPermute128Scalars(0, 0, xh) // 2 copies of elements 32-47
	hh := xh.ConcatPermute128Scalars(1, 1, xh) // 2 copies of elements 48-63
	ll4 := z.SetLo(ll).SetHi(ll)               // 4 copies of elements 0-15 etc
	lh4 := z.SetLo(lh).SetHi(lh)
	hl4 := z.SetLo(hl).SetHi(hl)
	hh4 := z.SetLo(hh).SetHi(hh)

	c70 := BroadcastUint8x64(0x70)
	c16 := BroadcastUint8x64(16)
	y0 := indices.AddSaturated(c70)
	y := indices.Sub(c16)
	y1 := y.AddSaturated(c70)
	y = y.Sub(c16)
	y2 := y.AddSaturated(c70)
	y = y.Sub(c16)
	y3 := y.AddSaturated(c70)
	// y0..y3 have bit 7 == 0 (with low 4 bits == indices-16*k) iff indices is in 16*k..16*k+15.
	ll4 = ll4.LookupModOrZeroGrouped(y0.BitsToInt8()) // Elements with 0 <= indexes < 16, zero elsewhere.
	lh4 = lh4.LookupModOrZeroGrouped(y1.BitsToInt8()) // Elements with 16 <= indexes < 32, zero elsewhere.
	hl4 = hl4.LookupModOrZeroGrouped(y2.BitsToInt8()) // Elements with 32 <= indexes < 48, zero elsewhere.
	hh4 = hh4.LookupModOrZeroGrouped(y3.BitsToInt8()) // Elements with 48 <= indexes < 64, zero elsewhere.
	return ll4.Or(lh4).Or(hl4.Or(hh4))
}
