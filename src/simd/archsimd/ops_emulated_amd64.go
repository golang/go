// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable && amd64

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
// As in Go, if shift is greater than or equal to the element width, the
// result is 0 or -1, depending on the sign of the element.
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
// As in Go, if shift is greater than or equal to the element width, the
// result is 0 or -1, depending on the sign of the element.
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
var popcnt4x64 = [64]int8{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
//
// Asm: VPOPCNTB if CPU feature AVX512BITALG, Emulated otherwise, CPU Feature: AVX
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
// Asm: VPOPCNTB if CPU feature AVX512BITALG, Emulated otherwise, CPU Feature: AVX
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
// Asm: VPOPCNTB if CPU feature AVX512BITALG, Emulated otherwise, CPU Feature: AVX2
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
// Asm: VPOPCNTB if CPU feature AVX512BITALG, Emulated otherwise, CPU Feature: AVX2
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
// Asm: VPOPCNTB if CPU feature AVX512BITALG, Emulated otherwise, CPU Feature: AVX512
func (x Int8x64) OnesCount() (z Int8x64) {
	if X86.AVX512BITALG() {
		return x.onesCount()
	}
	lut := LoadInt8x64Array(&popcnt4x64)
	mask0f := BroadcastInt8x64(0x0f)
	lo := x.And(mask0f)
	hi := x.ToBits().ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().BitsToInt8().And(mask0f)
	return lut.LookupModOrZeroGrouped(lo).Add(lut.LookupModOrZeroGrouped(hi))
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
//
// Asm: VPOPCNTB if CPU feature AVX512BITALG, Emulated otherwise, CPU Feature: AVX512
func (x Uint8x64) OnesCount() (z Uint8x64) {
	if X86.AVX512BITALG() {
		return x.onesCount()
	}
	lut := LoadInt8x64Array(&popcnt4x64).ToBits()
	mask0f := BroadcastInt8x64(0x0f).ToBits()
	lo := x.And(mask0f)
	hi := x.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0f)
	return lut.LookupModOrZeroGrouped(lo.BitsToInt8()).Add(lut.LookupModOrZeroGrouped(hi.BitsToInt8()))
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
	return x.Elem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1]
//
// Emulated, CPU Feature: AVX
func (x Float64x2) ReduceSum() (z float64) {
	return x.ConcatAddPairs(x).Elem(0) // [x0+x1, x0+x1]
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
	return x.Lo().Add(x.Hi()).Elem(0)
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
	return x.Lo().Add(x.Hi()).Elem(0)
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
	lo, hi := x.Lo(), x.Hi()
	y := lo.ConcatAddPairsGrouped(hi) // [x0+x1, x2+x3, x8+x9, x10+x11 | x4+x5, x6+x7, x12+x13, x14+x15]
	y = y.ConcatAddPairsGrouped(y)    // [x0..3, x8..11, ... | x4..7, x12..15, ...]
	w := y.Lo().Add(y.Hi())           // [x0..7, x8..15, ...]
	return w.ConcatAddPairs(w).Elem(0)
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
	lo, hi := x.Lo(), x.Hi()
	y := lo.ConcatAddPairsGrouped(hi) // [x0+x1, x4+x5 | x2+x3, x6+x7]
	w := y.Lo().Add(y.Hi())           // [x0..3, x4..7]
	return w.ConcatAddPairs(w).Elem(0)
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
	return x.Elem(0)
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
	return x.Elem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Int32x4) ReduceSum() (z int32) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.Elem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Uint32x4) ReduceSum() (z uint32) {
	x = x.ConcatAddPairs(x)
	x = x.ConcatAddPairs(x)
	return x.Elem(0)
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Int16x16) ReduceSum() (z int16) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Uint16x16) ReduceSum() (z uint16) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Int32x8) ReduceSum() (z int32) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Uint32x8) ReduceSum() (z uint32) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Int16x32) ReduceSum() (z int16) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Uint16x32) ReduceSum() (z uint16) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Int32x16) ReduceSum() (z int32) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Uint32x16) ReduceSum() (z uint32) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Int8x16) ReduceSum() (z int8) {
	s := x.ToBits().SumOf8AbsDiff(Uint8x16{})
	return int8(s.Elem(0) + s.Elem(1))
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) ReduceSum() (z uint8) {
	s := x.SumOf8AbsDiff(Uint8x16{})
	return uint8(s.Elem(0) + s.Elem(1))
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) ReduceSum() (z int8) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) ReduceSum() (z uint8) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) ReduceSum() (z int8) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) ReduceSum() (z uint8) {
	return x.Lo().Add(x.Hi()).ReduceSum()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint16x8) ReduceMin() (z uint16) {
	return x.minPosUnsigned().Elem(0)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint16x8) ReduceMax() (z uint16) {
	return ^x.Not().minPosUnsigned().Elem(0)
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int16x8) ReduceMin() (z int16) {
	return int16(x.ToBits().Xor(BroadcastUint16x8(0x8000)).minPosUnsigned().Elem(0) ^ 0x8000)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int16x8) ReduceMax() (z int16) {
	return int16(x.ToBits().Xor(BroadcastUint16x8(0x7fff)).minPosUnsigned().Elem(0) ^ 0x7fff)
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint16x16) ReduceMin() (z uint16) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint16x16) ReduceMax() (z uint16) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int16x16) ReduceMin() (z int16) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int16x16) ReduceMax() (z int16) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint16x32) ReduceMin() (z uint16) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint16x32) ReduceMax() (z uint16) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int16x32) ReduceMin() (z int16) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int16x32) ReduceMax() (z int16) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) ReduceMin() (z uint8) {
	// Shift odd bytes down into low bytes of 16-bit lanes with zero high bytes.
	// Bytewise min puts min(x[2i], x[2i+1]) in the low bytes and clears every
	// high byte to min(x[2i+1], 0) == 0, giving 8 zero-extended uint16s for
	// VPHMINPOSUW without constants or shuffles.
	t := x.ReshapeToUint16s().ShiftAllRight(8).ReshapeToUint8s()
	return uint8(x.Min(t).ReshapeToUint16s().minPosUnsigned().Elem(0))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint8x16) ReduceMax() (z uint8) {
	return ^x.Not().ReduceMin()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int8x16) ReduceMin() (z int8) {
	return int8(x.ToBits().Xor(BroadcastUint8x16(0x80)).ReduceMin() ^ 0x80)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int8x16) ReduceMax() (z int8) {
	return int8(x.ToBits().Xor(BroadcastUint8x16(0x7f)).ReduceMin() ^ 0x7f)
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) ReduceMin() (z uint8) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint8x32) ReduceMax() (z uint8) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) ReduceMin() (z int8) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int8x32) ReduceMax() (z int8) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) ReduceMin() (z uint8) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint8x64) ReduceMax() (z uint8) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) ReduceMin() (z int8) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int8x64) ReduceMax() (z int8) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

func (x Int32x4) reduceMaxVec() Int32x4 {
	m := x.Max(x.permuteScalars(2 + 3<<2 + 0<<4 + 1<<6))
	return m.Max(m.permuteScalars(1 + 0<<2 + 3<<4 + 2<<6))
}

func (x Int32x4) reduceMinVec() Int32x4 {
	m := x.Min(x.permuteScalars(2 + 3<<2 + 0<<4 + 1<<6))
	return m.Min(m.permuteScalars(1 + 0<<2 + 3<<4 + 2<<6))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int32x4) ReduceMax() (z int32) {
	return x.reduceMaxVec().Elem(0)
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int32x4) ReduceMin() (z int32) {
	return x.reduceMinVec().Elem(0)
}

func (x Uint32x4) reduceMaxVec() Uint32x4 {
	m := x.Max(x.permuteScalars(2 + 3<<2 + 0<<4 + 1<<6))
	return m.Max(m.permuteScalars(1 + 0<<2 + 3<<4 + 2<<6))
}

func (x Uint32x4) reduceMinVec() Uint32x4 {
	m := x.Min(x.permuteScalars(2 + 3<<2 + 0<<4 + 1<<6))
	return m.Min(m.permuteScalars(1 + 0<<2 + 3<<4 + 2<<6))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint32x4) ReduceMax() (z uint32) {
	return x.reduceMaxVec().Elem(0)
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint32x4) ReduceMin() (z uint32) {
	return x.reduceMinVec().Elem(0)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int32x8) ReduceMax() (z int32) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int32x8) ReduceMin() (z int32) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint32x8) ReduceMax() (z uint32) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint32x8) ReduceMin() (z uint32) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int32x16) ReduceMax() (z int32) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int32x16) ReduceMin() (z int32) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint32x16) ReduceMax() (z uint32) {
	return x.Lo().Max(x.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint32x16) ReduceMin() (z uint32) {
	return x.Lo().Min(x.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int64x2) ReduceMax() (z int64) {
	return max(x.Elem(0), x.Elem(1))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Int64x2) ReduceMin() (z int64) {
	return min(x.Elem(0), x.Elem(1))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint64x2) ReduceMax() (z uint64) {
	return max(x.Elem(0), x.Elem(1))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX
func (x Uint64x2) ReduceMin() (z uint64) {
	return min(x.Elem(0), x.Elem(1))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int64x4) ReduceMax() (z int64) {
	lo, hi := x.Lo(), x.Hi()
	return lo.IfElse(lo.Greater(hi), hi).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Int64x4) ReduceMin() (z int64) {
	lo, hi := x.Lo(), x.Hi()
	return hi.IfElse(lo.Greater(hi), lo).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint64x4) ReduceMax() (z uint64) {
	lo, hi := x.Lo(), x.Hi()
	return max(max(lo.Elem(0), lo.Elem(1)), max(hi.Elem(0), hi.Elem(1)))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX2
func (x Uint64x4) ReduceMin() (z uint64) {
	lo, hi := x.Lo(), x.Hi()
	return min(min(lo.Elem(0), lo.Elem(1)), min(hi.Elem(0), hi.Elem(1)))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int64x8) ReduceMax() (z int64) {
	// Fold directly with AVX-512 VPMAXSQ rather than calling m.ReduceMax(),
	// which would fall back to the AVX2 VPCMPGTQ/scalar path.
	m := x.Lo().Max(x.Hi())
	return m.Lo().Max(m.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Int64x8) ReduceMin() (z int64) {
	// Fold directly with AVX-512 VPMINSQ rather than calling m.ReduceMin().
	m := x.Lo().Min(x.Hi())
	return m.Lo().Min(m.Hi()).ReduceMin()
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint64x8) ReduceMax() (z uint64) {
	// Fold directly with AVX-512 VPMAXUQ rather than calling m.ReduceMax().
	m := x.Lo().Max(x.Hi())
	return m.Lo().Max(m.Hi()).ReduceMax()
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// Emulated, CPU Feature: AVX512
func (x Uint64x8) ReduceMin() (z uint64) {
	// Fold directly with AVX-512 VPMINUQ rather than calling m.ReduceMin().
	m := x.Lo().Min(x.Hi())
	return m.Lo().Min(m.Hi()).ReduceMin()
}

// Float ReduceMin and ReduceMax follow IEEE 754-2019 minimum/maximum, like Go's
// min and max built-ins, but VMINP[SD]/VMAXP[SD] return their second operand
// if either operand is NaN or both are zero, regardless of sign. Hence, the
// reductions below use a VMINP[SD]/VMAXP[SD] reduction tree to compute a result
// that is correct unless there are NaNs or signed zeroes, and in parallel build
// a bitwise "fixup" tree that recovers those cases:
//
//   - If there are no NaNs, the reduction tree computes the right value except
//     that the sign of a zero result may be wrong. For min, the correct result
//     has its sign bit set iff any element has its sign bit set: if the minimum
//     is negative, it's already set; if it's positive, no element has its sign
//     bit set; and if it's zero, then all elements are >= 0, so an element with
//     its sign bit set must be -0. Symmetrically, for max, the correct result
//     has its sign bit clear iff any element has its sign bit clear.
//
//   - If there are any NaNs, the result must be a NaN.
//
// The fixup tree ORs together a value that is all 1s in lanes that see a NaN,
// and otherwise has just the sign bit set if that lane sees a negative sign
// (for min) or a positive sign (for max). Reducing this vector yields all 1s if
// any element is NaN, and otherwise just the sign bit indicating the correct
// result sign.
//
// For min, ORing this fixup into the reduction tree result yields the correct
// sign or a NaN.
//
// For max, we flip the fixup's sign bit and take the signed integer maximum of
// that and the reduction tree result. Viewed as signed integers, floats with
// the sign bit clear are non-negative and floats with it set are negative. The
// flipped fixup is 0x7fff... if any element is NaN, which is larger than any
// other value and is itself a NaN. Otherwise, it's 0 if any element has its sign
// bit clear, which turns a -0 result into +0 and leaves a non-negative result
// alone (and the result can't be negative in this case). Otherwise, it's
// 0x8000..., which is smaller than any other value, so it leaves the result
// alone. There's no 64-bit VPMAXSQ before AVX-512, so for float64 we use 32-bit
// VPMAXSD and set the low half of the fixup to 0x8000_0000 (or 0x7fff_ffff for
// NaN). This leaves the low half of the result alone unless the high half is
// already a NaN, and only the high half matters for the sign of zero.
//
// The fixup tree runs in parallel with the reduction tree, so this adds just
// one operation (VPOR for min or VPMAXSD for max) to the critical path.
//
// For Float64x2, the reduction tree is a single VMINPD/VMAXPD, so the fixup
// tree would be longer than the reduction tree. Instead, it computes the
// pairwise operation in both operand orders. For min, the two results are equal
// unless an operand is NaN, in which case one result is that NaN, or both
// operands are zero, in which case one result is -0 if either operand is. So
// ORing them yields the IEEE minimum. For max, ANDing the two results yields
// +0 if either operand is +0, and ORing them and clearing the sign yields a NaN
// if either is a NaN. Combining these two yields the IEEE maximum.

// reduceMinTree returns a vector whose element 0 is the VMINPS reduction of x.
func (x Float32x4) reduceMinTree() Float32x4 {
	x = x.Min(x.concatPermuteScalars(2, 3, 2, 3, x))
	return x.Min(x.concatPermuteScalars(1, 1, 1, 1, x))
}

// reduceMaxTree returns a vector whose element 0 is the VMAXPS reduction of x.
func (x Float32x4) reduceMaxTree() Float32x4 {
	x = x.Max(x.concatPermuteScalars(2, 3, 2, 3, x))
	return x.Max(x.concatPermuteScalars(1, 1, 1, 1, x))
}

// reduceOr returns a vector whose element 0 is the bitwise OR of all elements
// of f.
func (f Uint32x4) reduceOr() Uint32x4 {
	f = f.Or(f.concatPermuteScalars(2, 3, 2, 3, f))
	return f.Or(f.concatPermuteScalars(1, 1, 1, 1, f))
}

// minFixup32 returns the IEEE min fixup vector for x and y.
func minFixup32(x, y Float32x4) Uint32x4 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt32x4().ToBits() // VCMPPS $3, x, y
	return x.ToBits().Or(y.ToBits()).And(BroadcastUint32x4(1 << 31)).Or(nan)
}

// maxFixup32 returns the IEEE max fixup vector for x and y.
func maxFixup32(x, y Float32x4) Uint32x4 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt32x4().ToBits() // VCMPPS $3, x, y
	return BroadcastUint32x4(1 << 31).AndNot(x.ToBits().And(y.ToBits())).Or(nan)
}

// minFixup32x8 is like minFixup32, but for 256-bit vectors.
func minFixup32x8(x, y Float32x8) Uint32x8 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt32x8().ToBits() // VCMPPS $3, x, y
	return x.ToBits().Or(y.ToBits()).And(BroadcastUint32x8(1 << 31)).Or(nan)
}

// maxFixup32x8 is like maxFixup32, but for 256-bit vectors.
func maxFixup32x8(x, y Float32x8) Uint32x8 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt32x8().ToBits() // VCMPPS $3, x, y
	return BroadcastUint32x8(1 << 31).AndNot(x.ToBits().And(y.ToBits())).Or(nan)
}

// applyMinFixup32 applies min fixup vector f to the result of a VMINPS
// reduction tree in element 0 of m.
func applyMinFixup32(m Float32x4, f Uint32x4) float32 {
	return m.ToBits().Or(f.reduceOr()).BitsToFloat32().Elem(0)
}

// applyMaxFixup32 applies max fixup vector f to the result of a VMAXPS
// reduction tree in element 0 of m.
func applyMaxFixup32(m Float32x4, f Uint32x4) float32 {
	z := f.reduceOr().Xor(BroadcastUint32x4(1 << 31)).BitsToInt32()
	return m.ToBits().BitsToInt32().Max(z).ToBits().BitsToFloat32().Elem(0)
}

// reduceMinTree returns a vector whose element 0 is the VMINPD reduction of x.
func (x Float64x2) reduceMinTree() Float64x2 {
	return x.Min(x.concatPermuteScalars(1, 1, x))
}

// reduceMaxTree returns a vector whose element 0 is the VMAXPD reduction of x.
func (x Float64x2) reduceMaxTree() Float64x2 {
	return x.Max(x.concatPermuteScalars(1, 1, x))
}

// reduceOr returns a vector whose element 0 is the bitwise OR of all elements
// of f.
func (f Uint64x2) reduceOr() Uint64x2 {
	return f.Or(f.concatPermuteScalars(1, 1, f))
}

// minFixup64 returns the IEEE min fixup vector for x and y.
func minFixup64(x, y Float64x2) Uint64x2 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt64x2().ToBits() // VCMPPD $3, x, y
	return x.ToBits().Or(y.ToBits()).And(BroadcastUint64x2(1 << 63)).Or(nan)
}

// maxFixup64 returns the IEEE max fixup vector for x and y.
func maxFixup64(x, y Float64x2) Uint64x2 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt64x2().ToBits() // VCMPPD $3, x, y
	return BroadcastUint64x2(1 << 63).AndNot(x.ToBits().And(y.ToBits())).Or(nan)
}

// minFixup64x4 is like minFixup64, but for 256-bit vectors.
func minFixup64x4(x, y Float64x4) Uint64x4 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt64x4().ToBits() // VCMPPD $3, x, y
	return x.ToBits().Or(y.ToBits()).And(BroadcastUint64x4(1 << 63)).Or(nan)
}

// maxFixup64x4 is like maxFixup64, but for 256-bit vectors.
func maxFixup64x4(x, y Float64x4) Uint64x4 {
	nan := x.IsNaN().Or(y.IsNaN()).ToInt64x4().ToBits() // VCMPPD $3, x, y
	return BroadcastUint64x4(1 << 63).AndNot(x.ToBits().And(y.ToBits())).Or(nan)
}

// applyMinFixup64 applies min fixup vector f to the result of a VMINPD
// reduction tree in element 0 of m.
func applyMinFixup64(m Float64x2, f Uint64x2) float64 {
	return m.ToBits().Or(f.reduceOr()).BitsToFloat64().Elem(0)
}

// applyMaxFixup64 applies max fixup vector f to the result of a VMAXPD
// reduction tree in element 0 of m.
func applyMaxFixup64(m Float64x2, f Uint64x2) float64 {
	z := f.reduceOr().Xor(BroadcastUint64x2(1<<63 | 1<<31)).ReshapeToUint32s().BitsToInt32()
	return m.ToBits().ReshapeToUint32s().BitsToInt32().Max(z).ToBits().ReshapeToUint64s().BitsToFloat64().Elem(0)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's max built-in and the IEEE 754-2019 maximum operation. Note
// that on AMD64 (and only AMD64), it differs from [Max].
//
// Emulated, CPU Feature: AVX
func (x Float32x4) ReduceMax() (z float32) {
	nan := x.IsNaN().ToInt32x4().ToBits()
	return applyMaxFixup32(x.reduceMaxTree(), BroadcastUint32x4(1<<31).AndNot(x.ToBits()).Or(nan))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's min built-in and the IEEE 754-2019 minimum operation. Note
// that on AMD64 (and only AMD64), it differs from [Min].
//
// Emulated, CPU Feature: AVX
func (x Float32x4) ReduceMin() (z float32) {
	nan := x.IsNaN().ToInt32x4().ToBits()
	return applyMinFixup32(x.reduceMinTree(), x.ToBits().And(BroadcastUint32x4(1<<31)).Or(nan))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's max built-in and the IEEE 754-2019 maximum operation. Note
// that on AMD64 (and only AMD64), it differs from [Max].
//
// Emulated, CPU Feature: AVX
func (x Float32x8) ReduceMax() (z float32) {
	lo, hi := x.Lo(), x.Hi()
	return applyMaxFixup32(lo.Max(hi).reduceMaxTree(), maxFixup32(lo, hi))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's min built-in and the IEEE 754-2019 minimum operation. Note
// that on AMD64 (and only AMD64), it differs from [Min].
//
// Emulated, CPU Feature: AVX
func (x Float32x8) ReduceMin() (z float32) {
	lo, hi := x.Lo(), x.Hi()
	return applyMinFixup32(lo.Min(hi).reduceMinTree(), minFixup32(lo, hi))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's max built-in and the IEEE 754-2019 maximum operation. Note
// that on AMD64 (and only AMD64), it differs from [Max].
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) ReduceMax() (z float32) {
	lo, hi := x.Lo(), x.Hi()
	m, f := lo.Max(hi), maxFixup32x8(lo, hi)
	return applyMaxFixup32(m.Lo().Max(m.Hi()).reduceMaxTree(), f.Lo().Or(f.Hi()))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's min built-in and the IEEE 754-2019 minimum operation. Note
// that on AMD64 (and only AMD64), it differs from [Min].
//
// Emulated, CPU Feature: AVX512
func (x Float32x16) ReduceMin() (z float32) {
	lo, hi := x.Lo(), x.Hi()
	m, f := lo.Min(hi), minFixup32x8(lo, hi)
	return applyMinFixup32(m.Lo().Min(m.Hi()).reduceMinTree(), f.Lo().Or(f.Hi()))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's max built-in and the IEEE 754-2019 maximum operation. Note
// that on AMD64 (and only AMD64), it differs from [Max].
//
// Emulated, CPU Feature: AVX
func (x Float64x2) ReduceMax() (z float64) {
	y := x.concatPermuteScalars(1, 1, x)
	m1, m2 := x.Max(y).ToBits(), y.Max(x).ToBits()
	return m1.And(m2).Or(m1.Or(m2).AndNot(BroadcastUint64x2(1 << 63))).BitsToFloat64().Elem(0)
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's min built-in and the IEEE 754-2019 minimum operation. Note
// that on AMD64 (and only AMD64), it differs from [Min].
//
// Emulated, CPU Feature: AVX
func (x Float64x2) ReduceMin() (z float64) {
	y := x.concatPermuteScalars(1, 1, x)
	return x.Min(y).ToBits().Or(y.Min(x).ToBits()).BitsToFloat64().Elem(0)
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's max built-in and the IEEE 754-2019 maximum operation. Note
// that on AMD64 (and only AMD64), it differs from [Max].
//
// Emulated, CPU Feature: AVX
func (x Float64x4) ReduceMax() (z float64) {
	lo, hi := x.Lo(), x.Hi()
	return applyMaxFixup64(lo.Max(hi).reduceMaxTree(), maxFixup64(lo, hi))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's min built-in and the IEEE 754-2019 minimum operation. Note
// that on AMD64 (and only AMD64), it differs from [Min].
//
// Emulated, CPU Feature: AVX
func (x Float64x4) ReduceMin() (z float64) {
	lo, hi := x.Lo(), x.Hi()
	return applyMinFixup64(lo.Min(hi).reduceMinTree(), minFixup64(lo, hi))
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's max built-in and the IEEE 754-2019 maximum operation. Note
// that on AMD64 (and only AMD64), it differs from [Max].
//
// Emulated, CPU Feature: AVX512
func (x Float64x8) ReduceMax() (z float64) {
	lo, hi := x.Lo(), x.Hi()
	m, f := lo.Max(hi), maxFixup64x4(lo, hi)
	return applyMaxFixup64(m.Lo().Max(m.Hi()).reduceMaxTree(), f.Lo().Or(f.Hi()))
}

// ReduceMin returns the smallest element of x.
//
//	z = min(x[0], x[1], ...)
//
// If any element is NaN, the result is NaN, and -0 is considered less than +0.
// This matches Go's min built-in and the IEEE 754-2019 minimum operation. Note
// that on AMD64 (and only AMD64), it differs from [Min].
//
// Emulated, CPU Feature: AVX512
func (x Float64x8) ReduceMin() (z float64) {
	lo, hi := x.Lo(), x.Hi()
	m, f := lo.Min(hi), minFixup64x4(lo, hi)
	return applyMinFixup64(m.Lo().Min(m.Hi()).reduceMinTree(), f.Lo().Or(f.Hi()))
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
	ll := x.concatPermute128Scalars(0, 0, x) // Double the low 128 bits of x. (The x argument is unused.)
	hh := x.concatPermute128Scalars(1, 1, x) // Double the high 128 bits of x.
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
	xl, xh := x.Lo(), x.Hi()
	ll := xl.concatPermute128Scalars(0, 0, xl) // 2 copies of elements 0-15
	lh := xl.concatPermute128Scalars(1, 1, xl) // 2 copies of elements 16-31
	hl := xh.concatPermute128Scalars(0, 0, xh) // 2 copies of elements 32-47
	hh := xh.concatPermute128Scalars(1, 1, xh) // 2 copies of elements 48-63
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

// LookupOrZero permutes x, returning 0 for out-of-range indices.
//
//	if indices[i] < 16 {
//	    z[i] = x[indices[i]]
//	} else {
//	    z[i] = 0
//	}
//
// Emulated
func (x Uint8x16) LookupOrZero(indices Uint8x16) (z Uint8x16) {
	return x.BitsToInt8().LookupOrZero(indices).ToBits()
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
func (x Uint8x32) LookupOrZero(indices Uint8x32) (z Uint8x32) {
	return x.BitsToInt8().LookupOrZero(indices).ToBits()
}

// LookupOrZero permutes x, returning 0 for out-of-range indices.
//
//	if indices[i] < 64 {
//	    z[i] = x[indices[i]]
//	} else {
//	    z[i] = 0
//	}
//
// Emulated
func (x Uint8x64) LookupOrZero(indices Uint8x64) (z Uint8x64) {
	return x.BitsToInt8().LookupOrZero(indices).ToBits()
}
