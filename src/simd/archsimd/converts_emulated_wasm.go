// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && wasm

package archsimd

// This file emulates the 2:1 narrowing integer conversions. Wasm has
// saturating narrowing from signed integers (narrow_s and narrow_u) but not
// from unsigned integers, and has no truncating narrowing.
//
// TODO: ExtendTo, TruncTo and SaturateTo are only provided for a 2:1 element
// size ratio, which is what wasm does in one instruction (or emulates here),
// but the spec defines every ratio. Add the 4:1 and 8:1 forms as sequences of
// 2:1 conversions, and perhaps peephole chains of 2:1 conversions into those
// sequences.

// Byte permutations for truncation. An out-of-range index (0xff) produces a
// zero.
var (
	trunc16To8Idx  = [16]uint8{0, 2, 4, 6, 8, 10, 12, 14, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
	trunc32To16Idx = [16]uint8{0, 1, 4, 5, 8, 9, 12, 13, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
	trunc64To32Idx = [16]uint8{0, 1, 2, 3, 8, 9, 10, 11, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
)

// truncBytes returns the bytes of x selected by idx.
func truncBytes(x Uint8x16, idx *[16]uint8) Uint8x16 {
	return x.BitsToInt8().LookupOrZero(LoadUint8x16Array(idx)).ToBits()
}

// TruncToInt8 converts each element of x to int8 by discarding its high
// bits.
//
//	z[i] = int8(x[i])
//
// The results are stored in the low 8 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int16x8) TruncToInt8() (z Int8x16) {
	return x.ToBits().TruncToUint8().BitsToInt8()
}

// TruncToUint8 converts each element of x to uint8 by discarding its high
// bits.
//
//	z[i] = uint8(x[i])
//
// The results are stored in the low 8 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Uint16x8) TruncToUint8() (z Uint8x16) {
	return truncBytes(x.ReshapeToUint8s(), &trunc16To8Idx)
}

// TruncToInt16 converts each element of x to int16 by discarding its high
// bits.
//
//	z[i] = int16(x[i])
//
// The results are stored in the low 4 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int32x4) TruncToInt16() (z Int16x8) {
	return x.ToBits().TruncToUint16().BitsToInt16()
}

// TruncToUint16 converts each element of x to uint16 by discarding its high
// bits.
//
//	z[i] = uint16(x[i])
//
// The results are stored in the low 4 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Uint32x4) TruncToUint16() (z Uint16x8) {
	return truncBytes(x.ReshapeToUint8s(), &trunc32To16Idx).ReshapeToUint16s()
}

// TruncToInt32 converts each element of x to int32 by discarding its high
// bits.
//
//	z[i] = int32(x[i])
//
// The results are stored in the low 2 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int64x2) TruncToInt32() (z Int32x4) {
	return x.ToBits().TruncToUint32().BitsToInt32()
}

// TruncToUint32 converts each element of x to uint32 by discarding its high
// bits.
//
//	z[i] = uint32(x[i])
//
// The results are stored in the low 2 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Uint64x2) TruncToUint32() (z Uint32x4) {
	return truncBytes(x.ReshapeToUint8s(), &trunc64To32Idx).ReshapeToUint32s()
}

// SaturateToInt8 converts each element of x to int8, saturating values
// that are outside the range of int8.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 8 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int16x8) SaturateToInt8() (z Int8x16) {
	var zero Int16x8
	return x.concatSaturateToInt8(zero)
}

// SaturateToUint8 converts each element of x to uint8, saturating values
// that are outside the range of uint8. Negative values become 0.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 8 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int16x8) SaturateToUint8() (z Uint8x16) {
	var zero Int16x8
	return x.concatSaturateToUint8(zero)
}

// SaturateToUint8 converts each element of x to uint8, saturating values
// that are outside the range of uint8.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 8 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Uint16x8) SaturateToUint8() (z Uint8x16) {
	// After the Min, x is non-negative as an int16, so the signed-to-unsigned
	// saturation doesn't change it.
	var zero Int16x8
	x = x.Min(BroadcastUint16x8(maxUint8))
	return x.BitsToInt16().concatSaturateToUint8(zero)
}

// SaturateToInt16 converts each element of x to int16, saturating values
// that are outside the range of int16.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 4 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int32x4) SaturateToInt16() (z Int16x8) {
	var zero Int32x4
	return x.concatSaturateToInt16(zero)
}

// SaturateToUint16 converts each element of x to uint16, saturating values
// that are outside the range of uint16. Negative values become 0.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 4 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int32x4) SaturateToUint16() (z Uint16x8) {
	var zero Int32x4
	return x.concatSaturateToUint16(zero)
}

// SaturateToUint16 converts each element of x to uint16, saturating values
// that are outside the range of uint16.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 4 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Uint32x4) SaturateToUint16() (z Uint16x8) {
	// After the Min, x is non-negative as an int32, so the signed-to-unsigned
	// saturation doesn't change it.
	var zero Int32x4
	x = x.Min(BroadcastUint32x4(maxUint16))
	return x.BitsToInt32().concatSaturateToUint16(zero)
}

// SaturateToInt32 converts each element of x to int32, saturating values
// that are outside the range of int32.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 2 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int64x2) SaturateToInt32() (z Int32x4) {
	hi, lo := BroadcastInt64x2(maxInt32), BroadcastInt64x2(minInt32)
	x = hi.IfElse(x.Greater(hi), x)
	x = lo.IfElse(x.Less(lo), x)
	return x.TruncToInt32()
}

// SaturateToUint32 converts each element of x to uint32, saturating values
// that are outside the range of uint32. Negative values become 0.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 2 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Int64x2) SaturateToUint32() (z Uint32x4) {
	var zero Int64x2
	hi := BroadcastInt64x2(maxUint32)
	x = x.Masked(x.Greater(zero))
	x = hi.IfElse(x.Greater(hi), x)
	return x.ToBits().TruncToUint32()
}

// SaturateToUint32 converts each element of x to uint32, saturating values
// that are outside the range of uint32.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low 2 elements of z.
// The remaining elements of z are zero.
//
// Emulated
func (x Uint64x2) SaturateToUint32() (z Uint32x4) {
	// x fits in a uint32 if its high 32 bits are zero.
	var zero Uint64x2
	hi := BroadcastUint64x2(maxUint32)
	return x.IfElse(x.ShiftAllRight(32).Equal(zero), hi).TruncToUint32()
}
