// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// Deprecated operations that are wrappers around their replacements.

// ExtendLo8ToInt16 sign-extends the low 8 elements of x to int16.
//
//	z[i] = int16(x[i])
//
// Asm: VPMOVSXBW, CPU Feature: AVX
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
// Asm: VPMOVSXBD, CPU Feature: AVX
//
// Deprecated: use x.ExtendToInt32() instead.
//
//go:fix inline
func (x Int8x16) ExtendLo4ToInt32() (z Int32x4) {
	return x.ExtendToInt32()
}

// ExtendLo2ToInt64 sign-extends the low 2 elements of x to int64.
//
//	z[i] = int64(x[i])
//
// Asm: VPMOVSXBQ, CPU Feature: AVX
//
// Deprecated: use x.ExtendToInt64() instead.
//
//go:fix inline
func (x Int8x16) ExtendLo2ToInt64() (z Int64x2) {
	return x.ExtendToInt64()
}

// ExtendLo4ToInt32 sign-extends the low 4 elements of x to int32.
//
//	z[i] = int32(x[i])
//
// Asm: VPMOVSXWD, CPU Feature: AVX
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
// Asm: VPMOVSXWQ, CPU Feature: AVX
//
// Deprecated: use x.ExtendToInt64() instead.
//
//go:fix inline
func (x Int16x8) ExtendLo2ToInt64() (z Int64x2) {
	return x.ExtendToInt64()
}

// ExtendLo2ToInt64 sign-extends the low 2 elements of x to int64.
//
//	z[i] = int64(x[i])
//
// Asm: VPMOVSXDQ, CPU Feature: AVX
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
// Asm: VPMOVZXBW, CPU Feature: AVX
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
// Asm: VPMOVZXBD, CPU Feature: AVX
//
// Deprecated: use x.ExtendToUint32() instead.
//
//go:fix inline
func (x Uint8x16) ExtendLo4ToUint32() (z Uint32x4) {
	return x.ExtendToUint32()
}

// ExtendLo2ToUint64 zero-extends the low 2 elements of x to uint64.
//
//	z[i] = uint64(x[i])
//
// Asm: VPMOVZXBQ, CPU Feature: AVX
//
// Deprecated: use x.ExtendToUint64() instead.
//
//go:fix inline
func (x Uint8x16) ExtendLo2ToUint64() (z Uint64x2) {
	return x.ExtendToUint64()
}

// ExtendLo4ToUint32 zero-extends the low 4 elements of x to uint32.
//
//	z[i] = uint32(x[i])
//
// Asm: VPMOVZXWD, CPU Feature: AVX
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
// Asm: VPMOVZXWQ, CPU Feature: AVX
//
// Deprecated: use x.ExtendToUint64() instead.
//
//go:fix inline
func (x Uint16x8) ExtendLo2ToUint64() (z Uint64x2) {
	return x.ExtendToUint64()
}

// ExtendLo2ToUint64 zero-extends the low 2 elements of x to uint64.
//
//	z[i] = uint64(x[i])
//
// Asm: VPMOVZXDQ, CPU Feature: AVX
//
// Deprecated: use x.ExtendToUint64() instead.
//
//go:fix inline
func (x Uint32x4) ExtendLo2ToUint64() (z Uint64x2) {
	return x.ExtendToUint64()
}
