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

// The ExtendLo* conversions from a 128-bit vector to a wider vector are
// equivalent to enlarging x to the result's width and using ExtendTo*, which
// the compiler lowers to the same single instruction.

// ExtendLo4ToInt64 sign-extends the low 4 elements of x to int64.
//
//	z[i] = int64(x[i])
//
// Asm: VPMOVSXBQ, CPU Feature: AVX2
//
// Deprecated: use x.Enlarge().ExtendToInt64() instead.
//
//go:fix inline
func (x Int8x16) ExtendLo4ToInt64() (z Int64x4) {
	return x.Enlarge().ExtendToInt64()
}

// ExtendLo4ToInt64 sign-extends the low 4 elements of x to int64.
//
//	z[i] = int64(x[i])
//
// Asm: VPMOVSXWQ, CPU Feature: AVX2
//
// Deprecated: use x.Enlarge().ExtendToInt64() instead.
//
//go:fix inline
func (x Int16x8) ExtendLo4ToInt64() (z Int64x4) {
	return x.Enlarge().ExtendToInt64()
}

// ExtendLo8ToInt32 sign-extends the low 8 elements of x to int32.
//
//	z[i] = int32(x[i])
//
// Asm: VPMOVSXBD, CPU Feature: AVX2
//
// Deprecated: use x.Enlarge().ExtendToInt32() instead.
//
//go:fix inline
func (x Int8x16) ExtendLo8ToInt32() (z Int32x8) {
	return x.Enlarge().ExtendToInt32()
}

// ExtendLo8ToInt64 sign-extends the low 8 elements of x to int64.
//
//	z[i] = int64(x[i])
//
// Asm: VPMOVSXBQ, CPU Feature: AVX512
//
// Deprecated: use x.Enlarge().Enlarge().ExtendToInt64() instead.
//
//go:fix inline
func (x Int8x16) ExtendLo8ToInt64() (z Int64x8) {
	return x.Enlarge().Enlarge().ExtendToInt64()
}

// ExtendLo4ToUint64 zero-extends the low 4 elements of x to uint64.
//
//	z[i] = uint64(x[i])
//
// Asm: VPMOVZXBQ, CPU Feature: AVX2
//
// Deprecated: use x.Enlarge().ExtendToUint64() instead.
//
//go:fix inline
func (x Uint8x16) ExtendLo4ToUint64() (z Uint64x4) {
	return x.Enlarge().ExtendToUint64()
}

// ExtendLo4ToUint64 zero-extends the low 4 elements of x to uint64.
//
//	z[i] = uint64(x[i])
//
// Asm: VPMOVZXWQ, CPU Feature: AVX2
//
// Deprecated: use x.Enlarge().ExtendToUint64() instead.
//
//go:fix inline
func (x Uint16x8) ExtendLo4ToUint64() (z Uint64x4) {
	return x.Enlarge().ExtendToUint64()
}

// ExtendLo8ToUint32 zero-extends the low 8 elements of x to uint32.
//
//	z[i] = uint32(x[i])
//
// Asm: VPMOVZXBD, CPU Feature: AVX2
//
// Deprecated: use x.Enlarge().ExtendToUint32() instead.
//
//go:fix inline
func (x Uint8x16) ExtendLo8ToUint32() (z Uint32x8) {
	return x.Enlarge().ExtendToUint32()
}

// ExtendLo8ToUint64 zero-extends the low 8 elements of x to uint64.
//
//	z[i] = uint64(x[i])
//
// Asm: VPMOVZXBQ, CPU Feature: AVX512
//
// Deprecated: use x.Enlarge().Enlarge().ExtendToUint64() instead.
//
//go:fix inline
func (x Uint8x16) ExtendLo8ToUint64() (z Uint64x8) {
	return x.Enlarge().Enlarge().ExtendToUint64()
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int8x16) ConcatPermute(y Int8x16, indices Uint8x16) Int8x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint8x16) ConcatPermute(y Uint8x16, indices Uint8x16) Uint8x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int8x32) ConcatPermute(y Int8x32, indices Uint8x32) Int8x32 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint8x32) ConcatPermute(y Uint8x32, indices Uint8x32) Uint8x32 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int8x64) ConcatPermute(y Int8x64, indices Uint8x64) Int8x64 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint8x64) ConcatPermute(y Uint8x64, indices Uint8x64) Uint8x64 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int16x8) ConcatPermute(y Int16x8, indices Uint16x8) Int16x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint16x8) ConcatPermute(y Uint16x8, indices Uint16x8) Uint16x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int16x16) ConcatPermute(y Int16x16, indices Uint16x16) Int16x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint16x16) ConcatPermute(y Uint16x16, indices Uint16x16) Uint16x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int16x32) ConcatPermute(y Int16x32, indices Uint16x32) Int16x32 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint16x32) ConcatPermute(y Uint16x32, indices Uint16x32) Uint16x32 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Float32x4) ConcatPermute(y Float32x4, indices Uint32x4) Float32x4 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int32x4) ConcatPermute(y Int32x4, indices Uint32x4) Int32x4 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint32x4) ConcatPermute(y Uint32x4, indices Uint32x4) Uint32x4 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Float32x8) ConcatPermute(y Float32x8, indices Uint32x8) Float32x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int32x8) ConcatPermute(y Int32x8, indices Uint32x8) Int32x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint32x8) ConcatPermute(y Uint32x8, indices Uint32x8) Uint32x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Float32x16) ConcatPermute(y Float32x16, indices Uint32x16) Float32x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int32x16) ConcatPermute(y Int32x16, indices Uint32x16) Int32x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint32x16) ConcatPermute(y Uint32x16, indices Uint32x16) Uint32x16 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Float64x2) ConcatPermute(y Float64x2, indices Uint64x2) Float64x2 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int64x2) ConcatPermute(y Int64x2, indices Uint64x2) Int64x2 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint64x2) ConcatPermute(y Uint64x2, indices Uint64x2) Uint64x2 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Float64x4) ConcatPermute(y Float64x4, indices Uint64x4) Float64x4 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int64x4) ConcatPermute(y Int64x4, indices Uint64x4) Int64x4 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint64x4) ConcatPermute(y Uint64x4, indices Uint64x4) Uint64x4 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Float64x8) ConcatPermute(y Float64x8, indices Uint64x8) Float64x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Int64x8) ConcatPermute(y Int64x8, indices Uint64x8) Int64x8 {
	return x.ConcatLookupMod(y, indices)
}

// ConcatPermute concatenates x and y and selects elements using indices modulo the concatenated length.
//
// Deprecated: use x.ConcatLookupMod(y, indices) instead.
//
//go:fix inline
func (x Uint64x8) ConcatPermute(y Uint64x8, indices Uint64x8) Uint64x8 {
	return x.ConcatLookupMod(y, indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int8x16) Permute(indices Uint8x16) Int8x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint8x16) Permute(indices Uint8x16) Uint8x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int8x32) Permute(indices Uint8x32) Int8x32 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint8x32) Permute(indices Uint8x32) Uint8x32 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int8x64) Permute(indices Uint8x64) Int8x64 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint8x64) Permute(indices Uint8x64) Uint8x64 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int16x8) Permute(indices Uint16x8) Int16x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint16x8) Permute(indices Uint16x8) Uint16x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int16x16) Permute(indices Uint16x16) Int16x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint16x16) Permute(indices Uint16x16) Uint16x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int16x32) Permute(indices Uint16x32) Int16x32 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint16x32) Permute(indices Uint16x32) Uint16x32 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Float32x8) Permute(indices Uint32x8) Float32x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int32x8) Permute(indices Uint32x8) Int32x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint32x8) Permute(indices Uint32x8) Uint32x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Float32x16) Permute(indices Uint32x16) Float32x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int32x16) Permute(indices Uint32x16) Int32x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint32x16) Permute(indices Uint32x16) Uint32x16 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Float64x4) Permute(indices Uint64x4) Float64x4 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int64x4) Permute(indices Uint64x4) Int64x4 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint64x4) Permute(indices Uint64x4) Uint64x4 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Float64x8) Permute(indices Uint64x8) Float64x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Int64x8) Permute(indices Uint64x8) Int64x8 {
	return x.LookupMod(indices)
}

// Permute selects elements from x using indices modulo len(x).
//
// Deprecated: use x.LookupMod(indices) instead.
//
//go:fix inline
func (x Uint64x8) Permute(indices Uint64x8) Uint64x8 {
	return x.LookupMod(indices)
}

// PermuteOrZero selects elements from x using indices modulo len(x), or 0 when indices[i] < 0.
//
// Deprecated: use x.LookupModOrZero(indices) instead.
//
//go:fix inline
func (x Int8x16) PermuteOrZero(indices Int8x16) Int8x16 {
	return x.LookupModOrZero(indices)
}

// PermuteOrZero selects elements from x using indices modulo len(x), or 0 when indices[i] < 0.
//
// Deprecated: use x.LookupModOrZero(indices) instead.
//
//go:fix inline
func (x Uint8x16) PermuteOrZero(indices Int8x16) Uint8x16 {
	return x.LookupModOrZero(indices)
}

// PermuteOrZeroGrouped divides x and indices into 128-bit groups and performs [PermuteOrZero] on each group.
//
// Deprecated: use x.LookupModOrZeroGrouped(indices) instead.
//
//go:fix inline
func (x Int8x32) PermuteOrZeroGrouped(indices Int8x32) Int8x32 {
	return x.LookupModOrZeroGrouped(indices)
}

// PermuteOrZeroGrouped divides x and indices into 128-bit groups and performs [PermuteOrZero] on each group.
//
// Deprecated: use x.LookupModOrZeroGrouped(indices) instead.
//
//go:fix inline
func (x Int8x64) PermuteOrZeroGrouped(indices Int8x64) Int8x64 {
	return x.LookupModOrZeroGrouped(indices)
}

// PermuteOrZeroGrouped divides x and indices into 128-bit groups and performs [PermuteOrZero] on each group.
//
// Deprecated: use x.LookupModOrZeroGrouped(indices) instead.
//
//go:fix inline
func (x Uint8x32) PermuteOrZeroGrouped(indices Int8x32) Uint8x32 {
	return x.LookupModOrZeroGrouped(indices)
}

// PermuteOrZeroGrouped divides x and indices into 128-bit groups and performs [PermuteOrZero] on each group.
//
// Deprecated: use x.LookupModOrZeroGrouped(indices) instead.
//
//go:fix inline
func (x Uint8x64) PermuteOrZeroGrouped(indices Int8x64) Uint8x64 {
	return x.LookupModOrZeroGrouped(indices)
}

// ConcatShiftBytesRight concatenates x and y and shifts it right by shift bytes.
// The result vector will be the lower half of the concatenated vector.
//
// Asm: VPALIGNR, CPU Feature: AVX
//
// Deprecated: use y.ConcatWindow(x, uint8(shift)) instead. ConcatWindow takes
// x and y in the opposite order.
//
//go:fix inline
func (x Uint8x16) ConcatShiftBytesRight(y Uint8x16, shift uint64) Uint8x16 {
	return y.ConcatWindow(x, uint8(shift))
}

// ConcatShiftBytesRightGrouped concatenates x and y and shifts it right by shift bytes.
// The result vector will be the lower half of the concatenated vector.
// This operation is performed grouped by each 16 byte.
//
// Asm: VPALIGNR, CPU Feature: AVX2
//
// Deprecated: use y.ConcatWindowGrouped(x, uint8(shift)) instead.
// ConcatWindowGrouped takes x and y in the opposite order.
//
//go:fix inline
func (x Uint8x32) ConcatShiftBytesRightGrouped(y Uint8x32, shift uint64) Uint8x32 {
	return y.ConcatWindowGrouped(x, uint8(shift))
}

// ConcatShiftBytesRightGrouped concatenates x and y and shifts it right by shift bytes.
// The result vector will be the lower half of the concatenated vector.
// This operation is performed grouped by each 16 byte.
//
// Asm: VPALIGNR, CPU Feature: AVX512
//
// Deprecated: use y.ConcatWindowGrouped(x, uint8(shift)) instead.
// ConcatWindowGrouped takes x and y in the opposite order.
//
//go:fix inline
func (x Uint8x64) ConcatShiftBytesRightGrouped(y Uint8x64, shift uint64) Uint8x64 {
	return y.ConcatWindowGrouped(x, uint8(shift))
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

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Float32x8) GetHi() (z Float32x4) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Float32x16) GetHi() (z Float32x8) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Float64x4) GetHi() (z Float64x2) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Float64x8) GetHi() (z Float64x4) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int8x32) GetHi() (z Int8x16) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int8x64) GetHi() (z Int8x32) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int16x16) GetHi() (z Int16x8) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int16x32) GetHi() (z Int16x16) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int32x8) GetHi() (z Int32x4) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int32x16) GetHi() (z Int32x8) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int64x4) GetHi() (z Int64x2) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Int64x8) GetHi() (z Int64x4) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint8x32) GetHi() (z Uint8x16) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint8x64) GetHi() (z Uint8x32) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint16x16) GetHi() (z Uint16x8) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint16x32) GetHi() (z Uint16x16) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint32x8) GetHi() (z Uint32x4) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint32x16) GetHi() (z Uint32x8) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint64x4) GetHi() (z Uint64x2) {
	return x.Hi()
}

// GetHi returns the upper half of x.
//
// Deprecated: use x.Hi() instead.
//
//go:fix inline
func (x Uint64x8) GetHi() (z Uint64x4) {
	return x.Hi()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Float32x8) GetLo() (z Float32x4) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Float32x16) GetLo() (z Float32x8) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Float64x4) GetLo() (z Float64x2) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Float64x8) GetLo() (z Float64x4) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int8x32) GetLo() (z Int8x16) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int8x64) GetLo() (z Int8x32) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int16x16) GetLo() (z Int16x8) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int16x32) GetLo() (z Int16x16) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int32x8) GetLo() (z Int32x4) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int32x16) GetLo() (z Int32x8) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int64x4) GetLo() (z Int64x2) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Int64x8) GetLo() (z Int64x4) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint8x32) GetLo() (z Uint8x16) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint8x64) GetLo() (z Uint8x32) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint16x16) GetLo() (z Uint16x8) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint16x32) GetLo() (z Uint16x16) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint32x8) GetLo() (z Uint32x4) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint32x16) GetLo() (z Uint32x8) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint64x4) GetLo() (z Uint64x2) {
	return x.Lo()
}

// GetLo returns the lower half of x.
//
// Deprecated: use x.Lo() instead.
//
//go:fix inline
func (x Uint64x8) GetLo() (z Uint64x4) {
	return x.Lo()
}

// CarrylessMultiplyEven computes the carryless multiplication of
// the even-indexed elements of x and y.
//
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
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
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
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
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
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
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
//
// Deprecated: use x.CarrylessMulEvenOdd(y) instead.
//
//go:fix inline
func (x Uint64x2) CarrylessMultiplyEvenOdd(y Uint64x2) (z Uint64x2) {
	return x.CarrylessMulEvenOdd(y)
}

// CarrylessMultiplyEven computes the carryless multiplication of
// the even-indexed elements of x and y.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
//
// Deprecated: use x.CarrylessMulEven(y) instead.
//
//go:fix inline
func (x Uint64x4) CarrylessMultiplyEven(y Uint64x4) (z Uint64x4) {
	return x.CarrylessMulEven(y)
}

// CarrylessMultiplyOdd computes the carryless multiplication of
// the odd-indexed elements of x and y.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
//
// Deprecated: use x.CarrylessMulOdd(y) instead.
//
//go:fix inline
func (x Uint64x4) CarrylessMultiplyOdd(y Uint64x4) (z Uint64x4) {
	return x.CarrylessMulOdd(y)
}

// CarrylessMultiplyOddEven computes the carryless multiplication of
// the odd-indexed elements of x and the even-indexed elements of y.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
//
// Deprecated: use x.CarrylessMulOddEven(y) instead.
//
//go:fix inline
func (x Uint64x4) CarrylessMultiplyOddEven(y Uint64x4) (z Uint64x4) {
	return x.CarrylessMulOddEven(y)
}

// CarrylessMultiplyEvenOdd computes the carryless multiplication of
// the even-indexed elements of x and the odd-indexed elements of y.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
//
// Deprecated: use x.CarrylessMulEvenOdd(y) instead.
//
//go:fix inline
func (x Uint64x4) CarrylessMultiplyEvenOdd(y Uint64x4) (z Uint64x4) {
	return x.CarrylessMulEvenOdd(y)
}

// CarrylessMultiplyEven computes the carryless multiplication of
// the even-indexed elements of x and y.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
//
// Deprecated: use x.CarrylessMulEven(y) instead.
//
//go:fix inline
func (x Uint64x8) CarrylessMultiplyEven(y Uint64x8) (z Uint64x8) {
	return x.CarrylessMulEven(y)
}

// CarrylessMultiplyOdd computes the carryless multiplication of
// the odd-indexed elements of x and y.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
//
// Deprecated: use x.CarrylessMulOdd(y) instead.
//
//go:fix inline
func (x Uint64x8) CarrylessMultiplyOdd(y Uint64x8) (z Uint64x8) {
	return x.CarrylessMulOdd(y)
}

// CarrylessMultiplyOddEven computes the carryless multiplication of
// the odd-indexed elements of x and the even-indexed elements of y.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
//
// Deprecated: use x.CarrylessMulOddEven(y) instead.
//
//go:fix inline
func (x Uint64x8) CarrylessMultiplyOddEven(y Uint64x8) (z Uint64x8) {
	return x.CarrylessMulOddEven(y)
}

// CarrylessMultiplyEvenOdd computes the carryless multiplication of
// the even-indexed elements of x and the odd-indexed elements of y.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
//
// Deprecated: use x.CarrylessMulEvenOdd(y) instead.
//
//go:fix inline
func (x Uint64x8) CarrylessMultiplyEvenOdd(y Uint64x8) (z Uint64x8) {
	return x.CarrylessMulEvenOdd(y)
}
