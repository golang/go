// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// ConcatPermuteScalars returns the selection of four elements from the two
// vectors x and y, where selector values in the range 0-3 specify
// elements from x and values in the range 4-7 specify the 0-3 elements
// of y. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
//
//	{1,2,4,8}.ConcatPermuteScalars(2,3,5,7,{9,25,49,81})
//
// returns {4,8,25,81}.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Int32x4) ConcatPermuteScalars(a, b, c, d uint8, y Int32x4) Int32x4 {
	return x.concatPermuteScalars(a, b, c, d, y)
}

// ConcatPermuteScalars returns the selection of four elements from the two
// vectors x and y, where selector values in the range 0-3 specify
// elements from x and values in the range 4-7 specify the 0-3 elements
// of y. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
//
//	{1,2,4,8}.ConcatPermuteScalars(2,3,5,7,{9,25,49,81})
//
// returns {4,8,25,81}.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Uint32x4) ConcatPermuteScalars(a, b, c, d uint8, y Uint32x4) Uint32x4 {
	return x.concatPermuteScalars(a, b, c, d, y)
}

// ConcatPermuteScalars returns the selection of four elements from the two
// vectors x and y, where selector values in the range 0-3 specify
// elements from x and values in the range 4-7 specify the 0-3 elements
// of y. a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
//
//	{1,2,4,8}.ConcatPermuteScalars(2,3,5,7,{9,25,49,81})
//
// returns {4,8,25,81}.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Float32x4) ConcatPermuteScalars(a, b, c, d uint8, y Float32x4) Float32x4 {
	return x.concatPermuteScalars(a, b, c, d, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
//
//	{1,2,4,8,16,32,64,128}.ConcatPermuteScalars(2,3,5,7,{9,25,49,81,121,169,225,289})
//
// returns {4,8,25,81,64,128,169,289}.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Int32x8) ConcatPermuteScalarsGrouped(a, b, c, d uint8, y Int32x8) Int32x8 {
	return x.concatPermuteScalarsGrouped(a, b, c, d, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
//
//	{1,2,4,8,16,32,64,128}.ConcatPermuteScalars(2,3,5,7,{9,25,49,81,121,169,225,289})
//
// returns {4,8,25,81,64,128,169,289}.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Uint32x8) ConcatPermuteScalarsGrouped(a, b, c, d uint8, y Uint32x8) Uint32x8 {
	return x.concatPermuteScalarsGrouped(a, b, c, d, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
// a is the source index of the least element in the
// output, and b, c, and d are the indices of the 2nd, 3rd, and 4th
// elements in the output.  For example,
//
//	{1,2,4,8,16,32,64,128}.ConcatPermuteScalars(2,3,5,7,{9,25,49,81,121,169,225,289})
//
// returns {4,8,25,81,64,128,169,289}.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX
func (x Float32x8) ConcatPermuteScalarsGrouped(a, b, c, d uint8, y Float32x8) Float32x8 {
	return x.concatPermuteScalarsGrouped(a, b, c, d, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX512
func (x Int32x16) ConcatPermuteScalarsGrouped(a, b, c, d uint8, y Int32x16) Int32x16 {
	return x.concatPermuteScalarsGrouped(a, b, c, d, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX512
func (x Uint32x16) ConcatPermuteScalarsGrouped(a, b, c, d uint8, y Uint32x16) Uint32x16 {
	return x.concatPermuteScalarsGrouped(a, b, c, d, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of four elements from  x and y,
// where selector values in the range 0-3 specify elements from x and
// values in the range 4-7 specify the 0-3 elements of y.
//
// Performance: When the selectors are constants and the selection can be
// implemented in a single instruction, it will be, otherwise it requires two.
// If the selectors are not constant this will translate to a function call.
//
// Asm: VSHUFPS, CPU Feature: AVX512
func (x Float32x16) ConcatPermuteScalarsGrouped(a, b, c, d uint8, y Float32x16) Float32x16 {
	return x.concatPermuteScalarsGrouped(a, b, c, d, y)
}

// ConcatPermuteScalars returns the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Uint64x2) ConcatPermuteScalars(a, b uint8, y Uint64x2) Uint64x2 {
	return x.concatPermuteScalars(a, b, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Uint64x4) ConcatPermuteScalarsGrouped(a, b uint8, y Uint64x4) Uint64x4 {
	return x.concatPermuteScalarsGrouped(a, b, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX512
func (x Uint64x8) ConcatPermuteScalarsGrouped(a, b uint8, y Uint64x8) Uint64x8 {
	return x.concatPermuteScalarsGrouped(a, b, y)
}

// ConcatPermuteScalars returns the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Float64x2) ConcatPermuteScalars(a, b uint8, y Float64x2) Float64x2 {
	return x.concatPermuteScalars(a, b, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Float64x4) ConcatPermuteScalarsGrouped(a, b uint8, y Float64x4) Float64x4 {
	return x.concatPermuteScalarsGrouped(a, b, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX512
func (x Float64x8) ConcatPermuteScalarsGrouped(a, b uint8, y Float64x8) Float64x8 {
	return x.concatPermuteScalarsGrouped(a, b, y)
}

// ConcatPermuteScalars returns the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Int64x2) ConcatPermuteScalars(a, b uint8, y Int64x2) Int64x2 {
	return x.concatPermuteScalars(a, b, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the two 128-bit halves of
// the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX
func (x Int64x4) ConcatPermuteScalarsGrouped(a, b uint8, y Int64x4) Int64x4 {
	return x.concatPermuteScalarsGrouped(a, b, y)
}

// ConcatPermuteScalarsGrouped returns, for each of the four 128-bit subvectors
// of the vectors x and y, the selection of two elements from the two
// vectors x and y, where selector values in the range 0-1 specify
// elements from x and values in the range 2-3 specify the 0-1 elements
// of y.
//
// Performance: When the selectors are constants the selection can be
// implemented in a single instruction. If the selectors are not constant this
// will translate to a function call.
//
// Asm: VSHUFPD, CPU Feature: AVX512
func (x Int64x8) ConcatPermuteScalarsGrouped(a, b uint8, y Int64x8) Int64x8 {
	return x.concatPermuteScalarsGrouped(a, b, y)
}

/* PermuteScalars */

// PermuteScalars performs a permutation of vector x's elements using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX
func (x Int32x4) PermuteScalars(a, b, c, d uint8) Int32x4 {
	return x.permuteScalars(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalars performs a permutation of vector x's elements using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX
func (x Uint32x4) PermuteScalars(a, b, c, d uint8) Uint32x4 {
	return x.permuteScalars(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsGrouped */

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX2
func (x Int32x8) PermuteScalarsGrouped(a, b, c, d uint8) Int32x8 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		 {  x[a], x[b], x[c], x[d],         x[a+4], x[b+4], x[c+4], x[d+4],
//			x[a+8], x[b+8], x[c+8], x[d+8], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table may be generated.
//
// Asm: VPSHUFD, CPU Feature: AVX512
func (x Int32x16) PermuteScalarsGrouped(a, b, c, d uint8) Int32x16 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFD, CPU Feature: AVX2
func (x Uint32x8) PermuteScalarsGrouped(a, b, c, d uint8) Uint32x8 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		 {  x[a], x[b], x[c], x[d],         x[a+4], x[b+4], x[c+4], x[d+4],
//			x[a+8], x[b+8], x[c+8], x[d+8], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFD, CPU Feature: AVX512
func (x Uint32x16) PermuteScalarsGrouped(a, b, c, d uint8) Uint32x16 {
	return x.permuteScalarsGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsHi */

// PermuteScalarsHi performs a permutation of vector x using the supplied indices:
//
//	result = {x[0], x[1], x[2], x[3], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX
func (x Int16x8) PermuteScalarsHi(a, b, c, d uint8) Int16x8 {
	return x.permuteScalarsHi(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHi performs a permutation of vector x using the supplied indices:
//
//	result = {x[0], x[1], x[2], x[3], x[a+4], x[b+4], x[c+4], x[d+4]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX
func (x Uint16x8) PermuteScalarsHi(a, b, c, d uint8) Uint16x8 {
	return x.permuteScalarsHi(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsHiGrouped */

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		  {x[0], x[1], x[2], x[3],   x[a+4], x[b+4], x[c+4], x[d+4],
//			x[8], x[9], x[10], x[11], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX2
func (x Int16x16) PermuteScalarsHiGrouped(a, b, c, d uint8) Int16x16 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		  {x[0], x[1], x[2], x[3],     x[a+4], x[b+4], x[c+4], x[d+4],
//			x[8], x[9], x[10], x[11],   x[a+12], x[b+12], x[c+12], x[d+12],
//			x[16], x[17], x[18], x[19], x[a+20], x[b+20], x[c+20], x[d+20],
//			x[24], x[25], x[26], x[27], x[a+28], x[b+28], x[c+28], x[d+28]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX512
func (x Int16x32) PermuteScalarsHiGrouped(a, b, c, d uint8) Int16x32 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	  {x[0], x[1], x[2], x[3],   x[a+4], x[b+4], x[c+4], x[d+4],
//		x[8], x[9], x[10], x[11], x[a+12], x[b+12], x[c+12], x[d+12]}
//
// Each group is of size 128-bit.
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX2
func (x Uint16x16) PermuteScalarsHiGrouped(a, b, c, d uint8) Uint16x16 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsHiGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//		 {  x[0], x[1], x[2], x[3],     x[a+4], x[b+4], x[c+4], x[d+4],
//			x[8], x[9], x[10], x[11],   x[a+12], x[b+12], x[c+12], x[d+12],
//			x[16], x[17], x[18], x[19], x[a+20], x[b+20], x[c+20], x[d+20],
//			x[24], x[25], x[26], x[27], x[a+28], x[b+28], x[c+28], x[d+28]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFHW, CPU Feature: AVX512
func (x Uint16x32) PermuteScalarsHiGrouped(a, b, c, d uint8) Uint16x32 {
	return x.permuteScalarsHiGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsLo */

// PermuteScalarsLo performs a permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[4], x[5], x[6], x[7]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX
func (x Int16x8) PermuteScalarsLo(a, b, c, d uint8) Int16x8 {
	return x.permuteScalarsLo(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLo performs a permutation of vector x using the supplied indices:
//
//	result = {x[a], x[b], x[c], x[d], x[4], x[5], x[6], x[7]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX
func (x Uint16x8) PermuteScalarsLo(a, b, c, d uint8) Uint16x8 {
	return x.permuteScalarsLo(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* PermuteScalarsLoGrouped */

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	 {x[a], x[b], x[c], x[d],         x[4], x[5], x[6], x[7],
//		 x[a+8], x[b+8], x[c+8], x[d+8], x[12], x[13], x[14], x[15]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX2
func (x Int16x16) PermuteScalarsLoGrouped(a, b, c, d uint8) Int16x16 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	 {x[a], x[b], x[c], x[d],    x[4], x[5], x[6], x[7],
//		x[a+8], x[b+8], x[c+8], x[d+8],     x[12], x[13], x[14], x[15],
//		x[a+16], x[b+16], x[c+16], x[d+16], x[20], x[21], x[22], x[23],
//		x[a+24], x[b+24], x[c+24], x[d+24], x[28], x[29], x[30], x[31]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX512
func (x Int16x32) PermuteScalarsLoGrouped(a, b, c, d uint8) Int16x32 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result = {x[a], x[b], x[c], x[d],         x[4], x[5], x[6], x[7],
//		x[a+8], x[b+8], x[c+8], x[d+8], x[12], x[13], x[14], x[15]}
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX2
func (x Uint16x16) PermuteScalarsLoGrouped(a, b, c, d uint8) Uint16x16 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

// PermuteScalarsLoGrouped performs a grouped permutation of vector x using the supplied indices:
//
//	 result =
//	 {x[a], x[b], x[c], x[d],    x[4], x[5], x[6], x[7],
//		x[a+8], x[b+8], x[c+8], x[d+8],     x[12], x[13], x[14], x[15],
//		x[a+16], x[b+16], x[c+16], x[d+16], x[20], x[21], x[22], x[23],
//		x[a+24], x[b+24], x[c+24], x[d+24], x[28], x[29], x[30], x[31]}
//
// Each group is of size 128-bit.
//
// Parameters a,b,c,d should have values between 0 and 3.
//
// Performance: If a through d are constants, then an instruction will be
// inlined, otherwise a jump table is generated.
//
// Asm: VPSHUFLW, CPU Feature: AVX512
func (x Uint16x32) PermuteScalarsLoGrouped(a, b, c, d uint8) Uint16x32 {
	return x.permuteScalarsLoGrouped(a&3 | (b&3)<<2 | (c&3)<<4 | d<<6)
}

/* InterleaveHiGrouped */

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHBW, CPU Feature: AVX2
func (x Int8x32) InterleaveHiGrouped(y Int8x32) Int8x32 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHBW, CPU Feature: AVX512
func (x Int8x64) InterleaveHiGrouped(y Int8x64) Int8x64 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHWD, CPU Feature: AVX2
func (x Int16x16) InterleaveHiGrouped(y Int16x16) Int16x16 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHWD, CPU Feature: AVX512
func (x Int16x32) InterleaveHiGrouped(y Int16x32) Int16x32 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHDQ, CPU Feature: AVX2
func (x Int32x8) InterleaveHiGrouped(y Int32x8) Int32x8 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHDQ, CPU Feature: AVX512
func (x Int32x16) InterleaveHiGrouped(y Int32x16) Int32x16 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHQDQ, CPU Feature: AVX2
func (x Int64x4) InterleaveHiGrouped(y Int64x4) Int64x4 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHQDQ, CPU Feature: AVX512
func (x Int64x8) InterleaveHiGrouped(y Int64x8) Int64x8 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHBW, CPU Feature: AVX2
func (x Uint8x32) InterleaveHiGrouped(y Uint8x32) Uint8x32 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHBW, CPU Feature: AVX512
func (x Uint8x64) InterleaveHiGrouped(y Uint8x64) Uint8x64 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHWD, CPU Feature: AVX2
func (x Uint16x16) InterleaveHiGrouped(y Uint16x16) Uint16x16 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHWD, CPU Feature: AVX512
func (x Uint16x32) InterleaveHiGrouped(y Uint16x32) Uint16x32 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHDQ, CPU Feature: AVX2
func (x Uint32x8) InterleaveHiGrouped(y Uint32x8) Uint32x8 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHDQ, CPU Feature: AVX512
func (x Uint32x16) InterleaveHiGrouped(y Uint32x16) Uint32x16 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHQDQ, CPU Feature: AVX2
func (x Uint64x4) InterleaveHiGrouped(y Uint64x4) Uint64x4 {
	return x.interleaveHiGrouped(y)
}

// InterleaveHiGrouped interleaves the elements of the high half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKHQDQ, CPU Feature: AVX512
func (x Uint64x8) InterleaveHiGrouped(y Uint64x8) Uint64x8 {
	return x.interleaveHiGrouped(y)
}

/* InterleaveLoGrouped */

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLBW, CPU Feature: AVX2
func (x Int8x32) InterleaveLoGrouped(y Int8x32) Int8x32 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLBW, CPU Feature: AVX512
func (x Int8x64) InterleaveLoGrouped(y Int8x64) Int8x64 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLWD, CPU Feature: AVX2
func (x Int16x16) InterleaveLoGrouped(y Int16x16) Int16x16 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLWD, CPU Feature: AVX512
func (x Int16x32) InterleaveLoGrouped(y Int16x32) Int16x32 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLDQ, CPU Feature: AVX2
func (x Int32x8) InterleaveLoGrouped(y Int32x8) Int32x8 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLDQ, CPU Feature: AVX512
func (x Int32x16) InterleaveLoGrouped(y Int32x16) Int32x16 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLQDQ, CPU Feature: AVX2
func (x Int64x4) InterleaveLoGrouped(y Int64x4) Int64x4 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLQDQ, CPU Feature: AVX512
func (x Int64x8) InterleaveLoGrouped(y Int64x8) Int64x8 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLBW, CPU Feature: AVX2
func (x Uint8x32) InterleaveLoGrouped(y Uint8x32) Uint8x32 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLBW, CPU Feature: AVX512
func (x Uint8x64) InterleaveLoGrouped(y Uint8x64) Uint8x64 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLWD, CPU Feature: AVX2
func (x Uint16x16) InterleaveLoGrouped(y Uint16x16) Uint16x16 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLWD, CPU Feature: AVX512
func (x Uint16x32) InterleaveLoGrouped(y Uint16x32) Uint16x32 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLDQ, CPU Feature: AVX2
func (x Uint32x8) InterleaveLoGrouped(y Uint32x8) Uint32x8 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLDQ, CPU Feature: AVX512
func (x Uint32x16) InterleaveLoGrouped(y Uint32x16) Uint32x16 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLQDQ, CPU Feature: AVX2
func (x Uint64x4) InterleaveLoGrouped(y Uint64x4) Uint64x4 {
	return x.interleaveLoGrouped(y)
}

// InterleaveLoGrouped interleaves the elements of the low half of each 128-bit subvector of x and y.
//
// Asm: VPUNPCKLQDQ, CPU Feature: AVX512
func (x Uint64x8) InterleaveLoGrouped(y Uint64x8) Uint64x8 {
	return x.interleaveLoGrouped(y)
}
