// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package math provides basic constants and mathematical functions.
package math

// Mathematical constants.
const (
	E   = 2.71828182845904523536028747135266249775724709369995957496696763 // http://oeis.org/A001113
	Pi  = 3.14159265358979323846264338327950288419716939937510582097494459 // http://oeis.org/A000796
	Phi = 1.61803398874989484820458683436563811772030917980576286213544862 // http://oeis.org/A001622

	Sqrt2   = 1.41421356237309504880168872420969807856967187537694807317667974 // http://oeis.org/A002193
	SqrtE   = 1.64872127070012814684865078781416357165377610071014801157507931 // http://oeis.org/A019774
	SqrtPi  = 1.77245385090551602729816748334114518279754945612238712821380779 // http://oeis.org/A002161
	SqrtPhi = 1.27201964951406896425242246173749149171560804184009624861664038 // http://oeis.org/A139339

	Ln2    = 0.693147180559945309417232121458176568075500134360255254120680009 // http://oeis.org/A002162
	Log2E  = 1 / Ln2
	Ln10   = 2.30258509299404568401799145468436420760110148862877297603332790 // http://oeis.org/A002392
	Log10E = 1 / Ln10
)

// Floating-point limit values.
// Max is the largest finite value representable by the type.
// SmallestNonzero is the smallest positive, non-zero value representable by the type.
const (
	MaxFloat32             = 3.40282346638528859811704183484516925440e+38  // 2**127 * (2**24 - 1) / 2**23
	SmallestNonzeroFloat32 = 1.401298464324817070923729583289916131280e-45 // 1 / 2**(127 - 1 + 23)

	MaxFloat64             = 1.797693134862315708145274237317043567981e+308 // 2**1023 * (2**53 - 1) / 2**52
	SmallestNonzeroFloat64 = 4.940656458412465441765687928682213723651e-324 // 1 / 2**(1023 - 1 + 52)
)

// Integer limit values.
const (
	MaxInt8   = 1<<7 - 1
	MinInt8   = -1 << 7
	MaxInt16  = 1<<15 - 1
	MinInt16  = -1 << 15
	MaxInt32  = 1<<31 - 1
	MinInt32  = -1 << 31
	MaxInt64  = 1<<63 - 1
	MinInt64  = -1 << 63
	MaxUint8  = 1<<8 - 1
	MaxUint16 = 1<<16 - 1
	MaxUint32 = 1<<32 - 1
	MaxUint64 = 1<<64 - 1
)
