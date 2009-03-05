// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// implemented in C, in ../../runtime
// perhaps one day the implementations will move here.

// Float32bits returns the IEEE 754 binary representation of f.
func Float32bits(f float32) (b uint32)

// Float32frombits returns the floating point number corresponding
// to the IEEE 754 binary representation b.
func Float32frombits(b uint32) (f float32)

// Float64bits returns the IEEE 754 binary representation of f.
func Float64bits(f float64) (b uint64)

// Float64frombits returns the floating point number corresponding
// the IEEE 754 binary representation b.
func Float64frombits(b uint64) (f float64)

// Frexp breaks f into a normalized fraction
// and an integral power of two.
// It returns frac and exp satisfying f == frac × 2<sup>exp</sup>,
// with the absolute value of frac in the interval [½, 1).
func Frexp(f float64) (frac float64, exp int)

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int32) (f float64)

// IsInf returns whether f is an infinity, according to sign.
// If sign > 0, IsInf returns whether f is positive infinity.
// If sign < 0, IsInf returns whether f is negative infinity.
// If sign == 0, IsInf returns whether f is either infinity.
func IsInf(f float64, sign int) (is bool)

// IsNaN returns whether f is an IEEE 754 ``not-a-number'' value.
func IsNaN(f float64) (is bool)

// Ldexp is the inverse of Frexp.
// It returns frac × 2<sup>exp</sup>.
func Ldexp(frac float64, exp int) (f float64)

// Modf returns integer and fractional floating-point numbers
// that sum to f.
// Integer and frac have the same sign as f.
func Modf(f float64) (integer float64, frac float64)

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() (f float64)
