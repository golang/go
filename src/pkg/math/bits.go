// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

const (
	uvnan    = 0x7FF0000000000001
	uvinf    = 0x7FF0000000000000
	uvneginf = 0xFFF0000000000000
	mask     = 0x7FF
	shift    = 64 - 11 - 1
	bias     = 1022
)

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) float64 {
	var v uint64
	if sign >= 0 {
		v = uvinf
	} else {
		v = uvneginf
	}
	return Float64frombits(v)
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() float64 { return Float64frombits(uvnan) }

// IsNaN returns whether f is an IEEE 754 ``not-a-number'' value.
func IsNaN(f float64) (is bool) {
	// IEEE 754 says that only NaNs satisfy f != f.
	// To avoid the floating-point hardware, could use:
	//	x := Float64bits(f);
	//	return uint32(x>>shift)&mask == mask && x != uvinf && x != uvneginf
	return f != f
}

// IsInf returns whether f is an infinity, according to sign.
// If sign > 0, IsInf returns whether f is positive infinity.
// If sign < 0, IsInf returns whether f is negative infinity.
// If sign == 0, IsInf returns whether f is either infinity.
func IsInf(f float64, sign int) bool {
	// Test for infinity by comparing against maximum float.
	// To avoid the floating-point hardware, could use:
	//	x := Float64bits(f);
	//	return sign >= 0 && x == uvinf || sign <= 0 && x == uvneginf;
	return sign >= 0 && f > MaxFloat64 || sign <= 0 && f < -MaxFloat64
}

// Frexp breaks f into a normalized fraction
// and an integral power of two.
// It returns frac and exp satisfying f == frac × 2<sup>exp</sup>,
// with the absolute value of frac in the interval [½, 1).
func Frexp(f float64) (frac float64, exp int) {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	switch {
	case f == 0:
		return
	case f < -MaxFloat64 || f > MaxFloat64 || f != f: // IsInf(f, 0) || IsNaN(f):
		frac = f
		return
	}
	x := Float64bits(f)
	exp = int((x>>shift)&mask) - bias
	x &^= mask << shift
	x |= bias << shift
	frac = Float64frombits(x)
	return
}

// Ldexp is the inverse of Frexp.
// It returns frac × 2<sup>exp</sup>.
func Ldexp(frac float64, exp int) float64 {
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	// special cases
	if frac != frac { // IsNaN(frac)
		return NaN()
	}
	x := Float64bits(frac)
	exp += int(x>>shift) & mask
	if exp <= 0 {
		return 0 // underflow
	}
	if exp >= mask { // overflow
		if frac < 0 {
			return Inf(-1)
		}
		return Inf(1)
	}
	x &^= mask << shift
	x |= uint64(exp) << shift
	return Float64frombits(x)
}
